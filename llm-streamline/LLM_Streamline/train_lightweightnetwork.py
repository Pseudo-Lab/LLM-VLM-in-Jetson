import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from itertools import chain
import gc

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup
from LLM_Streamline.get_cosine import get_cosine_similarity
from LLM_Streamline.get_train_data import get_data

class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
    
    def __getitem__(self,index):
        return self.input_data[index].clone().detach(), self.output_data[index].clone().detach()

    def __len__(self):
        return len(self.input_data)

def process_datasets(dataset, train_num_data, tokenizer):
    '''
    We divided the proportions of RedPajamaCommonCrawl, RedPajamaArXiv, 
    and RedPajamaBook by a normalization value because the data length 
    in these domains is higher than in other domains.
    '''
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200
    }
    
    filtered_datasets = {
        name: dataset.filter(lambda x: x['meta'] == {"redpajama_set_name": f"{name}"})
        for name in proportions.keys()
    }
    
    test_datasets = []
    train_datasets = []

    for name, proportion in proportions.items():   
        split = filtered_datasets[name].train_test_split(test_size=(3000 * proportion) / len(filtered_datasets[name]))
        test_datasets.append(split['test'])
        train_split = split['train'].train_test_split(test_size=1-(train_num_data * proportion) / len(split['train']))['train']
        train_datasets.append(train_split)

    dataset, test_dataset = concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    
    block_size = 2048
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    dataset = dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    
    return dataset, test_dataset

def valid_model(model, model_name, test_dataloader, device):        # FIX: add arg: model_name
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = []
    
    with torch.no_grad():
        for input_data, output_data in tqdm(test_dataloader):
            input_data = input_data.to(device)
            output_data = output_data.to(device)

            # FIX: position_ids is only valid with LLaMA
            if "llama" in model_name or "Llama" in model_name:
                position_ids = torch.arange(0, 2048).repeat(input_data.shape[0], 1).to(device)
                pred = model(hidden_states=input_data, position_ids=position_ids)
            else:
                pred = model(hidden_states=input_data)
            
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn(pred, output_data)
            total_loss.append(loss.item())
    
    return sum(total_loss) / len(total_loss)

def init_layer(replace_model_name, config, device, model, best_layer, layer_intervals):
    
    def init_opt_layer(model_dict, layer_dict, layer_idx):
        prefix = 'model.decoder.layers.{}'.format(layer_idx)
        mappings = {
            'self_attn.q_proj': ['weight', 'bias'],
            'self_attn.k_proj': ['weight', 'bias'],
            'self_attn.v_proj': ['weight', 'bias'],
            'self_attn.out_proj': ['weight', 'bias'],
            'self_attn_layer_norm': ['weight', 'bias'],
            'fc1': ['weight', 'bias'],
            'fc2': ['weight', 'bias'],
            'final_layer_norm': ['weight', 'bias']
        }
        
        for module, params in mappings.items():
            for param in params:
                layer_dict[f'{module}.{param}'] = model_dict[f'{prefix}.{module}.{param}']
        
        return layer_dict

    def init_llama_layer(model_dict, layer_dict, layer_idx):
        prefix = 'model.layers.{}'.format(layer_idx)
        mappings = {
            'self_attn.q_proj.weight': f'{prefix}.self_attn.q_proj.weight',
            'self_attn.k_proj.weight': f'{prefix}.self_attn.k_proj.weight',
            'self_attn.v_proj.weight': f'{prefix}.self_attn.v_proj.weight',
            'self_attn.o_proj.weight': f'{prefix}.self_attn.o_proj.weight',
            'mlp.gate_proj.weight': f'{prefix}.mlp.gate_proj.weight',
            'mlp.up_proj.weight': f'{prefix}.mlp.up_proj.weight',
            'mlp.down_proj.weight': f'{prefix}.mlp.down_proj.weight',
            'input_layernorm.weight': f'{prefix}.input_layernorm.weight',
            'post_attention_layernorm.weight': f'{prefix}.post_attention_layernorm.weight'
        }
        
        for target, source in mappings.items():
            layer_dict[target] = model_dict[source]
            
        return layer_dict

    if replace_model_name == "opt_layer":
        replace_model = OPTDecoderLayer(config).to(device)
    elif replace_model_name == "llama_layer":
        replace_model = LlamaDecoderLayer(config, 0).to(device)

    model_dict = model.state_dict()
    layer_dict = replace_model.state_dict()
    
    if replace_model_name == "opt_layer":       # FIX: typo:opt_Layer -> opt_layer
        layer_dict = init_opt_layer(model_dict, layer_dict, best_layer)
    elif replace_model_name == "llama_layer": 
        layer_dict = init_llama_layer(model_dict, layer_dict, best_layer)

    replace_model.load_state_dict(layer_dict)
    
    return replace_model
        
def lightweight_model_train(model, tokenizer, device, layer_intervals, num_layer, cosine_num_data, 
              train_num_data, batch_size, epochs, lr, min_lr, wd, config, model_name, gradient_accumulation_step):
    
    dataset = load_dataset('DKYoon/SlimPajama-6B')['train']
    dataset, test_dataset = process_datasets(dataset, train_num_data, tokenizer)
    
    best_layer = get_cosine_similarity(model, dataset, cosine_num_data, device, 
                                     layer_intervals, num_layer)

    if "opt" in model_name or "OPT" in model_name:
        replace_model = init_layer("opt_layer", config, device, model, best_layer, layer_intervals)
    elif "llama" in model_name or "Llama" in model_name:
        replace_model = init_layer("llama_layer", config, device, model, best_layer, layer_intervals)
    else:
        raise ValueError(f"Unknown model type: {replace_model}")
    
    def prepare_dataset_for_training(dataset, model, device):
        input_list, output_list = get_data(model, dataset, device, layer_intervals, best_layer, tokenizer, batch_size)
        return CustomDataset(input_list, output_list)
    
    test_dataset = prepare_dataset_for_training(test_dataset, model, device)
    train_dataset = prepare_dataset_for_training(dataset, model, device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(replace_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader)*epochs*0.01*0.5,
        num_training_steps=len(train_dataloader)*epochs*0.5,
        max_learning_rate=lr,
        min_learning_rate=min_lr,
    )

    best_loss = valid_model(replace_model, model_name, test_dataloader, device)     # FIX: add arg: model_name
    print("Before training, Validation_Loss:", best_loss)
    print("Starting training...")
    best_state_dict = None
    
    for epoch in range(epochs):
        replace_model.train()
        step = 0
        optimizer.zero_grad()
        
        for input_data, output_data in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            # FIX: position_ids is only valid with LLaMA
            if "llama" in model_name or "Llama" in model_name:
                position_ids = torch.arange(0, 2048).repeat(input_data.shape[0], 1).to(device)
                output = replace_model(hidden_states=input_data, position_ids=position_ids)
            else:
                output = replace_model(hidden_states=input_data)
            output = output[0] if isinstance(output, tuple) else output
            
            loss = criterion(output, output_data)
            loss /= gradient_accumulation_step
            loss.backward()

            if (step + 1) % gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(replace_model.parameters(), max_norm=5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            del input_data, output_data, output, loss
            torch.cuda.empty_cache()

            step += 1
        
        valid_loss = valid_model(replace_model, model_name, test_dataloader, device)    # FIX: add arg: model_name
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = replace_model.state_dict()
            
        print(f"Epoch: {epoch}, Validation Loss: {valid_loss:.6f}")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    replace_model.load_state_dict(best_state_dict)
    
    return replace_model, best_layer





