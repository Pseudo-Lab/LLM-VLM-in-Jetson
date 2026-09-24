from transformers import (
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer, 
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from args import TrainingArguments, ModelArguments
import torch

from accelerate import Accelerator
import deepspeed
from tqdm import tqdm

from LLM_Streamline.get_cosine import get_cosine_similarity
from LLM_Streamline.train_lightweightnetwork import process_datasets
from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup


def prune_model(model, pruned_model, best_layer, layer_intervals, model_name, num_layers):
    pruned_layers = [i for i in range(best_layer+1, best_layer+layer_intervals)]

    pruned_weight = pruned_model.state_dict()
    weight = model.state_dict()

    if "llama" in model_name or "Llama" in model_name:
        pruned_weight['model.norm.weight'] = weight['model.norm.weight']
        pruned_weight['model.embed_tokens.weight'] = weight['model.embed_tokens.weight']
        pruned_weight['lm_head.weight'] = weight['lm_head.weight']

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                continue
            else:
                pruned_weight['model.layers.{}.self_attn.q_proj.weight'.format(j)] = weight['model.layers.{}.self_attn.q_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.self_attn.k_proj.weight'.format(j)] = weight['model.layers.{}.self_attn.k_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.self_attn.v_proj.weight'.format(j)] = weight['model.layers.{}.self_attn.v_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.self_attn.o_proj.weight'.format(j)] = weight['model.layers.{}.self_attn.o_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.mlp.gate_proj.weight'.format(j)] = weight['model.layers.{}.mlp.gate_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.mlp.up_proj.weight'.format(j)] = weight['model.layers.{}.mlp.up_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.mlp.down_proj.weight'.format(j)] = weight['model.layers.{}.mlp.down_proj.weight'.format(i)]
                pruned_weight['model.layers.{}.input_layernorm.weight'.format(j)] = weight['model.layers.{}.input_layernorm.weight'.format(i)]
                pruned_weight['model.layers.{}.post_attention_layernorm.weight'.format(j)] = weight['model.layers.{}.post_attention_layernorm.weight'.format(i)]       
            j += 1
    
    elif "opt" in model_name or "OPT" in model_name:
        pruned_weight['model.decoder.embed_tokens.weight'] = weight['model.decoder.embed_tokens.weight']
        pruned_weight['model.decoder.embed_positions.weight'] = weight['model.decoder.embed_positions.weight']
        pruned_weight['model.decoder.final_layer_norm.weight'] = weight['model.decoder.final_layer_norm.weight']
        pruned_weight['model.decoder.final_layer_norm.bias'] = weight['model.decoder.final_layer_norm.bias']
        pruned_weight['lm_head.weight'] = weight['lm_head.weight']

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                continue
            else:
                pruned_weight['model.decoder.layers.{}.self_attn.q_proj.weight'.format(j)] = weight['model.decoder.layers.{}.self_attn.q_proj.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.k_proj.weight'.format(j)] = weight['model.decoder.layers.{}.self_attn.k_proj.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.v_proj.weight'.format(j)] = weight['model.decoder.layers.{}.self_attn.v_proj.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.out_proj.weight'.format(j)] = weight['model.decoder.layers.{}.self_attn.out_proj.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.q_proj.bias'.format(j)] = weight['model.decoder.layers.{}.self_attn.q_proj.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.k_proj.bias'.format(j)] = weight['model.decoder.layers.{}.self_attn.k_proj.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.v_proj.bias'.format(j)] = weight['model.decoder.layers.{}.self_attn.v_proj.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn.out_proj.bias'.format(j)] = weight['model.decoder.layers.{}.self_attn.out_proj.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn_layer_norm.weight'.format(j)] = weight['model.decoder.layers.{}.self_attn_layer_norm.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.self_attn_layer_norm.bias'.format(j)] = weight['model.decoder.layers.{}.self_attn_layer_norm.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.fc1.weight'.format(j)] = weight['model.decoder.layers.{}.fc1.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.fc1.bias'.format(j)] = weight['model.decoder.layers.{}.fc1.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.fc2.weight'.format(j)] = weight['model.decoder.layers.{}.fc2.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.fc2.bias'.format(j)] = weight['model.decoder.layers.{}.fc2.bias'.format(i)]
                pruned_weight['model.decoder.layers.{}.final_layer_norm.weight'.format(j)] = weight['model.decoder.layers.{}.final_layer_norm.weight'.format(i)]
                pruned_weight['model.decoder.layers.{}.final_layer_norm.bias'.format(j)] = weight['model.decoder.layers.{}.final_layer_norm.bias'.format(i)]     
            j += 1
    
    pruned_model.load_state_dict(pruned_weight)  
    return pruned_model

def valid_model(model, eval_dataloader, accelerator):
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss.append(accelerator.gather_for_metrics(loss).cpu())
            del outputs
            
    total_loss = torch.cat(total_loss)
    eval_loss = torch.mean(total_loss)
    return eval_loss
    
def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    return args, training_args

def run():
    args, training_args = parse_hf_args()
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=training_args.gradient_accumulation_step)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('DKYoon/SlimPajama-6B')['train']
    dataset, test_dataset = process_datasets(dataset, training_args.train_num_data, tokenizer)
       
    best_layer = get_cosine_similarity(model, dataset, training_args.cosine_num_data, 'cuda', 
                                     training_args.layer_intervals+1, config.num_hidden_layers)
    
    config.num_hidden_layers -=  training_args.layer_intervals
    pruned_model = AutoModelForCausalLM.from_config(config)

    pruned_model = prune_model(model, pruned_model, best_layer, training_args.layer_intervals+1, args.model_name, config.num_hidden_layers+training_args.layer_intervals)
    
    for name,p in pruned_model.named_parameters():
        if "layers.{}".format(best_layer) in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.batch_size)
    eval_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=training_args.batch_size)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pruned_model.parameters()), lr = training_args.lr, weight_decay = training_args.wd, betas=(0.9, 0.95))
    
    scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(train_dataloader)*training_args.epoches*0.01,
            num_training_steps=len(train_dataloader)*training_args.epoches*0.5,
            max_learning_rate=training_args.lr,
            min_learning_rate=training_args.min_lr,
        )

    train_dataloader, eval_dataloader, pruned_model, optimizer = accelerator.prepare(
            train_dataloader, eval_dataloader, pruned_model, optimizer
        )
 
    best_loss = valid_model(pruned_model, eval_dataloader, accelerator)

    print("Before training, Validation_Loss:", best_loss)
    print("Starting training...")

    for epoch in range(training_args.epoches):
        pruned_model.train()
        for step, batch in tqdm(enumerate(train_dataloader)):
            with accelerator.accumulate(pruned_model):
                outputs = pruned_model(**batch)     
                loss = outputs.loss
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
        valid_loss = valid_model(pruned_model, eval_dataloader, accelerator)
         
        if valid_loss < best_loss:
            best_loss = valid_loss          
            pruned_model.module.save_pretrained(training_args.output_dir)        # FIX: Set the output directory manually
            
        print(f"Epoch: {epoch}, Validation Loss:", best_loss)

    