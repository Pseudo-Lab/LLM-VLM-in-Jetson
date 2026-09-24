from transformers import DataCollatorForLanguageModeling
import torch
from tqdm import tqdm
import gc
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

@torch.no_grad() 
def get_data(model, dataset, device, layer_intervals, best_layer, tokenizer, batch_size):
    input_list = []
    output_list = []

    accelerator = Accelerator()
    device = accelerator.device

    model = model.to(device)
    model.eval()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )
    dataloader = accelerator.prepare(dataloader)

    try:
        for step, batch in tqdm(enumerate(dataloader)):
        
            hidden_states = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True
            ).hidden_states

            input_tensor = hidden_states[best_layer].cpu()
            output_tensor = hidden_states[best_layer + layer_intervals].cpu()

            input_list += torch.unbind(input_tensor, dim=0)
            output_list += torch.unbind(output_tensor, dim=0)

            del hidden_states

    finally:
        accelerator.free_memory()
        torch.cuda.empty_cache()
        model.cpu()
        del model
        gc.collect()
    
    return input_list, output_list