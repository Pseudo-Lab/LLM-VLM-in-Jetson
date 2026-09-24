from transformers import (
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer, 
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn

from datasets import load_dataset, load_from_disk, concatenate_datasets
from args import TrainingArguments, ModelArguments
from LLM_Streamline.train_lightweightnetwork import lightweight_model_train

def replace_lightweight_network(model, lightweight_network, pruned_model, best_layer, layer_intervals, model_name, num_layers):
    pruned_layers = [i for i in range(best_layer+1, best_layer+layer_intervals)]

    replace_weight = lightweight_network.state_dict()
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
            elif i == best_layer:
                pruned_weight['model.layers.{}.self_attn.q_proj.weight'.format(j)] = replace_weight['self_attn.q_proj.weight']
                pruned_weight['model.layers.{}.self_attn.k_proj.weight'.format(j)] = replace_weight['self_attn.k_proj.weight']
                pruned_weight['model.layers.{}.self_attn.v_proj.weight'.format(j)] = replace_weight['self_attn.v_proj.weight']
                pruned_weight['model.layers.{}.self_attn.o_proj.weight'.format(j)] = replace_weight['self_attn.o_proj.weight']
                pruned_weight['model.layers.{}.mlp.gate_proj.weight'.format(j)] = replace_weight['mlp.gate_proj.weight']
                pruned_weight['model.layers.{}.mlp.up_proj.weight'.format(j)] = replace_weight['mlp.up_proj.weight']
                pruned_weight['model.layers.{}.mlp.down_proj.weight'.format(j)] = replace_weight['mlp.down_proj.weight']
                pruned_weight['model.layers.{}.input_layernorm.weight'.format(j)] = replace_weight['input_layernorm.weight']
                pruned_weight['model.layers.{}.post_attention_layernorm.weight'.format(j)] = replace_weight['post_attention_layernorm.weight'] 
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
            elif i == best_layer:
                pruned_weight['model.decoder.layers.{}.self_attn.q_proj.weight'.format(j)] = replace_weight['self_attn.q_proj.weight']
                pruned_weight['model.decoder.layers.{}.self_attn.k_proj.weight'.format(j)] = replace_weight['self_attn.k_proj.weight']
                pruned_weight['model.decoder.layers.{}.self_attn.v_proj.weight'.format(j)] = replace_weight['self_attn.v_proj.weight']
                pruned_weight['model.decoder.layers.{}.self_attn.out_proj.weight'.format(j)] = replace_weight['self_attn.out_proj.weight']
                pruned_weight['model.decoder.layers.{}.self_attn.q_proj.bias'.format(j)] = replace_weight['self_attn.q_proj.bias']
                pruned_weight['model.decoder.layers.{}.self_attn.k_proj.bias'.format(j)] = replace_weight['self_attn.k_proj.bias']
                pruned_weight['model.decoder.layers.{}.self_attn.v_proj.bias'.format(j)] = replace_weight['self_attn.v_proj.bias']
                pruned_weight['model.decoder.layers.{}.self_attn.out_proj.bias'.format(j)] = replace_weight['self_attn.out_proj.bias']
                pruned_weight['model.decoder.layers.{}.self_attn_layer_norm.weight'.format(j)] = replace_weight['self_attn_layer_norm.weight']
                pruned_weight['model.decoder.layers.{}.self_attn_layer_norm.bias'.format(j)] = replace_weight['self_attn_layer_norm.bias']
                pruned_weight['model.decoder.layers.{}.fc1.weight'.format(j)] = replace_weight['fc1.weight']
                pruned_weight['model.decoder.layers.{}.fc1.bias'.format(j)] = replace_weight['fc1.bias']
                pruned_weight['model.decoder.layers.{}.fc2.weight'.format(j)] = replace_weight['fc2.weight']
                pruned_weight['model.decoder.layers.{}.fc2.bias'.format(j)] = replace_weight['fc2.bias']
                pruned_weight['model.decoder.layers.{}.final_layer_norm.weight'.format(j)] = replace_weight['final_layer_norm.weight']
                pruned_weight['model.decoder.layers.{}.final_layer_norm.bias'.format(j)] = replace_weight['final_layer_norm.bias']     
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

def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    return args, training_args

def run():
    args, training_args = parse_hf_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    lightweight_network, best_layer = lightweight_model_train(model, tokenizer, 'cuda', training_args.layer_intervals+1, config.num_hidden_layers, training_args.cosine_num_data, training_args.train_num_data, training_args.batch_size, training_args.epoches, training_args.lr, training_args.min_lr, training_args.wd, config, args.model_name, training_args.gradient_accumulation_step)

    config.num_hidden_layers -=  training_args.layer_intervals
    pruned_model = AutoModelForCausalLM.from_config(config)
    pruned_model = replace_lightweight_network(model, lightweight_network, pruned_model, best_layer, training_args.layer_intervals+1, args.model_name, config.num_hidden_layers+training_args.layer_intervals)
    
    pruned_model.save_pretrained(training_args.output_dir)      # FIX: Set the output directory manually
    