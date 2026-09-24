"""
Assembles and saves the full, ready-to-use pruned model (as opposed to just
the trained replace_layer) for all three --replace modes (none / mlp / tf).

For --replace none/tf, the result is a fully standard OPT/Llama checkpoint —
loadable with plain `AutoModelForCausalLM.from_pretrained(dir)`.

For --replace mlp, the replacement is a bare 2-layer MLP with no
attention/layernorm, which doesn't fit the standard OPTDecoderLayer /
LlamaDecoderLayer shape. To keep the saved checkpoint genuinely load-able by
anyone (e.g. after pushing to the Hub), a small self-contained modeling file
is written alongside it and wired up via config.json's `auto_map`, so
`AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` works
out of the box.
"""

import os
import re
import shutil
import time

import torch.nn as nn
from safetensors import SafetensorError
from transformers import AutoConfig, AutoModelForCausalLM


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MLPReplaceLayer(nn.Module):
    """
    Drop-in replacement for a standard decoder layer in a stock HF layer
    stack. Trained as a bare MLP(hidden_states) with no residual/attention/
    layernorm, so forward must reproduce that exactly — the decoder-layer
    kwargs (attention_mask, position_ids, past_key_value, ...) are ignored.
    Still returns the (hidden_states, [attn_weights], [present_key_value])
    tuple shape the parent decoder loop expects, so output_attentions/
    use_cache toggles used during .generate() don't break indexing.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = MLP(hidden_size)

    def forward(self, hidden_states, *args, output_attentions=False, use_cache=False, **kwargs):
        outputs = (self.mlp(hidden_states),)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)
        return outputs


_FAMILY_INFO = {
    "llama": {
        "layer_re": re.compile(r"^model\.layers\.(\d+)\.(.*)"),
        "layer_fmt": "model.layers.{j}.{rest}",
        "hf_module": "llama",
        "base_class": "LlamaForCausalLM",
        "layers_path": "self.model.layers",
    },
    "opt": {
        "layer_re": re.compile(r"^model\.decoder\.layers\.(\d+)\.(.*)"),
        "layer_fmt": "model.decoder.layers.{j}.{rest}",
        "hf_module": "opt",
        "base_class": "OPTForCausalLM",
        "layers_path": "self.model.decoder.layers",
    },
}


def _get_layers(model, model_family):
    return model.model.layers if model_family == "llama" else model.model.decoder.layers


def assemble_pruned_model(pretrained_dict, model_name, model_family, replace, pruning_start_layer, pruning_end_layer, replace_layer_state_dict=None):
    """
    Build the full-size deployable pruned model: the original model with
    layers [pruning_start_layer, pruning_end_layer] collapsed into either
    nothing ("none") or a single replacement layer ("mlp"/"tf"); every other
    layer is copied unchanged and re-indexed.

    `pretrained_dict` should be `AutoModelForCausalLM.from_pretrained(model_name).state_dict()`,
    loaded once by the caller and reused across calls (this function does not
    re-download/re-load the pretrained checkpoint).

    Returns (model, replace_layer_index_or_None).
    """
    info = _FAMILY_INFO[model_family]
    n_pruned = pruning_end_layer - pruning_start_layer + 1
    insert_replacement = replace != "none"

    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = config.num_hidden_layers - n_pruned + (1 if insert_replacement else 0)

    pruned_model = AutoModelForCausalLM.from_config(config)

    replace_layer_index = pruning_start_layer if insert_replacement else None

    if replace == "mlp":
        _get_layers(pruned_model, model_family)[replace_layer_index] = MLPReplaceLayer(config.hidden_size)

    pruned_dict = pruned_model.state_dict()
    layer_re, layer_fmt = info["layer_re"], info["layer_fmt"]

    # Layers removed outright vs. collapsed into one new layer shift tail
    # layers back by a different amount.
    shift = n_pruned - (1 if insert_replacement else 0)
    for key, value in pretrained_dict.items():
        m = layer_re.match(key)
        if m:
            i, rest = int(m.group(1)), m.group(2)
            if pruning_start_layer <= i <= pruning_end_layer:
                continue  # part of the pruned block — handled separately below
            j = i if i < pruning_start_layer else i - shift
            new_key = layer_fmt.format(j=j, rest=rest)
        else:
            new_key = key
        if new_key in pruned_dict:
            pruned_dict[new_key] = value

    if replace == "tf" and replace_layer_state_dict is not None:
        prefix = layer_fmt.format(j=replace_layer_index, rest="")
        for k, v in replace_layer_state_dict.items():
            pruned_dict[prefix + k] = v

    if replace == "mlp" and replace_layer_state_dict is not None:
        prefix = layer_fmt.format(j=replace_layer_index, rest="mlp.")
        for k, v in replace_layer_state_dict.items():
            pruned_dict[prefix + k] = v

    pruned_model.load_state_dict(pruned_dict)
    return pruned_model, replace_layer_index


_REMOTE_CODE_TEMPLATE = '''\
"""
Auto-generated by replace_and_retrain.py — do not hand-edit.

Loads this checkpoint with layer {layer_index} replaced by a lightweight
2-layer MLP (trained to approximate the block of layers this model
originally pruned). Requires trust_remote_code=True.
"""
import torch.nn as nn
from transformers.models.{hf_module}.modeling_{hf_module} import {base_class}


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MLPReplaceLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = MLP(hidden_size)

    def forward(self, hidden_states, *args, output_attentions=False, use_cache=False, **kwargs):
        outputs = (self.mlp(hidden_states),)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)
        return outputs


class {model_class}({base_class}):
    def __init__(self, config):
        super().__init__(config)
        {layers_path}[{layer_index}] = MLPReplaceLayer(config.hidden_size)
'''


def _write_remote_code(output_dir, model_family, replace_layer_index):
    info = _FAMILY_INFO[model_family]
    model_class = f"Pruned{info['base_class']}"
    module_name = f"modeling_pruned_{model_family}"
    source = _REMOTE_CODE_TEMPLATE.format(
        layer_index=replace_layer_index,
        hf_module=info["hf_module"],
        base_class=info["base_class"],
        layers_path=info["layers_path"],
        model_class=model_class,
    )
    with open(os.path.join(output_dir, f"{module_name}.py"), "w") as f:
        f.write(source)
    return f"{module_name}.{model_class}"


def save_pruned_model(model, tokenizer, output_dir, model_family, replace, replace_layer_index):
    """
    Saves `model` as a ready-to-use HF checkpoint at `output_dir`. For
    --replace mlp, also ships the custom layer's modeling code and wires up
    config.json's auto_map so AutoModelForCausalLM.from_pretrained(...,
    trust_remote_code=True) works for anyone who downloads it — plain
    from_pretrained (no trust_remote_code) is NOT enough for this mode,
    since the MLP replacement layer isn't a stock decoder layer.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        if replace == "mlp":
            target = _write_remote_code(output_dir, model_family, replace_layer_index)
            model.config.auto_map = {"AutoModelForCausalLM": target}

        try:
            model.save_pretrained(output_dir)
            break
        except SafetensorError:
            # Transient I/O errors (e.g. ENOENT from a network filesystem
            # hiccup mid-write) have been observed here; retry from a freshly
            # recreated output_dir rather than losing hours of training.
            if attempt == max_attempts:
                raise
            time.sleep(5)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def replace_best_checkpoint(tmp_dir, final_dir):
    """
    Atomically (best-effort) swap `tmp_dir` in as the new `final_dir`,
    removing whatever previous best checkpoint was there. Used to keep only
    the single best-eval-loss checkpoint on disk.
    """
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.rename(tmp_dir, final_dir)
