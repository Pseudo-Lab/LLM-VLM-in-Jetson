from transformers.models.opt.modeling_opt import OPTDecoder, OPTDecoderLayer, OPTModel
from transformers import OPTConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Constants – set both values from the cosine similarity step before running.
# ─────────────────────────────────────────────────────────────────────────────

# 0-indexed position of the layer whose weights initialise replace_layer.
# Run the cosine similarity analysis first (mseloss_entry.py with your OPT model)
# to find this.  Example: if cosine similarity identifies the (12, 17) pair as
# most similar, set BEST_LAYER = 12.
BEST_LAYER = 2

# 0-indexed position of the last layer that will be pruned.
# replace_layer is trained to approximate layers [BEST_LAYER+1 … LAST_PRUNED_LAYER].
# config.num_hidden_layers in train.py is set to LAST_PRUNED_LAYER + 1.
LAST_PRUNED_LAYER = 11


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight replacement layer
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Custom OPT classes
# ─────────────────────────────────────────────────────────────────────────────

class CustomOPTDecoder(OPTDecoder):
    """
    OPTDecoder with an embedded lightweight replace_layer for memory-efficient
    MSE training.

    The standard parent forward() is called normally (no hidden-state caching
    to disk/RAM).  Two minimal forward hooks intercept:

      1. layers[BEST_LAYER] output  →  input to replace_layer
      2. layers[-1] output          →  MSE training target (raw, before final_layer_norm)

    Returning both through last_hidden_state keeps train.py simple.
    """

    def __init__(self, config: OPTConfig, start_pruned_layer: int = BEST_LAYER + 1):
        super().__init__(config)
        self.start_pruned_layer = start_pruned_layer
        self.replace_layer = MLP(config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        captured = {}

        def _hook_best_layer(module, inp, out):
            # Hidden states immediately after BEST_LAYER — fed into replace_layer.
            captured["best_layer_out"] = out[0]

        def _hook_last_layer(module, inp, out):
            # Raw hidden states after the final training layer (before final_layer_norm).
            # This is the MSE target: what replace_layer must learn to approximate.
            captured["last_layer_out"] = out[0]

        h_best = self.layers[self.start_pruned_layer - 1].register_forward_hook(_hook_best_layer)
        h_last = self.layers[-1].register_forward_hook(_hook_last_layer)

        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        finally:
            h_best.remove()
            h_last.remove()

        # ── Replace-layer forward ──────────────────────────────────────────
        hs = captured["best_layer_out"]         # (bsz, seq_len, hidden_size)
        replace_hidden_states = self.replace_layer(hs)

        # ── Pack return value ──────────────────────────────────────────────
        # last_hidden_state[0] = frozen target  (raw output of last training layer)
        # last_hidden_state[1] = replace_layer prediction
        # train.py unpacks these to compute MSE loss.
        return BaseModelOutputWithPast(
            last_hidden_state=[captured["last_layer_out"], replace_hidden_states],
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CustomOPTModel(OPTModel):
    """OPTModel that swaps in CustomOPTDecoder."""

    def __init__(self, config: OPTConfig, start_pruned_layer: int = BEST_LAYER + 1):
        super().__init__(config)
        self.decoder = CustomOPTDecoder(config, start_pruned_layer=start_pruned_layer)
