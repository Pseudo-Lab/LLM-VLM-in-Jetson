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
# None: Just prune the layers, no replacement layer.


# ─────────────────────────────────────────────────────────────────────────────
# Custom OPT classes
# ─────────────────────────────────────────────────────────────────────────────

class CustomOPTDecoder(OPTDecoder):
    """
    OPTDecoder that runs the truncated model with no replacement layer.
    Layers [start_pruned_layer … last_pruned_layer] are simply absent from
    config.num_hidden_layers; the decoder forwards normally through the
    remaining layers and returns their last hidden state.
    """

    def __init__(self, config: OPTConfig, start_pruned_layer: int = BEST_LAYER + 1):
        super().__init__(config)
        self.start_pruned_layer = start_pruned_layer

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

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CustomOPTModel(OPTModel):
    """OPTModel that swaps in CustomOPTDecoder."""

    def __init__(self, config: OPTConfig, start_pruned_layer: int = BEST_LAYER + 1):
        super().__init__(config)
        self.decoder = CustomOPTDecoder(config, start_pruned_layer=start_pruned_layer)
