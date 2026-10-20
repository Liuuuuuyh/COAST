import torch
import torch.nn.functional as F
import math
from typing import Tuple, Callable, List, Optional, Union
from numpy import mean

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaModel, 
    _prepare_4d_causal_attention_mask, 
    _prepare_4d_causal_attention_mask_for_sdpa, 
    Cache, 
    DynamicCache
)
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

import torch
import torch.nn.functional as F
import math
from contextlib import contextmanager

# Temporary storage for attention weights captured from SDPA.
_captured_attention = None

@contextmanager
def capture_sdpa_attention():
    """
    通过劫持底层 F.scaled_dot_product_attention，
    在不触发 HF Eager Fallback 的前提下，捞出 Q 和 K 并计算 Attention Weights。
    """
    global _captured_attention
    _captured_attention = None
    original_sdpa = F.scaled_dot_product_attention

    def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        global _captured_attention
        
        # Capture Q and K, then explicitly compute the attention matrix.
        d_k = query.size(-1)
        # query, key shape: [batch, heads, seq_len, head_dim]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if attn_mask is not None:
            # The Hugging Face attention mask may require broadcasting or already have the target shape.
            scores = scores + attn_mask
            
        _captured_attention = torch.softmax(scores, dim=-1)
        
        # Return the original SDPA output to preserve the optimized forward pass.
        return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

    try:
        # 替换底层的 sdpa
        F.scaled_dot_product_attention = hooked_sdpa
        yield
    finally:
        # 退出上下文时，务必恢复原函数
        F.scaled_dot_product_attention = original_sdpa


# COAST: Contrastive Adaptive Semantic Token Pruning (progressive inference).
class COASTLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.last_attention = None

        # Optional efficiency metrics.
        self.all_FLOPs = 0.0
        self.num_forward = 0
        self.enable_tflops = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        else:
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        # print("self._use_flash_attention_2:", self._use_flash_attention_2)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_token = []
        
        # ========================================================================
        # COAST pruning schedule and routing hyperparameters.
        # ========================================================================
        RUN_COAST = True
        
        coast_config = {
            # optional: "scalar", "maxsim", "contrastive"
            "routing_mode": "contrastive", 
            "image_token_start": 35,
            "image_token_end": 611,
            "keep_tokens_by_layer": {
                2: 214,  # 214, 352
                12: 78,  # 78, 144
                22: 6,
                28: 4
            },
            "anchor_fraction": 0.30,
            "negative_fraction": 0.05, 
            "context_fraction_min": 0.05,
            "context_fraction_max": 0.60,
            "attention_mass_threshold": 1e-4,
            "max_anchor_fraction_of_budget": 0.8,
        }

        image_token_start = coast_config["image_token_start"] 
        img_seq_len = coast_config["image_token_end"] - coast_config["image_token_start"] 

        all_indices_mask = torch.ones(img_seq_len, dtype=torch.bool, device=hidden_states.device)
        # output_attentions = True
        _eff_output_attentions = output_attentions  # may be upgraded to True inside the loop

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__, hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache,
                )
            else:
                current_layer_idx = decoder_layer.self_attn.layer_idx
                
                # ========================================================================
                # Progressive COAST pruning logic.
                # ========================================================================
                if RUN_COAST and current_layer_idx in coast_config["keep_tokens_by_layer"] and seq_length > 1:
                    device = hidden_states.device
                    
                    current_img_end = image_token_start + img_seq_len
                    current_text_start = current_img_end
                    target_keep_num = coast_config["keep_tokens_by_layer"][current_layer_idx]
                    
                    if target_keep_num < img_seq_len:
                        if target_keep_num > 0 :
                            # 1. Extract positive and negative attention anchors.
                            text_to_image_attention = self.last_attention.mean(dim=1)[0][current_text_start:, image_token_start:current_img_end]
                            image_relevance_score = text_to_image_attention.max(dim=0)[0] 
                            last_query_attention = self.last_attention.mean(dim=1)[0][-1][image_token_start:current_img_end]

                            max_anchor_allowance = int(target_keep_num * coast_config["max_anchor_fraction_of_budget"])
                            theoretical_anchor = int(img_seq_len * coast_config["anchor_fraction"])
                            top_k_num = min(theoretical_anchor, max_anchor_allowance)
                            top_k_num_bottom = min(int(img_seq_len * coast_config["negative_fraction"]), img_seq_len - target_keep_num)
                            
                            # 2. Select top-attention semantic anchors.
                            anchor_indices = last_query_attention.topk(top_k_num).indices 
                            negative_anchor_indices = image_relevance_score.topk(top_k_num_bottom, largest=False).indices                 

                            # Compute normalized attention entropy.

                            scores_f32 = image_relevance_score.to(torch.float32)
                            total_img_attn = scores_f32.sum()
                            if total_img_attn < coast_config["attention_mass_threshold"]:
                                normalized_attention_entropy = 1.0  
                            else:
                                attn_probs = scores_f32 / total_img_attn
                                entropy = torch.special.entr(attn_probs).sum()
                                normalized_attention_entropy = (entropy / math.log(img_seq_len)).item()
                                normalized_attention_entropy = max(0.0, min(1.0, normalized_attention_entropy))

                            context_fraction_min = coast_config["context_fraction_min"]
                            context_fraction_max = coast_config["context_fraction_max"]
                            context_retention_fraction = context_fraction_min + (context_fraction_max - context_fraction_min) * normalized_attention_entropy
                            
                            remaining_token_budget = target_keep_num - top_k_num
                            context_token_count = int(remaining_token_budget * context_retention_fraction)
                            semantic_token_count = max(0, remaining_token_budget - context_token_count)

                            # 3. Build the remaining candidate pool.
                            all_indices = torch.arange(img_seq_len, device=device)
                            all_indices_mask = torch.ones(img_seq_len, dtype=torch.bool, device=device)
                            all_indices_mask[anchor_indices] = False
                            candidate_indices = all_indices[all_indices_mask]

                            image_hs = hidden_states[0, image_token_start:current_img_end, :] 
                            semantic_anchors = image_hs[anchor_indices] 
                            negative_anchors = image_hs[negative_anchor_indices] 
                            candidate_tokens = image_hs[candidate_indices] 
                            candidate_attention_scores = last_query_attention[candidate_indices] # Used by the scalar ablation.

                         # 4. Fine-grained contrastive routing (also supports ablations).
                            def select_by_contrastive_score(anchor_hs, negative_hs, candidate_hs, other_idx, n1, n2, mode, candidate_attn):
                                device = candidate_hs.device
                                
                                if mode == "scalar":
                                    # Scalar ablation: rank candidates by previous-layer attention.
                                    scores = candidate_attn
                                else:
                                    # Normalize token features.
                                    anchor_norm = F.normalize(anchor_hs, p=2, dim=1)       
                                    candidate_norm = F.normalize(candidate_hs, p=2, dim=1) 
                                    
                                    # MaxSim pull toward at least one positive anchor.
                                    sim_anchor = torch.mm(candidate_norm, anchor_norm.t()) 
                                    anchor_score = sim_anchor.max(dim=1)[0] 
                                    
                                    if mode == "maxsim":
                                        # MaxSim ablation: use positive-anchor similarity only.
                                        scores = anchor_score
                                    elif mode == "contrastive":
                                        # COAST: positive MaxSim minus mean negative similarity.
                                        negative_norm = F.normalize(negative_hs, p=2, dim=1)   
                                        sim_negative = torch.mm(candidate_norm, negative_norm.t()) 
                                        negative_score = sim_negative.mean(dim=1) 
                                        scores = anchor_score - negative_score
                                    else:
                                        raise ValueError(f"Unknown routing mode: {mode}")

                                # A. Select semantic evidence (top routing scores).
                                if n1 > 0:
                                    n1_idx = other_idx[scores.topk(n1, largest=True).indices]
                                else:
                                    n1_idx = torch.tensor([], dtype=other_idx.dtype, device=device)

                                # B. Select complementary context (bottom routing scores).
                                # Keep largest=False to preserve the original logic and bipolar retention behavior.
                                if n2 > 0:
                                    n2_idx = other_idx[scores.topk(n2, largest=False).indices]
                                else:
                                    n2_idx = torch.tensor([], dtype=other_idx.dtype, device=device)
                                    
                                return n1_idx, n2_idx


                            # Route candidates using the configured mode and attention scores for the scalar ablation.
                            n1_idx, n2_idx = select_by_contrastive_score(
                                semantic_anchors, negative_anchors, candidate_tokens, 
                                candidate_indices, n1=semantic_token_count, n2=context_token_count,
                                mode=coast_config.get("routing_mode", "contrastive"), # Default to the full COAST method.
                                candidate_attn=candidate_attention_scores
                            )
                            retained_image_indices = torch.cat([anchor_indices, n1_idx, n2_idx])
                            
                            # Rebuild the sequence with retained image tokens.
                            retained_image_indices = retained_image_indices.sort().values
                            img_seq_len = retained_image_indices.shape[0] # Update the image-token count for the next pruning stage.
                            retained_sequence_indices = torch.cat((torch.arange(image_token_start, device=device), retained_image_indices + image_token_start, torch.arange(current_img_end, seq_length, device=device)))

                        else:
                            img_seq_len = 0
                            retained_sequence_indices = torch.cat((torch.arange(image_token_start, device=device), torch.arange(current_img_end, seq_length, device=device)))
                        
                        hidden_states = hidden_states[:, retained_sequence_indices, :]
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        retained_sequence_indices = retained_sequence_indices.to(position_ids.device)
                        position_ids = position_ids[:, retained_sequence_indices]
                        seq_length = hidden_states.shape[1]

                # ========================================================================
                # Capture attention for the next pruning stage in the same forward pass.
                # ========================================================================
                is_next_prune_stage = (current_layer_idx + 1) in coast_config["keep_tokens_by_layer"]


                _need_coast_attn = RUN_COAST and is_next_prune_stage and seq_length > 1
                _need_attn = _need_coast_attn and seq_length > 1
                # Do not force output_attentions=True, which would trigger the eager fallback.
                _eff_output_attentions = output_attentions # Keep the default False value so Hugging Face can use SDPA.

                if seq_length > 1:
                    all_token.append(img_seq_len)

                # Wrap the current layer forward pass in the attention-capture context manager.
                if _need_attn:
                    with capture_sdpa_attention():
                        layer_outputs = decoder_layer(
                            hidden_states, 
                            attention_mask=attention_mask, 
                            position_ids=position_ids, 
                            past_key_value=past_key_values, 
                            output_attentions=_eff_output_attentions, 
                            use_cache=use_cache,
                        )
                    # Retrieve the captured attention weights from the module-level variable.
                    global _captured_attention
                    self.last_attention = _captured_attention 
                else:
                    # Run the standard forward path when attention capture is unnecessary.
                    layer_outputs = decoder_layer(
                        hidden_states, 
                        attention_mask=attention_mask, 
                        position_ids=position_ids, 
                        past_key_value=past_key_values, 
                        output_attentions=_eff_output_attentions, 
                        use_cache=use_cache,
                    )
    
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if _eff_output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if seq_length > 1 and self.enable_tflops:
                n = hidden_states.shape[1]
                d = hidden_states.shape[2]
                m = self.config.intermediate_size

                self.all_FLOPs += 4 * n * (d**2) + 2 * (n**2) * d + 3 * n * d * m

        if seq_length > 1:
            method_name = "COAST-Progressive" if RUN_COAST else "Dense"
            print(f"[Run Info] Method: {method_name} | Stages: {list(coast_config['keep_tokens_by_layer'].keys())} | Actual Avg Image Tokens/Layer: {mean(all_token):.1f}")
            print(f"[Run Info] Max Anchor Allowance Ratio: {coast_config['max_anchor_fraction_of_budget']}")

            if self.enable_tflops:
                self.num_forward += 1
                average_tflops_per_forward = (self.all_FLOPs / self.num_forward) * 1e-12
                print(f"Self_num_forward: {self.num_forward}")
                print(f"[Efficiency INFO] TFLOPs: {average_tflops_per_forward:.2f}")



        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns,
        )