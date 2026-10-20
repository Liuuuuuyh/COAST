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

# 全局变量或类属性用于暂存捕获的 attention
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
        
        # 顺手捞出 Q 和 K，手动计算 Attention Matrix
        d_k = query.size(-1)
        # query, key shape: [batch, heads, seq_len, head_dim]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if attn_mask is not None:
            # 注意：HF 传进来的 attn_mask 可能需要 broadcast，或者本身就是正确的 shape
            scores = scores + attn_mask
            
        _captured_attention = torch.softmax(scores, dim=-1)
        
        # 照常返回原版 SDPA 的输出，保证向前传播的极速性能
        return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

    try:
        # 替换底层的 sdpa
        F.scaled_dot_product_attention = hooked_sdpa
        yield
    finally:
        # 退出上下文时，务必恢复原函数
        F.scaled_dot_product_attention = original_sdpa


# COAST: Contrastive Adaptive Semantic Token-routing (Progressive Edition)
class COASTLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.last_attention = None

        # Efficiency metrics variables
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
        # [Experiment Config] 多阶段渐进式 COAST 面板 (Progressive Settings)
        # ========================================================================
        RUN_COAST = True
        
        coast_config = {
            # optional: "scalar", "maxsim", "contrastive"
            "routing_mode": "contrastive", 
            "img_start": 35,
            "original_img_end": 611,
            "prune_stages": {
                2: 214,  # 214, 352
                12: 78,  # 78, 144
                22: 6,
                28: 4
            },
            "anchor_ratio": 0.30,
            "negative_ratio": 0.05, 
            "alpha_min": 0.05,
            "alpha_max": 0.60,
            "attn_threshold": 1e-4,
            "max_anchor_allow_ratio": 0.8,
        }

        img_start = coast_config["img_start"] 
        img_seq_len = coast_config["original_img_end"] - coast_config["img_start"] 

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
                #  渐进式 COAST 核心执行区
                # ========================================================================
                if RUN_COAST and current_layer_idx in coast_config["prune_stages"] and seq_length > 1:
                    device = hidden_states.device
                    
                    current_img_end = img_start + img_seq_len
                    current_text_start = current_img_end
                    target_keep_num = coast_config["prune_stages"][current_layer_idx]
                    
                    if target_keep_num < img_seq_len:
                        if target_keep_num > 0 :
                            # 1. 动态双轨极性提取
                            text_to_image_attn = self.last_attention.mean(dim=1)[0][current_text_start:, img_start:current_img_end]
                            image_attention_score = text_to_image_attn.max(dim=0)[0] 
                            last_token_attn = self.last_attention.mean(dim=1)[0][-1][img_start:current_img_end]

                            max_anchor_allowance = int(target_keep_num * coast_config["max_anchor_allow_ratio"])
                            theoretical_anchor = int(img_seq_len * coast_config["anchor_ratio"])
                            top_k_num = min(theoretical_anchor, max_anchor_allowance)
                            top_k_num_bottom = min(int(img_seq_len * coast_config["negative_ratio"]), img_seq_len - target_keep_num)
                            
                            # 2. 锚点 Top-K
                            top_attention_index = last_token_attn.topk(top_k_num).indices 
                            bottom_attention_rank_index = image_attention_score.topk(top_k_num_bottom, largest=False).indices                 

                            # 无损熵计算

                            scores_f32 = image_attention_score.to(torch.float32)
                            total_img_attn = scores_f32.sum()
                            if total_img_attn < coast_config["attn_threshold"]:
                                normalized_entropy = 1.0  
                            else:
                                attn_probs = scores_f32 / total_img_attn
                                entropy = torch.special.entr(attn_probs).sum()
                                normalized_entropy = (entropy / math.log(img_seq_len)).item()
                                normalized_entropy = max(0.0, min(1.0, normalized_entropy))

                            alpha_min = coast_config["alpha_min"]
                            alpha_max = coast_config["alpha_max"]
                            dynamic_n2_ratio = alpha_min + (alpha_max - alpha_min) * normalized_entropy
                            
                            rest_token_retained_num = target_keep_num - top_k_num
                            n2_count = int(rest_token_retained_num * dynamic_n2_ratio)
                            n1_count = max(0, rest_token_retained_num - n2_count)

                            # 3. 构造剩余候选池
                            all_indices = torch.arange(img_seq_len, device=device)
                            all_indices_mask = torch.ones(img_seq_len, dtype=torch.bool, device=device)
                            all_indices_mask[top_attention_index] = False
                            other_indices = all_indices[all_indices_mask]

                            image_hs = hidden_states[0, img_start:current_img_end, :] 
                            text_aware_token = image_hs[top_attention_index] 
                            negative_token = image_hs[bottom_attention_rank_index] 
                            candidate_token = image_hs[other_indices] 
                            candidate_attn_scores = last_token_attn[other_indices] # 用于 Scalar 消融实验

                         # ---------------- 4. 细粒度对比路由 (支持消融实验) ----------------
                            def select_by_contrastive_score(anchor_hs, negative_hs, candidate_hs, other_idx, n1, n2, mode, candidate_attn):
                                device = candidate_hs.device
                                
                                if mode == "scalar":
                                    # Baseline: 直接用上一层的注意力分数
                                    scores = candidate_attn
                                else:
                                    # 归一化特征
                                    anchor_norm = F.normalize(anchor_hs, p=2, dim=1)       
                                    candidate_norm = F.normalize(candidate_hs, p=2, dim=1) 
                                    
                                    # 计算 MaxSim (Pull)
                                    sim_anchor = torch.mm(candidate_norm, anchor_norm.t()) 
                                    anchor_score = sim_anchor.max(dim=1)[0] 
                                    
                                    if mode == "maxsim":
                                        # Variant 1: 仅考虑与 Anchor 的相似度
                                        scores = anchor_score
                                    elif mode == "contrastive":
                                        # COAST 完整版: MaxSim - MeanSim
                                        negative_norm = F.normalize(negative_hs, p=2, dim=1)   
                                        sim_negative = torch.mm(candidate_norm, negative_norm.t()) 
                                        negative_score = sim_negative.mean(dim=1) 
                                        scores = anchor_score - negative_score
                                    else:
                                        raise ValueError(f"Unknown routing mode: {mode}")

                                # --- A. 提取 Top-n1 (核心语义 Exploitation) ---
                                if n1 > 0:
                                    n1_idx = other_idx[scores.topk(n1, largest=True).indices]
                                else:
                                    n1_idx = torch.tensor([], dtype=other_idx.dtype, device=device)

                                # --- B. 提取 Bottom-n2 (空间脚手架/双极保留 Bipolar Retention) ---
                                # 保持原始逻辑 largest=False 不变，完美对齐论文的 Bipolar 理论
                                if n2 > 0:
                                    n2_idx = other_idx[scores.topk(n2, largest=False).indices]
                                else:
                                    n2_idx = torch.tensor([], dtype=other_idx.dtype, device=device)
                                    
                                return n1_idx, n2_idx


                            # 调用路由函数 (传入 config 中的 mode 和 用于 Scalar 模式的 attn_scores)
                            n1_idx, n2_idx = select_by_contrastive_score(
                                text_aware_token, negative_token, candidate_token, 
                                other_indices, n1=n1_count, n2=n2_count,
                                mode=coast_config.get("routing_mode", "contrastive"), # 默认使用完整版
                                candidate_attn=candidate_attn_scores
                            )
                            combined_indices = torch.cat([top_attention_index, n1_idx, n2_idx])
                            
                            # 重组与压缩
                            combined_indices = combined_indices.sort().values
                            img_seq_len = combined_indices.shape[0] # 更新长度给下一阶段用
                            keep_indexs = torch.cat((torch.arange(img_start, device=device), combined_indices + img_start, torch.arange(current_img_end, seq_length, device=device)))

                        else:
                            img_seq_len = 0
                            keep_indexs = torch.cat((torch.arange(img_start, device=device), torch.arange(current_img_end, seq_length, device=device)))
                        
                        hidden_states = hidden_states[:, keep_indexs, :]
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        keep_indexs = keep_indexs.to(position_ids.device)
                        position_ids = position_ids[:, keep_indexs]
                        seq_length = hidden_states.shape[1]

                # ========================================================================
                # 预获取 Attention (服务于下一个剪枝层) - 合并进单次 forward，不再二次调用
                # ========================================================================
                is_next_prune_stage = (current_layer_idx + 1) in coast_config["prune_stages"]


                _need_coast_attn = RUN_COAST and is_next_prune_stage and seq_length > 1
                _need_attn = _need_coast_attn and seq_length > 1
                # 【修改点 1】不再强制修改 output_attentions 触发 Eager 回退
                _eff_output_attentions = output_attentions # 保持默认 False，让 HF 安心走 SDPA

                if seq_length > 1:
                    all_token.append(img_seq_len)

                # 【修改点 2】使用上下文管理器包裹当前层的 forward
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
                    # 从全局变量中把截获的 attention 拿出来
                    global _captured_attention
                    self.last_attention = _captured_attention 
                else:
                    # 不需要提取 attention，正常极速 forward
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
            print(f"[Run Info] Method: {method_name} | Stages: {list(coast_config['prune_stages'].keys())} | Actual Avg Image Tokens/Layer: {mean(all_token):.1f}")
            print(f"[Run Info] Max Anchor Allowance Ratio: {coast_config['max_anchor_allow_ratio']}")

            if self.enable_tflops:
                self.num_forward += 1
                FLOPs_avg_sample = (self.all_FLOPs / self.num_forward) * 1e-12
                print(f"Self_num_forward: {self.num_forward}")
                print(f"[Efficiency INFO] TFLOPs: {FLOPs_avg_sample:.2f}")



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