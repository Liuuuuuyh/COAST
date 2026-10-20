import torch
import torch.nn.functional as F
import math
from typing import Tuple, Callable, List, Optional, Union
from numpy import mean
import os
import json

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


# COAST: Contrastive Adaptive Semantic Token-routing (Progressive Edition)
class FastVLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.last_attention = None

        # Efficiency metrics variables
        # self.total_cuda_time = 0.0
        self.all_FLOPs = 0.0
        self.num_forward = 0
        self.enable_prefill_time = False
        self.enable_tflops = False
        # self.enable_prefill_time = True
        # self.enable_tflops = True

        self.disable_pruning = True
    
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

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_token = []
        
        # ========================================================================
        # [Experiment Config] 多阶段渐进式 COAST 面板 (Progressive Settings)
        # ========================================================================
        # RUN_COAST = True
        RUN_COAST = not getattr (self, "disable_pruning",False)
        # RUN_COAST = False
        RUN_FASTV = False
        
        coast_config = {
            "img_start": 35,
            "original_img_end": 611,
            # ====================== 7B =====================
            "prune_stages": {
                2: 214,   # 结束对齐期后，砍掉大部分低频背景
                12: 78,    # 进入深层前，二次提纯
                22: 6,
                28: 4
            },
            # ====================== 13B ============== =======
            # "prune_stages": {
            #     2: 194,  # 结束对齐期后，砍掉大部分低频背景
            #     16: 76,    # 进入深层前，二次提纯
            #     32: 6,
            # },
            # ======== # 192
            # "prune_stages": {
            #     4: 256,    # 第一阶段：剥离明显无关背景
            #     12: 128,   # 第二阶段：提纯次要物体
            #     20: 64     # 第三阶段：极限压缩至核心极性
            # },
            # "prune_stages": {
            #     3: 288,    # 第一阶段：剥离明显无关背景
            #     22: 0,   # 第二阶段：提纯次要物体
            #     # 28: 0     # 第三阶段：极限压缩至核心极性
            # },
            # ======== # 128
            # "prune_stages": {
            #     3: 122,   # 结束对齐期后，砍掉大部分低频背景
            #     22: 16,    # 进入深层前，二次提纯
            #     28: 0,    # 进入深层前，二次提纯
            # },
            # "prune_stages": {
            #     3: 122,   # 结束对齐期后，砍掉大部分低频背景
            #     22: 16,    # 进入深层前，二次提纯
            #     28: 0,    # 进入深层前，二次提纯
            # },
            # "prune_stages": {
            #     3: 128,   # 结束对齐期后，砍掉大部分低频背景
            #     20: 28,    # 进入深层前，二次提纯
            #     27: 2   # 最后阶段：极限压缩至核心极性
            # },
            # ======== # 288
            # "prune_stages": {
            #     4: 288,   # 延长满载对齐期至 4 层，确保极小文字特征不丢失
            #     16: 216   # 深层仅作极轻量化压缩
            # },
            # "prune_stages": { # 方案三
            #     2: 340,   # 延长满载对齐期至 4 层，确保极小文字特征不丢失
            #     22: 110,   # 深层仅作极轻量化压缩
            #     28: 92,   # 深层仅作极轻量化压缩
            # },
            # ======== # 64
            # "prune_stages": {
            #     2: 64,    # 极早起刀：剥离 90% 的纯净背景
            #     12: 13    # 深层极限：只保留最核心的语义极性锚点
            # },
            # "prune_stages": {
            #     2: 46,    # 极早起刀：剥离 90% 的纯净背景
            #     22: 18,   # 深层极限：只保留最核心的语义极性锚点
            #     28: 0 
            # },
            "anchor_ratio": 0.30,
            "negative_ratio": 0.05,  # 0.05
            "alpha_min": 0.05,  # 0.05      
            "alpha_max": 0.80,  # 0.60           
            "attn_threshold": 1e-4,
            "max_anchor_allow_ratio": 0.8,  # 0.8s
        }
        

        # ── 环境变量覆盖（供 search_params.py 使用）──────────────────────────
        # _env_ps = os.environ.get("COAST_PRUNE_STAGES")
        # if _env_ps:
        #     coast_config["prune_stages"] = {int(k): int(v) for k, v in json.loads(_env_ps).items()}
        # for _key, _env in [("anchor_ratio",   "COAST_ANCHOR_RATIO"),
        #                     ("negative_ratio", "COAST_NEGATIVE_RATIO"),
        #                     ("alpha_min",      "COAST_ALPHA_MIN"),
        #                     ("alpha_max",      "COAST_ALPHA_MAX")]:
        #     _val = os.environ.get(_env)
        #     if _val is not None:
        #         coast_config[_key] = float(_val)
        # ─────────────────────────────────────────────────────────────────────

        # 动态游标初始化
        img_start = coast_config["img_start"] 
        img_seq_len = coast_config["original_img_end"] - coast_config["img_start"] 

        if RUN_FASTV:
            fastv_config = {
                "img_start": 35,
                "img_end": 611,
                "prune_layer": 3, 
                "target_avg_tokens": 288,
                "layer_nums": 32
            }
            unpruned_layers = fastv_config["prune_layer"]
            pruned_layers = fastv_config["layer_nums"] - unpruned_layers
            
            # 计算全局总 Token 预算
            total_token_budget = fastv_config["target_avg_tokens"] * fastv_config["layer_nums"]
            # 计算前置层已经消耗的 Token 数量
            consumed_tokens = unpruned_layers * img_seq_len
            
            # 剩余的 Token 预算分摊到剩下的层中
            remaining_budget_per_layer = (total_token_budget - consumed_tokens) / pruned_layers
            
            # 3. 换算成最终给算法使用的 ratio (后续传给 Stage 1 和 Stage 2 使用)
            keep_ratio = remaining_budget_per_layer / img_seq_len

        # Efficiency: Start timing for prefill
        if self.enable_prefill_time and seq_length > 1:
            total_start_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            total_start_event.record()


        all_indices_mask = torch.ones(img_seq_len, dtype=torch.bool, device=hidden_states.device)
        
        # ========================================================================
        # Decoder Loop
        # ========================================================================
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
                    
                    # 动态游标计算当前层的真实边界 (因为序列在不断变短)
                    current_img_end = img_start + img_seq_len
                    current_text_start = current_img_end
                    target_keep_num = coast_config["prune_stages"][current_layer_idx]
                    
                    # 安全锁：只在需要压缩时执行
                    if target_keep_num < img_seq_len:
                        if target_keep_num > 0 :
                            # ---------------- 1. 动态双轨极性提取 ----------------
                            text_to_image_attn = self.last_attention.mean(dim=1)[0][current_text_start:, img_start:current_img_end]
                            image_attention_score = text_to_image_attn.max(dim=0)[0] 
                            
                            last_token_attn = self.last_attention.mean(dim=1)[0][-1][img_start:current_img_end]

                            # 动态防越界预算控制
                            max_anchor_allowance = int(target_keep_num * coast_config["max_anchor_allow_ratio"])
                            theoretical_anchor = int(img_seq_len * coast_config["anchor_ratio"])
                            top_k_num = min(theoretical_anchor, max_anchor_allowance)
                            top_k_num_bottom = min(int(img_seq_len * coast_config["negative_ratio"]), img_seq_len - target_keep_num)
                            
                            # ---------------- 2. 锚点 Top-K ----------------
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

                            # ---------------- 3. 构造剩余候选池 ----------------
                            all_indices = torch.arange(img_seq_len, device=device)
                            all_indices_mask = torch.ones(img_seq_len, dtype=torch.bool, device=device)
                            all_indices_mask[top_attention_index] = False
                            other_indices = all_indices[all_indices_mask]

                            image_hs = hidden_states[0, img_start:current_img_end, :] 
                            text_aware_token = image_hs[top_attention_index] 
                            negative_token = image_hs[bottom_attention_rank_index] 
                            candidate_token = image_hs[other_indices] 

                            # ---------------- 4. 细粒度 MaxSim 对比路由 ----------------
                            def select_by_contrastive_score(anchor_hs, negative_hs, candidate_hs, other_idx, n1, n2):
                                anchor_norm = F.normalize(anchor_hs, p=2, dim=1)       
                                negative_norm = F.normalize(negative_hs, p=2, dim=1)   
                                candidate_norm = F.normalize(candidate_hs, p=2, dim=1) 

                                sim_anchor = torch.mm(candidate_norm, anchor_norm.t()) 
                                anchor_score = sim_anchor.max(dim=1)[0] # 只以最大的来算

                                sim_negative = torch.mm(candidate_norm, negative_norm.t()) 
                                negative_score = sim_negative.mean(dim=1) 
                                
                                contrastive_score = anchor_score - negative_score
                                
                                n1_idx = other_idx[contrastive_score.topk(n1, largest=True).indices] if n1 > 0 else torch.tensor([], dtype=other_idx.dtype, device=other_idx.device)
                                n2_idx = other_idx[contrastive_score.topk(n2, largest=False).indices] if n2 > 0 else torch.tensor([], dtype=other_idx.dtype, device=other_idx.device)
                                return n1_idx, n2_idx
                                
                            n1_idx, n2_idx = select_by_contrastive_score(
                                text_aware_token, negative_token, candidate_token, 
                                other_indices, n1=n1_count, n2=n2_count
                            )
                            
                            # 汇总本阶段幸存的 Token
                            combined_indices = torch.cat([top_attention_index, n1_idx, n2_idx])
                            
                            # 重组与压缩
                            combined_indices = combined_indices.sort().values
                            img_seq_len = combined_indices.shape[0] # 更新长度给下一阶段用
                        
                            # keep_indexs 现在是用来切片【当前 Tensor】的相对索引
                            keep_indexs = torch.cat((torch.arange(img_start, device=device), combined_indices + img_start, torch.arange(current_img_end, seq_length, device=device)))
                        
                        else:
                            img_seq_len = 0
                            keep_indexs = torch.cat((torch.arange(img_start, device=device), torch.arange(current_img_end, seq_length, device=device)))
                        
                        # 1. 切片 hidden_states
                        hidden_states = hidden_states[:, keep_indexs, :]
                        
                        # 2. 切片 attention_mask
                        if attention_mask is not None:
                            # attention_mask = attention_mask[:, :, keep_indexs, :][:, :, :, keep_indexs]
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        # 3. 【神级修复】：用相对索引切片 position_ids，完美保留原始绝对物理位置！
                        keep_indexs = keep_indexs.to(position_ids.device)
                        position_ids = position_ids[:, keep_indexs]
                        
                        # 全局长度减小
                        seq_length = hidden_states.shape[1]


                # ========================================================================
                # 预获取 Attention (服务于下一个剪枝层)
                # ========================================================================
                # 检查下一层是否在剪枝计划里，如果是，我们必须在当前层保留 Attention
                is_next_prune_stage = (current_layer_idx + 1) in coast_config["prune_stages"]
                
                if RUN_COAST and is_next_prune_stage and seq_length > 1:
                    temp_layer_outputs = decoder_layer(
                        hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values, output_attentions=True, use_cache=use_cache,
                    )
                    self.last_attention = temp_layer_outputs[1]


                if RUN_FASTV == True:
                    ratio = keep_ratio
                    K = unpruned_layers
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1:
                        device = hidden_states.device
                        image_attention_score = self.last_attention.mean(dim=1)[0][-1][35:611]  
                        top_attention_rank_index = image_attention_score.topk(int(576 * ratio)).indices + 35
                        keep_indexs = torch.cat((torch.arange(35,device=device), top_attention_rank_index, torch.arange(611,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values
                        hidden_states = hidden_states[:,keep_indexs,:]
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        position_ids = keep_indexs.unsqueeze(0)
                        img_seq_len = top_attention_rank_index.shape[0]

                    if decoder_layer.self_attn.layer_idx == K - 1:
                        temp_layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        self.last_attention = temp_layer_outputs[1]


                if seq_length > 1:
                    all_token.append(img_seq_len)

                    # Efficiency: Calculate FLOPs
                    if self.enable_tflops:
                        n = hidden_states.shape[1]
                        d = hidden_states.shape[2]
                        m = self.config.intermediate_size

                        # MHA FLOPs
                        self.all_FLOPs += 4 * n * (d**2) + 2 * (n**2) * d + 3 * n * d * m

                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if seq_length > 1:
            method_name = "COAST-Progressive" if RUN_COAST else "FastV-Progressive" if RUN_FASTV else "Dense"
            print(f"[Run Info] Method: {method_name} | Stages: {list(coast_config['prune_stages'].keys())} | Actual Avg Image Tokens/Layer: {mean(all_token):.1f}")
            # print(f"[Run Info] Max Anchor Allowance Ratio: {coast_config['max_anchor_allow_ratio']}")
            print(f"[Run Info] Anchor Ratio: {coast_config['anchor_ratio']} | Negative Ratio: {coast_config['negative_ratio']} | Alpha Min: {coast_config['alpha_min']} | Alpha Max: {coast_config['alpha_max']} | Max Anchor Allowance Ratio: {coast_config['max_anchor_allow_ratio']}")

            # Efficiency: End timing for prefill and log
            if self.enable_prefill_time:
                total_end_event.record()
                torch.cuda.synchronize()
                total_cuda_time_ms = total_start_event.elapsed_time(total_end_event)
                # self.total_cuda_time += total_cuda_time_ms
                print(f"[Efficiency INFO] Prefill Time (ms): {total_cuda_time_ms:.2f}")

            if self.enable_tflops:
                self.num_forward += 1
                FLOPs_avg_sample = (self.all_FLOPs / self.num_forward) * 1e-12
                print(f"[Efficiency INFO] TFLOPs: {FLOPs_avg_sample:.2f}")


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