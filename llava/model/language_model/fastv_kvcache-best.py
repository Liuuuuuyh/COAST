import torch
from typing import Tuple, Callable
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa, Cache, DynamicCache
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union
from numpy import *
import torch.nn.functional as F
import math
#COntrastive Adaptive Semantic Token-routing
class FastVLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        super().__init__(config)
        # self.model.layers = nn.ModuleList([IQDecoderLayyer(config, layer_idx) for layer_idx in range(LlamaConfig.num_hidden_layers)])
    
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_token = []
        all_indices = torch.arange(576, device=hidden_states.device)
        all_indices_mask = torch.ones(576, dtype=torch.bool, device=hidden_states.device)
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:

                fastv = False
                K = 3
                # ========================================================================
                # COAST: Contrastive Adaptive Semantic Token-Routing
                # ========================================================================
                two_stage = True
                token_retained_num = 288
                if two_stage:
                    ratio = 0.3
                    if decoder_layer.self_attn.layer_idx == K and seq_length > 1:
                        device = hidden_states.device
                        
                        # ================= Stage 1 =================
                        # 提取文本 Token 对图像的注意力
                        text_to_image_attn = self.last_attention.mean(dim=1)[0][611:, 35:611] # [1, 32, 625, 625]

                        # 只要有任意一个文本 Token 强烈关注了某个图像区域，该区域就重要
                        image_attention_score = text_to_image_attn.max(dim=0)[0] # 维度变为 [576]
                        top_attention_rank_index = image_attention_score.topk(int(576 * ratio)).indices 
                        rest_token_retained_num = token_retained_num - int(576 * ratio)
                        
                        # stage 2
                        top_k_num_bottom = int(576 * 0.1)   
                        img_start = 35
                        img_end = 611
                        
                        # 选出最不受关注的 Bottom-K 作为 Negative (背景/噪声信息)
                        bottom_attention_rank_index = image_attention_score.topk(top_k_num_bottom, largest=False).indices
                        
                        # ================= 无损熵计算逻辑 =================
                        scores_f32 = image_attention_score.to(torch.float32)

                        # 总关注度实在太低，直接最大熵
                        total_img_attn = scores_f32.sum()
                        if total_img_attn < 1e-4:
                            normalized_entropy = 1.0  # 图像完全被无视，保留最大多样性
                        else:
                            attn_probs = scores_f32 / total_img_attn
                            entropy = torch.special.entr(attn_probs).sum()
                            max_entropy = math.log(576) # math.log(576)
                            normalized_entropy = (entropy / max_entropy).item()
                            normalized_entropy = max(0.0, min(1.0, normalized_entropy))

                        # 动态映射：熵越大 (越分散)，n2 比例越高；熵越小 (越集中)，n2 比例越低。
                        # 设定一个基础范围，例如 n2 占比在 5% 到 60% 之间浮动
                        min_n2_ratio = 0.05
                        max_n2_ratio = 0.6 
                        dynamic_n2_ratio = min_n2_ratio + (max_n2_ratio - min_n2_ratio) * normalized_entropy
                        n2_count = int(rest_token_retained_num * dynamic_n2_ratio)
                        n1_count = rest_token_retained_num - n2_count
                        # print("n2_count: ", n2_count, "n1_count:", n1_count, "normalized_entropy: ", normalized_entropy)

                        # ================= 准备候选池 =================
                        all_indices_mask[top_attention_rank_index] = False
                        other_indices = all_indices[all_indices_mask]

                        # 获取对应的 hidden_states 特征
                        text_aware_token = hidden_states[0][img_start:img_end][top_attention_rank_index] # Anchor
                        negative_token = hidden_states[0][img_start:img_end][bottom_attention_rank_index] # Negative
                        candidate_token = hidden_states[0][img_start:img_end][other_indices] # Candidates

                        # ================= Stage 2: 对比引导选择 =================
                        def select_by_contrastive_score(anchor_hs, negative_hs, candidate_hs, other_idx, n1, n2):
                            # 计算核心特征与背景特征的“原型 (Prototype)”
                            anchor_proto = anchor_hs.mean(dim=0, keepdim=True)
                            negative_proto = negative_hs.mean(dim=0, keepdim=True)
                            
                            candidate_norm = F.normalize(candidate_hs, p=2, dim=1)
                            
                            # 合并原型，只需一次矩阵乘法 (Fused MatMul)
                            protos = F.normalize(torch.cat([anchor_proto, negative_proto], dim=0), p=2, dim=1)
                            
                            # sims shape: [候选数量, 2]
                            sims = torch.mm(candidate_norm, protos.t()) 
                            
                            # 核心对比分数 = 与Anchor的相似度 - 与Negative的相似度
                            contrastive_score = sims[:, 0] - sims[:, 1]
                            
                            # 再次使用 argsort 替代双重 topk
                            sorted_contrastive_idx = contrastive_score.argsort(descending=True)
                            
                            n1_original_indices = other_idx[sorted_contrastive_idx[:n1]]
                            n2_original_indices = other_idx[sorted_contrastive_idx[-n2:]]
                            
                            return n1_original_indices, n2_original_indices

                        # 执行对比选择
                        n1_idx, n2_idx = select_by_contrastive_score(
                            text_aware_token, 
                            negative_token,
                            candidate_token, 
                            other_indices, 
                            n1=n1_count, 
                            n2=n2_count
                        )
                        combined_indices = torch.cat([top_attention_rank_index, n1_idx, n2_idx])

                        keep_indexs = torch.cat((torch.arange(35,device=device), combined_indices+35, torch.arange(611,seq_length,device=device)))
                        keep_indexs = keep_indexs.sort().values
                        hidden_states = hidden_states[:,keep_indexs,:]
                        if attention_mask is not None:
                            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                        position_ids = keep_indexs.unsqueeze(0)

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








                if fastv == True:
                    ratio = 0.5

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
                    all_token.append(hidden_states.shape[1])
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if seq_length > 1:
            print("num: ", len(all_token), ", average_token: ", mean(all_token))
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

