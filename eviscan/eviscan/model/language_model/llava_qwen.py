#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import List, Optional, Tuple, Union, Dict
import torch
import time

total_forward_time = 0
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from eviscan.eviscan.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
# from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from .modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
import pdb

import torch.nn.functional as F

def pooling_kvs(kvs, cmpr):
    # kvs shape: (bsz, 4, seq_len, head_dim)
    kernel_size = cmpr
    stride = cmpr
    kvs_permuted = kvs.permute(0, 1, 3, 2) # (batch_size, num_heads, feature_dim, sequence_length)
    # 然后展平 batch_size 和 num_heads
    N_flat = kvs_permuted.shape[0] * kvs_permuted.shape[1]
    C = kvs_permuted.shape[2]
    L = kvs_permuted.shape[3]
    kvs_for_pool = kvs_permuted.reshape(N_flat, C, L)
    pooled_kvs = F.avg_pool1d(kvs_for_pool, kernel_size=kernel_size, stride=stride)
    # 再恢复形状
    pooled_kvs_restored = pooled_kvs.view(kvs.shape[0], kvs.shape[1], pooled_kvs.shape[1], pooled_kvs.shape[2]).permute(0, 1, 3, 2)
    return pooled_kvs_restored

def sample_interval_kvs(kvs, cmpr):
    # kvs shape: (bsz, 4, seq_len, head_dim)
    # cmpr 表示采样间隔，例如 cmpr=2 表示每隔一个元素取一个
    
    # 获取原始形状信息
    bsz, num_heads, seq_len, head_dim = kvs.shape
    
    # 计算采样后的序列长度
    new_seq_len = seq_len // cmpr
    
    # 创建索引，每隔 cmpr 取一个元素
    # 例如，如果 cmpr=2，则取索引 [0, 2, 4, 6, ...]
    indices = torch.arange(0, seq_len, cmpr, device=kvs.device)
    
    # 如果需要确保新序列长度正好是 new_seq_len，可以限制索引数量
    # indices = indices[:new_seq_len]
    
    # 使用索引从原始序列中采样
    sampled_kvs = kvs[:, :, indices, :]
    
    return sampled_kvs

def sample_first_kvs(kvs, cmpr):
    # kvs shape: (bsz, 4, seq_len, head_dim)
    # cmpr 表示采样间隔，例如 cmpr=2 表示每隔一个元素取一个
    
    # 获取原始形状信息
    bsz, num_heads, seq_len, head_dim = kvs.shape
    
    # 计算采样后的序列长度
    new_seq_len = seq_len // cmpr
    
    # # 创建索引，每隔 cmpr 取一个元素
    # # 例如，如果 cmpr=2，则取索引 [0, 2, 4, 6, ...]
    # indices = torch.arange(0, seq_len, cmpr, device=kvs.device)
    
    # # 如果需要确保新序列长度正好是 new_seq_len，可以限制索引数量
    # indices = indices[:new_seq_len]
    
    # 使用索引从原始序列中采样
    sampled_kvs = kvs[:, :, :new_seq_len, :]
    
    return sampled_kvs
    

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        
        self.original_seq_len = 0
        
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        time_embedding=None,
        return_input_embeds: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:  

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, time_embedding)
        
        global total_forward_time
        start_time = time.perf_counter() * 1000

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

        end_time = time.perf_counter() * 1000
        execution_time = end_time - start_time
        
        total_forward_time += execution_time
        print(total_forward_time)
        # 输出执行时间（可以根据需要记录到日志文件）
        print(f"Model forward execution time: {execution_time:.2f} ms")
        
        if return_input_embeds:
            return outputs, inputs_embeds
        else:
            return outputs
    
            
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        time_embedding = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes,time_embedding=time_embedding)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    
    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, past_key_values=None, **kwargs):
        if past_key_values is not None and input_ids.numel() != 0: # when start decoding in 1st step —— predict timestamps
            input_ids = input_ids[:, -1:]
        inputs = {}
        if input_ids.numel() != 0:
            inputs['input_ids'] = input_ids
        else:
            inputs['inputs_embeds'] = inputs_embeds
        inputs['past_key_values'] = past_key_values
        inputs.update(kwargs)
        return inputs
    
AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
