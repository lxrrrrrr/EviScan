from transformers.cache_utils import Cache, DynamicCache, StaticCache
import torch
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import pdb


def pooling_kvs(kvs, position_ids, start=0, end=-1, cmpr=2):
    # 处理 position_ids
    if position_ids is not None:
        target_position_ids = position_ids[:, start: end] if end != -1 else position_ids[:, start:]
        # 取每个池化窗口的中心位置
        sampled_position_ids = target_position_ids[:, cmpr//2::cmpr]
        # 如果最后一个窗口不完整，需要调整
        expected_length = (target_position_ids.shape[1] + cmpr - 1) // cmpr
        if sampled_position_ids.shape[1] < expected_length:
            # 补充最后一个位置
            last_pos = target_position_ids[:, -1:]
            sampled_position_ids = torch.cat([sampled_position_ids, last_pos], dim=1)
    else:
        sampled_position_ids = None
    
    target_kvs = kvs[:, :, start: end]
    # kvs shape: (bsz, 4, seq_len, head_dim)
    kernel_size = cmpr
    stride = cmpr
    kvs_permuted = target_kvs.permute(0, 1, 3, 2) # (batch_size, num_heads, feature_dim, sequence_length)
    # 然后展平 batch_size 和 num_heads
    N_flat = kvs_permuted.shape[0] * kvs_permuted.shape[1]
    C = kvs_permuted.shape[2]
    L = kvs_permuted.shape[3]
    kvs_for_pool = kvs_permuted.reshape(N_flat, C, L)
    pooled_kvs = F.avg_pool1d(kvs_for_pool, kernel_size=kernel_size, stride=stride)
    # 再恢复形状
    pooled_kvs_restored = pooled_kvs.view(target_kvs.shape[0], target_kvs.shape[1], pooled_kvs.shape[1], pooled_kvs.shape[2]).permute(0, 1, 3, 2)

    return pooled_kvs_restored, sampled_position_ids

def uniform_sample_kvs(kvs, position_ids, start=0, end=-1, cmpr=2):
    # kvs shape: (bsz, 4, seq_len, head_dim)
    seq_len = end - start
    # 创建索引，每隔 cmpr 取一个元素
    # 例如，如果 cmpr=2，则取索引 [0, 2, 4, 6, ...]
    indices = torch.arange(start, start + seq_len, cmpr, device=kvs.device)
    # 使用索引从原始序列中采样
    sampled_kvs = kvs[:, :, indices, :]
    sampled_position_ids = position_ids[:, indices]
    return sampled_kvs, sampled_position_ids

class MyCache(Cache):
    """
    自定义缓存类，继承自 transformers 的 Cache 基类
    """
    def __init__(self, cmpr=2, sparse_mode='pooling', reforge_pos=False, max_cache_len: Optional[int] = 128000, **kwargs):
        super().__init__()
        # sparse config
        self.cmpr = cmpr
        self.sparse_mode = sparse_mode # pooling, uniform_sample
        self.reforge_pos = reforge_pos 
        self.max_cache_len = max_cache_len
        
        self.layer_num = 28
        # 初始化你的缓存存储结构
        self.key_cache: List[torch.Tensor] = [None] * self.layer_num
        self.value_cache: List[torch.Tensor] = [None] * self.layer_num
        self.past_kvs_position_ids: Optional[torch.Tensor] = None

        self.system_prompt_k: List[torch.Tensor] = [None] * self.layer_num
        self.system_prompt_v: List[torch.Tensor] = [None] * self.layer_num
        self.past_sp_position_ids: Optional[torch.Tensor] = None
        
        self.fine_video_rep_k: List[torch.Tensor] = [None] * self.layer_num
        self.fine_video_rep_v: List[torch.Tensor] = [None] * self.layer_num
        self.past_fvr_position_ids: Optional[torch.Tensor] = None

        self.coarse_video_rep_k: List[torch.Tensor] = [None] * self.layer_num
        self.coarse_video_rep_v: List[torch.Tensor] = [None] * self.layer_num
        self.past_cvr_position_ids: Optional[torch.Tensor] = None
        
        # self.record_whole_fvr_times_indices = []

    def set_infer_stage(self, stage):
        assert stage in ['prefill_t', 'prefill_v', 'coarse_decode', 'fine_decode'], "stage must be 'prefill_v' or 'decode'"
        self.infer_stage = stage
        
    def _sparse(self, key_states, value_states, position_ids, layer_idx):
        
        self.sparse_cache_pos_list = []
        
        frames_groups = self.video_info['frames_groups']
        visual_token_start_pos = self.video_info['visual_token_start_pos']
        visual_token_end_pos = self.video_info['visual_token_end_pos']
        
        if self.sparse_mode == "pooling":
            downsample_method = pooling_kvs
        elif self.sparse_mode == "uniform_sample":
            downsample_method = uniform_sample_kvs         
        
        coarse_key_states = []
        coarse_value_states = []
        sparse_kv_pos_list = []
        for idx, group_infos in enumerate(frames_groups):
            time_start, time_end, group_end = group_infos
            
            timestamps_pos = position_ids[:, time_start:time_end]
            downsampled_pos = None
    
            # 时间戳部分（直接切片引用，不复制）
            time_keys = key_states[:, :, time_start:time_end]
            time_vals = value_states[:, :, time_start:time_end]
            
            # 视觉token下采样
            downsampled_visual_keys, downsampled_pos = downsample_method(
                key_states, position_ids, 
                start=time_end, end=group_end, cmpr=self.cmpr
            )
            downsampled_visual_values, _ = downsample_method(
                value_states, position_ids, 
                start=time_end, end=group_end, cmpr=self.cmpr
            )
            
            sparse_kv_pos_list.append(torch.cat([timestamps_pos, downsampled_pos], dim=1))

            sparse_kv_thisgroup = torch.cat([time_keys, downsampled_visual_keys], dim=2)
            sparse_vv_thisgroup = torch.cat([time_vals, downsampled_visual_values], dim=2)
            
            coarse_key_states.append(sparse_kv_thisgroup)
            coarse_value_states.append(sparse_vv_thisgroup)
            
            # 及时清理临时变量
            del downsampled_visual_keys, downsampled_visual_values
                
        coarse_key_states = torch.cat(coarse_key_states, dim=2)
        coarse_value_states = torch.cat(coarse_value_states, dim=2)
        
        sparse_kv_pos = torch.cat(sparse_kv_pos_list, dim=1)
        
        if layer_idx == 0:
            self.past_cvr_position_ids = self._concat_pos(self.past_cvr_position_ids, sparse_kv_pos, coarse_key_states.shape[-2])
                
        # 实现你的缓存逻辑
        k_cache, v_cache = self._concat(self.coarse_video_rep_k[layer_idx], self.coarse_video_rep_v[layer_idx], coarse_key_states, coarse_value_states)
        self.coarse_video_rep_k[layer_idx] = k_cache
        self.coarse_video_rep_v[layer_idx] = v_cache
        
    def _concat_pos(self, past_pos, position_ids=None, updated_len=0):
        
        if position_ids is not None:
            updated_position_ids = position_ids
        else:
            past_last_pos = past_pos[0, -1].item() if past_pos is not None else -1
            updated_position_ids = torch.arange(past_last_pos + 1, past_last_pos + updated_len + 1, dtype=torch.long).unsqueeze(0)
        
        past_pos = torch.concat([past_pos, updated_position_ids], dim=-1) if past_pos is not None else updated_position_ids
        
        return past_pos
    
    def _concat(self, k_cache, v_cache, key_states, value_states):
        # 实现你的缓存逻辑
        if k_cache is None:
            # 第一次缓存
            k_cache = key_states
            v_cache = value_states
        else:
            # 拼接新的 key 和 value
            k_cache = torch.cat([k_cache, key_states], dim=-2)
            v_cache = torch.cat([v_cache, value_states], dim=-2)
            
            # 如果超过最大长度，进行截断或其他处理
            if self.max_cache_len is not None:
                if k_cache.shape[-2] > self.max_cache_len:
                    k_cache = k_cache[..., -self.max_cache_len:, :]
                    v_cache = v_cache[..., -self.max_cache_len:, :]
                    
        return k_cache, v_cache
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存的核心方法
        """
        if self.infer_stage == 'prefill_t':

            k_cache, v_cache = self._concat(self.system_prompt_k[layer_idx], self.system_prompt_v[layer_idx], key_states, value_states)
            self.system_prompt_k[layer_idx] = k_cache
            self.system_prompt_v[layer_idx] = v_cache
            
            if layer_idx == 0:
                self.past_sp_position_ids = self._concat_pos(self.past_sp_position_ids, position_ids, key_states.shape[-2])
            
            # return self.system_prompt_k[layer_idx], self.system_prompt_v[layer_idx]
        
        elif self.infer_stage == 'prefill_v':
            # 一次forward只更新一次
            if layer_idx == 0:
                self.past_fvr_position_ids  = self._concat_pos(self.past_fvr_position_ids, position_ids, key_states.shape[-2])
            
            k_cache, v_cache = self._concat(self.fine_video_rep_k[layer_idx], self.fine_video_rep_v[layer_idx], key_states, value_states)
            self.fine_video_rep_k[layer_idx] = k_cache
            self.fine_video_rep_v[layer_idx] = v_cache
            
            # pdb.set_trace()
            self._sparse(key_states, value_states, position_ids, layer_idx)
            
            # k_cache, v_cache = self._concat(self.key_cache[layer_idx], self.value_cache[layer_idx], key_states, value_states)
            # self.key_cache[layer_idx] = k_cache
            # self.value_cache[layer_idx] = v_cache
            
            # if layer_idx == 0:
            #     self.past_kvs_position_ids = self._concat_pos(self.past_kvs_position_ids, position_ids, key_states.shape[-2])
            
            # return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        # elif self.infer_stage == 'decode':
        k_cache, v_cache = self._concat(self.key_cache[layer_idx], self.value_cache[layer_idx], key_states, value_states)
        self.key_cache[layer_idx] = k_cache
        self.value_cache[layer_idx] = v_cache
        
        if layer_idx == 0:
            self.past_kvs_position_ids = self._concat_pos(self.past_kvs_position_ids, position_ids, key_states.shape[-2])
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """返回指定层的序列长度"""
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]
    
    def get_fvr_length(self, layer_idx: Optional[int] = 0) -> int:
        """返回指定层的序列长度"""
        if len(self.fine_video_rep_k) <= layer_idx or self.fine_video_rep_k[layer_idx] is None:
            return 0
        return self.fine_video_rep_k[layer_idx].shape[-2]
    
    def get_cvr_length(self, layer_idx: Optional[int] = 0) -> int:
        """返回指定层的序列长度"""
        if len(self.coarse_video_rep_k) <= layer_idx or self.coarse_video_rep_k[layer_idx] is None:
            return 0
        return self.coarse_video_rep_k[layer_idx].shape[-2]
    
    def get_max_length(self) -> Optional[int]:
        """返回最大缓存长度"""
        return self.max_cache_len
    
    def preprocess_video_info(self, input_ids=None, time_stamps=None, visual_token_start_pos=0):
        # IMAGE_TOKEN_INDEX = -200
        # TOKEN_PERFRAME = 36
        # visual_token_start_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
        num_v_tokens = time_stamps[0].size(0)
        visual_token_end_pos = visual_token_start_pos + num_v_tokens
        time_token_start_indices = (time_stamps[0] == 1462).nonzero(as_tuple=True)[0].cpu().tolist()
        time_token_start_indices = [idx + visual_token_start_pos for idx in time_token_start_indices]
        time_token_indices = (time_stamps[0] != 151654).nonzero(as_tuple=True)[0].cpu().tolist()
        time_token_indices = [idx + visual_token_start_pos for idx in time_token_indices]
        time_token_end_indices = (time_stamps[0] == 25).nonzero(as_tuple=True)[0].cpu().tolist()
        time_token_end_indices = [idx + visual_token_start_pos + 1 for idx in time_token_end_indices]
        
        frames_groups = []
        for idx, (time_start, time_end) in enumerate(zip(time_token_start_indices, time_token_end_indices)):
            if idx + 1 < len(time_token_start_indices):
                frames_group_end = time_token_start_indices[idx + 1]
            else:
                frames_group_end = visual_token_end_pos
            frames_groups.append(
                (time_start, time_end, frames_group_end)
            )    
        self.video_info = {
            "frames_groups": frames_groups,
            "visual_token_start_pos": visual_token_start_pos,
            "visual_token_end_pos": visual_token_end_pos
        }
        
        # time_tuple_indices = []
        # shift = self.get_fvr_length()
        # for time_start, time_end, _ in frames_groups:
        #     time_tuple_indices.append(
        #         (shift+time_start, shift+time_end)
        #     )
        # self.record_whole_fvr_times_indices.extend(
        #     time_tuple_indices
        # )
        # pdb.set_trace()  # (0,1-)
    
    def construct_past_kvs(self, use_sparse_past: bool = False, coarse_time_token_index: Optional[tuple] = None):
        
        if self.infer_stage == 'prefill_t':
            for layer_idx in range(self.layer_num):
                self.key_cache[layer_idx] = self.system_prompt_k[layer_idx].clone()
                self.value_cache[layer_idx] = self.system_prompt_v[layer_idx].clone()
            updated_past_kvs_position_ids = self.past_sp_position_ids.clone()
            self.past_kvs_position_ids = updated_past_kvs_position_ids
        
        elif self.infer_stage in ['prefill_v', 'coarse_decode']:
            # reload
            if use_sparse_past:
                for layer_idx in range(self.layer_num):
                    self.key_cache[layer_idx] = torch.concat([self.system_prompt_k[layer_idx], self.coarse_video_rep_k[layer_idx]], dim=2)
                    self.value_cache[layer_idx] = torch.concat([self.system_prompt_v[layer_idx], self.coarse_video_rep_v[layer_idx]], dim=2)
                if self.reforge_pos:
                    updated_past_kvs_position_ids = torch.arange(
                        self.key_cache[0].size(2), device=self.key_cache[0].device, dtype=torch.long
                    ).unsqueeze(dim=0)
                else:
                    updated_past_kvs_position_ids = torch.concat([self.past_sp_position_ids, self.past_cvr_position_ids], dim=-1)
            else:            
                for layer_idx in range(self.layer_num):
                    self.key_cache[layer_idx] = torch.concat([self.system_prompt_k[layer_idx], self.fine_video_rep_k[layer_idx]], dim=2)
                    self.value_cache[layer_idx] = torch.concat([self.system_prompt_v[layer_idx], self.fine_video_rep_v[layer_idx]], dim=2)
                if self.reforge_pos:
                    updated_past_kvs_position_ids = torch.arange(
                        self.key_cache[0].size(2), device=self.key_cache[0].device, dtype=torch.long
                    ).unsqueeze(dim=0)
                else:
                    updated_past_kvs_position_ids = torch.concat([self.past_sp_position_ids, self.past_fvr_position_ids], dim=-1)

            self.past_kvs_position_ids = updated_past_kvs_position_ids
        
        elif self.infer_stage == 'fine_decode':            
            coarse_token_index_start, coarse_token_index_end = coarse_time_token_index

            # reload fine video representation between (coarse_token_index_start, coarse_token_index_end)           
            for layer_idx in range(self.layer_num):
                reloaded_fine_video_rep_k = self.fine_video_rep_k[layer_idx][:, :, coarse_token_index_start:coarse_token_index_end, :]
                reloaded_fine_video_rep_v = self.fine_video_rep_v[layer_idx][:, :, coarse_token_index_start:coarse_token_index_end, :]
                self.key_cache[layer_idx] = torch.concat([self.system_prompt_k[layer_idx], reloaded_fine_video_rep_k], dim=2)
                self.value_cache[layer_idx] = torch.concat([self.system_prompt_v[layer_idx], reloaded_fine_video_rep_v], dim=2)
            # if self.reforge_pos:
            updated_past_kvs_position_ids = torch.arange(
                self.key_cache[0].size(2), device=self.key_cache[0].device, dtype=torch.long
            ).unsqueeze(dim=0)
            # else:
            #     updated_past_kvs_position_ids = torch.concat([self.past_sp_position_ids, self.past_fvr_position_ids[:, coarse_token_index_start:coarse_token_index_end]], dim=-1)
            
            self.past_kvs_position_ids = updated_past_kvs_position_ids

    def get_cache_position(self, input_ids=None, input_embeds=None):
        start_pos = self.past_kvs_position_ids[:, -1].max().item() + 1
        if input_embeds is not None:
            cache_position = torch.arange(start_pos, start_pos + input_embeds.size(1), device=input_embeds.device, dtype=torch.long)
        elif input_ids is not None:
            cache_position = torch.arange(start_pos, start_pos + input_ids.size(1), device=input_ids.device, dtype=torch.long)
        else:
            raise ValueError("Either input_ids or input_embeds must be provided.")
        return cache_position