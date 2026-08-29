import argparse
import re
import transformers.modeling_utils as modeling_utils
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers import GenerationConfig

# VideoXL was written against Transformers 4.43, where this helper was
# re-exported from modeling_utils. Keep the compatibility change local to this
# inference entry point instead of changing the installed Transformers package.
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

for _name, _value in {
    "apply_chunking_to_forward": apply_chunking_to_forward,
    "find_pruneable_heads_and_indices": find_pruneable_heads_and_indices,
    "prune_linear_layer": prune_linear_layer,
}.items():
    if not hasattr(modeling_utils, _name):
        setattr(modeling_utils, _name, _value)

# Transformers 4.57 requires Cache(layers=...), whereas EviScan's MyCache
# owns its per-layer tensors itself and was constructed with Cache() in 4.43.
_cache_init = Cache.__init__


def _timescope_cache_init(self, *args, **kwargs):
    if self.__class__.__name__ == "MyCache" and not args and not kwargs:
        return _cache_init(self, layers=[])
    return _cache_init(self, *args, **kwargs)


Cache.__init__ = _timescope_cache_init
_cache_max_cache_len = Cache.max_cache_len


def _timescope_get_max_cache_len(self):
    if hasattr(self, "_timescope_max_cache_len"):
        return self._timescope_max_cache_len
    return _cache_max_cache_len.fget(self)


def _timescope_set_max_cache_len(self, value):
    self._timescope_max_cache_len = value


Cache.max_cache_len = property(_timescope_get_max_cache_len, _timescope_set_max_cache_len)

# The local VideoXL model predates Transformers 4.50 and expects
# PreTrainedModel to provide GenerationMixin. Restore that contract locally.
if not hasattr(modeling_utils.PreTrainedModel, "generate"):
    for _name, _value in GenerationMixin.__dict__.items():
        if not _name.startswith("__") and not hasattr(modeling_utils.PreTrainedModel, _name):
            setattr(modeling_utils.PreTrainedModel, _name, _value)

from eviscan.eviscan.model.builder import load_pretrained_model
from eviscan.eviscan.mm_utils import tokenizer_image_token
from eviscan.eviscan.constants import IMAGE_TOKEN_INDEX
from decord import VideoReader, cpu
import torch
from eviscan.eviscan.model import MyCache
import yaml

def load_video(video_path, max_frames_num=None):
    # 初始化 VideoReader
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()

    # 如果未指定 max_frames_num，则按 1 FPS 采样整个视频
    if max_frames_num is None:
        video_duration = total_frame_num / fps  # 视频总时长（秒）
        max_frames_num = int(video_duration)   # 1 FPS 采样，总帧数 ≈ 视频时长（秒）

    # 计算采样间隔（按 1 FPS）
    # 例如：fps=30，则每隔 30 帧采 1 帧
    sampling_interval = int(fps)  # 确保 1 秒采 1 帧
    frame_idx = list(range(0, total_frame_num, sampling_interval))[:max_frames_num]

    # 获取帧数据
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    timestamps = [round(frame_index / fps, 1) for frame_index in frame_idx]

    return spare_frames, timestamps


def chunk_prefilling(model, image_processor, chunk_frames, chunk_times, input_ids, mycache, tokenizer):
    chunk_time_stamps = []
    token_frames_sum=(len(chunk_times)+3)//4
    compress_frame = chunk_times[::4]
    chunk_time_embedding = []
    for time in compress_frame:
        time="{:06.1f}".format(time)
        item = f"Time {time}s:"
        chunk_time_embedding.append(tokenizer(item).input_ids)
        chunk_time_embedding.append([151654]*144)

    chunk_time_embedding = [item for sublist in chunk_time_embedding for item in sublist]

    chunk_time_embedding = torch.tensor(chunk_time_embedding, dtype=torch.long).to(model.device)
    chunk_time_stamps.append(chunk_time_embedding)

    chunk_video_tensor = image_processor.preprocess(chunk_frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    
    # Prefill coarse and fine video representation
    with torch.inference_mode():
        visual_token_start_pos = mycache.get_fvr_length()
        # cache_position = mycache.get_cache_position() # modeling_qwen2 中有这一步
        mycache.preprocess_video_info(input_ids=input_ids, time_stamps=chunk_time_stamps, visual_token_start_pos=0)
        output = model(input_ids, images=[chunk_video_tensor], modalities=['video'], time_embedding=chunk_time_stamps, return_dict=True, use_cache=True, past_key_values=mycache)
        mycache = output.past_key_values
        
    return mycache


def extract_time_range(text):
    values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(values) < 2:
        raise ValueError(f"Expected two timestamps in model output, got: {text!r}")
    return values[-2:]


def mapped_to_token_indices(coarse_time_scope, times, tokenizer):
    grouped_times = times[::4]
    if not grouped_times:
        raise ValueError("No sampled timestamps are available for fine decoding")

    coarse_start, coarse_end = coarse_time_scope
    start_idx = max(index for index, time in enumerate(grouped_times) if time <= coarse_start) if any(
        time <= coarse_start for time in grouped_times
    ) else 0
    end_idx = next((index for index, time in enumerate(grouped_times) if time >= coarse_end), len(grouped_times) - 1)
    if end_idx < start_idx:
        end_idx = start_idx

    token_offsets = []
    token_count = 0
    for time in grouped_times:
        token_offsets.append(token_count)
        token_count += len(tokenizer(f"Time {time:06.1f}s:").input_ids) + 144
    return (grouped_times[start_idx], grouped_times[end_idx]), (token_offsets[start_idx], token_count if end_idx + 1 == len(grouped_times) else token_offsets[end_idx + 1])

def main():
    parser = argparse.ArgumentParser(description="Run EviScan coarse temporal grounding on one video.")
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--question", default="When does the main action happen?", help="Temporal grounding question.")
    parser.add_argument("--model-path", default="/comp_robot/lxr/LXRRRRRR/timescope")
    parser.add_argument("--max-frames", type=int, default=None, help="Sample at most this many 1-FPS frames.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    mem_config_path = "memory_config/uniform_cmpr2_reforge_mid.yaml"
    with open(mem_config_path, 'r', encoding='utf-8') as file:
        mem_config = yaml.safe_load(file)

    gen_kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}
    print(f"Loading model from {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, "llava_qwen", device_map=args.device)
    if model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    frames, times = load_video(args.video, args.max_frames)
    if not len(frames):
        raise ValueError(f"No frames decoded from {args.video}")
    print(f"Loaded {len(frames)} frames from {args.video}")
        
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    visual_place_holder = "<image>"
    
    mycache = MyCache(**mem_config)
    mycache.set_infer_stage('prefill_t')
    input_ids = tokenizer_image_token(system_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    output = model(input_ids, return_dict=True, use_cache=True, past_key_values=mycache)
    mycache = output.past_key_values
    mycache.construct_past_kvs()
        
    mycache.set_infer_stage('prefill_v')
    chunk_size = 12
    use_sparse_past = True
    for start_idx in range(0, len(frames), chunk_size):
            end_idx = min(start_idx + chunk_size, len(times))
            chunk_frames = frames[start_idx:end_idx]
            chunk_times = times[start_idx:end_idx]
            input_ids = tokenizer_image_token(visual_place_holder, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            print(f'------------Chunk-{start_idx//chunk_size} before prefilling------------')
            print('fvr_length:', mycache.get_fvr_length())
            print('cvr_length:', mycache.get_cvr_length())
            print('seq_length:', mycache.get_seq_length())
            mycache = chunk_prefilling(model, image_processor, chunk_frames, chunk_times, input_ids, mycache, tokenizer)
            print(f'------------Chunk-{start_idx//chunk_size} after prefilling------------')
            print('fvr_length:', mycache.get_fvr_length())
            print('cvr_length:', mycache.get_cvr_length())
            print('seq_length:', mycache.get_seq_length())
            mycache.construct_past_kvs(use_sparse_past=use_sparse_past)
            print(f'------------Chunk-{start_idx//chunk_size} after constructing------------')
            print('fvr_length:', mycache.get_fvr_length())
            print('cvr_length:', mycache.get_cvr_length())
            print('seq_length:', mycache.get_seq_length())
        
    print("Coarse Timescope Prediction")
        # print(f'------------Coarse Timescope Prediction------------')
    mycache.set_infer_stage('coarse_decode')
        # NOTE: 为了实现在 generate 过程中使用 past_key_values， 必须要修改 transformers 的内部代码，建议直接使用这个环境: "/share/project/minghao/Envs/lmms"
        # coarse time scope prediction:
    mycache.construct_past_kvs(use_sparse_past=use_sparse_past)
    user_prompt = '\n' + args.question + ' Please answer the approximate time period.<|im_end|>\n<|im_start|>assistant\n'
    input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(model.device)
    cache_position = mycache.get_cache_position(input_ids=input_ids)
    output = model.generate(input_ids, return_dict_in_generate=True, use_cache=True, past_key_values=mycache, cache_position=cache_position, **gen_kwargs)
    coarse_pred = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
    print(f"Coarse prediction: {coarse_pred}")

    coarse_time_scope = extract_time_range(coarse_pred)
    mapped_times, mapped_token_index = mapped_to_token_indices(coarse_time_scope, times, tokenizer)
    print(f"Coarse times after mapping: {mapped_times}")

    print("Fine Timescope Prediction")
    mycache.set_infer_stage('fine_decode')
    mycache.construct_past_kvs(coarse_time_token_index=mapped_token_index)
    user_prompt = '\n' + args.question + ' Please answer the exact time.<|im_end|>\n<|im_start|>assistant\n'
    input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(model.device)
    cache_position = mycache.get_cache_position(input_ids=input_ids)
    output = model.generate(input_ids, return_dict_in_generate=True, use_cache=True, past_key_values=mycache, cache_position=cache_position, **gen_kwargs)
    fine_pred = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
    print(f"Fine Pred: {fine_pred}")
        
if __name__ == '__main__':
    main()
        # coarse_time_scope = extract_time_range(coarse_pred)
        # assert len(coarse_time_scope) == 2 and type(coarse_time_scope[0]) == float and type(coarse_time_scope[1]) == float, f'coarse_time_scope: {coarse_pred}'
        # print(f'Extracted Coarse Time Scope: {coarse_time_scope}')
        # print('kv seq_length when coarse decoding:', mycache.get_seq_length())
        
        # print_header("Fine Timescope Prediction", level=1)
        # # print(f'------------Fine Timescope Prediction------------')
        # # fine time scope prediction:
        # # 将预测出来的时间映射到 4帧一组的 时间戳上
        # mapped_times, mapped_token_index = mapped_to_token_indices(coarse_time_scope, times)
        # print(f'Coarse Times After Mapped: {mapped_times}')  # print(f'Extracted Mapped Times Index: {mapped_times_index}')
        # print(f'Coarse Times After Mapped Token Index: {mapped_token_index}')  # print(f'Extracted Mapped Times Index: {mapped_times_index}')
        # mycache.set_infer_stage('fine_decode')
        # mycache.construct_past_kvs(coarse_time_token_index=mapped_token_index)
        # user_prompt = '\n' + ques['question'] +' Please answer the exact time.<|im_end|>\n<|im_start|>assistant\n'
        # input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(model.device)
        # cache_position = mycache.get_cache_position(input_ids=input_ids)
        # output = model.generate(input_ids, return_dict_in_generate=True, use_cache=True, past_key_values=mycache, cache_position=cache_position, **gen_kwargs)
        # output_ids = output.sequences
        # fine_pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(f'Fine Pred: {fine_pred}')
        # fine_time_scope = extract_time_range(fine_pred)
        # print(f'Extracted Fine Time Scope: {fine_time_scope}')
        # print('kv seq_length when fine decoding:', mycache.get_seq_length())

# kv length:
# sys: 14 
# fvr: 462(1232-770) 
# user: 24 
# output: 17
# total: 516
