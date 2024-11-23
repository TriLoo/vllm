
from functools import lru_cache, cached_property
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers import Idefics2Config
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig, Idefics2PerceiverConfig
from xformers import ops as xops

from PIL import Image

from vllm.attention import AttentionMetadata
from vllm.attention.selector import _Backend
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, InputContext,
                         token_inputs)
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.pooler import PoolingMetadata, Pooler, PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SequenceData, PoolerOutput
from vllm.transformers_utils.processor import get_processor

from .interfaces import SupportsMultiModal, SupportsPP
from .idefics2_vision_model import Idefics2VisionTransformer
from .utils import AutoWeightsLoader, merge_multimodal_embeddings, get_vit_attn_backend, is_pp_missing_parameter, PPMissingLayer

cached_get_processor = lru_cache(get_processor)

def get_resampler_n_latents(ctx: InputContext) -> int:
    hf_config = ctx.get_hf_config(Idefics2Config)
    return hf_config.perceiver_config.resampler_n_latents

cached_get_resampler_n_latents = lru_cache(get_resampler_n_latents)

class Idefics2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""
    pixel_attn_mask: torch.Tensor
    """shape: `(batch_size * num_images, height, width)`"""


class Idefics2ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """

Idefics2ImageInputs = Union[Idefics2ImagePixelInputs,
                             Idefics2ImageEmbeddingInputs]

def get_idefics2_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size

def get_idefics2_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_idefics2_patch_grid_length(image_size=image_size,
                                                 patch_size=patch_size)
    return grid_length ** 2

def get_idefics2_image_feature_size(hf_config: Idefics2Config) -> int:
    # return get_idefics2_num_patches(image_size=hf_config.image_size,
    #                                 patch_size=hf_config.patch_size)
    # TODO: check the image feature size
    return hf_config.perceiver_config.resampler_n_latents

def dummy_seq_data_for_idefics2(
    hf_config: Idefics2Config,
    seq_len: int,
    num_images: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_idefics2_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    return SequenceData.from_prompt_token_counts(
        (image_token_id, image_feature_size * num_images),      # IMAGE
        (0, seq_len - image_feature_size * num_images),         # TEXT
    )


def dummy_image_for_idefics2(
    hf_config: Idefics2VisionConfig,
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    width = height = 378
    image = Image.new("RGB", (width, height), color=0)
    # print('---- num_images: ', num_images)
    return {"image": image if num_images == 1 else [image] * num_images}
    

def dummy_data_for_idefics2(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(Idefics2Config)
    vision_config = hf_config.vision_config
    mm_images = mm_counts["image"]

    seq_data = dummy_seq_data_for_idefics2(
        hf_config,
        seq_len,
        mm_images,
        image_token_id=hf_config.image_token_id,
    )
    mm_data = dummy_image_for_idefics2(vision_config, mm_images)
    return seq_data, mm_data

    
def get_max_idefics2_image_tokens(ctx: InputContext):
    return cached_get_resampler_n_latents(ctx)


# TODO: change this function to class
def input_processor_for_idefics2(ctx: InputContext,     # model_config
                                 inputs: DecoderOnlyInputs) -> DecoderOnlyInputs:
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs
    
    image_inputs = multi_modal_data.get("image", None)
    if image_inputs is None:
        return inputs
    
    processor = cached_get_processor(ctx.model_config.model)
    image_processor = processor.image_processor
    prompt_token_ids = inputs.get("prompt_token_ids", None)
    print('---- current prompt_token_ids: ', prompt_token_ids)
    fake_image_token = processor.fake_image_token
    image_token = processor.image_token
    image_seq_len = cached_get_resampler_n_latents(ctx)
    text = inputs['prompt']

    # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
    fake_image_token = fake_image_token.content
    image_token = image_token.content
    fake_image_token_id = processor.tokenizer.added_tokens_encoder[fake_image_token]
    image_token_id = processor.tokenizer.added_tokens_encoder[image_token]
    image_str = f"{fake_image_token}{image_token * image_seq_len}{fake_image_token}"
    if image_processor.do_image_splitting:
        # A single image token is split into 4 patches + 1 original image
        image_str = image_str * 5
    # if isinstance(text, str):
    #     text = [text]
    # elif not isinstance(text, list) and not isinstance(text[0], str):
    #     raise ValueError("Invalid input text. Please provide a string, or a list of strings")
    new_prompt_token_ids = []
    new_prompt = text.replace(image_token, image_str).replace(f"{fake_image_token}{fake_image_token}", f"{fake_image_token}")
    print('current new_prompt: ', new_prompt)
    for token_id in prompt_token_ids:
        if token_id == image_token_id:
            new_prompt_token_ids.extend([fake_image_token_id] + [image_token_id] * image_seq_len + [fake_image_token_id])
        else:
            new_prompt_token_ids.append(token_id)
    print('current new_prompt_token_ids: ', new_prompt_token_ids)

    return token_inputs(prompt_token_ids=new_prompt_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data)
    
    #Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py/#L748
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to 
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class Idefics2MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int,
                 output_size: int, hidden_act: str,
                 quant_config: QuantizationConfig):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = get_act_fn(hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Idefics2PerceiverAttention(nn.Module):

    def __init__(self,
                 config,
                 quant_config,
                 layer_idx: Optional[int] = None) -> None:
        """Perceiver Cross-Attention Module --> 
                    let long-form inputs be `context`,
         resampled embeddings be `latents`"""
        super().__init__()

        self.layer_idx = None
        self.hidden_size = config.text_config.hidden_size
        self.num_heads = config.perceiver_config.resampler_n_heads
        self.head_dim = config.perceiver_config.resampler_head_dim
        self.num_key_value_heads = config.perceiver_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.perceiver_config.attention_dropout

        self.q_proj = ColumnParallelLinear(self.hidden_size,
                                           self.num_heads * self.head_dim,
                                           bias=False,
                                           quant_config=quant_config)
        self.k_proj = ColumnParallelLinear(self.hidden_size,
                                           self.num_key_value_heads *
                                           self.head_dim,
                                           bias=False,
                                           quant_config=quant_config)
        self.v_proj = ColumnParallelLinear(self.hidden_size,
                                           self.num_key_value_heads *
                                           self.head_dim,
                                           bias=False,
                                           quant_config=quant_config)
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.is_causal = False
        self.attn_backend: _Backend = get_vit_attn_backend()

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) 
        appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, 
            embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] 
            representing  long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape 
            [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape 
            [bsz, seq] representing position indices of each input token.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Tuple 
            of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): 
            Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use 
            past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        print('shape of latents: ', latents.shape)
        print('shape of context: ', context.shape)
        print('device of latents: ', latents.device)
        print('device of context: ', context.device)
        print('dtype of latents: ', latents.dtype)
        print('dtype of context: ', context.dtype)
        print('type of latents: ', type(latents))
        print('type of context: ', type(context))
        hidden_states = torch.concat([context, latents], dim=-2)

        query_states = self.q_proj(latents)[0]
        key_states = self.k_proj(hidden_states)[0]
        value_states = self.v_proj(hidden_states)[0]

        ## flash attention
        # not support here

        ## F.scaled_dot_product_attention
        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len,
                                         self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)
        ## repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # print('---- shape of query_states: ', query_states.shape)
        # print('shape of key_states: ', key_states.shape)
        # print('shape of value_states: ', value_states.shape)
        # print('shape of attention_mask: ', attention_mask.shape)
        # print('dtype of attention_mask: ', attention_mask.dtype)
        attn_output = F.scaled_dot_product_attention(query_states, 
                                                     key_states, 
                                                     value_states,
                                                     attn_mask=attention_mask,
                                                     dropout_p=0.0)

        ## xops.memory_efficient_attention_forward
        # not supported
        # query_states = query_states.view(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups,
        #                                  self.head_dim)
        # key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, 1,
        #                              self.head_dim).expand([bsz, kv_seq_len, self.num_key_value_heads, self.num_key_value_groups, self.head_dim])
        # value_states = value_states.view(bsz, kv_seq_len,
        #                                  self.num_key_value_heads, 1,
        #                                  self.head_dim).expand([bsz, kv_seq_len, self.num_key_value_heads, self.num_key_value_groups, self.head_dim])
        # context_layer = xops.memory_efficient_attention_forward(
        #     query_states, key_states, value_states, attn_bias=attention_mask.to(torch.float32), p=0, scale=None)
        # attn_output = rearrange(context_layer,
        #                           "b s h d -> b s (h d)").contiguous()
        # attn_output = context_layer.reshape(bsz, q_len, latent_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        # print('shape of attn_output: ', attn_output.shape)

        attn_output = self.o_proj(attn_output)
        return attn_output


class Idefics2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config, quant_config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        # self.input_latents_norm = RMSNorm(self.hidden_size,
        self.input_latents_norm = Idefics2RMSNorm(self.hidden_size,
                                          eps=self.rms_norm_eps)
        # self.input_context_norm = RMSNorm(self.hidden_size,
        self.input_context_norm = Idefics2RMSNorm(self.hidden_size,
                                          eps=self.rms_norm_eps)
        self.self_attn = Idefics2PerceiverAttention(config,
                                                    quant_config,
                                                    layer_idx=layer_idx)
        # self.post_attention_layernorm = RMSNorm(self.hidden_size,
        self.post_attention_layernorm = Idefics2RMSNorm(self.hidden_size,
                                                eps=self.rms_norm_eps)
        self.mlp = Idefics2MLP(
            hidden_size=config.text_config.hidden_size,
            intermediate_size=config.text_config.hidden_size * 4,
            output_size=config.text_config.hidden_size,
            hidden_act=config.perceiver_config.hidden_act,
            quant_config=quant_config)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        residual = latents
        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=attention_mask,
        )

        latents = residual + latents[0]
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        outputs = (latents, )
        return outputs


class Idefics2PerceiverResampler(nn.Module):

    def __init__(self, config, quant_config) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of 
        embeddings (say from a ResNet or ViT or MAE) of a given dimension, 
        performs `depth` blocks of cross-attention with a fixed `n_latents` 
        inputs, then returns a Tensor of shape [bsz, n_latents, embed_dim]. 
        The Resampler acts as a form of learned pooling and is derived from 
        [Perceiver: General Perception with Iterative Attention]
        (https://arxiv.org/abs/2103.03206).
        """
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps
        self.quant_config = quant_config
        # Create Latents for Perceiver
        self.latents = nn.Parameter(
            torch.ones(self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        self.layers = nn.ModuleList([
            Idefics2PerceiverLayer(config, quant_config, idx)
            for idx in range(self.depth)
        ])
        # self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self._use_flash_attention_2 =\
             config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        context: torch.Tensor,
        attention_mask,
    ) -> torch.Tensor:
        # seq embed -> bsz seq embed
        latents = self.latents.unsqueeze(0).expand(
            (context.shape[0], *self.latents.size()))

        latent_attention_mask = torch.ones(
            (attention_mask.size(0), latents.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latent_attention_mask],
                                   dim=-1)
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, latents.dtype, tgt_len=self.n_latents)

        compressed_context = latents
        for perceiver_layer in self.layers:
            layer_outputs = perceiver_layer(
                compressed_context,
                context,
                attention_mask=attention_mask,
            )
            compressed_context = layer_outputs[0]

        compressed_context = self.norm(compressed_context)

        return compressed_context


class Idefics2Connector(nn.Module):

    def __init__(self, config: Idefics2Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.modality_projection = Idefics2MLP(
            hidden_size=config.vision_config.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            output_size=config.text_config.hidden_size,
            hidden_act=config.text_config.hidden_act,
            quant_config=quant_config)
        self.perceiver_resampler = Idefics2PerceiverResampler(
            config, quant_config)
        self.quant_config = quant_config

    def forward(self, image_hidden_states, attention_mask):
        image_hidden_states = self.modality_projection(image_hidden_states)
        image_hidden_states = self.perceiver_resampler(
            context=image_hidden_states, attention_mask=attention_mask)
        return image_hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper()  # default mapper
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_idefics2_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_idefics2)
@INPUT_REGISTRY.register_input_processor(input_processor_for_idefics2)
class Idefics2ForConditionalGeneration(nn.Module, SupportsMultiModal, 
                                       SupportsPP):
    def __init__(self,
                 config: Idefics2Config,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_model = Idefics2VisionTransformer(config.vision_config)
        self.connector = Idefics2Connector(config)
        # self.text_model = LlamaForCausalLM(config.text_config, cache_config, quant_config)
        self.text_model = LlamaModel(config.text_config, cache_config, quant_config)
        # self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.text_config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.text_config.hidden_size,
                org_num_embeddings=config.text_config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                ),
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config.text_config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.text_config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.text_model.make_empty_intermediate_tensors)
    
    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        longest_edge = self.config.vision_config.image_size
        actual_dims = tuple(data.shape[1:])

        # if max(actual_dims) > longest_edge:
        #     expected_expr = ("longest edge", *map(str, longest_edge))
        #     raise ValueError(
        #         f"The expected longest edge of pixel values is {expected_expr}. "
        #         f"You supplied {tuple(data.shape)}.")

        return data
    
    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Idefics2ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)

        if pixel_attention_mask is None:
            return None
        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            # Remove the N dimension until multiple images are supported.
            pixel_values = pixel_values.squeeze(1)
            pixel_attention_mask = pixel_attention_mask.squeeze(1)

            return Idefics2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
                pixel_attn_mask=pixel_attention_mask,
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            # Remove the N dimension until multiple images are supported.
            image_embeds = image_embeds.squeeze(1)
            return Idefics2ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
                pixel_attn_mask = pixel_attention_mask,
            )
        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        vision_model: Idefics2VisionTransformer,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        target_dtype = vision_model.get_input_embeddings().patch_embedding.weight.dtype
        image_features = vision_model(pixel_values.to(dtype=target_dtype), patch_attention_mask)

        return image_features       # last hidden state
    
    def _process_image_input(
        self,
        image_input: Idefics2ImageInputs,
    ) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None
        pixel_values = image_input["data"]
        pixel_attention_mask = image_input["pixel_attn_mask"]
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
        print('shape of pixel attention mask: ', pixel_attention_mask.shape)
        pixel_attention_mask = pixel_attention_mask.reshape(-1, *pixel_attention_mask.shape[-2:])
        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)         # (B, H // size, W, size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)              # (B, H // size, W // size, size, size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) == patch_size * patch_size).bool()         # (B, H // size, W // size)
        print('shape of pixel_values: ', pixel_values.shape)
        print('shape of patch attention mask: ', patch_attention_mask.shape)
        # print('shape of pixel_attention_mask: ', pixel_attention_mask.shape)
        # print('shape of patch_attention_mask: ', patch_attention_mask.shape)
        # print('dtype of patch attention mask: ', patch_attention_mask.dtype)
        image_features = self._image_pixels_to_features(
            self.vision_model,
            pixel_values,
            patch_attention_mask
        )
        # image_features = torch.rand(256, 729, 1152).cuda().to(torch.float16)
        return self.connector(image_features, patch_attention_mask.view(pixel_values.size(0), -1))

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs: object) -> Union[SamplerOutput, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            parsed_image_input = self._parse_and_validate_image_input(**kwargs)

            if parsed_image_input is not None:
                vision_embeddings = self._process_image_input(
                    parsed_image_input)
                inputs_embeds = self.text_model.get_input_embeddings(
                    input_ids)
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.config.image_token_id)

                input_ids = None
            else:
                inputs_embeds = None

        hidden_states = self.text_model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    # def pooler(
    #     self,
    #     hidden_states: torch.Tensor,
    #     pooling_metadata: PoolingMetadata,
    # ) -> Optional[PoolerOutput]:
    #     return self._pooler(hidden_states, pooling_metadata)
    
    def drop_model_prefix(self, 
                    name: str, 
                    loaded_weight: torch.Tensor)-> Tuple[str, torch.Tensor]:
        if name.startswith("model."):
            # name = name.replace("model.", "")
            name = name[6:]
        # if name.startswith('text_model.'):
        #     name = 'text_model.model.' + name[11:]
        # if name == 'lm_head.weight':
        #     name = 'text_model.lm_head.weight'
        return name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self,             
                                    skip_prefixes=(["vision_model."]),)
                                   # skip_prefixes=(["lm_head."]
                                   # if self.config.text_config.tie_word_embeddings else None),)
        # loader = AutoWeightsLoader(self)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        weights = [self.drop_model_prefix(name, loaded_weight) for name, loaded_weight in weights]
        remain_weights = []
        for name, loaded_weight in weights:
            if not name.startswith('vision_model.'):
                remain_weights.append((name, loaded_weight))
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        # loader.load_weights(self.drop_model_prefix(name, loaded_weight) for name, loaded_weight in weights)
        loader.load_weights(remain_weights)
        # print('shape of lm_head: ', self.lm_head.weight.data.shape)
        # print('shape of text_model.lm_head: ', self.text_model.lm_head.weight.data.shape)
        # self.text_model.lm_head.weight.data.copy_(self.lm_head.weight.data)
    