# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA attention backend that avoids Triton unified decode kernels."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class CudaPagedAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None


class CudaPagedAttentionMetadataBuilder(
    AttentionMetadataBuilder[CudaPagedAttentionMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> CudaPagedAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CudaPagedAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        use_cascade = common_prefix_len > 0
        prefix_scheduler_metadata = None

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None

        return CudaPagedAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )


class CudaPagedAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32]

    @staticmethod
    def get_name() -> str:
        return "CUDA_PAGED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["CudaPagedAttentionImpl"]:
        return CudaPagedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["CudaPagedAttentionMetadataBuilder"]:
        return CudaPagedAttentionMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 112, 120, 128, 160, 192, 224, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(6, 0)


class CudaPagedAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.fp8_dtype = current_platform.fp8_dtype()
        self.sinks = sinks

        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: CudaPagedAttentionMetadata,
    ) -> torch.Tensor:
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=attn_metadata.query_start_loc,
            b_seq_len=attn_metadata.seq_lens,
            max_input_len=attn_metadata.max_query_len,
            is_causal=False,
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def _run_sdpa_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        start_req_idx: int,
        num_reqs: int,
    ) -> None:
        enable_gqa = self.num_heads > self.num_kv_heads
        for req_idx in range(start_req_idx, num_reqs):
            start = int(query_start_loc[req_idx].item())
            end = int(query_start_loc[req_idx + 1].item())
            q = query[start:end].movedim(0, 1).unsqueeze(0)
            k = key[start:end].movedim(0, 1).unsqueeze(0)
            v = value[start:end].movedim(0, 1).unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale,
                enable_gqa=enable_gqa,
            )
            output[start:end].copy_(out.squeeze(0).movedim(1, 0))

    def _run_cuda_decode_attention(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: CudaPagedAttentionMetadata,
        output: torch.Tensor,
    ) -> bool:
        if self.sliding_window[0] != -1 or self.sinks is not None:
            return False

        query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        if attn_metadata.max_query_len <= 1:
            num_decodes = attn_metadata.seq_lens.shape[0]
        elif query_lens[0].item() > 1:
            num_decodes = 0
        else:
            is_prefill = query_lens > 1
            num_decodes = (
                is_prefill.int().argmax(dim=-1).item()
                if torch.any(is_prefill)
                else attn_metadata.seq_lens.shape[0]
            )

        if num_decodes == 0:
            return True

        num_decode_tokens = int(attn_metadata.query_start_loc[num_decodes].item())
        if num_decode_tokens != num_decodes:
            return False

        decode_query = query[:num_decode_tokens]
        decode_output = output[:num_decode_tokens]
        decode_block_table = attn_metadata.block_table[:num_decodes].to(torch.int32)
        decode_seq_lens = attn_metadata.seq_lens[:num_decodes]
        decode_max_seq_len = int(decode_seq_lens.max().item())
        block_size = value_cache.shape[3]

        from vllm import _custom_ops as ops

        if decode_max_seq_len <= 8192:
            ops.paged_attention_v1(
                decode_output,
                decode_query,
                key_cache,
                value_cache,
                self.num_kv_heads,
                self.scale,
                decode_block_table,
                decode_seq_lens,
                block_size,
                decode_max_seq_len,
                self.alibi_slopes,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        else:
            partition_size = 512
            num_partitions = (decode_max_seq_len + partition_size - 1) // partition_size
            tmp_output = torch.empty(
                size=(num_decodes, self.num_heads, num_partitions, self.head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_decodes, self.num_heads, num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                decode_output,
                exp_sums,
                max_logits,
                tmp_output,
                decode_query,
                key_cache,
                value_cache,
                self.num_kv_heads,
                self.scale,
                decode_block_table,
                decode_seq_lens,
                block_size,
                decode_max_seq_len,
                self.alibi_slopes,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        return True

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CudaPagedAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for CudaPagedAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False
        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
            )

        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens] if key is not None else None
        value = value[:num_actual_tokens] if value is not None else None
        output = output[:num_actual_tokens]

        query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        has_prefill = bool(torch.any(query_lens > 1).item())
        has_extend = has_prefill and bool(
            torch.any(attn_metadata.seq_lens != query_lens).item()
        )

        used_fast_paths = False
        if (
            key is not None
            and value is not None
            and output_scale is None
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and not has_extend
        ):
            if has_prefill:
                if query_lens[0].item() > 1:
                    num_decodes = 0
                else:
                    is_prefill = query_lens > 1
                    num_decodes = (
                        is_prefill.int().argmax(dim=-1).item()
                        if torch.any(is_prefill)
                        else attn_metadata.seq_lens.shape[0]
                    )
                self._run_sdpa_prefill(
                    query,
                    key,
                    value,
                    output,
                    attn_metadata.query_start_loc,
                    num_decodes,
                    attn_metadata.seq_lens.shape[0],
                )
            if self._run_cuda_decode_attention(
                layer, query, key_cache, value_cache, attn_metadata, output
            ):
                used_fast_paths = True

        if not used_fast_paths:
            chunked_prefill_paged_decode(
                query=query,
                key=key,
                value=value,
                output=output,
                kv_cache_dtype=self.kv_cache_dtype,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=attn_metadata.block_table,
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                max_seq_len=attn_metadata.max_seq_len,
                max_query_len=attn_metadata.max_query_len,
                k_scale=layer._k_scale,
                v_scale=layer._v_scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window[0],
                sm_scale=self.scale,
                output_scale=output_scale,
                sinks=self.sinks,
            )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )
        PagedAttention.write_to_paged_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
