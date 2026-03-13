# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from typing import DefaultDict

import torch

from vllm import SamplingParams
from vllm.config import VllmConfig

from .interface import BatchUpdate, LogitsProcessor, MoveDirectionality

_NO_REPEAT_NGRAM_ARG = "no_repeat_ngram_size"


def _parse_no_repeat_ngram_size(params: SamplingParams) -> int:
    extra_args = params.extra_args
    if not isinstance(extra_args, dict):
        return 0

    value = extra_args.get(_NO_REPEAT_NGRAM_ARG)
    if value is None:
        return 0
    if not isinstance(value, int):
        raise ValueError(
            f"{_NO_REPEAT_NGRAM_ARG} must be an integer, got {type(value).__name__}."
        )
    return max(value, 0)


@dataclass
class _RequestState:
    ngram_size: int
    output_token_ids: MutableSequence[int]
    seen_ngrams: DefaultDict[tuple[int, ...], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Applies no-repeat n-gram masking based on `SamplingParams.extra_args`.

    Requests opt in by passing `extra_args={"no_repeat_ngram_size": N}`.
    This is used by MinerU's vLLM client path and is also generally useful for
    other multimodal extraction workloads.
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        _parse_no_repeat_ngram_size(sampling_params)

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None:
        del vllm_config, device, is_pin_memory
        self.request_states: dict[int, _RequestState] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for batch_idx in batch_update.removed:
            self.request_states.pop(batch_idx, None)

        for batch_idx, params, _, output_token_ids in batch_update.added:
            ngram_size = _parse_no_repeat_ngram_size(params)
            self.request_states[batch_idx] = _RequestState(
                ngram_size=ngram_size,
                output_token_ids=output_token_ids,
            )

        for src_idx, dst_idx, direction in batch_update.moved:
            src_state = self.request_states.pop(src_idx, None)
            dst_state = self.request_states.pop(dst_idx, None)
            if src_state is not None:
                self.request_states[dst_idx] = src_state
            if direction == MoveDirectionality.SWAP and dst_state is not None:
                self.request_states[src_idx] = dst_state

    def _banned_tokens(self, state: _RequestState) -> set[int]:
        output_token_ids = state.output_token_ids
        ngram_size = state.ngram_size

        if ngram_size <= 0 or not output_token_ids:
            return set()

        if ngram_size == 1:
            return set(output_token_ids)

        if len(output_token_ids) < ngram_size:
            return set()

        previous_prefix = tuple(output_token_ids[-ngram_size:-1])
        state.seen_ngrams[previous_prefix].add(output_token_ids[-1])

        current_prefix = tuple(output_token_ids[-ngram_size + 1 :])
        return state.seen_ngrams.get(current_prefix, set())

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for batch_idx, state in self.request_states.items():
            banned_tokens = self._banned_tokens(state)
            if banned_tokens:
                logits[batch_idx, list(banned_tokens)] = float("-inf")

        return logits


class MinerULogitsProcessor(NoRepeatNGramLogitsProcessor):
    """Compatibility alias for MinerU's documented logits processor name."""

