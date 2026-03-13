# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.logits_processor.no_repeat_ngram import (
    MinerULogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)


def test_no_repeat_ngram_masks_seen_completion() -> None:
    output_token_ids: list[int] = []
    processor = NoRepeatNGramLogitsProcessor(
        VllmConfig(), torch.device("cpu"), is_pin_memory=False
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[
                (
                    0,
                    SamplingParams(extra_args={"no_repeat_ngram_size": 3}),
                    None,
                    output_token_ids,
                )
            ],
            moved=[],
        )
    )

    for token_id in [1, 2, 3, 1, 2]:
        output_token_ids.append(token_id)
        logits = torch.zeros((1, 8), dtype=torch.float32)
        processor.apply(logits)

    assert logits[0, 3].item() == float("-inf")
    assert logits[0, 4].item() == 0.0


def test_no_repeat_ngram_validates_extra_arg_type() -> None:
    try:
        NoRepeatNGramLogitsProcessor.validate_params(
            SamplingParams(extra_args={"no_repeat_ngram_size": "3"})
        )
    except ValueError as exc:
        assert "must be an integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-integer ngram size")


def test_mineru_alias_uses_same_processor() -> None:
    processor = MinerULogitsProcessor(
        VllmConfig(), torch.device("cpu"), is_pin_memory=False
    )
    assert isinstance(processor, NoRepeatNGramLogitsProcessor)
