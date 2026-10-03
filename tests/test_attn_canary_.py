"""
Unit tests for the attention output canary (EXL3_ATTN_CANARY) in triton_paged.py.

The canary is the port's live tripwire for the open transient-wrong-prefill-output
bug (docs/rocm.md), yet no test executed its own code paths: mode parsing, the
non-finite raise, the all-zero-row warn/raise, and warn-once behavior ran for the
first time in production serving. These tests call _attn_output_canary directly
on synthetic tensors.

    PYTHONPATH=. python -m pytest tests/test_attn_canary_.py -q
"""
import pytest
import torch

from exllamav3.modules.attention_fn import triton_paged as tp


@pytest.fixture(autouse=True)
def _restore_canary_state():
    saved = (tp._attn_canary, tp._canary_warned)
    yield
    tp._attn_canary, tp._canary_warned = saved


def _q():
    return torch.randn(2, 8, 1, 128, device="cuda")


def test_disabled_is_silent():
    tp._attn_canary = 0
    out = torch.zeros(2, 8, 1, 128, device="cuda")  # would be "dead" if enabled
    tp._attn_output_canary("decode", out, _q())     # must not raise


def test_healthy_output_passes():
    tp._attn_canary = 2
    out = torch.randn(2, 8, 1, 128, device="cuda")
    tp._attn_output_canary("prefill", out, _q())    # must not raise


def test_nonfinite_raises_in_strict_mode():
    tp._attn_canary = 2
    out = torch.randn(2, 8, 1, 128, device="cuda")
    out[0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        tp._attn_output_canary("decode", out, _q())


def test_all_zero_row_raises_in_strict_mode():
    tp._attn_canary = 2
    out = torch.randn(2, 8, 1, 128, device="cuda")
    out[1] = 0.0  # one whole dead row
    with pytest.raises(RuntimeError, match="all-zero"):
        tp._attn_output_canary("prefill", out, _q())


def test_all_zero_row_warns_once_in_warn_mode(capsys):
    tp._attn_canary = 1
    tp._canary_warned = False
    out = torch.randn(2, 8, 1, 128, device="cuda")
    out[1] = 0.0
    tp._attn_output_canary("decode", out, _q())     # warn, not raise
    tp._attn_output_canary("decode", out, _q())     # second failure: still no raise
    captured = capsys.readouterr()
    assert captured.out.count("[attn-canary] WARNING") == 1, captured.out
