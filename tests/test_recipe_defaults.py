"""Production recipe defaults agree across the ComfyUI schema and node callables."""

from inspect import signature
from pathlib import Path

from nodes.loader import UniVidXLoader
from nodes.sampler import UniVidXSampler


def test_attention_defaults_to_sage():
    assert UniVidXSampler.INPUT_TYPES()["optional"]["attention"][1]["default"] == "sage"
    assert signature(UniVidXSampler.sample).parameters["attention"].default == "sage"


def test_context_stride_defaults_to_16():
    assert UniVidXSampler.INPUT_TYPES()["optional"]["context_stride_frames"][1]["default"] == 16
    assert signature(UniVidXSampler.sample).parameters["context_stride_frames"].default == 16


def test_context_blend_defaults_to_cosine():
    assert UniVidXSampler.INPUT_TYPES()["optional"]["context_blend"][1]["default"] == "cosine"
    assert signature(UniVidXSampler.sample).parameters["context_blend"].default == "cosine"


def test_rope_precision_defaults_to_float32():
    assert UniVidXSampler.INPUT_TYPES()["optional"]["rope_precision"][1]["default"] == "float32"
    assert signature(UniVidXSampler.sample).parameters["rope_precision"].default == "float32"


def test_persistent_parameter_cap_defaults_to_2e9():
    widget = UniVidXLoader.INPUT_TYPES()["optional"]["num_persistent_param_in_dit"]
    assert widget[1]["default"] == 2_000_000_000
    default = signature(UniVidXLoader.load).parameters["num_persistent_param_in_dit"].default
    assert default == 2_000_000_000


def test_readme_attributes_historical_measurement_to_sage_and_explicit_cap():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    history = readme.split("**Historical measurement (2026-09-10):**", 1)[1].split("\n\n", 1)[0]
    history = " ".join(history.split())
    for setting in (
        "189 frames at 704x1248 in 40:11",
        "`attention=sage` (SageAttention 1.0.6)",
        "`num_persistent_param_in_dit=8e9`",
        "`distillation=lightx2v` at 4 steps",
        "an empty prompt (CFG 1.0)",
        "context windows 21/12",
        "`triangular` blending and `float64` RoPE",
        "`vram_limit` unset",
        "8e9 cap was withdrawn on 2026-09-12",
        "<= 4e9 with SageAttention 1.0.6",
    ):
        assert setting in history
    assert "2026-09-10 numbers were measured with" not in readme
