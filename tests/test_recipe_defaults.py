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
    widget = UniVidXLoader.INPUT_TYPES()["required"]["num_persistent_param_in_dit"]
    assert widget[1]["default"] == 2_000_000_000
    default = signature(UniVidXLoader.load).parameters["num_persistent_param_in_dit"].default
    assert default == 2_000_000_000


def test_readme_attributes_historical_measurement_to_sage_and_explicit_cap():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    recipe = readme.split("## Production recipe", 1)[1].split("\n## ", 1)[0]
    recipe = " ".join(recipe.split())
    for setting in (
        "Measured **2026-09-13**",
        "**704x1248, 189 frames**",
        "**26:17 total**",
        "**28.9 s per window-step**",
        "**15.2 GiB peak** with SageAttention 2.2.0",
        "`num_persistent_param_in_dit=2e9`",
        "`vram_limit=0` (unset)",
        "`distillation_lora=Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`",
        "`steps=4`",
        "Never set the residency cap to `4e9` with SageAttention 2.2.0",
    ):
        assert setting in recipe
