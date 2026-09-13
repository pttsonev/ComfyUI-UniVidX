"""Node contracts: the CFG rule, the encoding split, and decoder alignment.

These run against a stub pipeline. The three behaviours pinned here are the ones
that are silent when wrong: CFG quietly staying at 5 for an empty prompt, a
normal pass quietly passing through a colour transfer, and a decoder quietly
handing back a 1080p condition beside 480p generated passes.

Negative cases carry as much weight as positive ones — a rule that always fires
is indistinguishable from a hardcoded constant.
"""

import sys
import types

import pytest

from univid import attention, context, modes, rope

import __init__ as pack  # the pack's V1 registration, loaded flat by pytest
from nodes.decoder import UniVidXAlphaDecoder, UniVidXIntrinsicDecoder
from nodes.sampler import _DEFAULT_NEGATIVE_PROMPT, UniVidXSampler


torch = pytest.importorskip("torch")


class _Task:
    def __init__(self, name):
        self.name = name
        self.family = modes.get_mode(name).family


class _Pipe:
    """Records the kwargs it was called with and returns plausible outputs."""

    def __init__(self, frames=2, height=4, width=5):
        self.calls = []
        self._shape = (frames, height, width)
        self.model_fn = lambda *args, **kwargs: kwargs["latents"]

    def check_resize_height_width(self, height, width, num_frames=None):
        return height, width, num_frames

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames, height, width = self._shape
        mode = modes.get_mode(kwargs["training_mode"])
        return {key: torch.zeros(3, frames, height, width) for key in mode.result_keys}


class _Model:
    def __init__(self, variant, pipe):
        self.variant = variant
        self.pipe = pipe


def _image(frames=2, height=4, width=5, fill=0.5):
    return torch.full((frames, height, width, 3), fill)


def _fake_dit(name):
    module = types.ModuleType(name)
    module.flash_attention = "upstream-layout"
    module.F = types.SimpleNamespace(scaled_dot_product_attention=lambda *a, **k: "sdpa-out")
    module.rope_apply = lambda *args: "upstream-rope"
    module.torch = torch
    module.rearrange = lambda x, pattern, n: x.reshape(x.shape[0], x.shape[1], n, -1)
    return module


@pytest.fixture(autouse=True)
def loaded_dit(monkeypatch):
    """Model stubs load their own family into sys.modules; use real DiT discovery.

    An alpha-only session must not inherit an intrinsic RoPE module supplied by
    the fixture. The optional SageAttention package itself remains a stub.
    """
    for name in attention.DIT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    intrinsic_name = "src.models.wan_video_dit_intrinsic"
    module = _fake_dit(intrinsic_name)
    modules = {intrinsic_name: module}
    original_init = _Model.__init__

    def load_variant(self, variant, pipe):
        original_init(self, variant, pipe)
        name = f"src.models.wan_video_dit_{variant}"
        if name not in modules:
            modules[name] = _fake_dit(name)
        monkeypatch.setitem(sys.modules, name, modules[name])

    monkeypatch.setattr(_Model, "__init__", load_variant)
    monkeypatch.setattr(attention, "_import_sageattn", lambda: (lambda *a, **k: "sage-out"))
    return module


def _sample(mode="R2AIN", variant="intrinsic", pipe=None, **kwargs):
    pipe = _Pipe() if pipe is None else pipe
    call = {
        "model": _Model(variant, pipe), "task": _Task(mode),
        "height": 4, "width": 5, "num_frames": 2,
    }
    call.update(kwargs)
    (result,) = UniVidXSampler().sample(**call)
    return result, pipe


# --- the empty-prompt CFG rule ------------------------------------------------


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t "])
def test_empty_prompt_forces_cfg_to_one_and_reports_it(prompt):
    result, pipe = _sample(prompt=prompt, cfg_scale=5.0, rgb=_image())
    assert result.cfg_scale == 1.0
    assert result.cfg_forced is True
    assert pipe.calls[0]["cfg_scale"] == 1.0
    assert any("CFG forced to 1.0" in message for message in result.messages)


def test_non_empty_prompt_leaves_cfg_untouched():
    """The negative case: a rule that always fires is a hardcoded constant."""
    result, pipe = _sample(prompt="a street at night", cfg_scale=5.0, rgb=_image())
    assert result.cfg_scale == 5.0
    assert result.cfg_forced is False
    assert pipe.calls[0]["cfg_scale"] == 5.0


def test_positive_prompt_is_passed_as_four_copies():
    """Upstream embeds one prompt per modality latent; a bare string is batch 1."""
    _, pipe = _sample(prompt="a street", rgb=_image())
    assert pipe.calls[0]["prompt"] == ["a street"] * 4
    assert isinstance(pipe.calls[0]["negative_prompt"], str)


def test_default_negative_prompt_is_upstreams():
    assert "最差质量" in _DEFAULT_NEGATIVE_PROMPT
    schema = UniVidXSampler.INPUT_TYPES()["required"]["negative_prompt"][1]
    assert schema["default"] == _DEFAULT_NEGATIVE_PROMPT


# --- input encoding -----------------------------------------------------------


def test_display_referred_applies_range_map_only():
    result, _ = _sample(rgb=_image(fill=0.5), input_encoding="display_referred_srgb")
    assert pytest.approx(float(result.conditions["rgb"].mean()), abs=1e-6) == 0.0


def test_linear_setting_applies_the_srgb_transfer():
    """0.5 linear encodes near 0.7354 sRGB, so the conditioned value must move."""
    result, _ = _sample(rgb=_image(fill=0.5), input_encoding="linear_rec709_to_srgb")
    encoded = float(result.conditions["rgb"].mean()) * 0.5 + 0.5
    assert pytest.approx(encoded, abs=1e-3) == 0.7354


@pytest.mark.parametrize("modality", ["normal"])
@pytest.mark.parametrize("encoding", ["display_referred_srgb", "linear_rec709_to_srgb"])
def test_normal_conditioning_never_passes_through_a_transfer(modality, encoding):
    """Geometry is not colour — asserted under BOTH settings, per modality."""
    result, _ = _sample(
        mode="N2RAI", rgb=None, normal=_image(fill=0.5), input_encoding=encoding,
    )
    assert pytest.approx(float(result.conditions[modality].mean()), abs=1e-6) == 0.0


@pytest.mark.parametrize("encoding", ["display_referred_srgb", "linear_rec709_to_srgb"])
def test_matte_conditioning_never_passes_through_a_transfer(encoding):
    result, _ = _sample(
        mode="P2RFB", variant="alpha", pha=_image(fill=0.5), input_encoding=encoding,
    )
    assert pytest.approx(float(result.conditions["pha"].mean()), abs=1e-6) == 0.0


def test_irradiance_is_treated_as_rgb_like():
    """Omitting irradiance would leave it unconverted beside its converted neighbours."""
    result, _ = _sample(
        mode="I2RAN", irradiance=_image(fill=0.5), input_encoding="linear_rec709_to_srgb",
    )
    encoded = float(result.conditions["irradiance"].mean()) * 0.5 + 0.5
    assert pytest.approx(encoded, abs=1e-3) == 0.7354


def test_out_of_range_linear_input_is_refused_naming_modality_and_value():
    with pytest.raises(ValueError, match="albedo") as exc:
        _sample(mode="RA2IN", rgb=_image(), albedo=_image(fill=4.0),
                input_encoding="linear_rec709_to_srgb")
    assert "4.0" in str(exc.value)


def test_out_of_range_refusal_points_at_an_existing_recovery_path():
    """There is no standalone Gamut tonemap node; the message must not invent one."""
    with pytest.raises(ValueError, match="Gamut: Load EXR with tonemap_preview=True"):
        _sample(rgb=_image(fill=9.0), input_encoding="linear_rec709_to_srgb")


def test_in_range_linear_input_is_accepted():
    result, _ = _sample(rgb=_image(fill=1.0), input_encoding="linear_rec709_to_srgb")
    assert result.input_encoding == "linear_rec709_to_srgb"


def test_display_referred_does_not_range_check():
    """Only the linear setting declares a [0,1] prerequisite."""
    result, _ = _sample(rgb=_image(fill=4.0), input_encoding="display_referred_srgb")
    assert result.cfg_forced is True


# --- validation order ---------------------------------------------------------


def test_family_mismatch_is_refused():
    with pytest.raises(ValueError, match="requires family"):
        _sample(mode="R2PFB", variant="intrinsic", rgb=_image())


def test_missing_required_input_is_refused_before_conversion():
    pipe = _Pipe()
    with pytest.raises(ValueError, match="albedo"):
        UniVidXSampler().sample(
            model=_Model("intrinsic", pipe), task=_Task("RA2IN"), rgb=_image(),
            height=4, width=5, num_frames=2,
        )
    assert pipe.calls == [], "the pipeline must not be reached"


def test_unused_modality_inputs_are_ignored():
    result, _ = _sample(rgb=_image(), pha=_image(), bgr=_image())
    assert set(result.conditions) == {"rgb"}


# --- decoder alignment --------------------------------------------------------


def test_every_decoder_output_shares_one_shape():
    """The condition is deliberately supplied at a different size and length."""
    result, _ = _sample(rgb=_image(frames=7, height=16, width=20))
    images = UniVidXIntrinsicDecoder().decode(result)
    assert len({tuple(image.shape) for image in images}) == 1
    assert tuple(images[0].shape) == (2, 4, 5, 3)


def test_condition_output_is_the_fitted_tensor_not_the_source():
    result, _ = _sample(rgb=_image(frames=7, height=16, width=20, fill=0.25))
    rgb = UniVidXIntrinsicDecoder().decode(result)[0]
    assert rgb.shape[0] == 2 and rgb.shape[1] == 4


def test_missing_carried_condition_raises_rather_than_substituting():
    result, _ = _sample(rgb=_image())
    result.conditions.pop("rgb")
    with pytest.raises(ValueError, match="missing carried condition"):
        UniVidXIntrinsicDecoder().decode(result)


def test_missing_generated_result_raises():
    result, _ = _sample(rgb=_image())
    result.outputs.pop("albedo")
    with pytest.raises(ValueError, match="missing generated result 'albedo'"):
        UniVidXIntrinsicDecoder().decode(result)


def test_normal_output_is_read_from_the_normal_unit_key():
    result, _ = _sample(rgb=_image())
    assert "normal_unit" in result.outputs and "normal" not in result.outputs
    assert len(UniVidXIntrinsicDecoder().decode(result)) == 4


def test_wrong_family_decoder_is_refused():
    result, _ = _sample(rgb=_image())
    with pytest.raises(ValueError, match="not an alpha decoder result"):
        UniVidXAlphaDecoder().decode(result)


def test_alpha_decoder_returns_its_four_passes():
    result, _ = _sample(mode="R2PFB", variant="alpha", rgb=_image())
    assert len(UniVidXAlphaDecoder().decode(result)) == 4


# --- registration -------------------------------------------------------------


def test_all_five_nodes_register_with_matching_display_names():
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert len(pack.NODE_CLASS_MAPPINGS) == 5
    assert all(key.startswith("UniVidX_") for key in pack.NODE_CLASS_MAPPINGS)
    assert all(name.startswith("UniVidX: ") for name in pack.NODE_DISPLAY_NAME_MAPPINGS.values())


@pytest.mark.parametrize("cls", [UniVidXIntrinsicDecoder, UniVidXAlphaDecoder, UniVidXSampler])
def test_return_names_match_return_types_arity(cls):
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)


def test_task_node_offers_every_mode():
    from nodes.task import UniVidXTask

    offered = UniVidXTask.INPUT_TYPES()["required"]["mode"][0]
    assert set(offered) == set(modes.MODES)


def test_teacache_and_cfg_merge_are_not_exposed():
    """Both are dead wiring upstream; exposing them would promise a speedup that isn't there."""
    schema = UniVidXSampler.INPUT_TYPES()
    widgets = set(schema["required"]) | set(schema.get("optional", {}))
    assert not widgets & {"tea_cache_l1_thresh", "tea_cache_model_id", "cfg_merge"}


# --- the attention kernel and the conditioning encode ------------------------


class _Vae:
    def __init__(self):
        self.calls = []

    def encode(self, videos, device, **kwargs):
        self.calls.append((videos, device, kwargs))
        return "latents"


class _EncodingPipe(_Pipe):
    """A pipeline that encodes its conditioning the way upstream does: bare."""

    def __init__(self):
        super().__init__()
        self.vae = _Vae()

    def __call__(self, **kwargs):
        self.vae.encode("video", device="cuda")
        return super().__call__(**kwargs)


def test_sampler_keeps_sdpa_selectable_and_defaults_to_untiled_encode():
    optional = UniVidXSampler.INPUT_TYPES()["optional"]
    assert optional["attention"][0] == list(attention.KERNELS)
    assert optional["attention"][1]["default"] == "sage"
    assert optional["tiled_encode"][1]["default"] is False
    result, _ = _sample(rgb=_image(), attention="sdpa")
    assert (result.attention_kernel, result.tiled_encode) == ("sdpa", False)
    assert not any("Attention kernel" in m or "SageAttention" in m for m in result.messages)


def test_unknown_attention_kernel_is_refused():
    with pytest.raises(ValueError, match="Unknown attention kernel"):
        _sample(rgb=_image(), attention="flash_attn_3")


def test_rope_keeps_float64_selectable_and_reports_nothing_for_it():
    optional = UniVidXSampler.INPUT_TYPES()["optional"]
    assert optional["rope_precision"][0] == ["float64", "float32"]
    assert optional["rope_precision"][1]["default"] == "float32"
    result, _ = _sample(rgb=_image(), rope_precision="float64")
    assert result.rope_precision == "float64"
    assert not any("RoPE" in message for message in result.messages)


@pytest.mark.parametrize("variant,mode", [("intrinsic", "R2AIN"), ("alpha", "R2PFB")])
@pytest.mark.parametrize("with_other_variant", [False, True])
def test_default_rope_reaches_each_family_and_reports_actual_calls(
    monkeypatch, variant, mode, with_other_variant,
):
    name = f"src.models.wan_video_dit_{variant}"
    other_name = "src.models.wan_video_dit_" + ("alpha" if variant == "intrinsic" else "intrinsic")
    other = _fake_dit(other_name)
    other_original = other.rope_apply
    if with_other_variant:
        monkeypatch.setitem(sys.modules, other_name, other)
    wrappers = []

    class _RopePipe(_Pipe):
        def __call__(self, **kwargs):
            module = sys.modules[name]
            assert isinstance(module.rope_apply, rope.RopeFP32)
            wrappers.append(module.rope_apply)
            if with_other_variant:
                assert isinstance(other.rope_apply, rope.RopeFP32)
                wrappers.append(other.rope_apply)
            else:
                assert other_name not in sys.modules
            module.rope_apply(torch.ones(1, 6, 16), torch.ones(6, 1, 4, dtype=torch.complex128), 2)
            assert "rope_precision" not in kwargs
            return super().__call__(**kwargs)

    result, _ = _sample(mode=mode, variant=variant, pipe=_RopePipe(), rgb=_image())
    assert result.rope_precision == "float32"
    assert f"RoPE applied in float32 on 1 calls ({variant})." in result.messages
    for wrapper in wrappers:
        assert wrapper.module.rope_apply is wrapper.original
        assert not wrapper.cache
    assert other.rope_apply is other_original


def test_unknown_rope_precision_is_refused_before_pipeline():
    pipe = _Pipe()
    with pytest.raises(ValueError, match="Unknown RoPE precision"):
        _sample(pipe=pipe, rgb=_image(), rope_precision="float16")
    assert not pipe.calls


def test_sage_without_the_package_is_refused_before_the_pipeline_runs(monkeypatch):
    monkeypatch.setattr(attention, "_import_sageattn", lambda: None)
    pipe = _Pipe()
    with pytest.raises(ValueError, match="sageattention"):
        _sample(pipe=pipe, rgb=_image())
    assert pipe.calls == []


def test_sage_is_live_during_the_call_restored_after_and_its_engagement_is_reported(monkeypatch):
    class _Half:
        dtype = "torch.bfloat16"

        def contiguous(self):
            return self

        def clone(self):
            return _Half()

    module = types.SimpleNamespace(F=types.SimpleNamespace(
        scaled_dot_product_attention=lambda *a, **k: "sdpa-out", silu="the-silu",
    ))
    original = module.F
    seen = {}

    class _AttendingPipe(_Pipe):
        def __call__(self, **kwargs):
            seen["F"] = module.F
            seen["out"] = module.F.scaled_dot_product_attention(_Half(), _Half(), _Half())
            return super().__call__(**kwargs)

    monkeypatch.setattr(attention, "_import_sageattn", lambda: (lambda *a, **k: "sage-out"))
    monkeypatch.setattr(attention, "dit_modules", lambda: [module])
    result, _ = _sample(pipe=_AttendingPipe(), rgb=_image(), attention="sage")
    assert isinstance(seen["F"], attention.FunctionalProxy)
    assert seen["out"] == "sage-out"
    assert module.F is original
    assert result.attention_kernel == "sage"
    assert any(m.startswith("Attention kernel: sage") for m in result.messages)
    assert "SageAttention handled 1 of 1 attention calls." in result.messages


def test_shared_kv_defaults_on_is_live_during_the_call_restored_after_and_reported(loaded_dit):
    optional = UniVidXSampler.INPUT_TYPES()["optional"]
    assert optional["shared_kv"][1]["default"] is True
    seen = {}

    class _AttendingPipe(_Pipe):
        def __call__(self, **kwargs):
            seen["layout"] = loaded_dit.flash_attention
            return super().__call__(**kwargs)

    result, _ = _sample(pipe=_AttendingPipe(), rgb=_image())
    assert isinstance(seen["layout"], attention.SharedKVAttention)
    assert seen["layout"].original == "upstream-layout"
    assert loaded_dit.flash_attention == "upstream-layout"
    assert result.shared_kv is True
    # the stub pipeline makes no attention calls, so the honest count is zero
    assert (
        "Shared K/V attention applied on 0 of 0 cross-modal attention calls; "
        "0 per-row cross-attention calls unaffected."
    ) in result.messages


def test_shared_kv_off_leaves_upstreams_layout_alone_and_says_nothing(loaded_dit):
    seen = {}

    class _AttendingPipe(_Pipe):
        def __call__(self, **kwargs):
            seen["layout"] = loaded_dit.flash_attention
            return super().__call__(**kwargs)

    result, _ = _sample(pipe=_AttendingPipe(), rgb=_image(), shared_kv=False)
    assert seen["layout"] == "upstream-layout"
    assert result.shared_kv is False
    assert not any("Shared K/V" in m for m in result.messages)


def test_shared_kv_with_no_loaded_dit_is_refused_before_the_pipeline_runs(monkeypatch):
    monkeypatch.setattr(attention, "shared_kv_modules", lambda: [])
    pipe = _Pipe()
    with pytest.raises(RuntimeError, match="no vendored DiT module"):
        _sample(pipe=pipe, rgb=_image())
    assert pipe.calls == []
    result, pipe = _sample(rgb=_image(), shared_kv=False)   # off needs nothing to patch
    assert len(pipe.calls) == 1 and result.shared_kv is False


def test_shared_kv_must_be_a_boolean():
    with pytest.raises(ValueError, match="shared_kv must be a boolean"):
        _sample(rgb=_image(), shared_kv="yes")


def test_tiled_encode_forwards_the_decode_tiling_into_the_conditioning_encode():
    pipe = _EncodingPipe()
    result, _ = _sample(
        pipe=pipe, rgb=_image(), tiled_encode=True,
        tile_size_height=7, tile_size_width=9, tile_stride_height=3, tile_stride_width=4,
    )
    assert pipe.vae.calls == [
        ("video", "cuda", {"tiled": True, "tile_size": (7, 9), "tile_stride": (3, 4)}),
    ]
    assert "encode" not in vars(pipe.vae)  # the wrapper never outlives the call
    assert result.tiled_encode is True
    assert any(m.startswith("Conditioning VAE encode: tiled (7x9") for m in result.messages)


def test_untiled_encode_leaves_the_vae_exactly_as_upstream_calls_it():
    pipe = _EncodingPipe()
    result, _ = _sample(pipe=pipe, rgb=_image())
    assert pipe.vae.calls == [("video", "cuda", {})]
    assert "encode" not in vars(pipe.vae)
    assert not any("VAE encode" in m for m in result.messages)


def test_tiled_encode_without_a_vae_is_refused():
    with pytest.raises(ValueError, match="vae"):
        _sample(rgb=_image(), tiled_encode=True)


def test_encode_wrapper_is_removed_when_the_pipeline_raises():
    class _FailingPipe(_EncodingPipe):
        def __call__(self, **kwargs):
            raise RuntimeError("cuda out of memory")

    pipe = _FailingPipe()
    with pytest.raises(RuntimeError, match="out of memory"):
        _sample(pipe=pipe, rgb=_image(), tiled_encode=True)
    assert "encode" not in vars(pipe.vae)


# --- tile overlap ---------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"tiled": True, "tile_size_height": 30, "tile_stride_height": 30},
    {"tiled": True, "tile_size_width": 52, "tile_stride_width": 52},
    {"tiled": False, "tiled_encode": True, "tile_size_height": 30, "tile_stride_height": 30},
])
def test_zero_overlap_tiling_is_refused_before_the_pipeline_runs(kwargs):
    """Upstream's blend mask is `x[-border:]`; a zero border crashes after denoising."""
    pipe = _EncodingPipe() if kwargs.get("tiled_encode") else _Pipe()
    with pytest.raises(ValueError, match="overlap"):
        _sample(pipe=pipe, rgb=_image(), **kwargs)
    assert pipe.calls == []


def test_equal_stride_and_size_are_harmless_when_nothing_is_tiled():
    _, pipe = _sample(rgb=_image(), tiled=False, tile_size_height=30, tile_stride_height=30)
    assert pipe.calls[0]["tiled"] is False


def test_stride_larger_than_size_is_still_refused_untiled():
    """A gap is nonsense regardless of tiling; the older check stays."""
    with pytest.raises(ValueError, match="must not exceed"):
        _sample(rgb=_image(), tiled=False, tile_size_height=30, tile_stride_height=31)


# --- cancellation -------------------------------------------------------------


class _Interrupted(Exception):
    pass


class _PredictingPipe(_Pipe):
    """Exercise two denoising predictions through the pipeline's instance hook."""

    def __init__(self):
        super().__init__()
        self.latents = torch.ones(4, 1, 48, 1, 1)
        self.timestep = object()
        self.marker = object()
        self.predictions = []

        def predict(marker, *, latents, timestep):
            self.predictions.append((marker, latents, timestep))
            return latents

        self.model_fn = predict

    def __call__(self, **kwargs):
        for _ in range(2):
            self.last_prediction = self.model_fn(
                self.marker, latents=self.latents, timestep=self.timestep,
            )
        return super().__call__(**kwargs)


def _install_interrupt_check(monkeypatch, check):
    comfy = types.ModuleType("comfy")
    management = types.ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = check
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)


@pytest.mark.parametrize("context_enabled", [False, True])
def test_interrupt_precedes_first_prediction_and_restores_model_fn(
    monkeypatch, loaded_dit, context_enabled,
):
    pipe = _PredictingPipe()
    original = pipe.model_fn
    checks = []
    cancelled = _Interrupted("cancelled before prediction")

    def check():
        checks.append(len(pipe.predictions))
        raise cancelled

    _install_interrupt_check(monkeypatch, check)
    with pytest.raises(_Interrupted) as caught:
        _sample(pipe=pipe, rgb=_image(), num_frames=189, context_enabled=context_enabled)
    assert caught.value is cancelled
    assert checks == [0]
    assert pipe.predictions == []
    assert pipe.model_fn is original
    assert loaded_dit.flash_attention == "upstream-layout"


@pytest.mark.parametrize("context_enabled", [False, True])
def test_interrupt_stops_before_next_prediction_or_context_window(monkeypatch, context_enabled):
    pipe = _PredictingPipe()
    original = pipe.model_fn

    def check():
        if pipe.predictions:
            raise _Interrupted("cancelled after first prediction")

    _install_interrupt_check(monkeypatch, check)
    with pytest.raises(_Interrupted, match="after first prediction"):
        _sample(pipe=pipe, rgb=_image(), num_frames=189, context_enabled=context_enabled)
    assert len(pipe.predictions) == 1
    assert pipe.predictions[0][1].shape[2] == (6 if context_enabled else 48)
    assert pipe.model_fn is original


@pytest.mark.parametrize("context_enabled", [False, True])
def test_sampling_without_comfy_delegates_and_restores_model_fn(monkeypatch, context_enabled):
    # Explicitly force ImportError even if the suite is run inside ComfyUI.
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    pipe = _PredictingPipe()
    original = pipe.model_fn
    result, _ = _sample(
        pipe=pipe, rgb=_image(), num_frames=189, context_enabled=context_enabled,
    )
    assert len(pipe.calls) == 1
    assert result.context_window_count == (12 if context_enabled else 1)
    assert len(pipe.predictions) == (24 if context_enabled else 2)
    assert all(
        marker is pipe.marker and timestep is pipe.timestep
        for marker, _, timestep in pipe.predictions
    )
    assert torch.equal(pipe.last_prediction, pipe.latents)
    assert pipe.model_fn is original


# --- context blend ------------------------------------------------------------


def test_context_blend_widget_defaults_to_cosine():
    widget = UniVidXSampler.INPUT_TYPES()["optional"]["context_blend"]
    assert widget[0] == list(context.BLEND_SHAPES)
    assert widget[1]["default"] == "cosine"
    result, _ = _sample(rgb=_image())
    assert result.context_blend == "cosine"


@pytest.mark.parametrize("shape", context.BLEND_SHAPES)
def test_context_blend_reaches_dispatcher_and_report(monkeypatch, shape):
    seen = []
    original_weights = context.blend_weights

    def weights(length, blend="triangular"):
        seen.append((length, blend))
        return original_weights(length, blend)

    monkeypatch.setattr(context, "blend_weights", weights)
    pipe = _PredictingPipe()
    original = pipe.model_fn
    result, _ = _sample(
        pipe=pipe, rgb=_image(), num_frames=189, context_enabled=True,
        context_window_frames=21, context_stride_frames=16, context_blend=shape,
    )
    assert seen == [(6, shape)] * 24  # twelve windows on each of two predictions
    assert result.context_blend == shape
    assert result.context_window_count == 12
    assert (
        "Context windows: 12 per noise prediction "
        f"(48 latent frames; window 6, stride 4, blend {shape})."
    ) in result.messages
    assert "context_blend" not in pipe.calls[0]
    assert pipe.model_fn is original


@pytest.mark.parametrize("shape", context.BLEND_SHAPES)
def test_context_blend_is_ignored_when_context_is_disabled(monkeypatch, shape):
    def unexpected_weights(*args, **kwargs):
        pytest.fail("disabled context must not blend predictions")

    monkeypatch.setattr(context, "blend_weights", unexpected_weights)
    pipe = _PredictingPipe()
    result, _ = _sample(
        pipe=pipe, rgb=_image(), num_frames=189, context_enabled=False, context_blend=shape,
    )
    assert result.context_blend == shape
    assert result.context_window_count == 1
    assert len(pipe.predictions) == 2
    assert all(latents is pipe.latents for _, latents, _ in pipe.predictions)
    assert pipe.last_prediction is pipe.latents
    assert not any(message.startswith("Context windows:") for message in result.messages)
    assert "context_blend" not in pipe.calls[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_unknown_context_blend_is_refused_even_when_disabled(enabled):
    pipe = _Pipe()
    with pytest.raises(ValueError, match="Unknown context blend") as exc:
        _sample(pipe=pipe, rgb=_image(), context_enabled=enabled, context_blend="unknown")
    assert "unknown" in str(exc.value)
    assert ", ".join(context.BLEND_SHAPES) in str(exc.value)
    assert pipe.calls == []
