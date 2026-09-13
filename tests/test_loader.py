"""Loader behaviour that can be proven without 30 GB of weights.

`_load_dit_state` takes `torch` and `load_file` as parameters, so the FP8
dequantisation — the one piece of numerics this pack owns rather than borrows —
is testable against real tensors on tiny shapes. Everything requiring an
assembled 14B model is a Phase 6 live check, not a test.

The argument-validation tests matter more than they look: they assert the order
of the checks, because rejecting a bad variant *after* spending two minutes
loading a DiT would be a poor trade.
"""

from pathlib import Path

import pytest

from univid import loader, paths


torch = pytest.importorskip("torch", reason="dequantisation tests need real tensors")


def _source(kind, scale_weights=None, count=1):
    return paths.WeightSource(
        kind=kind,
        paths=tuple(Path(f"/fake/shard{i}.safetensors") for i in range(count)),
        headers=tuple({} for _ in range(count)),
        scale_weights=scale_weights or {},
    )


# --- FP8 dequantisation -------------------------------------------------------


def test_scaled_weights_are_multiplied_by_their_scale():
    """weight * scale, rounded once into the compute dtype."""
    state = {
        "blocks.0.q.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "blocks.0.q.scale_weight": torch.tensor([2.0]),
        "blocks.0.norm.weight": torch.tensor([7.0]),
    }
    out, consumed = loader._load_dit_state(
        _source("scaled_single_file", {"blocks.0.q.weight": "blocks.0.q.scale_weight"}),
        dtype=torch.float32, torch=torch, load_file=lambda path, device: dict(state),
    )
    assert torch.equal(out["blocks.0.q.weight"], torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
    assert torch.equal(out["blocks.0.norm.weight"], torch.tensor([7.0]))
    assert consumed == ("blocks.0.q.scale_weight",)


def test_scale_tensors_are_removed_from_the_loaded_state():
    """A leftover scale_weight would surface as an unexpected key and abort load."""
    state = {
        "w.weight": torch.tensor([[1.0]]),
        "w.scale_weight": torch.tensor([3.0]),
    }
    out, _ = loader._load_dit_state(
        _source("scaled_single_file", {"w.weight": "w.scale_weight"}),
        dtype=torch.float32, torch=torch, load_file=lambda path, device: dict(state),
    )
    assert "w.scale_weight" not in out


def test_dequantisation_multiplies_before_rounding():
    """Rounding to bf16 first would lose the scale's precision.

    0.1 is not representable in bfloat16; scaling in float32 and rounding once
    lands closer to the true product than rounding both factors beforehand.
    """
    state = {"w.weight": torch.tensor([[0.1]]), "w.scale_weight": torch.tensor([3.0])}
    out, _ = loader._load_dit_state(
        _source("scaled_single_file", {"w.weight": "w.scale_weight"}),
        dtype=torch.bfloat16, torch=torch, load_file=lambda path, device: dict(state),
    )
    once = float(out["w.weight"].to(torch.float32))
    twice = float((torch.tensor(0.1).to(torch.bfloat16) * torch.tensor(3.0).to(torch.bfloat16)))
    assert abs(once - 0.3) <= abs(twice - 0.3)


def test_unscaled_source_leaves_every_tensor_alone():
    state = {"w.weight": torch.tensor([[5.0]])}
    out, consumed = loader._load_dit_state(
        _source("single_file"), dtype=torch.float32, torch=torch,
        load_file=lambda path, device: dict(state),
    )
    assert torch.equal(out["w.weight"], torch.tensor([[5.0]]))
    assert consumed == ()


def test_all_tensors_are_cast_to_the_compute_dtype():
    state = {"a": torch.tensor([1.0], dtype=torch.float32), "b": torch.tensor([2.0], dtype=torch.float16)}
    out, _ = loader._load_dit_state(
        _source("single_file"), dtype=torch.bfloat16, torch=torch,
        load_file=lambda path, device: dict(state),
    )
    assert all(tensor.dtype is torch.bfloat16 for tensor in out.values())


def test_duplicate_keys_across_shards_are_refused():
    """Six shards must partition the DiT; an overlap means the wrong file set."""
    shard = {"w.weight": torch.tensor([[1.0]])}
    with pytest.raises(ValueError, match="Duplicate DiT keys"):
        loader._load_dit_state(
            _source("canonical_shards", count=6), dtype=torch.float32, torch=torch,
            load_file=lambda path, device: dict(shard),
        )


def test_unknown_source_kind_is_refused():
    with pytest.raises(ValueError, match="Unknown DiT weight source kind"):
        loader._load_dit_state(
            _source("something_else"), dtype=torch.float32, torch=torch,
            load_file=lambda path, device: {},
        )


# --- checkpoint attachment ----------------------------------------------------


class _FakeModule:
    """Minimal load_state_dict target; mirrors torch's return shape."""

    def __init__(self, keys, missing=(), unexpected=()):
        self._keys = set(keys)
        self._missing = list(missing)
        self._unexpected = list(unexpected)

    def load_state_dict(self, state, strict=False, assign=False):
        class _Result:
            missing_keys = self._missing
            unexpected_keys = self._unexpected
        return _Result()

    def state_dict(self):
        return {key: None for key in self._keys}


def test_checkpoint_dit_prefix_is_stripped(tmp_path):
    captured = {}

    class _Recorder(_FakeModule):
        def load_state_dict(self, state, strict=False, assign=False):
            captured.update(state)
            return super().load_state_dict(state, strict, assign)

    loader._attach_checkpoint(
        _Recorder({"blocks.0.lora_A.weight"}),
        tmp_path / "univid_intrinsic.safetensors",
        load_file=lambda path, device: {"dit.blocks.0.lora_A.weight": 1},
    )
    assert "blocks.0.lora_A.weight" in captured
    assert not any(key.startswith("dit.") for key in captured)


def test_missing_lora_adapters_abort_the_load(tmp_path):
    """A wrong-family checkpoint leaves adapters unfilled; that must not pass."""
    with pytest.raises(RuntimeError, match="All modality adapters are required"):
        loader._attach_checkpoint(
            _FakeModule({"x"}, missing=["blocks.0.self_attn.q.lora_A.rgb.weight"]),
            tmp_path / "univid_alpha.safetensors",
            load_file=lambda path, device: {"x": 1},
        )


def test_unexpected_checkpoint_keys_abort_the_load(tmp_path):
    with pytest.raises(RuntimeError, match="All modality adapters are required"):
        loader._attach_checkpoint(
            _FakeModule({"x"}, unexpected=["stray.weight"]),
            tmp_path / "univid_intrinsic.safetensors",
            load_file=lambda path, device: {"x": 1},
        )


def test_missing_base_weights_are_tolerated(tmp_path):
    """A LoRA checkpoint legitimately omits base weights — only adapters matter."""
    report = loader._attach_checkpoint(
        _FakeModule({"x"}, missing=["blocks.0.self_attn.q.weight"]),
        tmp_path / "univid_intrinsic.safetensors",
        load_file=lambda path, device: {"x": 1},
    )
    assert report.missing_keys == ("blocks.0.self_attn.q.weight",)


def test_duplicate_key_after_prefix_strip_is_refused(tmp_path):
    """`dit.w` and a bare `w` would silently collide into one entry."""
    with pytest.raises(ValueError, match="duplicate key after removing"):
        loader._attach_checkpoint(
            _FakeModule({"w"}),
            tmp_path / "univid_intrinsic.safetensors",
            load_file=lambda path, device: {"dit.w": 1, "w": 2},
        )


# --- argument validation happens before any expensive work --------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"variant": "nope"}, "Unknown UniVidX variant"),
        ({"variant": "intrinsic", "compute_dtype": "int8"}, "Unsupported UniVidX compute dtype"),
        ({"variant": "intrinsic", "vram_buffer": -1}, "finite, non-negative"),
        ({"variant": "intrinsic", "vram_buffer": float("nan")}, "finite, non-negative"),
    ],
)
def test_bad_arguments_are_refused_before_touching_torch(kwargs, match, monkeypatch):
    """These must raise without importing torch or probing CUDA at all."""
    def _explode():
        raise AssertionError("torch must not be reached for an argument error")

    monkeypatch.setattr(loader, "_get_torch", _explode)
    variant = kwargs.pop("variant")
    with pytest.raises(ValueError, match=match):
        loader.load_model(variant, **kwargs)


# --- vendor import guard ------------------------------------------------------


def test_foreign_src_package_is_refused(monkeypatch, tmp_path):
    """Another pack owning the top-level `src` name must be an error, not a silent wrong import.

    `src` is a generic name; if some other custom node imported its own, blindly
    importing ours would either fail obscurely or load their code.
    """
    foreign = type(loader)("src")
    foreign.__file__ = str(tmp_path / "elsewhere" / "src" / "__init__.py")
    monkeypatch.setitem(__import__("sys").modules, "src", foreign)
    with pytest.raises(RuntimeError, match="another package already owns 'src'"):
        loader._get_upstream("intrinsic", tmp_path / "vendor" / "univid")


# --- cache ---------------------------------------------------------------------


def test_evict_reports_how_many_handles_it_dropped(monkeypatch):
    monkeypatch.setattr(loader, "_MODEL_CACHE", {"a": object(), "b": object()})
    assert loader.evict_model() == 2
    assert loader._MODEL_CACHE == {}


def test_evict_targets_a_single_handle(monkeypatch):
    keep, drop = object(), object()
    monkeypatch.setattr(loader, "_MODEL_CACHE", {"k": keep, "d": drop})
    assert loader.evict_model(drop) == 1
    assert loader._MODEL_CACHE == {"k": keep}


def test_module_imports_without_torch_at_module_level():
    """Torch is imported lazily so the pack loads on a bare interpreter."""
    source = (Path(loader.__file__)).read_text(encoding="utf-8")
    head = source.split("def ")[0]
    assert "import torch" not in head


# --- bounded retention --------------------------------------------------------
#
# Reproduces the reviewer's scenario directly: changing only vram_buffer used to
# leave two full handles cached, stranding ~28 GB of host RAM per change with no
# reachable way to release it.


def _stub_build(monkeypatch, built):
    """Replace model construction and the CUDA preflight with cheap stubs."""
    monkeypatch.setattr(loader, "_MODEL_CACHE", loader.OrderedDict())
    monkeypatch.setattr(loader, "_vendor_config", lambda: Path("/fake/arch.jsn"))
    monkeypatch.setattr(
        loader.paths, "resolve_models",
        lambda variant, **kw: paths.ModelPaths(
            dit=_source("single_file"), vae=Path("/fake/vae"),
            text_encoder=Path("/fake/te"), checkpoint=Path("/fake/ckpt"),
            tokenizer=Path("/fake/tok"),
        ),
    )

    def _build(variant, resolved, cfg, *, dtype, device, vram_buffer, **kwargs):
        marker = object()
        built.append(marker)
        return marker, ()

    monkeypatch.setattr(loader, "_build_model", _build)

    class _Cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def current_device(): return 0
        @staticmethod
        def synchronize(device=None): return None
        @staticmethod
        def empty_cache(): return None
        @staticmethod
        def get_device_properties(index):
            return type("P", (), {"total_memory": 32 * 1024 ** 3})()

    class _Torch:
        cuda = _Cuda
        bfloat16 = torch.bfloat16
        float16 = torch.float16
        float32 = torch.float32
        device = torch.device

        @staticmethod
        def zeros(*args, **kwargs):
            # The CUDA preflight probe, satisfied on CPU: the torch in this
            # interpreter is CPU-only, and these tests are about cache
            # retention rather than about the probe itself.
            kwargs.pop("device", None)
            return torch.zeros(*args, **kwargs)

    monkeypatch.setattr(loader, "_get_torch", lambda: _Torch)


def test_changing_vram_buffer_does_not_retain_the_previous_model(monkeypatch):
    """The exact reported case: buffer 0.5 -> 1.0 must not leave two handles."""
    built = []
    _stub_build(monkeypatch, built)
    loader.load_model("intrinsic", vram_buffer=0.5)
    loader.load_model("intrinsic", vram_buffer=1.0)
    assert len(built) == 2, "a changed buffer should build a new model"
    assert len(loader._MODEL_CACHE) == 1, "the previous handle must not be retained"


def test_switching_variant_does_not_retain_the_previous_model(monkeypatch):
    built = []
    _stub_build(monkeypatch, built)
    loader.load_model("intrinsic")
    loader.load_model("alpha")
    assert len(loader._MODEL_CACHE) == 1


def test_identical_settings_reuse_the_cached_handle(monkeypatch):
    """Bounding must not turn every queue into a 28 GB reload."""
    built = []
    _stub_build(monkeypatch, built)
    first = loader.load_model("intrinsic", vram_buffer=0.5)
    second = loader.load_model("intrinsic", vram_buffer=0.5)
    assert first is second
    assert len(built) == 1


def test_eviction_happens_before_the_replacement_is_built(monkeypatch):
    """Evicting afterwards would peak at two models in host RAM."""
    built, observed = [], []
    _stub_build(monkeypatch, built)

    def _build(variant, resolved, cfg, *, dtype, device, vram_buffer, **kwargs):
        observed.append(len(loader._MODEL_CACHE))
        marker = object()
        built.append(marker)
        return marker, ()

    monkeypatch.setattr(loader, "_build_model", _build)
    loader.load_model("intrinsic", vram_buffer=0.5)
    loader.load_model("intrinsic", vram_buffer=1.0)
    assert observed == [0, 0], "the cache must be empty while a replacement is built"


def test_cache_limit_is_configurable(monkeypatch):
    monkeypatch.setenv("UNIVIDX_MODEL_CACHE_MAX", "3")
    assert loader._cache_limit() == 3
    monkeypatch.setenv("UNIVIDX_MODEL_CACHE_MAX", "nonsense")
    assert loader._cache_limit() == 1
    monkeypatch.setenv("UNIVIDX_MODEL_CACHE_MAX", "0")
    assert loader._cache_limit() == 1, "zero would disable caching entirely"


def test_raised_limit_retains_more_than_one(monkeypatch):
    built = []
    _stub_build(monkeypatch, built)
    monkeypatch.setenv("UNIVIDX_MODEL_CACHE_MAX", "2")
    loader.load_model("intrinsic", vram_buffer=0.5)
    loader.load_model("intrinsic", vram_buffer=1.0)
    assert len(loader._MODEL_CACHE) == 2


def test_reuse_refreshes_recency(monkeypatch):
    """The older-but-recently-used handle must survive, not the stale one."""
    built = []
    _stub_build(monkeypatch, built)
    monkeypatch.setenv("UNIVIDX_MODEL_CACHE_MAX", "2")
    first = loader.load_model("intrinsic", vram_buffer=0.5)
    loader.load_model("intrinsic", vram_buffer=1.0)
    loader.load_model("intrinsic", vram_buffer=0.5)
    loader.load_model("intrinsic", vram_buffer=2.0)
    assert first in loader._MODEL_CACHE.values()


# --- VRAM strategy selection (CD-47) ------------------------------------------
#
# The two knobs select mutually exclusive strategies upstream: setting
# num_persistent_param_in_dit makes it force vram_limit to None. Accepting both
# would leave the user believing a limit applied that upstream discarded.


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"vram_limit": -1.0}, "finite and non-negative"),
        ({"vram_limit": float("nan")}, "finite and non-negative"),
        ({"num_persistent_param_in_dit": -2}, "non-negative integer or -1"),
        ({"num_persistent_param_in_dit": 1.5}, "non-negative integer or -1"),
        ({"vram_limit": 8.0, "num_persistent_param_in_dit": 1000},
         "mutually exclusive"),
    ],
)
def test_vram_arguments_are_validated_before_torch(kwargs, match, monkeypatch):
    def _explode():
        raise AssertionError("torch must not be reached for an argument error")

    monkeypatch.setattr(loader, "_get_torch", _explode)
    with pytest.raises(ValueError, match=match):
        loader.load_model("intrinsic", **kwargs)


def test_unset_sentinels_are_not_treated_as_both_set(monkeypatch):
    """vram_limit=0 and num_persistent=-1 both mean 'unset', not a conflict.

    Mapping these after the exclusivity check would reject the legacy widget defaults.
    """
    built = []
    _stub_build(monkeypatch, built)
    loader.load_model("intrinsic", vram_limit=0.0, num_persistent_param_in_dit=-1)
    assert len(built) == 1


@pytest.mark.parametrize(
    "first,second",
    [
        ({"vram_limit": 8.0}, {"vram_limit": 12.0}),
        ({"num_persistent_param_in_dit": 0}, {"num_persistent_param_in_dit": 5000}),
        ({}, {"distillation": "lightx2v"}),
        ({"distillation_strength": 1.0}, {"distillation_strength": 0.5}),
    ],
)
def test_changing_a_residency_or_distill_setting_rebuilds(first, second, monkeypatch):
    """A merged or differently-resident model cannot be reused as another."""
    built = []
    _stub_build(monkeypatch, built)
    monkeypatch.setattr(
        loader.paths, "resolve_lightx2v",
        lambda *a, **k: paths.LoRASource(
            path=Path("/fake/lx2v.safetensors"), header={}, pairs={}, deltas={},
            alpha_keys={}, alpha=None,
        ),
        raising=False,
    )
    loader.load_model("intrinsic", **first)
    loader.load_model("intrinsic", **second)
    assert len(built) == 2, "the second settings must not reuse the first handle"
