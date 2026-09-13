"""Pure wrapper contracts plus tiny CPU comparisons to upstream's exact formula."""

import sys
import types

import pytest

from univid import rope


class _Tensor:
    def __init__(self, shape=(1, 6, 2, 8), dtype="bfloat16", device="cpu", pointer=123):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.pointer = pointer
        self.casts = []

    def data_ptr(self):
        return self.pointer

    def to(self, dtype):
        self.casts.append(dtype)
        return _Tensor(self.shape, dtype, self.device)

    def reshape(self, *shape):
        return _Tensor(shape, self.dtype, self.device)

    def flatten(self, dim):
        assert dim == 2
        return self

    def __mul__(self, other):
        assert self.dtype == other.dtype == "complex64"
        return self


def _module(name="src.models.wan_video_dit_intrinsic"):
    def rearrange(x, pattern, n):
        assert pattern == "b s (n d) -> b s n d" and n == 2
        return x

    def view_as_complex(x):
        assert x.dtype == "float32"
        return _Tensor(x.shape, "complex64")

    return types.SimpleNamespace(
        __name__=name,
        torch=types.SimpleNamespace(
            float32="float32", complex64="complex64",
            view_as_complex=view_as_complex,
            view_as_real=lambda x: _Tensor(x.shape, "float32"),
        ),
        rearrange=rearrange, rope_apply=lambda *args: "upstream",
    )


@pytest.mark.parametrize("raises", [False, True])
def test_swap_restores_and_clears_cache_on_every_exit(raises):
    module = _module()
    original = module.rope_apply
    wrapper = None
    try:
        with rope.patched_rope([module], "float32") as wrappers:
            wrapper, = wrappers
            assert module.rope_apply is wrapper
            assert wrapper.original is original
            assert module.rope_apply(_Tensor(), _Tensor(), 2).dtype == "bfloat16"
            assert wrapper.cache
            if raises:
                raise LookupError("body failed")
    except LookupError:
        assert raises
    assert module.rope_apply is original
    assert wrapper.calls == 1 and not wrapper.cache


def test_float64_is_a_noop_even_without_a_module():
    module = _module()
    original = module.rope_apply
    with rope.patched_rope([module], "float64") as wrappers:
        assert wrappers is None and module.rope_apply is original
    with rope.patched_rope([], "float64") as wrappers:
        assert wrappers is None
    assert rope.rope_report(None) == []


def test_refuses_reentry_without_disturbing_outer_swap():
    module = _module()
    with rope.patched_rope([module], "float32") as wrappers:
        with pytest.raises(RuntimeError, match="already patched"):
            with rope.patched_rope([module], "float32"):
                pytest.fail("entered twice")
        assert module.rope_apply is wrappers[0]


@pytest.mark.parametrize("modules", [[], [types.SimpleNamespace()]])
def test_refuses_missing_rope_apply(modules):
    with pytest.raises(RuntimeError, match="rope_apply"):
        with rope.patched_rope(modules, "float32"):
            pytest.fail("missing seam")


def test_refuses_unknown_precision():
    with pytest.raises(ValueError, match="Unknown RoPE precision"):
        with rope.patched_rope([_module()], "float16"):
            pytest.fail("invalid precision")


def test_counts_calls_and_casts_once_per_frequency_tensor_with_bounded_cache():
    module = _module()
    # Same pointer/shape/device but different objects must not share a cast.
    freqs = [_Tensor(dtype="complex128") for _ in range(7)]
    with rope.patched_rope([module], "float32") as wrappers:
        wrapper, = wrappers
        assert rope.rope_report(wrappers) == ["RoPE applied in float32 on 0 calls."]
        for f in freqs:
            for _ in range(2):
                module.rope_apply(_Tensor(), f, 2)
            assert f.casts == ["complex64"]
            assert len(wrapper.cache) <= 4
        assert wrapper.calls == 14
        assert rope.rope_report(wrappers) == ["RoPE applied in float32 on 14 calls (intrinsic)."]
    assert not wrapper.cache


@pytest.mark.parametrize("attribute,value", [
    ("pointer", 456), ("shape", (6, 1, 4)), ("device", "fake-device"),
])
def test_cache_key_tracks_storage_shape_and_device(attribute, value):
    module = _module()
    freqs = _Tensor()
    with rope.patched_rope([module], "float32"):
        module.rope_apply(_Tensor(), freqs, 2)
        setattr(freqs, attribute, value)
        module.rope_apply(_Tensor(), freqs, 2)
    assert freqs.casts == ["complex64", "complex64"]


@pytest.mark.parametrize("name", rope.DIT_MODULE_NAMES)
def test_finds_each_loaded_dit_without_importing_the_others(monkeypatch, name):
    for candidate in rope.DIT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, candidate, raising=False)
    assert rope.dit_modules() == []
    module = _module(name)
    monkeypatch.setitem(sys.modules, name, module)
    assert rope.dit_modules() == [module]


@pytest.mark.parametrize("with_intrinsic", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_alpha_global_is_patched_and_restored_with_or_without_intrinsic(
    monkeypatch, with_intrinsic, raises,
):
    for name in rope.DIT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    names = ["src.models.wan_video_dit_alpha"]
    if with_intrinsic:
        names.insert(0, "src.models.wan_video_dit_intrinsic")
    modules = [_module(name) for name in names]
    for module in modules:
        # Match SelfAttention.forward's lookup: the called function resolves
        # rope_apply in its own module globals, not through an imported alias.
        exec("def forward(x, freqs): return rope_apply(x, freqs, 2)", vars(module))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    originals = [module.rope_apply for module in modules]
    try:
        with rope.patched_rope(rope.dit_modules(), "float32") as wrappers:
            assert len(wrappers) == len(modules)
            for module, wrapper in zip(modules, wrappers):
                assert module.rope_apply is wrapper
                assert module.forward(_Tensor(), _Tensor()).dtype == "bfloat16"
                assert wrapper.calls == 1 and wrapper.cache
            labels = "intrinsic, alpha" if with_intrinsic else "alpha"
            assert rope.rope_report(wrappers) == [
                f"RoPE applied in float32 on {len(modules)} calls ({labels})."
            ]
            if raises:
                raise LookupError("alpha pipeline failed")
    except LookupError:
        assert raises
    for module, original, wrapper in zip(modules, originals, wrappers):
        assert module.rope_apply is original
        assert not wrapper.cache


def test_all_three_namespaces_are_patched_but_report_only_counts_used_ones(monkeypatch):
    modules = [_module(name) for name in rope.DIT_MODULE_NAMES]
    originals = [module.rope_apply for module in modules]
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    with rope.patched_rope(rope.dit_modules(), "float32") as wrappers:
        assert len(wrappers) == 3
        for module, wrapper in zip(modules, wrappers):
            assert module.rope_apply is wrapper
        for _ in range(2):
            modules[1].rope_apply(_Tensor(), _Tensor(), 2)
        modules[2].rope_apply(_Tensor(), _Tensor(), 2)
        assert rope.rope_report(wrappers) == ["RoPE applied in float32 on 3 calls (alpha, base)."]
    assert [module.rope_apply for module in modules] == originals
    assert all(not wrapper.cache for wrapper in wrappers)


def test_reentry_checks_all_modules_before_changing_any():
    intrinsic, alpha = [_module(name) for name in rope.DIT_MODULE_NAMES[:2]]
    original = intrinsic.rope_apply
    with rope.patched_rope([alpha], "float32") as outer:
        with pytest.raises(RuntimeError, match="re-entry refused"):
            with rope.patched_rope([intrinsic, alpha], "float32"):
                pytest.fail("entered with an already patched alpha module")
        assert intrinsic.rope_apply is original
        assert alpha.rope_apply is outer[0]


def test_patch_setup_failure_restores_previously_patched_modules(monkeypatch):
    modules = [_module(name) for name in rope.DIT_MODULE_NAMES[:2]]
    originals = [module.rope_apply for module in modules]
    original_init = rope.RopeFP32.__init__

    def fail_on_alpha(self, module, original):
        if module is modules[1]:
            raise LookupError("wrapper setup failed")
        original_init(self, module, original)

    monkeypatch.setattr(rope.RopeFP32, "__init__", fail_on_alpha)
    with pytest.raises(LookupError, match="wrapper setup failed"):
        with rope.patched_rope(modules, "float32"):
            pytest.fail("entered after setup failure")
    assert [module.rope_apply for module in modules] == originals


def test_loaded_module_without_rope_does_not_hide_a_usable_alpha_module():
    alpha = _module("src.models.wan_video_dit_alpha")
    original = alpha.rope_apply
    with rope.patched_rope([types.SimpleNamespace(), alpha], "float32") as wrappers:
        assert len(wrappers) == 1 and alpha.rope_apply is wrappers[0]
    assert alpha.rope_apply is original


def _torch_module():
    torch = pytest.importorskip("torch")

    def rearrange(x, pattern, n):
        assert pattern == "b s (n d) -> b s n d"
        return x.reshape(x.shape[0], x.shape[1], n, -1)

    def upstream(x, freqs, num_heads):
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        out = torch.view_as_complex(x.to(torch.float64).reshape(
            x.shape[0], x.shape[1], x.shape[2], -1, 2,
        ))
        return torch.view_as_real(out * freqs).flatten(2).to(x.dtype)

    return types.SimpleNamespace(torch=torch, rearrange=rearrange, rope_apply=upstream)


def test_cpu_numerics_match_upstream_float64_formula():
    module = _torch_module()
    torch = module.torch
    generator = torch.Generator(device="cpu").manual_seed(7)
    phases = torch.randn(6, 1, 4, dtype=torch.float64, generator=generator)
    freqs = torch.polar(torch.ones_like(phases), phases)
    x = torch.randn(1, 6, 16, generator=generator)
    wrapper = rope.RopeFP32(module, module.rope_apply)
    for dtype in (torch.bfloat16, torch.float32):
        inputs = x.to(dtype)
        actual = wrapper(inputs, freqs, 2)
        expected = module.rope_apply(inputs, freqs, 2)
        assert actual.dtype == dtype and actual.shape == inputs.shape
        if dtype == torch.bfloat16:
            assert torch.allclose(actual, expected, atol=1e-3)
        else:
            assert float((actual - expected).abs().max()) < 1e-5


def test_cpu_multiply_stays_complex64():
    module = _torch_module()
    torch = module.torch
    seen = []

    def view_as_real(value):
        seen.append(value.dtype)
        return torch.view_as_real(value)

    module.torch = types.SimpleNamespace(
        float32=torch.float32, complex64=torch.complex64,
        view_as_complex=torch.view_as_complex, view_as_real=view_as_real,
    )
    freqs = torch.ones(6, 1, 4, dtype=torch.complex128)
    with rope.patched_rope([module], "float32") as wrappers:
        wrapper, = wrappers
        module.rope_apply(torch.ones(1, 6, 16), freqs, 2)
        assert next(iter(wrapper.cache.values()))[1].dtype == torch.complex64
    assert seen == [torch.complex64]
    assert freqs.dtype == torch.complex128
