"""The attention swap: kernel routing, fallback, and restoration of the vendored `F`.

Pure tests with fake tensors and fake modules. The behaviours pinned are the
ones that would be silent when wrong: a swap that leaks past the run, a fallback
that hides behind the requested kernel, and a call that SageAttention cannot
take being sent to it anyway.
"""

import logging
import sys
import types

import pytest

from univid import attention


class _Tensor:
    """Already contiguous: `contiguous()` hands back the same object, as torch does."""

    def __init__(self, dtype="torch.bfloat16"):
        self.dtype = dtype
        self.contiguous_calls = 0
        self.clone_calls = 0

    def contiguous(self):
        self.contiguous_calls += 1
        return self

    def clone(self):
        self.clone_calls += 1
        return _Tensor(self.dtype)


class _StridedTensor(_Tensor):
    """A view: `contiguous()` materialises a fresh tensor."""

    def contiguous(self):
        self.contiguous_calls += 1
        return _Tensor(self.dtype)


def _functional(record):
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        record.append({"attn_mask": attn_mask, "dropout_p": dropout_p,
                       "is_causal": is_causal, "scale": scale, **kwargs})
        return "sdpa-out"

    return types.SimpleNamespace(scaled_dot_product_attention=sdpa, silu="the-silu")


def _module(record):
    return types.SimpleNamespace(F=_functional(record))


def _sage(record, fail=None):
    def sageattn(q, k, v, **kwargs):
        if fail is not None:
            raise fail
        record.append(kwargs)
        return "sage-out"

    return sageattn


# --- the proxy ----------------------------------------------------------------


def test_proxy_replaces_sdpa_and_delegates_everything_else():
    record = []
    functional = _functional(record)
    proxy = attention.FunctionalProxy(functional, "the-kernel")
    assert proxy.scaled_dot_product_attention == "the-kernel"
    assert proxy.silu == "the-silu"
    with pytest.raises(AttributeError):
        proxy.does_not_exist


# --- the kernel ---------------------------------------------------------------


def test_eligible_calls_take_sage_with_hnd_layout_and_contiguous_inputs():
    sdpa_calls, sage_calls = [], []
    kernel = attention.SageKernel(_sage(sage_calls), _functional(sdpa_calls).scaled_dot_product_attention)
    q, k, v = _Tensor(), _Tensor(), _Tensor()
    assert kernel(q, k, v) == "sage-out"
    assert sage_calls == [{"tensor_layout": "HND", "is_causal": False, "sm_scale": None}]
    assert (q.contiguous_calls, k.contiguous_calls, v.contiguous_calls) == (1, 1, 1)
    assert sdpa_calls == []
    assert (kernel.calls, kernel.sage_calls) == (1, 1)


def test_sage_never_receives_the_callers_storage_because_it_smooths_k_in_place():
    """`sageattn` does `k -= km`. With shared K/V that would corrupt rows 2-4 of a call."""
    seen = []

    def sageattn(q, k, v, **kwargs):
        seen.append((q, k, v))
        return "sage-out"

    kernel = attention.SageKernel(sageattn, _functional([]).scaled_dot_product_attention)
    q, k, v = _Tensor(), _Tensor(), _Tensor()          # contiguous: would alias the caller
    kernel(q, k, v)
    assert (k.clone_calls, v.clone_calls) == (1, 1)
    assert seen[0][1] is not k and seen[0][2] is not v
    q, k, v = _StridedTensor(), _StridedTensor(), _StridedTensor()   # a view: contiguous() already copied
    kernel(q, k, v)
    assert (k.clone_calls, v.clone_calls) == (0, 0)
    assert seen[1][1] is not k and seen[1][2] is not v


def test_scale_is_forwarded_as_sm_scale():
    sage_calls = []
    kernel = attention.SageKernel(_sage(sage_calls), _functional([]).scaled_dot_product_attention)
    kernel(_Tensor(), _Tensor(), _Tensor(), scale=0.125)
    assert sage_calls[0]["sm_scale"] == 0.125


@pytest.mark.parametrize("kwargs", [
    {"attn_mask": "mask"}, {"dropout_p": 0.1}, {"is_causal": True}, {"enable_gqa": True},
])
def test_calls_sage_cannot_take_go_to_the_real_kernel_unchanged(kwargs):
    sdpa_calls, sage_calls = [], []
    kernel = attention.SageKernel(_sage(sage_calls), _functional(sdpa_calls).scaled_dot_product_attention)
    assert kernel(_Tensor(), _Tensor(), _Tensor(), **kwargs) == "sdpa-out"
    assert sage_calls == []
    assert len(sdpa_calls) == 1
    for name, value in kwargs.items():
        assert sdpa_calls[0][name] == value
    assert kernel.disabled_reason is None  # ineligible is not failure
    assert (kernel.calls, kernel.sage_calls) == (1, 0)


def test_float32_inputs_disable_sage_for_the_run_and_log_once(caplog):
    sdpa_calls, sage_calls = [], []
    kernel = attention.SageKernel(_sage(sage_calls), _functional(sdpa_calls).scaled_dot_product_attention)
    with caplog.at_level(logging.WARNING, logger=attention.__name__):
        kernel(_Tensor("torch.float32"), _Tensor("torch.float32"), _Tensor("torch.float32"))
        kernel(_Tensor(), _Tensor(), _Tensor())  # half precision, but the run is already off sage
    assert sage_calls == []
    assert len(sdpa_calls) == 2
    assert "float32" in kernel.disabled_reason
    assert sum("SageAttention disabled" in r.message for r in caplog.records) == 1


def test_a_kernel_failure_falls_back_for_the_rest_of_the_run_and_logs_once(caplog):
    sdpa_calls = []
    kernel = attention.SageKernel(
        _sage([], fail=RuntimeError("no kernel for sm_120")),
        _functional(sdpa_calls).scaled_dot_product_attention,
    )
    with caplog.at_level(logging.WARNING, logger=attention.__name__):
        for _ in range(3):
            assert kernel(_Tensor(), _Tensor(), _Tensor()) == "sdpa-out"
    assert len(sdpa_calls) == 3
    assert kernel.disabled_reason == "RuntimeError: no kernel for sm_120"
    assert (kernel.calls, kernel.sage_calls) == (3, 0)
    assert sum("SageAttention disabled" in r.message for r in caplog.records) == 1


# --- the swap -----------------------------------------------------------------


def test_sdpa_is_a_no_op_that_touches_nothing():
    module = _module([])
    original = module.F
    with attention.patched_attention([module], "sdpa") as kernels:
        assert kernels is None
        assert module.F is original
    assert module.F is original


def test_sage_swaps_every_module_and_restores_even_when_the_body_raises(monkeypatch):
    monkeypatch.setattr(attention, "_import_sageattn", lambda: _sage([]))
    modules = [_module([]), _module([])]
    originals = [module.F for module in modules]
    with pytest.raises(RuntimeError, match="boom"):
        with attention.patched_attention(modules, "sage") as kernels:
            assert len(kernels) == 2
            for module in modules:
                assert isinstance(module.F, attention.FunctionalProxy)
                assert isinstance(module.F.scaled_dot_product_attention, attention.SageKernel)
                assert module.F.silu == "the-silu"
            raise RuntimeError("boom")
    for module, original in zip(modules, originals):
        assert module.F is original


def test_sage_without_the_package_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(attention, "_import_sageattn", lambda: None)
    assert attention.sage_available() is False
    module = _module([])
    with pytest.raises(RuntimeError, match="sageattention"):
        with attention.patched_attention([module], "sage"):
            pass
    assert not isinstance(module.F, attention.FunctionalProxy)


def test_sage_with_no_loaded_dit_module_is_refused(monkeypatch):
    monkeypatch.setattr(attention, "_import_sageattn", lambda: _sage([]))
    with pytest.raises(RuntimeError, match="no vendored DiT module"):
        with attention.patched_attention([], "sage"):
            pass


def test_unknown_kernel_is_refused_before_anything_is_touched():
    module = _module([])
    with pytest.raises(ValueError, match="Unknown attention kernel"):
        with attention.patched_attention([module], "flash_attn_3"):
            pass


def test_dit_modules_lists_only_the_vendored_modules_that_are_loaded(monkeypatch):
    fake = types.ModuleType(attention.DIT_MODULE_NAMES[1])
    for name in attention.DIT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, attention.DIT_MODULE_NAMES[1], fake)
    assert attention.dit_modules() == [fake]


# --- shared K/V ---------------------------------------------------------------


class _T:
    """Fake tensor for the layout tests: names carry the rearranges applied so far."""

    def __init__(self, name, batch=4):
        self.name = name
        self.shape = (batch, 7, 8)
        self.rows = {}

    def __getitem__(self, item):
        return _T(f"{self.name}[{item.start}:{item.stop}]", batch=1)

    def __setitem__(self, index, value):
        self.rows[index] = value


def _rearrange(x, pattern, **axes):
    return _T(f"{x.name} | {pattern}", batch=x.shape[0])


class _Torch:
    def __init__(self, draw):
        self.draw = draw          # what `rand(1) < drop_out` evaluates to
        self.rand_calls = 0

    def rand(self, n):
        self.rand_calls += 1
        torch = self

        class _Draw:
            def __lt__(self, other):
                return torch.draw

        return _Draw()

    @staticmethod
    def empty_like(x):
        return _T(f"empty_like({x.name})", batch=x.shape[0])


def _layout_module(record, draw=False, mode="scaled_dot_product"):
    """A vendored DiT module as the wrapper sees it: torch, rearrange, F, ATTENTION_MODE."""
    def sdpa(q, k, v, **kwargs):
        record.append((q, k, v, kwargs))
        return _T(f"attn({q.name})", batch=q.shape[0])

    def original(q, k, v, num_heads, compatibility_mode=False, drop_out=None):
        record.append(("original", q, k, v, num_heads, compatibility_mode, drop_out))
        return "upstream-out"

    return types.SimpleNamespace(
        torch=_Torch(draw), rearrange=_rearrange, ATTENTION_MODE=mode,
        F=types.SimpleNamespace(scaled_dot_product_attention=sdpa), flash_attention=original,
    )


def test_shared_path_attends_each_row_against_one_kv_with_no_repeat():
    record = []
    module = _layout_module(record)
    wrapper = attention.SharedKVAttention(module, module.flash_attention)
    q, k, v = _T("q"), _T("k"), _T("v")
    out = wrapper(q, k, v, 40, drop_out=0)
    assert len(record) == 4
    for i, (qi, ki, vi, kwargs) in enumerate(record):
        assert qi.name == f"q | b s (n d) -> b n s d[{i}:{i + 1}]"
        assert ki is record[0][1] and vi is record[0][2]          # ONE K, ONE V, for every row
        assert ki.name == "k | b s (n d) -> 1 n (b s) d" and vi.name == "v | b s (n d) -> 1 n (b s) d"
        assert "repeat" not in ki.name and "expand" not in ki.name
        assert kwargs == {}
    assert out.name == "empty_like(q)"
    assert sorted(out.rows) == [0, 1, 2, 3]
    assert out.rows[2].name == "attn(q | b s (n d) -> b n s d[2:3]) | 1 n s d -> s (n d)"
    assert (wrapper.calls, wrapper.shared_calls, wrapper.per_row_calls) == (1, 1, 0)
    assert module.torch.rand_calls == 1                          # upstream's RNG draw, preserved


def test_drop_out_branch_reproduces_upstreams_per_row_attention_and_is_not_counted_as_shared():
    record = []
    module = _layout_module(record, draw=True)
    wrapper = attention.SharedKVAttention(module, module.flash_attention)
    out = wrapper(_T("q"), _T("k"), _T("v"), 40, drop_out=0.25)
    assert len(record) == 1
    q, k, v, _ = record[0]
    assert (q.name, k.name, v.name) == tuple(f"{n} | b s (n d) -> b n s d" for n in "qkv")
    assert out.name == "attn(q | b s (n d) -> b n s d) | b n s d -> b s (n d)"
    assert (wrapper.calls, wrapper.shared_calls, wrapper.per_row_calls) == (1, 0, 1)
    assert module.torch.rand_calls == 1


def test_the_kernel_is_read_from_the_module_at_call_time_so_sage_still_routes():
    record, later = [], []
    module = _layout_module(record)
    wrapper = attention.SharedKVAttention(module, module.flash_attention)
    module.F = types.SimpleNamespace(
        scaled_dot_product_attention=lambda q, k, v, **kw: (later.append(q), _T("late", batch=1))[1],
    )
    wrapper(_T("q"), _T("k"), _T("v"), 40, drop_out=0)
    assert record == [] and len(later) == 4


def test_a_non_sdpa_attention_mode_delegates_to_upstream_untouched():
    record = []
    module = _layout_module(record, mode="flash_attn_3")
    wrapper = attention.SharedKVAttention(module, module.flash_attention)
    assert wrapper(_T("q"), _T("k"), _T("v"), 40, drop_out=0) == "upstream-out"
    assert record[0][0] == "original" and record[0][4:] == (40, False, 0)
    assert (wrapper.calls, wrapper.shared_calls, wrapper.per_row_calls) == (1, 0, 0)
    assert module.torch.rand_calls == 0
    record.clear()
    wrapper(_T("q"), _T("k"), _T("v"), 40, compatibility_mode=True, drop_out=0)
    assert len(record) == 4                                      # compatibility_mode forces the sdpa branch


def test_shared_kv_swaps_every_module_and_restores_even_when_the_body_raises():
    modules = [_layout_module([]), _layout_module([])]
    originals = [module.flash_attention for module in modules]
    with pytest.raises(RuntimeError, match="boom"):
        with attention.patched_shared_kv(modules, True) as wrappers:
            assert len(wrappers) == 2
            for module, wrapper, original in zip(modules, wrappers, originals):
                assert module.flash_attention is wrapper
                assert isinstance(wrapper, attention.SharedKVAttention)
                assert wrapper.original is original
            raise RuntimeError("boom")
    for module, original in zip(modules, originals):
        assert module.flash_attention is original


def test_shared_kv_disabled_touches_nothing_and_enabled_with_no_module_is_refused():
    module = _layout_module([])
    original = module.flash_attention
    with attention.patched_shared_kv([module], False) as wrappers:
        assert wrappers is None
        assert module.flash_attention is original
    with pytest.raises(RuntimeError, match="no vendored DiT module"):
        with attention.patched_shared_kv([], True):
            pass


def test_shared_kv_modules_lists_only_the_dits_that_repeat_kv(monkeypatch):
    assert "src.models.wan_video_dit" not in attention.SHARED_KV_MODULE_NAMES
    for name in attention.DIT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    fake = types.ModuleType(attention.SHARED_KV_MODULE_NAMES[0])
    monkeypatch.setitem(sys.modules, attention.SHARED_KV_MODULE_NAMES[0], fake)
    monkeypatch.setitem(sys.modules, "src.models.wan_video_dit", types.ModuleType("src.models.wan_video_dit"))
    assert attention.shared_kv_modules() == [fake]


def test_shared_kv_report_counts_across_wrappers():
    module = _layout_module([])
    first = attention.SharedKVAttention(module, module.flash_attention)
    second = attention.SharedKVAttention(module, module.flash_attention)
    first(_T("q"), _T("k"), _T("v"), 40, drop_out=0)
    module.torch.draw = True
    second(_T("q"), _T("k"), _T("v"), 40, drop_out=1)
    assert attention.shared_kv_report([first, second]) == [
        "Shared K/V attention applied on 1 of 1 cross-modal attention calls; "
        "1 per-row cross-attention calls unaffected.",
    ]
    module.ATTENTION_MODE = "flash_attn_3"
    first(_T("q"), _T("k"), _T("v"), 40, drop_out=0)
    assert (first.calls, first.shared_calls, first.per_row_calls) == (2, 1, 0)
    assert attention.shared_kv_report([first, second]) == [
        "Shared K/V attention applied on 1 of 2 cross-modal attention calls; "
        "1 per-row cross-attention calls unaffected.",
    ]
    assert attention.shared_kv_report(None) == []


def _torch_rearrange(x, pattern, n=None, **axes):
    """The four einops patterns upstream's layout uses, in plain torch (no einops on the test box)."""
    if pattern == "b s (n d) -> b n s d":
        b, s, nd = x.shape
        return x.view(b, s, n, nd // n).permute(0, 2, 1, 3)
    if pattern == "b s (n d) -> 1 n (b s) d":
        b, s, nd = x.shape
        return x.reshape(1, b * s, n, nd // n).permute(0, 2, 1, 3)
    if pattern == "b n s d -> b s (n d)":
        b, heads, s, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, s, heads * d)
    if pattern == "1 n s d -> s (n d)":
        _, heads, s, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(s, heads * d)
    raise AssertionError(pattern)


def test_shared_kv_matches_upstreams_repeat_formula_on_real_tensors():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    generator = torch.Generator().manual_seed(0)
    b, s, n, d = 4, 37, 4, 16
    q, k, v = (torch.randn(b, s, n * d, generator=generator).to(device=device, dtype=dtype) for _ in range(3))
    functional = torch.nn.functional
    module = types.SimpleNamespace(
        torch=torch, rearrange=_torch_rearrange, ATTENTION_MODE="scaled_dot_product", F=functional,
    )
    # upstream's cross-modal branch, verbatim but for the rearrange spelling
    q4 = _torch_rearrange(q, "b s (n d) -> b n s d", n=n)
    k4 = _torch_rearrange(k, "b s (n d) -> 1 n (b s) d", n=n).repeat(b, 1, 1, 1)
    v4 = _torch_rearrange(v, "b s (n d) -> 1 n (b s) d", n=n).repeat(b, 1, 1, 1)
    expected = _torch_rearrange(functional.scaled_dot_product_attention(q4, k4, v4), "b n s d -> b s (n d)")
    wrapper = attention.SharedKVAttention(module, original=None)
    actual = wrapper(q, k, v, n, drop_out=0)
    assert actual.shape == expected.shape
    if device == "cuda":
        assert torch.equal(actual, expected)                    # bit-identical on the fused kernels
    else:
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert (wrapper.calls, wrapper.shared_calls, wrapper.per_row_calls) == (1, 1, 0)


# --- the report ---------------------------------------------------------------


def test_engagement_report_counts_across_kernels_and_names_any_fallback():
    healthy = attention.SageKernel(_sage([]), _functional([]).scaled_dot_product_attention)
    healthy(_Tensor(), _Tensor(), _Tensor())
    broken = attention.SageKernel(
        _sage([], fail=ValueError("bad")), _functional([]).scaled_dot_product_attention,
    )
    broken(_Tensor(), _Tensor(), _Tensor())
    lines = attention.engagement_report([healthy, broken])
    assert lines[0] == "SageAttention handled 1 of 2 attention calls."
    assert lines[1] == "SageAttention fell back to sdpa: ValueError: bad"
    assert attention.engagement_report(None) == []


def test_module_imports_no_torch():
    import inspect

    source = inspect.getsource(attention)
    assert "import torch" not in source
    assert "sageattention" in source  # imported lazily, inside a function only
