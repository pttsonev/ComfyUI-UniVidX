"""Swap the DiT's attention kernel at run time without touching the vendored snapshot.

Upstream's `flash_attention` knows two kernels: flash_attn_3, which is not
installed here, and torch's scaled_dot_product_attention. SageAttention (INT8
quantised QK) is what ComfyUI itself runs on this class of GPU, and it is the
lever that matters at high resolution, where attention is the quadratic term.

Each vendored DiT module holds a module-level `F` bound to torch.nn.functional
and calls `F.scaled_dot_product_attention` from inside its own layout logic —
including the cross-modal repeat of K/V across the four modality latents.
Replacing `F` with a delegating proxy for the duration of one pipeline call keeps
that layout logic running byte-for-byte and changes only the kernel beneath it.
Nothing in the vendor tree is edited and nothing else in the process sees the
swap.

A separate, temporary `flash_attention` wrapper on the intrinsic and alpha
DiTs removes the cross-modal K/V repeat. Each modality's query row attends to
the same shared K/V views, preserving the CPU RNG draw and reading the module's
current `F` at call time so the layout swap composes with either kernel. The
base Wan DiT has no repeat and keeps its original layout function.

SageAttention is an approximation. It is off by default, named in the sampler's
messages when on, and the run reports how many attention calls it actually took
so a silent fallback can never pass for a speedup. This module imports no torch.
"""

from contextlib import contextmanager
import logging
import sys

_LOGGER = logging.getLogger(__name__)

KERNELS = ("sdpa", "sage")
DIT_MODULE_NAMES = (
    "src.models.wan_video_dit_intrinsic",
    "src.models.wan_video_dit_alpha",
    "src.models.wan_video_dit",
)
SHARED_KV_MODULE_NAMES = (
    "src.models.wan_video_dit_intrinsic",
    "src.models.wan_video_dit_alpha",
)
_SAGE_DTYPES = frozenset({"torch.float16", "torch.bfloat16"})


def _import_sageattn():
    """Return sageattention.sageattn, or None when the package cannot be imported."""
    try:
        from sageattention import sageattn
    except Exception:  # absent, or a broken CUDA extension raising at import
        return None
    return sageattn


def sage_available() -> bool:
    return _import_sageattn() is not None


def dit_modules():
    """The vendored DiT modules already imported into this process, in a stable order."""
    return [sys.modules[name] for name in DIT_MODULE_NAMES if name in sys.modules]


def shared_kv_modules():
    """Loaded vendored DiTs whose cross-modal attention repeats K/V."""
    return [sys.modules[name] for name in SHARED_KV_MODULE_NAMES if name in sys.modules]


class SharedKVAttention:
    """Attend each query row against shared K/V without changing the active kernel."""

    def __init__(self, module, original):
        self.module = module
        self.original = original
        self.calls = 0
        self.shared_calls = 0
        self.per_row_calls = 0

    def __call__(self, q, k, v, num_heads, compatibility_mode=False, drop_out=None):
        self.calls += 1
        module = self.module
        if not (compatibility_mode or module.ATTENTION_MODE == "scaled_dot_product"):
            return self.original(
                q, k, v, num_heads, compatibility_mode=compatibility_mode, drop_out=drop_out,
            )
        torch, rearrange, F = module.torch, module.rearrange, module.F
        # Even drop_out=0 consumes one CPU RNG draw in upstream's layout function.
        if torch.rand(1) < drop_out:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = F.scaled_dot_product_attention(q, k, v)
            self.per_row_calls += 1
            return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

        batch_size = q.shape[0]
        q4 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k1 = rearrange(k, "b s (n d) -> 1 n (b s) d", n=num_heads)
        v1 = rearrange(v, "b s (n d) -> 1 n (b s) d", n=num_heads)
        out = torch.empty_like(q)
        for i in range(batch_size):
            out[i] = rearrange(
                F.scaled_dot_product_attention(q4[i:i + 1], k1, v1),
                "1 n s d -> s (n d)", n=num_heads,
            )
        self.shared_calls += 1
        return out


class SageKernel:
    """`scaled_dot_product_attention` stand-in routing eligible calls to SageAttention.

    Eligibility is exactly what SageAttention supports: no mask, no dropout,
    non-causal, half-precision inputs. Anything else goes to the real kernel
    unchanged. A failure inside SageAttention is logged ONCE and the rest of the
    run stays on the real kernel — the same failure would otherwise repeat on
    every layer of every step, and the counts still tell the truth afterwards.
    """

    def __init__(self, sageattn, fallback):
        self.sageattn = sageattn
        self.fallback = fallback
        self.disabled_reason = None
        self.calls = 0
        self.sage_calls = 0

    def _disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            _LOGGER.warning("[ComfyUI-UniVidX] SageAttention disabled for this run: %s", reason)

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                 scale=None, **kwargs):
        self.calls += 1
        eligible = (
            self.disabled_reason is None and attn_mask is None
            and not dropout_p and not is_causal and not kwargs
        )
        if eligible:
            dtype = str(getattr(query, "dtype", ""))
            if dtype not in _SAGE_DTYPES:
                self._disable(f"needs float16/bfloat16 inputs, got {dtype or 'unknown'}")
            else:
                try:
                    # Sage smooths K in place. Keep caller storage intact for
                    # later query rows and for fallback if the kernel raises.
                    sage_key = key.contiguous()
                    if sage_key is key:
                        sage_key = sage_key.clone()
                    sage_value = value.contiguous()
                    if sage_value is value:
                        sage_value = sage_value.clone()
                    out = self.sageattn(
                        query.contiguous(), sage_key, sage_value,
                        tensor_layout="HND", is_causal=False, sm_scale=scale,
                    )
                except Exception as exc:  # the kernel is optional; the run is not
                    self._disable(f"{type(exc).__name__}: {exc}")
                else:
                    self.sage_calls += 1
                    return out
        return self.fallback(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, **kwargs,
        )


class FunctionalProxy:
    """`torch.nn.functional` with one name replaced; every other attribute is delegated."""

    def __init__(self, functional, kernel):
        self._functional = functional
        self.scaled_dot_product_attention = kernel

    def __getattr__(self, name):
        return getattr(self._functional, name)


@contextmanager
def patched_attention(modules, kernel: str):
    """Run the body with `kernel` under the given DiT modules, restoring `F` afterwards.

    Yields the SageKernel instances so the caller can report how many attention
    calls actually took the fast path; yields None for the reference kernel.
    """
    if kernel not in KERNELS:
        raise ValueError(
            f"Unknown attention kernel: {kernel!r} (choose from {', '.join(KERNELS)})."
        )
    if kernel == "sdpa":
        yield None
        return
    sageattn = _import_sageattn()
    if sageattn is None:
        raise RuntimeError(
            "attention=sage needs the sageattention package in ComfyUI's Python, and it "
            "is not importable there. Install it, or set attention back to sdpa."
        )
    modules = list(modules)
    if not modules:
        raise RuntimeError("attention=sage: no vendored DiT module is loaded to patch.")
    originals = [(module, module.F) for module in modules]
    kernels = []
    try:
        for module, functional in originals:
            proxy_kernel = SageKernel(sageattn, functional.scaled_dot_product_attention)
            kernels.append(proxy_kernel)
            module.F = FunctionalProxy(functional, proxy_kernel)
        yield kernels
    finally:
        for module, functional in originals:
            module.F = functional


@contextmanager
def patched_shared_kv(modules, enabled: bool):
    """Temporarily remove the cross-modal K/V repeat, yielding engagement counters."""
    if not enabled:
        yield None
        return
    modules = list(modules)
    if not modules:
        raise RuntimeError("shared_kv=True: no vendored DiT module is loaded to patch.")
    originals = [(module, module.flash_attention) for module in modules]
    wrappers = []
    try:
        for module, original in originals:
            wrapper = SharedKVAttention(module, original)
            wrappers.append(wrapper)
            module.flash_attention = wrapper
        yield wrappers
    finally:
        for module, original in originals:
            module.flash_attention = original


def shared_kv_report(wrappers) -> list[str]:
    """Cross-modal engagement, with upstream per-row calls reported separately."""
    if not wrappers:
        return []
    total = sum(wrapper.calls for wrapper in wrappers)
    shared = sum(wrapper.shared_calls for wrapper in wrappers)
    per_row = sum(wrapper.per_row_calls for wrapper in wrappers)
    # Non-sdpa calls delegated to upstream remain in the cross-modal denominator.
    delegated = total - shared - per_row
    return [
        f"Shared K/V attention applied on {shared} of {shared + delegated} cross-modal attention "
        f"calls; {per_row} per-row cross-attention calls unaffected."
    ]


def engagement_report(kernels) -> list[str]:
    """Human-readable lines on what SageAttention actually did during one run."""
    if not kernels:
        return []
    total = sum(kernel.calls for kernel in kernels)
    taken = sum(kernel.sage_calls for kernel in kernels)
    lines = [f"SageAttention handled {taken} of {total} attention calls."]
    for kernel in kernels:
        if kernel.disabled_reason is not None:
            lines.append(f"SageAttention fell back to sdpa: {kernel.disabled_reason}")
    return lines
