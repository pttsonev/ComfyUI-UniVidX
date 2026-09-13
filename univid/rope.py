"""Float32 RoPE across loaded DiTs without editing the pinned upstream snapshot.

Unlike the usual instance-attribute wrappers, SelfAttention resolves rope_apply
through its DiT module global: there is no instance seam. This module-attribute
swap lasts for one pipeline call, restores in finally, and refuses re-entry.
The small frequency cache belongs to that call and is cleared on exit. Torch
and rearrange are read from the vendored module at call time; no torch import
is needed to import this module or exercise the wrapper with fake tensors.
"""

from collections import OrderedDict
from contextlib import contextmanager
import sys

from .attention import DIT_MODULE_NAMES

PRECISIONS = ("float64", "float32")


def dit_modules():
    """Loaded DiTs whose SelfAttention resolves rope_apply in its module namespace.

    Intrinsic, alpha and base each define their own rope_apply and SelfAttention;
    replacing one namespace's attribute cannot affect either of the others.
    """
    return [sys.modules[name] for name in DIT_MODULE_NAMES if name in sys.modules]


class RopeFP32:
    """Upstream's RoPE layout with float32 pairs and complex64 multiplication."""

    def __init__(self, module, original):
        self.module = module
        self.original = original
        self.calls = 0
        self.cache = OrderedDict()

    def __call__(self, x, freqs, num_heads):
        self.calls += 1
        torch, rearrange = self.module.torch, self.module.rearrange
        key = (id(freqs), freqs.data_ptr(), tuple(freqs.shape), freqs.device)
        if key not in self.cache:
            # Retain the source too, preventing identity/storage reuse from
            # serving another tensor's cast. At most four sources are held.
            self.cache[key] = (freqs, freqs.to(torch.complex64))
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        freqs32 = self.cache[key][1]
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        x_out = torch.view_as_complex(x.to(torch.float32).reshape(
            x.shape[0], x.shape[1], x.shape[2], -1, 2,
        ))
        x_out = torch.view_as_real(x_out * freqs32).flatten(2)
        return x_out.to(x.dtype)


@contextmanager
def patched_rope(modules, precision: str):
    """Replace every loaded RoPE seam, yielding per-module engagement counters."""
    if precision not in PRECISIONS:
        raise ValueError(
            f"Unknown RoPE precision: {precision!r} (choose from {', '.join(PRECISIONS)})."
        )
    if precision == "float64":
        yield None
        return
    originals = [(module, module.rope_apply) for module in modules if hasattr(module, "rope_apply")]
    if not originals:
        raise RuntimeError("rope_precision=float32: no vendored DiT rope_apply is loaded to patch.")
    # Check every namespace before mutating any, so a nested call cannot disturb
    # an outer patch even when its module appears last in the discovery order.
    if any(isinstance(original, RopeFP32) for _, original in originals):
        raise RuntimeError("rope_precision=float32: rope_apply is already patched; re-entry refused.")
    wrappers = []
    try:
        for module, original in originals:
            wrapper = RopeFP32(module, original)
            wrappers.append(wrapper)
            module.rope_apply = wrapper
        yield wrappers
    finally:
        for wrapper in reversed(wrappers):
            wrapper.module.rope_apply = wrapper.original
            wrapper.cache.clear()


def rope_report(wrappers) -> list[str]:
    """Sum actual calls and name only the DiTs that used float32 RoPE."""
    if wrappers is None:
        return []
    calls = sum(wrapper.calls for wrapper in wrappers)
    names = ", ".join(
        wrapper.module.__name__.rsplit("wan_video_dit", 1)[-1].lstrip("_") or "base"
        for wrapper in wrappers if wrapper.calls
    )
    suffix = f" ({names})" if names else ""
    return [f"RoPE applied in float32 on {calls} calls{suffix}."]
