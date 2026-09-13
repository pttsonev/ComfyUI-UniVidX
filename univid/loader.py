"""Assemble the pinned UniVidX pipelines from absolute model paths.

Only model assembly is replaced; sampling and modality-specific LoRA routing stay
upstream. Models are assembled on CPU in the compute dtype before upstream's
VRAM manager takes over. Torch and the vendored engine are imported only on load.
Optional LightX2V updates are merged into the base state BEFORE UniVidX attaches
its per-modality adapters. A different selection or strength requires rebuilding;
the cached merged base cannot be un-merged. The reference default remains none.

VRAM management selects one of two upstream strategies: a persistent DiT
parameter cap, or CPU streaming under a VRAM limit minus the buffer. These are
mutually exclusive; the cap makes upstream ignore the limit and buffer. Lower
residency trades wall time for memory without changing weights or compute dtype.
The text encoder already onloads to CPU in both strategies.

The cache is keyed on fully resolved settings and bounded to ONE handle by
default, evicting the previous model *before* allocating a replacement — a BF16
DiT is ~28 GB of host RAM, and simply nudging `vram_buffer` in the UI produces a
new key. Raise `UNIVIDX_MODEL_CACHE_MAX` only with the RAM to match.

Eviction is still ADVISORY about *when* memory comes back: it drops this
module's reference, but a ComfyUI graph holding a handle keeps its models alive
until that graph is released. Dropping our reference is necessary for the memory
to be reclaimable at all; it is not sufficient to reclaim it immediately.
"""

from collections import OrderedDict
from dataclasses import dataclass
import gc
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import sys
from threading import RLock
from typing import Any

from . import paths


_UNSET = object()
_torch = _UNSET
_torch_error = None
_VENDOR_PATH_ADDED = False
_CACHE_LOCK = RLock()
# LRU-bounded, and bounded at ONE by default for a reason: a BF16 DiT is ~28 GB
# of host RAM, so retaining a second handle is not a cache, it is a leak that
# ordinary UI use triggers. Nudging vram_buffer or switching variant produces a
# new cache key, and nothing else holds a strong reference between graph runs —
# so an unbounded dict would strand every previous model for the life of the
# process. Weak references would be the opposite mistake: nothing holds a handle
# between runs, so every queue would reload 28 GB from disk.
# Raise via UNIVIDX_MODEL_CACHE_MAX only if you have the RAM for N models.
_MODEL_CACHE: "OrderedDict[tuple, ModelHandle]" = OrderedDict()


def _cache_limit() -> int:
    try:
        return max(1, int(os.environ.get("UNIVIDX_MODEL_CACHE_MAX", "1")))
    except ValueError:
        return 1


def _make_room(limit: int) -> None:
    """Drop least-recently-used handles BEFORE the replacement is allocated.

    Evicting after construction would put two full models in host RAM at once,
    which is precisely the peak this is meant to avoid.
    """
    while len(_MODEL_CACHE) >= limit:
        evicted_key, evicted = _MODEL_CACHE.popitem(last=False)
        _LOGGER.info(
            "[ComfyUI-UniVidX] Evicting cached model (variant=%s, dtype=%s, vram_buffer=%s) "
            "to stay within UNIVIDX_MODEL_CACHE_MAX=%d",
            evicted_key[0], evicted.compute_dtype, evicted.vram_buffer, limit,
        )
        del evicted
        gc.collect()
        if _torch not in (_UNSET, None) and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
_LOGGER = logging.getLogger(__name__)
_MODALITIES = {
    "intrinsic": ("rgb", "albedo", "irradiance", "normal"),
    "alpha": ("com", "pha", "fgr", "bgr"),
}
_LORA_TARGETS = "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"


@dataclass(frozen=True)
class LoadReport:
    """All incompatible keys, consumed format metadata and base merge counts."""

    stage: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    consumed_keys: tuple[str, ...] = ()
    merged_linears: int = 0
    merged_deltas: int = 0


@dataclass(frozen=True)
class ModelHandle:
    """Keep the upstream instance, resolved settings and load diagnostics alive."""

    variant: str
    model: Any
    model_paths: paths.ModelPaths
    compute_dtype: str
    device: str
    vram_buffer: float
    load_reports: tuple[LoadReport, ...]
    distillation: str = "none"
    distillation_strength: float = 1.0
    distillation_path: Path | None = None
    vram_limit: float | None = None
    num_persistent_param_in_dit: int | None = None

    @property
    def pipe(self):
        """The unmodified upstream pipeline used by the sampler."""
        return self.model.pipe


def _get_torch():
    """Cache both import success and failure; a pinned None is never retried."""
    global _torch, _torch_error
    if _torch is _UNSET:
        try:
            import torch as module
        except (ImportError, OSError) as exc:
            module = None
            _torch_error = exc
        _torch = module
    if _torch is None:
        raise RuntimeError("UniVidX requires a working PyTorch installation with CUDA.") from _torch_error
    return _torch


def _vendor_config() -> Path:
    """Find the materialised snapshot, including when the pack is junctioned."""
    root = Path(__file__).resolve().parents[1] / "vendor" / "univid"
    config = root / "configs" / "wan2_1_14b_t2v_dit_config.json"
    for path in (root / "src" / "__init__.py", config):
        if not path.is_file():
            raise paths.MissingModelFile(path.name, path.parent)
    return config


def _get_upstream(variant: str, vendor_root: Path):
    """Add the vendor root once, without replacing another pack's src package."""
    global _VENDOR_PATH_ADDED
    existing = sys.modules.get("src")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve().parent != vendor_root / "src":
            raise RuntimeError("UniVidX cannot import its vendor: another package already owns 'src'.")
    if not _VENDOR_PATH_ADDED:
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        _VENDOR_PATH_ADDED = True
    if variant == "intrinsic":
        from src.pipelines.univid_intrinsic import UniVidIntrinsic
        model_class = UniVidIntrinsic
    else:
        from src.pipelines.univid_alpha import UniVidAlpha
        model_class = UniVidAlpha
    return sys.modules[model_class.__module__], model_class


def _load_dit_state(source: paths.WeightSource, *, dtype, torch, load_file):
    """Merge shards or read one file, consuming only declared scaling metadata."""
    if source.kind not in {"canonical_shards", "single_file", "scaled_single_file"}:
        raise ValueError(f"Unknown DiT weight source kind: {source.kind!r}")
    state = {}
    for path in source.paths:
        shard = load_file(str(path), device="cpu")
        duplicates = state.keys() & shard.keys()
        if duplicates:
            raise ValueError(f"Duplicate DiT keys across shards: {sorted(duplicates)}")
        state.update(shard)
        del shard

    consumed = []
    if source.kind == "scaled_single_file":
        for weight_name, scale_name in source.scale_weights.items():
            # Multiply in float32, then round once to the compute dtype. Casting
            # the scale to BF16 first would lose precision before multiplication.
            weight = state[weight_name].to(dtype=torch.float32)
            weight.mul_(state[scale_name].to(dtype=torch.float32))
            state[weight_name] = weight.to(dtype=dtype)
            del weight
            del state[scale_name]
            consumed.append(scale_name)
        if "scaled_fp8" in state:
            del state["scaled_fp8"]
            consumed.append("scaled_fp8")
    # Replace entries in place so old FP8/F32 tensors are released one at a time.
    for name in state:
        state[name] = state[name].to(dtype=dtype)
    return state, tuple(sorted(consumed))


def _merge_lightx2v(
    state: dict, source: paths.LoRASource, *, strength: float, torch, load_file,
) -> LoadReport:
    """Merge on CPU in float32, one Linear at a time, then round to its dtype."""
    if not source.pairs:
        raise ValueError(f"LightX2V LoRA {source.path} would merge zero Linears.")
    lora = load_file(str(source.path), device="cpu")
    expected = source.header.keys() - {"__metadata__"}
    missing, unexpected = sorted(expected - lora.keys()), sorted(lora.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"LightX2V tensors differ from the validated header: missing keys={missing}; "
            f"unexpected keys={unexpected}."
        )
    # Validate every update before changing any base weight. In particular, a
    # wrong prefix must not silently turn a requested distillation into a no-op.
    for name, tensor in lora.items():
        if list(tensor.shape) != source.header[name]["shape"]:
            raise ValueError(f"LightX2V tensor {name!r} does not match its header shape.")
        if name not in source.alpha_keys.values() and not tensor.is_floating_point():
            raise ValueError(f"LightX2V tensor {name!r} must be floating point.")
    scales = {}
    for target, (a_key, b_key) in source.pairs.items():
        expected_shape = (lora[b_key].shape[0], lora[a_key].shape[1])
        if target not in state or tuple(state[target].shape) != expected_shape:
            raise ValueError(f"LightX2V target {target!r} is missing or has an incompatible base shape.")
        rank = lora[a_key].shape[0]
        alpha_key = source.alpha_keys.get(target)
        alpha = lora[alpha_key].item() if alpha_key is not None else source.alpha
        # LightX2V's SVD extractor absorbs the singular values into down/up and
        # writes no alpha. Absent alpha therefore means alpha=rank (scale 1),
        # NOT 1/rank. Honour a supplied per-Linear alpha or global metadata alpha.
        if alpha is None:
            alpha = rank
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError(f"LightX2V alpha for {target!r} must be finite and non-negative.")
        scales[target] = alpha / rank
    for target, key in source.deltas.items():
        if target not in state or state[target].shape != lora[key].shape:
            raise ValueError(f"LightX2V delta {key!r} has no matching base parameter {target!r}.")
    merged = 0
    with torch.no_grad():
        for target, (a_key, b_key) in source.pairs.items():
            if strength != 0:
                base = state[target]
                weight = base.to(dtype=torch.float32)
                weight.addmm_(
                    lora[b_key].to(dtype=torch.float32), lora[a_key].to(dtype=torch.float32),
                    beta=1.0, alpha=strength * scales[target],
                )
                state[target] = weight.to(dtype=base.dtype)
                del base, weight
            merged += 1
        for target, key in source.deltas.items():
            if strength != 0:
                base = state[target]
                weight = base.to(dtype=torch.float32)
                weight.add_(lora[key].to(dtype=torch.float32), alpha=strength)
                state[target] = weight.to(dtype=base.dtype)
                del base, weight
    if merged == 0:
        raise RuntimeError(f"LightX2V LoRA {source.path} merged zero Linears.")
    _LOGGER.info(
        "[ComfyUI-UniVidX] LightX2V %s: merged %d Linears and %d direct deltas "
        "into the base before UniVidX adapters; strength=%s%s",
        source.path, merged, len(source.deltas), strength,
        " (zero strength: base weights unchanged)" if strength == 0 else "",
    )
    return LoadReport(
        stage=f"LightX2V {source.path}", missing_keys=(), unexpected_keys=(),
        consumed_keys=tuple(sorted(lora)), merged_linears=merged, merged_deltas=len(source.deltas),
    )


def _apply_state(model, state: dict, *, stage: str, assign: bool = False, consumed_keys=()):
    """Report both key lists even when a shape mismatch makes loading raise."""
    try:
        result = model.load_state_dict(state, strict=False, assign=assign)
    except RuntimeError as exc:
        expected = model.state_dict().keys()
        missing = sorted(expected - state.keys())
        unexpected = sorted(state.keys() - expected)
        raise RuntimeError(
            f"{stage} failed: missing keys={missing}; unexpected keys={unexpected}. {exc}"
        ) from exc
    report = LoadReport(
        stage=stage,
        missing_keys=tuple(result.missing_keys),
        unexpected_keys=tuple(result.unexpected_keys),
        consumed_keys=tuple(consumed_keys),
    )
    _LOGGER.info(
        "[ComfyUI-UniVidX] %s: missing keys=%s; unexpected keys=%s; consumed metadata keys=%s",
        stage, report.missing_keys, report.unexpected_keys, report.consumed_keys,
    )
    return report


def _attach_checkpoint(model, checkpoint: Path, *, load_file) -> LoadReport:
    """Accept the selected family's adapters by their keys, never its filename."""
    raw_state = load_file(str(checkpoint), device="cpu")
    state = {}
    for name, tensor in raw_state.items():
        key = name.removeprefix("dit.")
        if key in state:
            raise ValueError(f"UniVidX checkpoint has duplicate key after removing 'dit.': {key!r}")
        state[key] = tensor
    del raw_state
    report = _apply_state(model, state, stage=f"UniVidX checkpoint {checkpoint}")
    # A LoRA checkpoint normally omits the base weights. Missing adapters or
    # unexpected keys instead indicate incompatible content (including a wrong family).
    missing_adapters = [
        name for name in report.missing_keys
        if ".lora_A." in name or ".lora_B." in name
    ]
    if missing_adapters or report.unexpected_keys:
        raise RuntimeError(
            f"Incompatible UniVidX checkpoint {checkpoint}: missing keys={report.missing_keys}; "
            f"unexpected keys={report.unexpected_keys}. All modality adapters are required."
        )
    return report


def _build_model(
    variant: str, resolved: paths.ModelPaths, config: Path, *, dtype, device, vram_buffer,
    distillation_source: paths.LoRASource | None = None, distillation_strength: float = 1.0,
    vram_limit: float | None = None, num_persistent_param_in_dit: int | None = None,
):
    torch = _get_torch()
    try:
        upstream, model_class = _get_upstream(variant, config.parent.parent)
        from safetensors.torch import load_file
        from src.models.util import init_weights_on_device
    except ImportError as exc:
        raise RuntimeError(
            f"UniVidX runtime dependency unavailable: {exc}. "
            "The pinned engine requires this dependency in the ComfyUI interpreter."
        ) from exc

    model = object.__new__(model_class)
    upstream.DiffusionTrainingModule.__init__(model)
    model.device = device
    model.torch_dtype = dtype
    model.pipe = upstream.WanVideoPipeline(device=device, torch_dtype=dtype)
    pipe = model.pipe

    manager = upstream.ModelManager(torch_dtype=dtype, device="cpu")
    manager.load_model(str(resolved.text_encoder), device="cpu", torch_dtype=dtype)
    manager.load_model(str(resolved.vae), device="cpu", torch_dtype=dtype)
    pipe.text_encoder = manager.fetch_model("wan_video_text_encoder")
    pipe.vae = manager.fetch_model("wan_video_vae")
    if pipe.text_encoder is None or pipe.vae is None:
        raise RuntimeError(
            "ModelManager did not detect the resolved Wan2.1 models: "
            f"text encoder={resolved.text_encoder}; VAE={resolved.vae}."
        )
    del manager

    with config.open(encoding="utf-8") as stream:
        dit_kwargs = json.load(stream)
    # Reuse upstream's empty-parameter construction, avoiding a temporary 56 GB
    # float32 DiT. Non-parameter data such as rotary frequencies remains real.
    with init_weights_on_device():
        pipe.dit = upstream.WanModel(**dit_kwargs)
    state, consumed = _load_dit_state(resolved.dit, dtype=dtype, torch=torch, load_file=load_file)
    distillation_reports = ()
    if distillation_source is not None:
        distillation_reports = (_merge_lightx2v(
            state, distillation_source, strength=distillation_strength, torch=torch, load_file=load_file,
        ),)
    dit_report = _apply_state(pipe.dit, state, stage="Wan2.1 DiT", assign=True, consumed_keys=consumed)
    del state
    if dit_report.missing_keys or dit_report.unexpected_keys:
        raise RuntimeError(
            f"Incompatible Wan2.1 DiT: missing keys={dit_report.missing_keys}; "
            f"unexpected keys={dit_report.unexpected_keys}."
        )

    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(str(resolved.tokenizer))
    lora_configs = [
        {"target_modules": _LORA_TARGETS, "lora_rank": 32, "adapter_name": modality}
        for modality in _MODALITIES[variant]
    ]
    pipe.dit = model.add_multiple_loras_to_model(model=pipe.dit, lora_configs=lora_configs)
    pipe.dit.to(dtype=dtype)
    checkpoint_report = _attach_checkpoint(pipe.dit, resolved.checkpoint, load_file=load_file)
    pipe.units = [
        upstream.WanVideoUnit_ShapeChecker(),
        upstream.WanVideoUnit_NoiseInitializer(),
        upstream.WanVideoUnit_InputVideoEmbedder(),
        upstream.WanVideoUnit_PromptEmbedder(),
    ]
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.freeze_except([])
    for name, param in pipe.dit.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.use_gradient_checkpointing = True
    model.use_gradient_checkpointing_offload = False
    model.extra_inputs = []
    # Upstream's __init__ leaves the module in train mode — its inference script
    # calls model.eval() separately, and we have replaced that script. Without
    # this the DiT samples in train mode with gradient checkpointing live, which
    # is both slower and not the reference path.
    model.eval()
    # vram_buffer stays at upstream's own default (0.5 GiB) unless the caller
    # overrides it. Phase 6 measures real residency; picking a different number
    # before then would be an unmeasured deviation from the reference path.
    pipe.enable_vram_management(
        num_persistent_param_in_dit=num_persistent_param_in_dit,
        vram_limit=vram_limit, vram_buffer=vram_buffer,
    )
    return model, (*distillation_reports, dit_report, checkpoint_report)


def load_model(
    variant: str,
    *,
    dit: str | PathLike[str] | None = None,
    vae: str | PathLike[str] | None = None,
    text_encoder: str | PathLike[str] | None = None,
    checkpoint: str | PathLike[str] | None = None,
    tokenizer: str | PathLike[str] = "umt5-xxl",
    compute_dtype: str = "bfloat16",
    device: str = "cuda",
    vram_buffer: float = 0.5,
    distillation: str = "none",
    distillation_strength: float = 1.0,
    vram_limit: float | None = None,
    num_persistent_param_in_dit: int | None = None,
) -> ModelHandle:
    """Resolve, construct or reuse a family handle; VRAM values are in GiB.

    Explicit absolute model paths also work outside ComfyUI. Device aliases such
    as cuda resolve to an indexed device before caching. Only successful loads
    are cached; sampler behaviour remains the responsibility of upstream.
    The node's vram_limit=0 and num_persistent_param_in_dit=-1 sentinels map to
    upstream's None defaults. A parameter cap of zero remains a real cap.
    """
    if variant not in _MODALITIES:
        raise ValueError(f"Unknown UniVidX variant: {variant!r}")
    if compute_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported UniVidX compute dtype: {compute_dtype!r}")
    if distillation not in {"none", "lightx2v"}:
        raise ValueError(f"Unknown UniVidX distillation option: {distillation!r}")
    distillation_strength = float(distillation_strength)
    if not math.isfinite(distillation_strength) or not 0 <= distillation_strength <= 2:
        raise ValueError("distillation_strength must be a finite number between 0.0 and 2.0.")
    vram_buffer = float(vram_buffer)
    if not math.isfinite(vram_buffer) or vram_buffer < 0:
        raise ValueError("vram_buffer must be a finite, non-negative number of GiB.")
    if vram_limit is not None:
        vram_limit = float(vram_limit)
        if not math.isfinite(vram_limit) or vram_limit < 0:
            raise ValueError("vram_limit must be finite and non-negative in GiB; 0 means unset.")
        if vram_limit == 0:
            vram_limit = None
    if num_persistent_param_in_dit is not None:
        if type(num_persistent_param_in_dit) is not int or num_persistent_param_in_dit < -1:
            raise ValueError("num_persistent_param_in_dit must be a non-negative integer or -1 (unset).")
        if num_persistent_param_in_dit == -1:
            num_persistent_param_in_dit = None
    if vram_limit is not None and num_persistent_param_in_dit is not None:
        raise ValueError(
            "vram_limit and num_persistent_param_in_dit are mutually exclusive: "
            "upstream would silently ignore vram_limit when num_persistent_param_in_dit is set."
        )
    with _CACHE_LOCK:
        torch = _get_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("UniVidX requires CUDA; PyTorch cannot access a CUDA device.")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("UniVidX requires a CUDA device (for example 'cuda:0').")
        dtype = getattr(torch, compute_dtype)
        try:
            index = torch.cuda.current_device() if target.index is None else target.index
            device = f"cuda:{index}"
            total_vram = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
        except (RuntimeError, AssertionError) as exc:
            raise RuntimeError(f"UniVidX cannot access CUDA device {device}: {exc}") from exc
        if vram_buffer >= total_vram:
            raise ValueError(f"vram_buffer must be smaller than {device}'s {total_vram:.2f} GiB.")
        resolved = paths.resolve_models(
            variant, dit=dit, vae=vae, text_encoder=text_encoder,
            checkpoint=checkpoint, tokenizer=tokenizer,
        )
        distillation_source = paths.resolve_lightx2v() if distillation == "lightx2v" else None
        config = _vendor_config()
        key = (
            variant, resolved.dit.kind, resolved.dit.paths, resolved.vae,
            resolved.text_encoder, resolved.checkpoint, resolved.tokenizer,
            config, compute_dtype, device, vram_buffer,
            distillation, distillation_strength,
            distillation_source.path if distillation_source is not None else None,
            vram_limit, num_persistent_param_in_dit,
        )
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
        else:
            # Driver visibility alone is insufficient (e.g. a new GPU with an
            # older torch wheel). Fail before allocating the models in host RAM.
            try:
                torch.zeros(1, dtype=dtype, device=device).add_(1)
                torch.cuda.synchronize(device=device)
            except (RuntimeError, AssertionError) as exc:
                raise RuntimeError(f"UniVidX requires working CUDA kernels on {device}: {exc}") from exc
            # Free the previous handle first: two BF16 DiTs will not fit in host
            # RAM on most machines, and a changed vram_buffer alone gets here.
            _make_room(_cache_limit())
            model, reports = _build_model(
                variant, resolved, config, dtype=dtype, device=device, vram_buffer=vram_buffer,
                distillation_source=distillation_source, distillation_strength=distillation_strength,
                vram_limit=vram_limit, num_persistent_param_in_dit=num_persistent_param_in_dit,
            )
            _MODEL_CACHE[key] = ModelHandle(
                variant=variant, model=model, model_paths=resolved,
                compute_dtype=compute_dtype, device=device, vram_buffer=vram_buffer,
                load_reports=reports,
                distillation=distillation, distillation_strength=distillation_strength,
                distillation_path=distillation_source.path if distillation_source is not None else None,
                vram_limit=vram_limit, num_persistent_param_in_dit=num_persistent_param_in_dit,
            )
        return _MODEL_CACHE[key]


def evict_model(handle: ModelHandle | None = None) -> int:
    """Drop one handle (or all with None); ADVISORY, graph references keep it alive."""
    with _CACHE_LOCK:
        keys = [key for key, value in _MODEL_CACHE.items() if handle is None or value is handle]
        for key in keys:
            del _MODEL_CACHE[key]
        return len(keys)
