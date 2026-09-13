"""Validate conditioning and call upstream's pipeline without changing sampling.

TeaCache and cfg_merge are deliberately not exposed. WanVideoUnit_TeaCache is
absent from the pipeline's units list, so tea_cache is always None. cfg_merge
chunks a batch dimension holding the four modalities, not a merged positive/
negative pair. Both kwargs stay at upstream's inert defaults.

An empty prompt forces CFG to 1.0, matching upstream's inference script. The
result carries this decision, frame-fitting reports and the exact condition
tensors passed to the pipeline, so decoders never return unfitted source clips.

Optional context windows blend noise predictions at every denoising step. Each
window still sees only its own temporal span, improving local continuity and
seam-free blending without global awareness of a 191-frame shot. Anchor-frame
conditioning is a known further mitigation, deliberately deferred until measured.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import math
from typing import Any

if "." in __package__:
    from ..univid import attention as attn_kernels
    from ..univid import context, modes, rope, tensors
else:
    from univid import attention as attn_kernels
    from univid import context, modes, rope, tensors


_LOGGER = logging.getLogger(__name__)
_INPUT_ENCODINGS = ("display_referred_srgb", "linear_rec709_to_srgb")
_RGB_MODALITIES = frozenset({"rgb", "albedo", "irradiance", "fgr", "bgr"})
_MODALITIES = ("rgb", "albedo", "irradiance", "normal", "pha", "fgr", "bgr")
_PREVIEW_HINT = (
    "Use Gamut: Load EXR with tonemap_preview=True, then select display_referred_srgb."
)
_VAE_TILING_TOOLTIP = (
    "Controls VAE decode only. Conditioning encode stays untiled, so at higher "
    "resolutions the encoder can run out of memory regardless of decode tiling."
)
# Upstream's inference scripts hardcode this negative prompt, so it is the
# reference default rather than an empty string. It is unused whenever CFG is
# 1.0 — including every empty-prompt decomposition, where the negative pass is
# never evaluated — but it matters for prompted generation modes.
_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass(frozen=True)
class SampleResult:
    """Pipeline outputs and fitted [1,3,T,H,W] conditions, with visible decisions."""

    outputs: dict[str, Any]
    mode: modes.Mode
    conditions: dict[str, Any]
    frame_fits: dict[str, tensors.FrameFit]
    shape: tuple[int, int, int, int]
    input_encoding: str
    cfg_scale: float
    cfg_forced: bool
    messages: tuple[str, ...]
    context_window_count: int = 1
    attention_kernel: str = "sdpa"
    tiled_encode: bool = False
    shared_kv: bool = True
    rope_precision: str = "float64"
    context_blend: str = "triangular"


def _get_interrupt_check():
    """Resolve ComfyUI's cancellation hook lazily; standalone runs have no flag."""
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return lambda: None
    return throw_exception_if_processing_interrupted


@contextmanager
def _interruptible_model_fn(pipe):
    """Poll before each prediction and restore the cached pipeline on every exit."""
    original = pipe.model_fn
    check_interrupt = _get_interrupt_check()

    def model_fn(*args, **kwargs):
        check_interrupt()
        return original(*args, **kwargs)

    try:
        pipe.model_fn = model_fn
        yield
    finally:
        pipe.model_fn = original


class _ContextWindowDispatcher:
    """Window only latent time; keep modalities, timestep and other kwargs intact."""

    def __init__(self, model_fn, *, window_length: int, stride: int, blend: str = "triangular"):
        self.model_fn = model_fn
        self.window_length = window_length
        self.stride = stride
        self.blend = blend

    def __call__(self, *args, latents, **kwargs):
        if latents.ndim != 5:
            raise ValueError("Context windows require latents shaped [modalities,C,T,H,W].")
        windows = context.plan_windows(
            latents.shape[2], window_length=self.window_length, stride=self.stride,
        )
        check_interrupt = _get_interrupt_check()
        if len(windows) == 1:
            # No slicing, copies or arithmetic on the reference short-clip path.
            check_interrupt()
            return self.model_fn(*args, latents=latents, **kwargs)

        torch = tensors._get_torch()
        accumulated = weight_sum = None
        prediction_dtype = None
        for start, end in windows:
            check_interrupt()
            window = latents[:, :, start:end, :, :]
            # In conditioning modes timestep is a list of FOUR modality steps.
            # It remains in kwargs unchanged, as do all conditioning arguments.
            prediction = self.model_fn(*args, latents=window, **kwargs)
            if prediction.shape != window.shape:
                raise ValueError(
                    f"Context window [{start}, {end}) returned shape {tuple(prediction.shape)}; "
                    f"expected {tuple(window.shape)}."
                )
            if accumulated is None:
                prediction_dtype = prediction.dtype
                accumulation_dtype = (
                    torch.float32 if prediction_dtype in (torch.float16, torch.bfloat16)
                    else prediction_dtype
                )
                accumulated = prediction.new_zeros(latents.shape, dtype=accumulation_dtype)
                weight_sum = prediction.new_zeros(
                    (1, 1, latents.shape[2], 1, 1), dtype=accumulation_dtype,
                )
            weights = torch.tensor(
                context.blend_weights(end - start, self.blend),
                dtype=accumulated.dtype, device=accumulated.device,
            ).view(1, 1, end - start, 1, 1)
            accumulated[:, :, start:end, :, :] += prediction.to(dtype=accumulated.dtype) * weights
            weight_sum[:, :, start:end, :, :] += weights
        return accumulated.div_(weight_sum).to(dtype=prediction_dtype)


@contextmanager
def _context_model_fn(
    pipe, *, enabled: bool, window_length: int, stride: int, blend: str = "triangular",
):
    """Restore the cached pipeline's original dispatcher even after cancellation."""
    if not enabled:
        yield
        return
    original = pipe.model_fn
    try:
        pipe.model_fn = _ContextWindowDispatcher(
            original, window_length=window_length, stride=stride, blend=blend,
        )
        yield
    finally:
        pipe.model_fn = original


@contextmanager
def _tiled_vae_encode(pipe, *, enabled: bool, tile_size, tile_stride):
    """Forward the decode tiling into the conditioning encode, which upstream calls bare.

    The VAE's `encode` accepts the same tiling arguments as `decode` and in the
    same latent units, but every conditioning call in the pipeline omits them.
    The wrapper is an instance attribute for the duration of one call and is
    removed afterwards, so a cached pipeline never keeps a stale setting.
    """
    if not enabled:
        yield
        return
    vae = getattr(pipe, "vae", None)
    if vae is None:
        raise ValueError("tiled_encode requires a pipeline with a vae; this one has none.")
    original = vae.encode
    had_instance_attr = "encode" in vars(vae)

    def encode(*args, **kwargs):
        kwargs.setdefault("tiled", True)
        kwargs.setdefault("tile_size", tile_size)
        kwargs.setdefault("tile_stride", tile_stride)
        return original(*args, **kwargs)

    vae.encode = encode
    try:
        yield
    finally:
        if had_instance_attr:
            vae.encode = original
        else:
            del vae.encode


def _prepare_condition(image, *, modality: str, input_encoding: str):
    tensors.validate_image(image)
    torch = tensors._get_torch()
    image = image.to(dtype=torch.float32)
    if input_encoding == "linear_rec709_to_srgb" and modality in _RGB_MODALITIES:
        minimum, maximum = image.amin().item(), image.amax().item()
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum < 0 or maximum > 1:
            value = maximum if not math.isfinite(maximum) or maximum > 1 else minimum
            raise ValueError(
                f"UniVidX {modality!r} has value {value!r}; linear_rec709_to_srgb "
                f"requires scene-linear Rec.709 values in [0, 1]. {_PREVIEW_HINT}"
            )
        image = torch.where(
            image <= 0.0031308,
            12.92 * image,
            1.055 * image.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055,
        )
    return image


class UniVidXSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIVIDX_MODEL",),
                "task": ("UNIVIDX_TASK",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {
                    "default": _DEFAULT_NEGATIVE_PROMPT, "multiline": True,
                    "tooltip": "Upstream's reference negative prompt. Unused whenever CFG is 1.0, "
                    "which includes every empty-prompt decomposition.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1}),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "step": 0.1,
                    "tooltip": "An empty or whitespace-only prompt always forces CFG to 1.0.",
                }),
                "width": ("INT", {"default": 640, "min": 16, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "step": 16}),
                "num_frames": ("INT", {"default": 21, "min": 1, "step": 4}),
                "input_encoding": (list(_INPUT_ENCODINGS), {
                    "default": "display_referred_srgb",
                    "tooltip": "Training-side encoding is unspecified upstream. The default matches "
                    "its inference-time video convention. Linear input must be Rec.709 in [0,1]. "
                    "This setting affects rgb, albedo, irradiance, fgr and bgr only; never normal or pha.",
                }),
            },
            "optional": {
                **{
                    name: ("IMAGE", {"tooltip": "Used only when the selected mode requires this modality."})
                    for name in _MODALITIES
                },
                "tiled": ("BOOLEAN", {"default": True, "tooltip": _VAE_TILING_TOOLTIP}),
                "tile_size_height": ("INT", {
                    "default": 30, "min": 1,
                    "tooltip": "Tile height in latent pixels. " + _VAE_TILING_TOOLTIP,
                }),
                "tile_size_width": ("INT", {
                    "default": 52, "min": 1,
                    "tooltip": "Tile width in latent pixels. " + _VAE_TILING_TOOLTIP,
                }),
                "tile_stride_height": ("INT", {
                    "default": 15, "min": 1,
                    "tooltip": "Height stride in latent pixels; smaller than the tile height, so "
                    "tiles overlap and blend. " + _VAE_TILING_TOOLTIP,
                }),
                "tile_stride_width": ("INT", {
                    "default": 26, "min": 1,
                    "tooltip": "Width stride in latent pixels; smaller than the tile width, so "
                    "tiles overlap and blend. " + _VAE_TILING_TOOLTIP,
                }),
                "context_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Blend overlapping noise predictions each denoising step for long clips. "
                    "Improves local continuity; each window still sees only its own temporal span.",
                }),
                "context_window_frames": ("INT", {
                    "default": 21, "min": 1, "step": 4,
                    "tooltip": "Context window in frames. 21 matches the model's trained length; "
                    "going higher is off-distribution.",
                }),
                "context_stride_frames": ("INT", {
                    "default": 16, "min": 1, "step": 4,
                    "tooltip": "Stride in frames, no greater than the window. Default 16 gives a "
                    "2-latent overlap with a 21-frame window. Smaller strides mean "
                    "more overlap and better continuity, but more windows and more processing time.",
                }),
                "attention": (list(attn_kernels.KERNELS), {
                    "default": "sage",
                    "tooltip": "default sage (SageAttention 2.2.0 measured 6x sdpa per call); "
                    "set sdpa if the package is not installed",
                }),
                "tiled_encode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also tile the conditioning VAE encode, using the tile settings "
                    "above. Upstream encodes untiled, which is what runs out of memory first at "
                    "high resolution. Tiling blends tile borders, so leave it off unless the "
                    "untiled encode does not fit.",
                }),
                "shared_kv": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Identical attention math with K and V held once instead of once "
                    "per modality: about a third less activation VRAM at every resolution, and "
                    "the difference between 1088x1920 fitting and not. Turn off only to A/B "
                    "against upstream's exact tensor layout.",
                }),
                "rope_precision": (list(rope.PRECISIONS), {
                    "default": "float32",
                    "tooltip": "Default float32 RoPE: -0.4 s/window and -1.4 GiB with 0.49 deg "
                    "mean normal difference (kernel-noise class), measured on RTX 5090 at "
                    "704x1248. float64 keeps upstream numerics.",
                }),
                "context_blend": (list(context.BLEND_SHAPES), {
                    "default": "cosine",
                    "tooltip": "Default cosine halves the seam penalty of stride 16 at zero cost. "
                    "How overlapping windows are mixed: triangular = upstream-style "
                    "taper across the whole window; cosine = smoother; flat = average; "
                    "sharp = uniform weights with half-weight window endpoints. Pick per shot together "
                    "with context_stride_frames. Ignored when context_enabled is off.",
                }),
            },
        }

    RETURN_TYPES = ("UNIVIDX_RESULT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "sample"
    CATEGORY = "UniVidX/Sampling"
    DESCRIPTION = "Decompose or generate video passes; report frame fitting and empty-prompt CFG."

    def sample(
        self, model, task, prompt="", negative_prompt="", seed=0, steps=50, cfg_scale=5.0,
        width=640, height=480, num_frames=21, input_encoding="display_referred_srgb",
        rgb=None, albedo=None, irradiance=None, normal=None, pha=None, fgr=None, bgr=None,
        tiled=True, tile_size_height=30, tile_size_width=52,
        tile_stride_height=15, tile_stride_width=26,
        context_enabled=False, context_window_frames=21, context_stride_frames=16,
        attention="sage", tiled_encode=False, shared_kv=True, rope_precision="float32",
        context_blend="cosine",
    ):
        mode = modes.get_mode(task.name)
        if task.family != mode.family or model.variant != mode.family:
            raise ValueError(
                f"UniVidX mode {mode.name!r} requires family {mode.family!r}; "
                f"task family={task.family!r}, model family={model.variant!r}."
            )
        supplied = {
            "rgb": rgb, "albedo": albedo, "irradiance": irradiance,
            "normal": normal, "pha": pha, "fgr": fgr, "bgr": bgr,
        }
        modes.validate_inputs(mode.name, supplied)
        # Family and required-input errors precede even importing torch or
        # accessing the pipeline. Extra images are never validated or converted.
        if input_encoding not in _INPUT_ENCODINGS:
            raise ValueError(f"Unknown UniVidX input encoding: {input_encoding!r}")
        for name, value in (
            ("width", width), ("height", height), ("num_frames", num_frames), ("steps", steps),
            ("tile_size_height", tile_size_height), ("tile_size_width", tile_size_width),
            ("tile_stride_height", tile_stride_height), ("tile_stride_width", tile_stride_width),
            ("context_window_frames", context_window_frames), ("context_stride_frames", context_stride_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not isinstance(tiled, bool):
            raise ValueError("tiled must be a boolean.")
        if tile_stride_height > tile_size_height:
            raise ValueError("tile_stride_height must not exceed tile_size_height.")
        if tile_stride_width > tile_size_width:
            raise ValueError("tile_stride_width must not exceed tile_size_width.")
        if not isinstance(context_enabled, bool):
            raise ValueError("context_enabled must be a boolean.")
        if context_stride_frames > context_window_frames:
            raise ValueError("context_stride_frames must not exceed context_window_frames.")
        if context_blend not in context.BLEND_SHAPES:
            raise ValueError(
                f"Unknown context blend: {context_blend!r} "
                f"(choose from {', '.join(context.BLEND_SHAPES)})."
            )
        if not math.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative.")
        if attention not in attn_kernels.KERNELS:
            raise ValueError(
                f"Unknown attention kernel: {attention!r} "
                f"(choose from {', '.join(attn_kernels.KERNELS)})."
            )
        if not isinstance(tiled_encode, bool):
            raise ValueError("tiled_encode must be a boolean.")
        if not isinstance(shared_kv, bool):
            raise ValueError("shared_kv must be a boolean.")
        if rope_precision not in rope.PRECISIONS:
            raise ValueError(
                f"Unknown RoPE precision: {rope_precision!r} "
                f"(choose from {', '.join(rope.PRECISIONS)})."
            )
        # Upstream blends tiles over a border of (size - stride) latent pixels and
        # builds that mask with `x[-border:]`; at zero overlap that slice is the
        # whole row and the assignment of an empty tensor raises. On the decode
        # path that would surface only after denoising has finished.
        if (tiled or tiled_encode) and (
            tile_stride_height >= tile_size_height or tile_stride_width >= tile_size_width
        ):
            raise ValueError(
                "Tiling needs a positive overlap: each tile stride must be smaller than its "
                f"tile size, got {tile_size_height}x{tile_size_width} tiles with "
                f"{tile_stride_height}x{tile_stride_width} strides."
            )
        # Refuse before any tensor work: a missing optional kernel must never
        # degrade silently into the slow path the artist asked to leave.
        if attention == "sage" and not attn_kernels.sage_available():
            raise ValueError(
                "attention=sage needs the sageattention package in ComfyUI's Python, and it "
                "is not importable there. Install it, or set attention back to sdpa."
            )
        messages = []
        if attention == "sage":
            messages.append(
                "Attention kernel: sage (SageAttention, INT8-quantised QK) — an approximation; "
                "A/B against sdpa before trusting the passes."
            )
        if tiled_encode:
            messages.append(
                f"Conditioning VAE encode: tiled ({tile_size_height}x{tile_size_width} latent "
                f"tiles, stride {tile_stride_height}x{tile_stride_width})."
            )
        context_window_count = 1
        latent_window = (context_window_frames - 1) // 4 + 1
        latent_stride = (context_stride_frames - 1) // 4 + 1
        cfg_forced = not prompt.strip()
        if cfg_forced:
            messages.append(f"Empty prompt: CFG forced to 1.0 (requested {cfg_scale}).")
            cfg_scale = 1.0
            prompt = ""

        torch = tensors._get_torch()
        with torch.no_grad():
            prepared = {
                modality: _prepare_condition(
                    supplied[modality], modality=modality, input_encoding=input_encoding,
                )
                for modality in mode.condition_kwargs
            }
            pipe = model.pipe
            requested_shape = (num_frames, height, width, 3)
            height, width, num_frames = pipe.check_resize_height_width(
                height, width, num_frames=num_frames,
            )
            shape = (num_frames, height, width, 3)
            if shape != requested_shape:
                messages.append(f"Upstream shape fitting: {requested_shape} -> {shape}.")
            if context_enabled:
                latent_length = (num_frames - 1) // 4 + 1
                context_window_count = len(context.plan_windows(
                    latent_length, window_length=latent_window, stride=latent_stride,
                ))
                messages.append(
                    f"Context windows: {context_window_count} per noise prediction "
                    f"({latent_length} latent frames; window {latent_window}, stride {latent_stride}, "
                    f"blend {context_blend})."
                )
            conditions, frame_fits, condition_kwargs = {}, {}, {}
            for modality, keyword in mode.condition_kwargs.items():
                video, fit = tensors.image_to_video(
                    prepared[modality], height=height, width=width, num_frames=num_frames,
                )
                conditions[modality] = video
                frame_fits[modality] = fit
                condition_kwargs[keyword] = video
                if fit.action != "unchanged":
                    messages.append(
                        f"{modality}: {fit.action}, {fit.source_frames} -> {fit.target_frames} frames."
                    )
            for message in messages:
                _LOGGER.info("[ComfyUI-UniVidX] %s", message)
            # The interrupt wrapper surrounds the optional dispatcher; unwinding
            # restores the dispatcher first, then the original model_fn.
            with _context_model_fn(
                pipe, enabled=context_enabled, window_length=latent_window, stride=latent_stride,
                blend=context_blend,
            ), _interruptible_model_fn(pipe), _tiled_vae_encode(
                pipe, enabled=tiled_encode,
                tile_size=(tile_size_height, tile_size_width),
                tile_stride=(tile_stride_height, tile_stride_width),
            ), attn_kernels.patched_attention(
                attn_kernels.dit_modules(), attention,
            ) as kernels, attn_kernels.patched_shared_kv(
                attn_kernels.shared_kv_modules(), shared_kv,
            ) as kv_wrappers, rope.patched_rope(
                rope.dit_modules(), rope_precision,
            ) as rope_wrappers:
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    torch.cuda.reset_peak_memory_stats()
                outputs = pipe(
                    # Upstream's inference scripts pass the positive prompt as FOUR
                    # copies — one per modality latent in the batch — while leaving
                    # the negative a single string (intrinsic:178, alpha:175, and
                    # base_prompter.process_prompt recurses over lists explicitly).
                    # A bare string here would embed at batch 1 against batch-4
                    # latents, so this asymmetry is reproduced rather than tidied.
                    training_mode=mode.name, is_inference=True, prompt=[prompt] * 4,
                    negative_prompt=negative_prompt, seed=seed, rand_device="cpu",
                    num_inference_steps=steps, cfg_scale=cfg_scale,
                    tiled=tiled, tile_size=(tile_size_height, tile_size_width),
                    tile_stride=(tile_stride_height, tile_stride_width),
                    height=height, width=width, num_frames=num_frames, **condition_kwargs,
                )
        if not isinstance(outputs, dict):
            raise TypeError(f"UniVidX mode {mode.name!r} returned no output dictionary.")
        # Reported after the run, from counts: a fallback inside the kernel must
        # never pass for the speedup that was requested.
        report_lines = (
            attn_kernels.engagement_report(kernels)
            + attn_kernels.shared_kv_report(kv_wrappers)
            + rope.rope_report(rope_wrappers)
        )
        if cuda_available:
            report_lines.append(
                f"Peak VRAM during sampling: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                f"(allocated), {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB (reserved)."
            )
        for line in report_lines:
            _LOGGER.info("[ComfyUI-UniVidX] %s", line)
            messages.append(line)
        return (SampleResult(
            outputs=outputs, mode=mode, conditions=conditions, frame_fits=frame_fits,
            shape=shape, input_encoding=input_encoding, cfg_scale=cfg_scale,
            cfg_forced=cfg_forced, messages=tuple(messages),
            context_window_count=context_window_count,
            attention_kernel=attention, tiled_encode=tiled_encode, shared_kv=shared_kv,
            rope_precision=rope_precision,
            context_blend=context_blend,
        ),)
