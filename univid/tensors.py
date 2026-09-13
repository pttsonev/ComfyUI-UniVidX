"""Float IMAGE/video conversion with explicit frame fitting and no ComfyUI imports.

The input path follows upstream's inference convention: bilinear resize with
align_corners=False, truncate or repeat the last frame, then *2-1. The return
path is *0.5+0.5 in float32, without clipping or an intermediate 8-bit image.
Torch is imported only when a conversion is requested.
"""

from dataclasses import dataclass
from typing import Literal


_UNSET = object()
_torch = _UNSET
_torch_error = None


@dataclass(frozen=True)
class FrameFit:
    """How a source clip was fitted to the requested frame count."""

    action: Literal["unchanged", "truncate", "repeat_last_frame"]
    source_frames: int
    target_frames: int


def _get_torch():
    """Cache both import success and unavailability; None is never retried."""
    global _torch, _torch_error
    if _torch is _UNSET:
        try:
            import torch as module
        except (ImportError, OSError) as exc:
            module = None
            _torch_error = exc
        _torch = module
    if _torch is None:
        raise RuntimeError("UniVidX tensor conversion requires PyTorch.") from _torch_error
    return _torch


def validate_image(image) -> None:
    """Require a non-empty floating IMAGE batch with exactly three channels."""
    torch = _get_torch()
    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise TypeError("IMAGE must be a floating-point torch tensor.")
    if image.ndim != 4 or image.shape[-1] != 3 or any(size == 0 for size in image.shape):
        raise ValueError(f"IMAGE must have non-empty shape [T,H,W,3], got {tuple(image.shape)}.")


def image_to_video(image, *, height: int, width: int, num_frames: int):
    """Convert [T,H,W,3] IMAGE codes to float32 [1,3,T,H,W], returning FrameFit."""
    validate_image(image)
    for name, value in (("height", height), ("width", width), ("num_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    torch = _get_torch()
    frames = torch.nn.functional.interpolate(
        image.to(dtype=torch.float32).permute(0, 3, 1, 2),
        size=(height, width), mode="bilinear", align_corners=False,
    )
    source_frames = frames.shape[0]
    if source_frames > num_frames:
        frames = frames[:num_frames]
        action = "truncate"
    elif source_frames < num_frames:
        last = frames[-1:].expand(num_frames - source_frames, -1, -1, -1)
        frames = torch.cat((frames, last), dim=0)
        action = "repeat_last_frame"
    else:
        action = "unchanged"
    video = (frames * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return video, FrameFit(action=action, source_frames=source_frames, target_frames=num_frames)


def video_to_image(video):
    """Convert upstream [3,T,H,W] to float32 [T,H,W,3] IMAGE codes on CPU."""
    torch = _get_torch()
    if not isinstance(video, torch.Tensor) or not video.is_floating_point():
        raise TypeError("Video must be a floating-point torch tensor.")
    if video.ndim != 4 or video.shape[0] != 3 or any(size == 0 for size in video.shape):
        raise ValueError(f"Video must have non-empty shape [3,T,H,W], got {tuple(video.shape)}.")
    image = video.to(device="cpu", dtype=torch.float32).permute(1, 2, 3, 0)
    return (image * 0.5 + 0.5).contiguous()
