"""Pure temporal window planning and positive blend weights.

Lengths and half-open spans are in latent time steps, never modality slots.
The tensor dispatcher normalises the weighted noise predictions at each step.
Each window still sees only its own temporal span: this improves local continuity
and seam-free blending, but gives no global awareness of a 191-frame shot.
Anchor-frame conditioning is a known further mitigation and is deliberately not
attempted without a measurement. This module has no runtime dependencies.
"""

import math


BLEND_SHAPES = ("triangular", "cosine", "flat", "sharp")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def plan_windows(total_length: int, *, window_length: int, stride: int) -> list[tuple[int, int]]:
    """Cover every latent index, shifting the last full window back to the end."""
    for name, value in (
        ("total_length", total_length), ("window_length", window_length), ("stride", stride),
    ):
        _positive_integer(value, name)
    if stride > window_length:
        raise ValueError("stride must not exceed window_length; larger strides leave gaps.")
    if total_length <= window_length:
        return [(0, total_length)]
    last_start = total_length - window_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window_length) for start in starts]


def blend_weights(length: int, shape: str = "triangular") -> list[float]:
    """Return one window's symmetric taper, with triangular as the reference.

    Cosine uses the interior of a Hann window: h[i] = 0.5 - 0.5 *
    cos(2*pi*(i+1)/(length+1)), i=0..length-1, then w[i] = h[i]/max(h).
    Mirrored indices give exact symmetry and normalisation makes the peak 1.0
    for odd and even lengths. Flat is all ones. Sharp is all ones except the
    first and last weights at 0.5 (lengths 1 and 2 have no interior peak).

    Endpoints stay positive so the ends of the shot and non-overlapping windows
    never have zero coverage. These are raw weights: divide accumulated weighted
    predictions by the summed weights to normalise contributions to 1 everywhere.
    """
    _positive_integer(length, "length")
    if shape not in BLEND_SHAPES:
        raise ValueError(
            f"Unknown blend shape: {shape!r} (choose from {', '.join(BLEND_SHAPES)})."
        )
    if shape == "flat":
        return [1.0] * length
    if shape == "cosine":
        weights = [
            0.5 - 0.5 * math.cos(2 * math.pi * min(index + 1, length - index) / (length + 1))
            for index in range(length)
        ]
        peak = max(weights)
        return [weight / peak for weight in weights]
    if shape == "sharp":
        weights = [1.0] * length
        weights[0] = weights[-1] = 0.5
        return weights
    peak = (length + 1) // 2
    return [min(index + 1, length - index) / peak for index in range(length)]
