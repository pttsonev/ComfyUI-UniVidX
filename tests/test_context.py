"""Context-window planning and the noise-prediction dispatcher.

Two properties carry the whole feature. Coverage: every latent index must be
inside at least one window and must end up with non-zero blend weight, or that
part of the shot is reconstructed from nothing. Passthrough: everything except
latent time must reach the model untouched — in conditioning modes `timestep`
is a list of four modality steps, and the four modalities on dim 0 are coupled
by cross-modal attention, so slicing either one breaks the model silently.

The short-clip path is asserted to be exactly the old path: same object, no
slicing, no arithmetic. A regression there would change every 21-frame result.
"""

import pytest

from univid.context import BLEND_SHAPES, blend_weights, plan_windows


torch = pytest.importorskip("torch")

from nodes.sampler import _ContextWindowDispatcher, _context_model_fn


# --- planning -----------------------------------------------------------------


@pytest.mark.parametrize(
    "total,window,stride",
    [(48, 6, 3), (49, 6, 3), (48, 6, 6), (6, 6, 3), (7, 6, 3), (100, 21, 5), (48, 6, 1)],
)
def test_every_index_is_covered(total, window, stride):
    covered = set()
    for start, end in plan_windows(total, window_length=window, stride=stride):
        covered.update(range(start, end))
    assert covered == set(range(total))


@pytest.mark.parametrize("total,window,stride", [(48, 6, 3), (49, 6, 3), (50, 21, 8)])
def test_no_window_runs_past_the_end(total, window, stride):
    windows = plan_windows(total, window_length=window, stride=stride)
    assert all(end <= total for _, end in windows)
    assert windows[-1][1] == total, "the last window must reach the end"


def test_short_total_is_one_window():
    """A clip at or under the trained length must not be windowed at all."""
    assert plan_windows(6, window_length=6, stride=3) == [(0, 6)]
    assert plan_windows(3, window_length=6, stride=3) == [(0, 3)]


def test_stride_larger_than_window_is_refused():
    """It would leave uncovered gaps — silent holes in the shot."""
    with pytest.raises(ValueError, match="leave gaps"):
        plan_windows(48, window_length=6, stride=7)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_non_positive_arguments_are_refused(bad):
    with pytest.raises(ValueError, match="positive integer"):
        plan_windows(bad, window_length=6, stride=3)


def test_the_users_case_is_fifteen_windows():
    """189 frames -> 48 latent frames at the trained window length."""
    assert len(plan_windows(48, window_length=6, stride=3)) == 15


# --- blend weights ------------------------------------------------------------


@pytest.mark.parametrize("shape", BLEND_SHAPES)
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 7, 8, 21])
def test_weights_are_symmetric_positive_and_peak_normalised(length, shape):
    """A zero at a window edge would leave that frame with no contribution."""
    weights = blend_weights(length, shape)
    assert len(weights) == length
    assert weights == weights[::-1]
    assert all(weight > 0 for weight in weights)
    # Sharp's first/last rule leaves no interior peak in these tiny windows.
    assert max(weights) == (0.5 if shape == "sharp" and length <= 2 else 1.0)


@pytest.mark.parametrize("shape", ["triangular", "cosine", "sharp"])
@pytest.mark.parametrize("length", [5, 6, 21])
def test_weights_peak_in_the_middle(length, shape):
    weights = blend_weights(length, shape)
    assert max(weights) == pytest.approx(1.0)
    assert weights[length // 2] == 1.0
    assert weights[0] < max(weights) and weights[-1] < max(weights)


@pytest.mark.parametrize("length", range(1, 9))
def test_triangular_is_exactly_the_previous_formula(length):
    expected = [min(index + 1, length - index) / ((length + 1) // 2) for index in range(length)]
    assert blend_weights(length) == expected
    assert blend_weights(length, "triangular") == expected


@pytest.mark.parametrize("shape,expected", [
    ("cosine", [0.25, 0.75, 1.0, 0.75, 0.25]),
    ("flat", [1.0, 1.0, 1.0, 1.0, 1.0]),
    ("sharp", [0.5, 1.0, 1.0, 1.0, 0.5]),
])
def test_blend_profiles(shape, expected):
    assert blend_weights(5, shape) == pytest.approx(expected)


def test_unknown_blend_shape_names_the_choices():
    with pytest.raises(ValueError, match="Unknown blend shape") as exc:
        blend_weights(6, "unknown")
    assert "unknown" in str(exc.value)
    assert ", ".join(BLEND_SHAPES) in str(exc.value)


@pytest.mark.parametrize("shape", BLEND_SHAPES)
@pytest.mark.parametrize("total,window,stride", [(48, 6, 3), (49, 6, 3), (100, 21, 7), (48, 6, 6)])
def test_summed_coverage_is_positive_everywhere(total, window, stride, shape):
    """Normalisation divides by this sum, so a zero would be a divide-by-zero."""
    sums = [0.0] * total
    for start, end in plan_windows(total, window_length=window, stride=stride):
        for offset, weight in enumerate(blend_weights(end - start, shape)):
            sums[start + offset] += weight
    assert all(value > 0 for value in sums)


# --- the dispatcher -----------------------------------------------------------


def _latents(modalities=4, channels=2, frames=48, height=3, width=3):
    return torch.arange(
        modalities * channels * frames * height * width, dtype=torch.float32
    ).reshape(modalities, channels, frames, height, width)


@pytest.mark.parametrize("shape", BLEND_SHAPES)
def test_single_window_takes_the_untouched_reference_path(shape):
    """At or under the window length nothing is sliced, copied or blended."""
    seen = {}

    def model_fn(**kwargs):
        seen["latents"] = kwargs["latents"]
        seen["prediction"] = kwargs["latents"] * 2
        return seen["prediction"]

    latents = _latents(frames=6)
    out = _ContextWindowDispatcher(model_fn, window_length=6, stride=3, blend=shape)(latents=latents)
    assert seen["latents"] is latents, "the reference path must pass the same object"
    assert out is seen["prediction"], "the reference prediction must bypass blend arithmetic"
    assert torch.equal(out, latents * 2)


@pytest.mark.parametrize("shape,overlap", [
    ("triangular", [12.5, 15.0, 17.5]),
    ("cosine", [12.0, 15.0, 18.0]),
    ("flat", [15.0, 15.0, 15.0]),
    ("sharp", [40.0 / 3, 15.0, 50.0 / 3]),
])
def test_dispatcher_mixes_different_predictions_with_the_selected_blend(shape, overlap):
    predictions = iter([10.0, 20.0])

    def model_fn(**kwargs):
        return torch.full_like(kwargs["latents"], next(predictions))

    latents = _latents(frames=7)
    out = _ContextWindowDispatcher(model_fn, window_length=5, stride=2, blend=shape)(latents=latents)
    expected = torch.tensor([10.0, 10.0, *overlap, 20.0, 20.0]).view(1, 1, 7, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), rtol=0, atol=2e-6)


def test_windows_slice_time_and_never_modalities():
    calls = []

    def model_fn(**kwargs):
        calls.append(kwargs["latents"].shape)
        return torch.zeros_like(kwargs["latents"])

    latents = _latents(frames=48)
    _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(latents=latents)
    assert len(calls) == 15
    for shape in calls:
        assert shape[0] == 4, "all four modalities must stay in every window"
        assert shape[1] == 2 and shape[3] == 3 and shape[4] == 3
        assert shape[2] == 6, "only latent time may be sliced"


def test_timestep_list_of_four_passes_through_untouched():
    """Conditioning modes pass 999 for frozen slots and the live step for the target."""
    timestep = [torch.tensor(999.0), torch.tensor(999.0), torch.tensor(999.0), torch.tensor(7.0)]
    seen = []

    def model_fn(**kwargs):
        seen.append(kwargs["timestep"])
        return torch.zeros_like(kwargs["latents"])

    _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(
        latents=_latents(frames=48), timestep=timestep,
    )
    assert all(step is timestep for step in seen), "timestep must not be sliced or rebuilt"


def test_other_kwargs_reach_every_window_unchanged():
    marker = object()
    seen = []

    def model_fn(**kwargs):
        seen.append(kwargs["context"])
        return torch.zeros_like(kwargs["latents"])

    _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(
        latents=_latents(frames=48), context=marker,
    )
    assert seen and all(item is marker for item in seen)


def test_blended_output_keeps_the_full_shape():
    def model_fn(**kwargs):
        return torch.ones_like(kwargs["latents"])

    latents = _latents(frames=48)
    out = _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(latents=latents)
    assert out.shape == latents.shape


def test_constant_prediction_survives_blending_exactly():
    """Normalisation must be exact: a constant field in gives that constant out.

    This is the strongest available check without a real model — any weighting
    or normalisation error shows up immediately as a non-constant result.
    """
    def model_fn(**kwargs):
        return torch.full_like(kwargs["latents"], 3.0)

    out = _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(latents=_latents(frames=48))
    assert torch.allclose(out, torch.full_like(out, 3.0), rtol=0, atol=1e-5)


def test_identity_prediction_is_reconstructed():
    """Returning the window unchanged must reassemble the original latents."""
    def model_fn(**kwargs):
        return kwargs["latents"]

    latents = _latents(frames=48)
    out = _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(latents=latents)
    assert torch.allclose(out, latents, rtol=0, atol=1e-3)


def test_low_precision_predictions_accumulate_in_float32():
    """Summing 15 bf16 windows at low precision would lose the blend."""
    def model_fn(**kwargs):
        return torch.full_like(kwargs["latents"], 3.0, dtype=torch.bfloat16)

    latents = _latents(frames=48).to(torch.bfloat16)
    out = _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(latents=latents)
    assert out.dtype is torch.bfloat16, "the result must return in the model's dtype"
    assert torch.allclose(out.float(), torch.full_like(out.float(), 3.0), rtol=0, atol=1e-2)


def test_wrong_prediction_shape_is_refused():
    def model_fn(**kwargs):
        return torch.zeros_like(kwargs["latents"])[:, :, :-1]

    with pytest.raises(ValueError, match="expected"):
        _ContextWindowDispatcher(model_fn, window_length=6, stride=3)(latents=_latents(frames=48))


def test_non_five_dimensional_latents_are_refused():
    with pytest.raises(ValueError, match=r"\[modalities,C,T,H,W\]"):
        _ContextWindowDispatcher(lambda **k: None, window_length=6, stride=3)(
            latents=torch.zeros(4, 2, 48),
        )


# --- the pipeline is cached, so restoration matters ---------------------------


class _Pipe:
    def __init__(self):
        self.model_fn = "original"


def test_dispatcher_is_removed_after_use():
    pipe = _Pipe()
    with _context_model_fn(pipe, enabled=True, window_length=6, stride=3):
        assert pipe.model_fn != "original"
    assert pipe.model_fn == "original"


def test_dispatcher_is_removed_even_on_failure():
    """The pipeline is cached — a leaked wrapper would poison every later run."""
    pipe = _Pipe()
    with pytest.raises(RuntimeError):
        with _context_model_fn(pipe, enabled=True, window_length=6, stride=3):
            raise RuntimeError("cancelled mid-sample")
    assert pipe.model_fn == "original"


def test_disabled_leaves_the_pipeline_alone():
    pipe = _Pipe()
    with _context_model_fn(pipe, enabled=False, window_length=6, stride=3):
        assert pipe.model_fn == "original"
    assert pipe.model_fn == "original"
