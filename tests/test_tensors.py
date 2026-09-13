"""IMAGE <-> video conversion: the range map, frame fitting, and what is reported.

Frame fitting is the part worth guarding. Silently truncating a 100-frame plate
to 21, or padding a short one by repeating its last frame, changes what the user
gets without changing anything they can see — so the conversion returns what it
did and the sampler surfaces it.
"""

import pytest

from univid.tensors import FrameFit, image_to_video, video_to_image


torch = pytest.importorskip("torch")


def _image(frames=3, height=4, width=5):
    count = frames * height * width * 3
    return torch.linspace(0.0, 1.0, count).reshape(frames, height, width, 3)


def test_range_map_and_layout():
    """[T,H,W,3] in [0,1] becomes [1,3,T,H,W] in [-1,1]."""
    video, _ = image_to_video(_image(), height=4, width=5, num_frames=3)
    assert tuple(video.shape) == (1, 3, 3, 4, 5)
    assert video.dtype is torch.float32
    assert pytest.approx(float(video.min()), abs=1e-6) == -1.0
    assert pytest.approx(float(video.max()), abs=1e-6) == 1.0


def test_round_trip_recovers_the_image():
    image = _image()
    video, _ = image_to_video(image, height=4, width=5, num_frames=3)
    assert torch.allclose(video_to_image(video[0]), image, rtol=0, atol=1e-6)


def test_return_path_is_float32_with_no_eight_bit_step():
    """A value between two 8-bit codes must survive; quantisation would lose it."""
    video = torch.full((3, 1, 1, 1), 2.0 / 255.0 - 1.0 + 1e-4)
    image = video_to_image(video)
    assert image.dtype is torch.float32
    assert float(image.flatten()[0] * 255.0) % 1.0 != 0.0


def test_return_path_does_not_clip():
    """Out-of-range model output is surfaced, not hidden behind a clamp."""
    video = torch.full((3, 1, 1, 1), 3.0)
    assert float(video_to_image(video).max()) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "source,target,action",
    [(3, 3, "unchanged"), (9, 3, "truncate"), (2, 5, "repeat_last_frame")],
)
def test_frame_fitting_reports_what_it_did(source, target, action):
    video, fit = image_to_video(_image(frames=source), height=4, width=5, num_frames=target)
    assert fit == FrameFit(action=action, source_frames=source, target_frames=target)
    assert video.shape[2] == target


def test_padding_repeats_the_last_frame_not_black():
    """Padding with black would inject content the plate never had."""
    image = _image(frames=2)
    video, fit = image_to_video(image, height=4, width=5, num_frames=4)
    assert fit.action == "repeat_last_frame"
    for index in (2, 3):
        assert torch.equal(video[0, :, index], video[0, :, 1])


def test_truncation_keeps_the_leading_frames():
    image = _image(frames=6)
    video, _ = image_to_video(image, height=4, width=5, num_frames=2)
    reference, _ = image_to_video(image[:2], height=4, width=5, num_frames=2)
    assert torch.equal(video, reference)


def test_resize_changes_spatial_dimensions_only():
    video, fit = image_to_video(_image(frames=3, height=8, width=9), height=4, width=5, num_frames=3)
    assert tuple(video.shape) == (1, 3, 3, 4, 5)
    assert fit.action == "unchanged"


@pytest.mark.parametrize(
    "bad", [torch.zeros(2, 3, 4), torch.zeros(2, 3, 4, 4), torch.zeros(0, 3, 4, 3)]
)
def test_bad_image_shapes_are_refused(bad):
    with pytest.raises(ValueError, match=r"\[T,H,W,3\]"):
        image_to_video(bad, height=2, width=2, num_frames=1)


def test_integer_images_are_refused():
    with pytest.raises(TypeError, match="floating-point"):
        image_to_video(torch.zeros(1, 2, 2, 3, dtype=torch.uint8), height=2, width=2, num_frames=1)


@pytest.mark.parametrize("kwargs", [{"height": 0}, {"width": -1}, {"num_frames": 0}])
def test_non_positive_targets_are_refused(kwargs):
    call = {"height": 2, "width": 2, "num_frames": 1, **kwargs}
    with pytest.raises(ValueError, match="must be a positive integer"):
        image_to_video(_image(frames=1, height=2, width=2), **call)


@pytest.mark.parametrize("bad", [torch.zeros(4, 2, 2, 2), torch.zeros(3, 2, 2)])
def test_bad_video_shapes_are_refused(bad):
    with pytest.raises(ValueError, match=r"\[3,T,H,W\]"):
        video_to_image(bad)
