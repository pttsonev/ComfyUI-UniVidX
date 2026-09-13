"""Mode metadata, checked against the vendored upstream rather than against itself.

`univid/modes.py` restates facts that live in upstream's pipelines: which mode
names exist, and which keys each one's decode branch writes. A test that only
asserted the module is self-consistent would pass forever while drifting away
from the model it describes.

So the first two tests parse `vendor/univid/src/pipelines/univid_*.py` and
compare. That makes them the guard `vendor/univid/VENDOR.md` promises: refresh
the snapshot to a newer upstream commit and these fail loudly if the mode set
or any decode branch moved, instead of the pack silently mislabelling outputs.
"""

import re
from pathlib import Path

import pytest

from univid.modes import (
    ALPHA_MODES,
    CONDITION_KWARGS,
    INTRINSIC_MODES,
    MODALITIES,
    MODES,
    get_mode,
    validate_inputs,
)


PIPELINES = Path(__file__).resolve().parent.parent / "vendor" / "univid" / "src" / "pipelines"
FAMILIES = [
    ("intrinsic", "univid_intrinsic.py", INTRINSIC_MODES, "RAIN"),
    ("alpha", "univid_alpha.py", ALPHA_MODES, "RPFB"),
]


def _source(filename):
    return (PIPELINES / filename).read_text(encoding="utf-8")


def _upstream_mode_names(src):
    return set(re.findall(r'training_mode\s*==\s*"([A-Za-z0-9]+)"', src))


def _upstream_decode_keys(src):
    """Map mode name -> the video_dict keys its decode branch writes, in order.

    Only the first `training_mode` chain after ``video_dict = {}`` is the decode
    chain; a later chain in the same file repeats the names without writing any
    keys, so the first non-empty capture per name wins.
    """
    decode = src[src.rindex("video_dict = {}"):]
    parts = re.split(r'(?:el)?if training_mode == "([A-Za-z0-9]+)":', decode)
    keys = {}
    for index in range(1, len(parts) - 1, 2):
        found = tuple(re.findall(r'video_dict\["([a-z_]+)"\]', parts[index + 1]))
        if found:
            keys.setdefault(parts[index], found)
    return keys


@pytest.mark.parametrize("family,filename,names,letters", FAMILIES)
def test_mode_names_match_vendored_upstream(family, filename, names, letters):
    """Our name tuple must equal upstream's branch names exactly — no extras."""
    assert set(names) == _upstream_mode_names(_source(filename))
    assert len(names) == 15
    assert len(set(names)) == len(names), "duplicate mode name"


@pytest.mark.parametrize("family,filename,names,letters", FAMILIES)
def test_result_keys_match_vendored_decode_branches(family, filename, names, letters):
    """Every mode's result keys must match its upstream decode branch, in order.

    Order matters: the decoder node splays these positionally, so a reordering
    upstream would silently swap two passes rather than fail.
    """
    upstream = _upstream_decode_keys(_source(filename))
    assert set(upstream) == set(names), "a decode branch went unparsed"
    for name in names:
        assert MODES[name].result_keys == upstream[name], name


def test_normal_result_key_is_normal_unit_not_normal():
    """Upstream decodes the normal through a unit-vector path under its own key.

    Getting this wrong yields a KeyError at decode time in every intrinsic mode
    that produces a normal, so it is pinned explicitly rather than relying on
    the generated table.
    """
    assert MODES["R2AIN"].result_keys == ("albedo", "irradiance", "normal_unit")
    assert "normal" not in MODES["t2RAIN"].result_keys
    assert MODES["N2RAI"].condition_kwargs == {"normal": "inference_normal"}


@pytest.mark.parametrize("family,filename,names,letters", FAMILIES)
def test_condition_kwargs_exist_upstream(family, filename, names, letters):
    """Each condition keyword we would pass must be a name upstream reads."""
    src = _source(filename)
    for letter in letters:
        assert CONDITION_KWARGS[letter] in src, CONDITION_KWARGS[letter]


@pytest.mark.parametrize("family,filename,names,letters", FAMILIES)
def test_conditions_and_results_partition_the_family(family, filename, names, letters):
    """Conditions and results are complementary: every letter is used once."""
    everything = {MODALITIES[letter] for letter in letters}
    for name in names:
        mode = MODES[name]
        results = {key.replace("normal_unit", "normal") for key in mode.result_keys}
        assert mode.conditions.isdisjoint(results), name
        assert mode.conditions | results == everything, name
        assert mode.family == family


def test_text_only_modes_require_no_conditions():
    for name in ("t2RAIN", "t2RPFB"):
        assert MODES[name].conditions == frozenset()
        assert validate_inputs(name, {}).name == name


def test_validate_inputs_names_the_missing_modality():
    with pytest.raises(ValueError, match="albedo"):
        validate_inputs("RA2IN", {"rgb": object()})


def test_validate_inputs_treats_explicit_none_as_missing():
    """A disconnected optional ComfyUI input arrives as None, not as absent."""
    with pytest.raises(ValueError, match="albedo"):
        validate_inputs("RA2IN", {"rgb": object(), "albedo": None})


def test_validate_inputs_accepts_extra_inputs():
    """The sampler offers every modality socket; unused ones are not an error."""
    mode = validate_inputs("R2AIN", {"rgb": object(), "pha": object()})
    assert mode.name == "R2AIN"


def test_validate_inputs_accepts_falsy_values():
    """Emptiness is not the test — only None means 'not supplied'.

    A zero-filled tensor is falsy under some conventions; treating it as
    missing would reject a legitimate black plate.
    """
    validate_inputs("R2AIN", {"rgb": 0})


def test_get_mode_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown UniVidX mode"):
        get_mode("R2XYZ")


def test_mode_is_immutable():
    """Modes are shared module state; a caller must not be able to edit one."""
    with pytest.raises(Exception):
        MODES["R2AIN"].name = "changed"  # type: ignore[misc]


def test_module_imports_without_torch_or_comfyui():
    """The pure layer must stay importable with nothing installed.

    Asserted on the source text, because torch happens to be installed in most
    environments and an import-based check would pass either way.
    """
    source = (Path(__file__).resolve().parent.parent / "univid" / "modes.py").read_text(
        encoding="utf-8"
    )
    for banned in ("import torch", "import numpy", "folder_paths", "from ..vendor", "import comfy"):
        assert banned not in source, banned
