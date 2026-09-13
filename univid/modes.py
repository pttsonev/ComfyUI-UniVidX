"""Mode metadata from the pinned UniVidX intrinsic and alpha pipeline branches.

Intrinsic letters: R=rgb, A=albedo, I=irradiance, N=normal. Alpha letters:
R=rgb (composite), P=pha (matte), F=fgr (foreground), B=bgr (background).
Letters before 2 name condition modalities; letters after 2 name results in
upstream's decode order. The lowercase t2* text-only modes, t2RAIN and t2RPFB,
have no condition images. Normal conditioning uses inference_normal, while
the decoded result key is normal_unit, never normal.

Names and result keys were read from vendor/univid/src/pipelines/univid_*.py
at acf0ef791ae6d0294dc5f37d61a82d9b294185b9. This module never imports upstream.
"""

from collections.abc import Mapping
from dataclasses import dataclass


INTRINSIC_MODES = (
    "t2RAIN", "R2AIN", "A2RIN", "I2RAN", "N2RAI",
    "RA2IN", "RI2AN", "RN2AI", "AI2RN", "AN2RI", "IN2RA",
    "AIN2R", "RIN2A", "RAN2I", "RAI2N",
)
ALPHA_MODES = (
    "t2RPFB", "R2PFB", "P2RFB", "F2RPB", "B2RPF",
    "RP2FB", "RF2PB", "RB2PF", "PF2RB", "PB2RF", "FB2RP",
    "PFB2R", "RFB2P", "RPB2F", "RPF2B",
)

MODALITIES = {
    "R": "rgb",
    "A": "albedo",
    "I": "irradiance",
    "N": "normal",
    "P": "pha",
    "F": "fgr",
    "B": "bgr",
}
CONDITION_KWARGS = {
    "R": "inference_rgb",
    "A": "inference_albedo",
    "I": "inference_irradiance",
    "N": "inference_normal",
    "P": "inference_pha",
    "F": "inference_fgr",
    "B": "inference_bgr",
}
RESULT_KEYS = {**MODALITIES, "N": "normal_unit"}


@dataclass(frozen=True)
class Mode:
    """One mode's required modalities and ordered pipeline result keys."""

    name: str
    family: str
    conditions: frozenset[str]
    result_keys: tuple[str, ...]

    @property
    def condition_kwargs(self) -> dict[str, str]:
        """Map this mode's condition modality names to pipeline keywords."""
        return {
            MODALITIES[letter]: keyword
            for letter, keyword in CONDITION_KWARGS.items()
            if MODALITIES[letter] in self.conditions
        }


def _make_mode(name: str, *, family: str) -> Mode:
    conditions, results = name.split("2")
    return Mode(
        name=name,
        family=family,
        conditions=frozenset(MODALITIES[letter] for letter in conditions if letter != "t"),
        result_keys=tuple(RESULT_KEYS[letter] for letter in results),
    )


MODES = {
    name: _make_mode(name, family=family)
    for family, names in (("intrinsic", INTRINSIC_MODES), ("alpha", ALPHA_MODES))
    for name in names
}


def get_mode(name: str) -> Mode:
    """Resolve an exact upstream mode name, rejecting unknown names."""
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(f"Unknown UniVidX mode: {name!r}") from None


def validate_inputs(name: str, inputs: Mapping[str, object]) -> Mode:
    """Require every condition by modality name; absent or None means missing.

    Supplied values are not interpreted, so tensors need no imports or boolean
    conversion here. Extra inputs are permitted; text-only modes require none.
    """
    mode = get_mode(name)
    missing = sorted(modality for modality in mode.conditions if inputs.get(modality) is None)
    if missing:
        raise ValueError(f"UniVidX mode {name!r} is missing required inputs: {', '.join(missing)}.")
    return mode
