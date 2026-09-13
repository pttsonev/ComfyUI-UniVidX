"""LightX2V merge: the arithmetic, the scale convention, and the silent failures.

This merge is destructive and unverifiable by eye — a wrong scale or a skipped
subset produces a model that still runs and still makes pictures, and the user
would attribute the resulting quality to distillation rather than to a bug. So
the tests here are mostly about the ways it could quietly do nothing, or do
half of what it claims.

The scale convention is the sharpest edge: LightX2V's extractor writes no alpha,
and absent alpha means alpha = rank (scale 1), NOT 1/rank. Reading it the other
way would under-apply every update by a factor of 64 and look merely "weak".
"""

from pathlib import Path

import pytest

from univid import loader, paths


torch = pytest.importorskip("torch")


def _source(pairs=None, deltas=None, alpha=None, alpha_keys=None, header=None):
    return paths.LoRASource(
        path=Path("/fake/lightx2v.safetensors"),
        header=header or {},
        pairs=pairs or {},
        deltas=deltas or {},
        alpha_keys=alpha_keys or {},
        alpha=alpha,
    )


def _header_for(tensors):
    return {
        name: {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [0, 0]}
        for name, tensor in tensors.items()
    }


def _merge(state, source, lora, strength=1.0):
    return loader._merge_lightx2v(
        state, source, strength=strength, torch=torch, load_file=lambda path, device: dict(lora),
    )


# --- the arithmetic -----------------------------------------------------------


def test_pair_update_is_base_plus_b_at_a():
    """W' = W + strength * (alpha/rank) * (B @ A), with alpha absent => scale 1."""
    base = torch.zeros(4, 3)
    down = torch.ones(2, 3)          # A: rank x in
    up = torch.ones(4, 2)            # B: out x rank
    lora = {"m.lora_A": down, "m.lora_B": up}
    state = {"m.weight": base.clone()}
    _merge(state, _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                          header=_header_for(lora)), lora)
    # B @ A is 4x3 of value 2 (inner dimension = rank 2); alpha absent -> scale 1.
    assert torch.allclose(state["m.weight"], torch.full((4, 3), 2.0))


def test_strength_scales_the_update_linearly():
    lora = {"m.lora_A": torch.ones(2, 3), "m.lora_B": torch.ones(4, 2)}
    source = _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")}, header=_header_for(lora))
    results = []
    for strength in (0.5, 1.0, 2.0):
        state = {"m.weight": torch.zeros(4, 3)}
        _merge(state, source, lora, strength=strength)
        results.append(float(state["m.weight"][0, 0]))
    assert results == pytest.approx([1.0, 2.0, 4.0])


def test_zero_strength_leaves_the_base_untouched():
    """Equivalent to not merging — must not drift the weights at all."""
    lora = {"m.lora_A": torch.ones(2, 3), "m.lora_B": torch.ones(4, 2)}
    base = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    state = {"m.weight": base.clone()}
    _merge(state, _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                          header=_header_for(lora)), lora, strength=0.0)
    assert torch.equal(state["m.weight"], base)


def test_absent_alpha_means_scale_one_not_one_over_rank():
    """The sharpest edge: 1/rank would under-apply this update 64-fold."""
    rank = 64
    lora = {"m.lora_A": torch.ones(rank, 2), "m.lora_B": torch.ones(2, rank)}
    state = {"m.weight": torch.zeros(2, 2)}
    _merge(state, _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                          header=_header_for(lora)), lora)
    # B @ A sums over rank -> every entry is `rank`; scale 1 keeps it at `rank`.
    assert float(state["m.weight"][0, 0]) == pytest.approx(float(rank))


def test_global_alpha_metadata_is_honoured():
    rank = 2
    lora = {"m.lora_A": torch.ones(rank, 3), "m.lora_B": torch.ones(4, rank)}
    state = {"m.weight": torch.zeros(4, 3)}
    _merge(state, _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                          alpha=1.0, header=_header_for(lora)), lora)
    # alpha 1 over rank 2 halves the update: (B@A)=2 -> 1.0
    assert float(state["m.weight"][0, 0]) == pytest.approx(1.0)


def test_per_linear_alpha_overrides_the_global_one():
    rank = 2
    lora = {
        "m.lora_A": torch.ones(rank, 3), "m.lora_B": torch.ones(4, rank),
        "m.alpha": torch.tensor(1.0),
    }
    state = {"m.weight": torch.zeros(4, 3)}
    _merge(state, _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                          alpha=8.0, alpha_keys={"m.weight": "m.alpha"},
                          header=_header_for(lora)), lora)
    assert float(state["m.weight"][0, 0]) == pytest.approx(1.0), "per-Linear alpha must win"


# --- direct deltas ------------------------------------------------------------


def _pair_and_delta(delta_value):
    """The real file's shape: Linear pairs AND direct deltas together.

    A deltas-only LoRA is deliberately not tested as valid — the merge refuses
    it, and that refusal is the guard that catches a prefix mismatch mapping no
    Linears at all.
    """
    lora = {
        "m.lora_A": torch.ones(2, 3), "m.lora_B": torch.zeros(4, 2),
        "n.delta": torch.full((3,), delta_value),
    }
    source = _source(
        pairs={"m.weight": ("m.lora_A", "m.lora_B")},
        deltas={"n.weight": "n.delta"},
        header=_header_for(lora),
    )
    return lora, source


def test_direct_deltas_are_applied_alongside_pairs():
    """LightX2V's extractor writes deltas for non-Linear parameters.

    Dropping them yields a partial merge that still reports success — the model
    would run, look slightly wrong, and the cause would be invisible.
    """
    lora, source = _pair_and_delta(0.5)
    state = {"m.weight": torch.zeros(4, 3), "n.weight": torch.ones(3)}
    _merge(state, source, lora)
    assert torch.allclose(state["n.weight"], torch.full((3,), 1.5))


def test_deltas_respect_strength():
    lora, source = _pair_and_delta(1.0)
    state = {"m.weight": torch.zeros(4, 3), "n.weight": torch.zeros(3)}
    _merge(state, source, lora, strength=0.25)
    assert torch.allclose(state["n.weight"], torch.full((3,), 0.25))


def test_a_deltas_only_lora_is_refused():
    """Pins the guard's intent: no mapped Linears means the merge did nothing."""
    lora = {"n.delta": torch.full((3,), 0.5)}
    with pytest.raises(ValueError, match="would merge zero"):
        _merge({"n.weight": torch.ones(3)},
               _source(deltas={"n.weight": "n.delta"}, header=_header_for(lora)), lora)


# --- the silent failures ------------------------------------------------------


def test_a_merge_that_would_do_nothing_is_refused():
    """The worst outcome: 'distilled' output that was never distilled."""
    with pytest.raises(ValueError, match="would merge zero"):
        _merge({"m.weight": torch.zeros(2, 2)}, _source(), {})


def test_missing_target_in_the_base_is_refused():
    lora = {"m.lora_A": torch.ones(2, 3), "m.lora_B": torch.ones(4, 2)}
    with pytest.raises(ValueError, match="missing or has an incompatible base shape"):
        _merge({}, _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                           header=_header_for(lora)), lora)


def test_shape_mismatch_against_the_base_is_refused():
    lora = {"m.lora_A": torch.ones(2, 3), "m.lora_B": torch.ones(4, 2)}
    with pytest.raises(ValueError, match="incompatible base shape"):
        _merge({"m.weight": torch.zeros(9, 9)},
               _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                       header=_header_for(lora)), lora)


def test_tensor_disagreeing_with_its_header_is_refused():
    """A header/tensor mismatch means the file is not what validation approved."""
    lora = {"m.lora_A": torch.ones(2, 3), "m.lora_B": torch.ones(4, 2)}
    header = _header_for(lora)
    header["m.lora_A"]["shape"] = [9, 9]
    with pytest.raises(ValueError, match="does not match its header shape"):
        _merge({"m.weight": torch.zeros(4, 3)},
               _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")}, header=header), lora)


def test_integer_lora_tensors_are_refused():
    lora = {"m.lora_A": torch.ones(2, 3, dtype=torch.int32), "m.lora_B": torch.ones(4, 2)}
    with pytest.raises(ValueError, match="must be floating point"):
        _merge({"m.weight": torch.zeros(4, 3)},
               _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                       header=_header_for(lora)), lora)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -1.0])
def test_invalid_alpha_is_refused(alpha):
    lora = {"m.lora_A": torch.ones(2, 3), "m.lora_B": torch.ones(4, 2)}
    with pytest.raises(ValueError, match="finite and non-negative"):
        _merge({"m.weight": torch.zeros(4, 3)},
               _source(pairs={"m.weight": ("m.lora_A", "m.lora_B")},
                       alpha=alpha, header=_header_for(lora)), lora)


# --- the cache must not hand back a merged model as an unmerged one -----------


def test_distillation_settings_are_part_of_the_cache_key():
    """A merged model cannot be un-merged, so the cache must not confuse them."""
    import inspect

    parameters = inspect.signature(loader.load_model).parameters
    distillation = [name for name in parameters if "distill" in name]
    assert distillation, "load_model must accept a distillation setting"

    key_source = inspect.getsource(loader.load_model)
    key_block = key_source[key_source.index("key = ("):key_source.index(")", key_source.index("key = ("))]
    for name in distillation:
        assert name in key_block, f"{name} must be part of the cache key"
