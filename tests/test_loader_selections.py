"""Explicit model selections and upstream path dispatch, without importing torch."""

from contextlib import nullcontext
from inspect import Parameter, signature
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from nodes.loader import UniVidXLoader
from univid import loader, paths
from .test_paths import _tensor, _wan_tensors, _write_safetensors


class FolderPaths:
    def __init__(self, root):
        self.models_dir = str(root)
        self.roots = {category: [str(root / category)] for category in (
            "diffusion_models", "vae", "text_encoders", "unividx", "loras",
        )}
        self.lookups = []

    def add_model_folder_path(self, category, directory):
        if directory not in self.roots.setdefault(category, []):
            self.roots[category].append(directory)

    def get_folder_paths(self, category):
        return self.roots[category]

    def get_filename_list(self, category):
        return sorted({
            path.relative_to(root).as_posix()
            for root in map(Path, self.roots[category])
            for path in root.rglob("*.safetensors")
        })

    def get_full_path_or_raise(self, category, name):
        self.lookups.append((category, name))
        for root in self.roots[category]:
            path = Path(root) / name
            if path.is_file():
                return str(path.absolute())
        raise FileNotFoundError(name)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    registry = FolderPaths(tmp_path)
    for roots in registry.roots.values():
        Path(roots[0]).mkdir()
    monkeypatch.setattr(paths, "folder_paths", registry)
    return registry


def _symlink(link, target):
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Host does not permit symlinks: {exc}")
    return link


def _lora_tensors(rank, target="blocks.0.self_attn.q"):
    return {
        f"{target}.lora_A.weight": _tensor([rank, 5120]),
        f"{target}.lora_B.weight": _tensor([5120, rank]),
    }


@pytest.fixture
def selections(registry, tmp_path, monkeypatch):
    files = {
        "dit": ("diffusion_models", _wan_tensors()),
        "vae": ("vae", {"vae.weight": _tensor([4])}),
        "text_encoder": ("text_encoders", {"text.weight": _tensor([4])}),
        "checkpoint": ("unividx", {"adapter.weight": _tensor([4])}),
    }
    selected = {}
    for name, (category, tensors) in files.items():
        filename = f"{name}.safetensors"
        _write_safetensors(tmp_path / category / filename, tensors)
        selected[name] = filename
        if name in ("vae", "text_encoder"):
            monkeypatch.setattr(paths, f"_{name.upper()}_HASHES", {paths.state_dict_key_hash(tensors)})
    tokenizer = tmp_path / "unividx" / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    (tokenizer / "spiece.model").write_bytes(b"tokenizer")
    selected["tokenizer"] = "google/umt5-xxl"
    return selected


def test_dropdowns_are_required_in_widget_order(registry, selections, tmp_path):
    _write_safetensors(tmp_path / "loras" / "step32.safetensors", _lora_tensors(32))
    widgets = UniVidXLoader.INPUT_TYPES()["required"]
    assert list(widgets) == [
        "variant", "compute_dtype", "dit", "vae", "text_encoder", "checkpoint", "tokenizer",
        "distillation_lora", "distillation_strength", "vram_buffer", "vram_limit",
        "num_persistent_param_in_dit",
    ]
    for name, value in selections.items():
        assert widgets[name][0] == [value]
    assert widgets["distillation_lora"][0] == ["none", "step32.safetensors"]
    assert widgets["distillation_lora"][1]["default"] == "none"
    assert "HIGHEST-RISK" in widgets["distillation_lora"][1]["tooltip"]


def test_checkpoint_dropdown_hides_tokenizer_and_cache_files(registry, tmp_path):
    root = tmp_path / "unividx"
    for name in (
        "univid_intrinsic.safetensors", "shows/sh010/univid_alpha.safetensors",
        "google/umt5-xxl/spiece.model", "google/umt5-xxl/tokenizer.json",
        ".cache/huggingface/download/univid_intrinsic.safetensors.metadata",
        ".cache/stale.safetensors",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"x")
    choices = UniVidXLoader.INPUT_TYPES()["required"]["checkpoint"][0]
    assert sorted(c.replace("\\", "/") for c in choices) == [
        "shows/sh010/univid_alpha.safetensors", "univid_intrinsic.safetensors",
    ]


def test_empty_registry_offers_no_discovery_fallback(registry):
    widgets = UniVidXLoader.INPUT_TYPES()["required"]
    for name in ("dit", "vae", "text_encoder", "checkpoint", "tokenizer"):
        assert widgets[name][0] == []
    assert widgets["distillation_lora"][0] == ["none"]


def test_bare_import_uses_explicit_text_paths(monkeypatch):
    monkeypatch.setattr(paths, "folder_paths", None)
    widgets = UniVidXLoader.INPUT_TYPES()["required"]
    for name in ("dit", "vae", "text_encoder", "checkpoint", "tokenizer", "distillation_lora"):
        assert widgets[name][0] == "STRING"
        assert "discover" not in widgets[name][1]["tooltip"]
    assert widgets["distillation_lora"][1]["default"] == "none"


@pytest.mark.parametrize("marker", ["spiece.model", "tokenizer.json", "tokenizer_config.json"])
def test_tokenizer_lists_all_roots_and_skips_hidden_trees(registry, tmp_path, marker):
    extra = tmp_path / "extra"
    registry.roots["unividx"].insert(0, str(extra))
    for root, name in ((extra, "other/tokenizer"), (tmp_path / "unividx", "google/umt5-xxl")):
        tokenizer = root / name
        tokenizer.mkdir(parents=True)
        (tokenizer / marker).write_bytes(b"tokenizer")
        hidden = root / ".cache" / "huggingface" / name
        hidden.mkdir(parents=True)
        (hidden / marker).write_bytes(b"stub")
        (root / "empty").mkdir()
    assert paths.tokenizer_choices() == ["google/umt5-xxl", "other/tokenizer"]
    assert paths.resolve_tokenizer("other/tokenizer") == extra / "other/tokenizer"


def test_tokenizer_directory_symlink_is_preserved(registry, tmp_path):
    real = tmp_path / "tokenizer-store"
    real.mkdir()
    (real / "spiece.model").write_bytes(b"tokenizer")
    link = _symlink(tmp_path / "unividx" / "selected-tokenizer", real)
    alias = _symlink(tmp_path / "unividx" / "alias-tokenizer", real)
    _symlink(real / "cycle", real)
    assert paths.tokenizer_choices() == ["alias-tokenizer", "selected-tokenizer"]
    assert paths.resolve_tokenizer("selected-tokenizer") == link
    assert paths.resolve_tokenizer("alias-tokenizer") == alias


def test_symlinked_model_roots_are_preserved(registry, tmp_path):
    store = tmp_path / "model-store"
    (store / "unividx").mkdir(parents=True)
    _write_safetensors(store / "unividx" / "checkpoint.safetensors", {"w": _tensor([1])})
    linked_models = _symlink(tmp_path / "linked-models", store)
    registry.models_dir = str(linked_models)
    registry.roots["unividx"] = []
    assert paths.register_model_folder() == linked_models / "unividx"
    assert paths._model_roots("unividx") == (linked_models / "unividx",)
    assert paths.resolve_checkpoint("intrinsic", filename="checkpoint.safetensors") == (
        linked_models / "unividx" / "checkpoint.safetensors"
    )


def test_every_relative_file_uses_comfy_registry(registry, selections):
    paths.resolve_models("intrinsic", **selections)
    assert registry.lookups == [
        ("diffusion_models", selections["dit"]), ("vae", selections["vae"]),
        ("text_encoders", selections["text_encoder"]), ("unividx", selections["checkpoint"]),
    ]


@pytest.mark.parametrize("value", [None, "", "   ", "auto (discover)"])
def test_empty_or_legacy_auto_selection_never_discovers(registry, selections, value):
    with pytest.raises((ValueError, paths.MissingModelFile)):
        paths.resolve_models("intrinsic", **{**selections, "dit": value})


def test_basenames_do_not_search_subdirectories(registry, tmp_path):
    nested = tmp_path / "diffusion_models" / "show"
    nested.mkdir()
    _write_safetensors(nested / "dit.safetensors", _wan_tensors())
    with pytest.raises(paths.MissingModelFile):
        paths.resolve_dit("dit.safetensors")
    assert paths.resolve_dit("show/dit.safetensors").paths == (nested / "dit.safetensors",)


def test_model_inputs_have_no_discovery_defaults():
    for function in (paths.resolve_models, loader.load_model):
        for name in ("dit", "vae", "text_encoder", "checkpoint", "tokenizer"):
            assert signature(function).parameters[name].default is Parameter.empty
    for function in (
        paths.resolve_dit, paths.resolve_vae, paths.resolve_text_encoder,
        paths.resolve_checkpoint, paths.resolve_lightx2v,
    ):
        assert signature(function).parameters["filename"].default is Parameter.empty


@pytest.mark.parametrize("resolve", [paths.resolve_vae, paths.resolve_text_encoder])
@pytest.mark.parametrize("suffix", [".pth", ".bin", ""])
def test_misnamed_safetensors_are_refused(tmp_path, resolve, suffix):
    file = _write_safetensors(tmp_path / f"misnamed{suffix}", {"w": _tensor([1])})
    with pytest.raises(ValueError, match="is a safetensors file with a non-.safetensors name") as exc:
        resolve(file)
    assert file.name in str(exc.value)
    assert "rename it to .safetensors" in str(exc.value)


@pytest.mark.parametrize("resolve", [paths.resolve_vae, paths.resolve_text_encoder])
def test_junk_pth_refusal_names_file_and_first_bytes(tmp_path, resolve):
    file = tmp_path / "junk.pth"
    file.write_bytes(b"garbage!")
    with pytest.raises(ValueError, match="not a zip or pickle") as exc:
        resolve(file)
    assert file.name in str(exc.value)
    assert b"garbage!".hex(" ") in str(exc.value)


@pytest.mark.parametrize("rank", [32, 64, 128])
@pytest.mark.parametrize("metadata", [False, True])
def test_lora_accepts_rank_from_pairs(tmp_path, rank, metadata):
    tensors = {**_lora_tensors(rank), **_lora_tensors(rank, "blocks.1.self_attn.q")}
    file = _write_safetensors(
        tmp_path / "step.safetensors", tensors, {"lora_rank": str(rank)} if metadata else None,
    )
    assert len(paths.classify_lightx2v(file).pairs) == 2


@pytest.mark.parametrize("same_pair", [False, True])
def test_mixed_lora_ranks_are_refused(tmp_path, same_pair):
    tensors = _lora_tensors(32)
    if same_pair:
        tensors["blocks.0.self_attn.q.lora_B.weight"] = _tensor([5120, 128])
    else:
        tensors.update(_lora_tensors(128, "blocks.1.self_attn.q"))
    file = _write_safetensors(tmp_path / "mixed.safetensors", tensors)
    with pytest.raises(ValueError, match="one consistent rank"):
        paths.classify_lightx2v(file)


def test_lora_metadata_rank_must_match_tensors(tmp_path):
    file = _write_safetensors(tmp_path / "step.safetensors", _lora_tensors(32), {"rank": "64"})
    with pytest.raises(ValueError, match="metadata rank.*does not match tensor rank"):
        paths.classify_lightx2v(file)


@pytest.mark.parametrize("defect,match", [
    ("shape", "positive rank and shape"), ("zero", "positive rank"),
    ("extra", "unexpected keys"), ("target", "unexpected keys"), ("missing", "missing pairs"),
])
def test_lora_shape_target_and_extra_tensor_refusals_remain(tmp_path, defect, match):
    tensors = _lora_tensors(32)
    if defect == "shape":
        tensors["blocks.0.self_attn.q.lora_A.weight"] = _tensor([32, 4096])
    elif defect == "zero":
        tensors = _lora_tensors(0)
    elif defect == "extra":
        tensors["stray"] = _tensor([1])
    elif defect == "target":
        tensors.update(_lora_tensors(32, "blocks.40.self_attn.q"))
    else:
        del tensors["blocks.0.self_attn.q.lora_B.weight"]
    file = _write_safetensors(tmp_path / "bad.safetensors", tensors)
    with pytest.raises(ValueError, match=match):
        paths.classify_lightx2v(file)


@pytest.mark.parametrize("selection", ["none", "step32.safetensors"])
def test_node_forwards_explicit_selections_and_distillation(monkeypatch, selections, selection):
    seen = {}

    def load(variant, **kwargs):
        seen.update(variant=variant, **kwargs)
        return "handle"

    monkeypatch.setattr(loader, "load_model", load)
    assert UniVidXLoader().load(
        "intrinsic", "bfloat16", **selections,
        distillation_lora=selection, distillation_strength=0.5,
    ) == ("handle",)
    assert all(seen[name] == value for name, value in selections.items())
    assert seen["distillation_lora"] == (None if selection == "none" else selection)
    assert seen["distillation_strength"] == 0.5


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    """Stub only the expensive runtime; exercise the real resolver and load cache."""
    cuda = SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0, synchronize=lambda **kw: None,
        get_device_properties=lambda index: SimpleNamespace(total_memory=32 * 1024 ** 3),
    )
    torch = SimpleNamespace(
        cuda=cuda, bfloat16="bf16", device=lambda name: SimpleNamespace(type="cuda", index=0),
        zeros=lambda *a, **kw: SimpleNamespace(add_=lambda value: None),
    )
    monkeypatch.setattr(loader, "_get_torch", lambda: torch)
    monkeypatch.setattr(loader, "_torch", loader._UNSET)
    monkeypatch.setattr(loader, "_MODEL_CACHE", loader.OrderedDict())
    monkeypatch.setenv("UNIVIDX_MODEL_CACHE_MAX", "1")
    config = tmp_path / "arch.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(loader, "_vendor_config", lambda: config)
    builds = []

    def build(variant, resolved, config, **kwargs):
        builds.append((resolved, kwargs))
        return object(), ()

    monkeypatch.setattr(loader, "_build_model", build)
    return builds


def test_none_skips_lora_and_file_selection_rebuilds_cache(fake_runtime, selections, tmp_path):
    file = _write_safetensors(tmp_path / "loras" / "step32.safetensors", _lora_tensors(32))
    plain = loader.load_model("intrinsic", **selections, distillation_lora="none")
    assert loader.load_model("intrinsic", **selections, distillation_lora=None) is plain
    assert fake_runtime[0][1]["distillation_source"] is None
    merged = loader.load_model("intrinsic", **selections, distillation_lora=file.name, distillation_strength=0.5)
    assert merged is not plain
    assert fake_runtime[1][1]["distillation_source"].path == file
    assert fake_runtime[1][1]["distillation_strength"] == 0.5
    assert loader.load_model(
        "intrinsic", **selections, distillation_lora=str(file), distillation_strength=0.5,
    ) is merged
    assert len(fake_runtime) == 2


def test_invalid_selected_lora_is_refused_before_build(fake_runtime, selections, tmp_path):
    file = _write_safetensors(tmp_path / "loras" / "bad.safetensors", {"stray": _tensor([1])})
    with pytest.raises(ValueError, match="unexpected keys"):
        loader.load_model("intrinsic", **selections, distillation_lora=file.name)
    assert not fake_runtime


def test_symlink_names_reach_cache_and_upstream_loaders(
    fake_runtime, selections, registry, tmp_path, monkeypatch,
):
    chosen = {}
    for name, category in (
        ("dit", "diffusion_models"), ("vae", "vae"),
        ("text_encoder", "text_encoders"), ("checkpoint", "unividx"),
    ):
        link = tmp_path / category / selections[name]
        blob = tmp_path / f"blob-{name}"
        link.rename(blob)
        _symlink(link, blob)
        chosen[name] = link
    lora_blob = _write_safetensors(tmp_path / "blob-lora", _lora_tensors(32))
    lora_link = _symlink(tmp_path / "loras" / "step32.safetensors", lora_blob)
    handle = loader.load_model("intrinsic", **selections, distillation_lora=lora_link.name)
    resolved, kwargs = fake_runtime[0]
    assert resolved.dit.paths == (chosen["dit"],)
    for name in ("vae", "text_encoder", "checkpoint"):
        assert getattr(resolved, name) == chosen[name]
    assert kwargs["distillation_source"].path == lora_link
    assert str(lora_link) in next(iter(loader._MODEL_CACHE))
    assert loader.load_model(
        "intrinsic", **selections, distillation_lora=str(lora_link),
    ) is handle

    # Run real assembly with minimal upstream stand-ins to observe the actual
    # arguments to ModelManager, safetensors.load_file and the tokenizer loader.
    seen = []
    dit = SimpleNamespace(
        to=lambda **kw: None, named_parameters=lambda: [],
        load_state_dict=lambda *a, **kw: SimpleNamespace(missing_keys=[], unexpected_keys=[]),
    )
    pipe = SimpleNamespace(
        prompter=SimpleNamespace(fetch_models=lambda model: None, fetch_tokenizer=lambda path: seen.append(path)),
        scheduler=SimpleNamespace(set_timesteps=lambda *a, **kw: None),
        freeze_except=lambda names: None, enable_vram_management=lambda **kw: None,
    )

    class Model:
        def add_multiple_loras_to_model(self, model, **kwargs): return model
        def eval(self): pass

    upstream = SimpleNamespace(
        DiffusionTrainingModule=object, WanVideoPipeline=lambda **kw: pipe, WanModel=lambda **kw: dit,
        ModelManager=lambda **kw: SimpleNamespace(
            load_model=lambda path, **kw: seen.append(path), fetch_model=lambda name: object(),
        ),
        **{f"WanVideoUnit_{name}": lambda: object() for name in (
            "ShapeChecker", "NoiseInitializer", "InputVideoEmbedder", "PromptEmbedder",
        )},
    )
    monkeypatch.setattr(loader, "_get_upstream", lambda *a: (upstream, Model))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(
        load_file=lambda path, **kw: seen.append(path) or {},
    ))
    monkeypatch.setitem(sys.modules, "src.models.util", SimpleNamespace(init_weights_on_device=nullcontext))
    # Retrieve the real function replaced by fake_runtime without undoing the
    # CUDA/cache fixtures. It is captured below at module import, before patches.
    _BUILD_MODEL("intrinsic", resolved, loader._vendor_config(), dtype="bf16", device="cuda:0", vram_buffer=0.5)
    assert seen == [
        str(chosen["text_encoder"]), str(chosen["vae"]), str(chosen["dit"]),
        str(resolved.tokenizer), str(chosen["checkpoint"]),
    ]


_BUILD_MODEL = loader._build_model


def test_symlinked_safetensors_still_get_hash_checked(tmp_path):
    blob = _write_safetensors(tmp_path / "hash-blob", {"wrong.weight": _tensor([4])})
    link = _symlink(tmp_path / "vae.safetensors", blob)
    with pytest.raises(ValueError, match="not a usable Wan2.1 VAE"):
        paths.resolve_vae(link)


def test_canonical_shards_use_selected_unresolved_parent(registry, tmp_path):
    directory = tmp_path / "diffusion_models" / "wan"
    directory.mkdir()
    tensors = list(_wan_tensors().items())
    shards = []
    for index, name in enumerate(paths.CANONICAL_SHARDS):
        blob = _write_safetensors(tmp_path / f"hash-{index}", dict(tensors[index::6]))
        shards.append(_symlink(directory / name, blob))
    for selected in shards:
        source = paths.resolve_dit(f"wan/{selected.name}")
        assert source.kind == "canonical_shards"
        assert source.paths == tuple(shards)


def test_incomplete_canonical_set_names_missing_sibling(tmp_path):
    selected = _write_safetensors(tmp_path / paths.CANONICAL_SHARDS[0], {"w": _tensor([1])})
    with pytest.raises(paths.MissingModelFile, match=paths.CANONICAL_SHARDS[1]):
        paths.resolve_dit(selected)
