"""Model resolution and weight-source classification, by content not filename.

Two kinds of test live here. Most build synthetic safetensors headers in a temp
directory, so they run anywhere. A few need the real multi-gigabyte checkpoints
and skip when absent — they are marked, and they never gate the suite.

The hash-constant test parses `vendor/univid/src/models/model_config.py` for the
same reason `test_modes.py` parses the pipelines: those constants restate a fact
that lives upstream, so a vendor refresh must fail here rather than silently
accept a look-alike model.
"""

import json
import re
import struct
from pathlib import Path

import pytest

from univid import paths
from univid.paths import (
    MissingModelFile,
    classify_dit,
    read_safetensors_header,
    resolve_text_encoder,
    resolve_vae,
    state_dict_key_hash,
)


PACK = Path(__file__).resolve().parent.parent
MODEL_CONFIG = PACK / "vendor" / "univid" / "src" / "models" / "model_config.py"
COMFY_MODELS = Path("D:/AI/ComfyUI/ComfyUI/models")

needs_models = pytest.mark.skipif(
    not COMFY_MODELS.is_dir(), reason="ComfyUI model tree not present on this machine"
)


def _write_safetensors(path, tensors, metadata=None):
    """Write a header-only safetensors file; tensor bytes are never read."""
    header = dict(tensors)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    return path


def _tensor(shape, dtype="BF16", start=0):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [start, start]}


def _wan_tensors(dtype="BF16", scaled=False):
    """A header satisfying the full Wan2.1-T2V-14B signature, all 40 blocks.

    When `scaled`, only the per-block Linears are FP8 with a `scale_weight`;
    the embeddings and head stay BF16. That mirrors the real Kijai file, where
    quantisation covers the Linears and leaves embeddings, norms and biases
    alone — and it is what the classifier requires, since any 2-D FP8 weight
    without a scale would be dequantised wrongly.
    """
    embedding_dtype = "BF16" if scaled else dtype
    tensors = {
        "patch_embedding.weight": _tensor([5120, 16, 1, 2, 2], embedding_dtype),
        "text_embedding.0.weight": _tensor([5120, 4096], embedding_dtype),
        "time_embedding.0.weight": _tensor([5120, 256], embedding_dtype),
        "head.head.weight": _tensor([64, 5120], embedding_dtype),
    }
    for index in range(40):
        tensors[f"blocks.{index}.self_attn.q.weight"] = _tensor([5120, 5120], dtype)
        tensors[f"blocks.{index}.ffn.0.weight"] = _tensor([13824, 5120], dtype)
        if scaled:
            tensors[f"blocks.{index}.self_attn.q.scale_weight"] = _tensor([1], "F32")
            tensors[f"blocks.{index}.ffn.0.scale_weight"] = _tensor([1], "F32")
    return tensors


# --- the constants must keep matching the vendored detector table -------------


def _vendored_detector_table():
    """Map hash -> model class name from the pinned model_config.py."""
    source = MODEL_CONFIG.read_text(encoding="utf-8")
    table = {}
    for match in re.finditer(
        r'\(None,\s*"([0-9a-f]{32})",\s*\[([^\]]+)\],\s*\[([^\]]+)\]', source
    ):
        table[match.group(1)] = (match.group(2), match.group(3).strip())
    return table


def test_text_encoder_hash_matches_the_vendored_detector():
    table = _vendored_detector_table()
    for digest in paths._TEXT_ENCODER_HASHES:
        assert digest in table, digest
        assert "wan_video_text_encoder" in table[digest][0]


def test_vae_hashes_are_exactly_the_wan21_vae_class():
    """Accept only hashes mapping to WanVideoVAE — not merely to the VAE *name*.

    Upstream registers the Wan2.2 VAE under the same `wan_video_vae` name, so a
    name-keyed allowlist would accept the wrong architecture. This asserts the
    distinction our resolver depends on.
    """
    table = _vendored_detector_table()
    wan21 = {d for d, (names, cls) in table.items() if "wan_video_vae" in names and cls == "WanVideoVAE"}
    assert paths._VAE_HASHES == wan21


def test_rejected_hashes_are_the_wan22_vae_class():
    table = _vendored_detector_table()
    for digest, reason in paths._REJECTED_HASHES.items():
        assert table[digest][1] == "WanVideoVAE38", digest
        assert "Wan2.2" in reason, "the refusal must say which model it actually is"


def test_state_dict_key_hash_reproduces_the_documented_algorithm():
    """Sorted `name:shape` plus bare `name`, comma-joined, MD5 — from util.py."""
    import hashlib

    header = {"b.weight": _tensor([2, 3]), "a.weight": _tensor([4])}
    expected = hashlib.md5(
        ",".join(sorted(["a.weight:4", "a.weight", "b.weight:2_3", "b.weight"])).encode("utf-8")
    ).hexdigest()
    assert state_dict_key_hash(header) == expected


def test_state_dict_key_hash_ignores_metadata(tmp_path):
    plain = {"a.weight": _tensor([4])}
    assert state_dict_key_hash(plain) == state_dict_key_hash({**plain, "__metadata__": {"x": "y"}})


# --- look-alike refusal -------------------------------------------------------


def test_vae_lookalike_is_refused_by_hash(tmp_path):
    """A file named like the Wan2.1 VAE but shaped otherwise must not pass."""
    fake = _write_safetensors(tmp_path / "wan_2.1_vae.safetensors", {"a.weight": _tensor([4])})
    with pytest.raises(ValueError, match="not a usable Wan2.1 VAE"):
        resolve_vae(fake)


def test_text_encoder_lookalike_is_refused_by_hash(tmp_path):
    fake = _write_safetensors(tmp_path / "umt5-xxl-enc-bf16.safetensors", {"a.weight": _tensor([4])})
    with pytest.raises(ValueError, match="not a usable umt5-xxl text encoder"):
        resolve_text_encoder(fake)


def test_refusal_names_the_actual_model_when_known(tmp_path, monkeypatch):
    """A recognised wrong model is named, not just rejected.

    "Wrong hash" sends someone hunting; "that is the Wan2.2 VAE" ends it.
    """
    fake = _write_safetensors(tmp_path / "vae.safetensors", {"a.weight": _tensor([4])})
    digest = state_dict_key_hash(read_safetensors_header(fake))
    monkeypatch.setitem(paths._REJECTED_HASHES, digest, "the Wan2.2 VAE (WanVideoVAE38)")
    with pytest.raises(ValueError, match="Wan2.2 VAE"):
        resolve_vae(fake)


def test_pth_files_bypass_hash_validation(tmp_path):
    """A .pth has no readable header; upstream's naming is the only signal."""
    legacy = tmp_path / "Wan2.1_VAE.pth"
    legacy.write_bytes(b"not safetensors")
    assert resolve_vae(legacy) == legacy.resolve()


# --- DiT classification -------------------------------------------------------


def test_plain_single_file_classifies_without_scales(tmp_path):
    path = _write_safetensors(tmp_path / "dit.safetensors", _wan_tensors("F8_E4M3"))
    source = classify_dit(path)
    assert source.kind == "single_file"
    assert source.scale_weights == {}


def test_scaled_single_file_is_detected_by_scale_tensors_not_filename(tmp_path):
    """The filename says nothing; the scale_weight entries decide."""
    path = _write_safetensors(tmp_path / "anything.safetensors", _wan_tensors("F8_E4M3", scaled=True))
    source = classify_dit(path)
    assert source.kind == "scaled_single_file"
    assert len(source.scale_weights) == 80
    assert source.scale_weights["blocks.0.self_attn.q.weight"] == "blocks.0.self_attn.q.scale_weight"


def test_incomplete_dit_is_refused(tmp_path):
    """A 39-block file is not a Wan2.1-14B DiT, whatever it is named."""
    tensors = _wan_tensors()
    del tensors["blocks.39.self_attn.q.weight"]
    path = _write_safetensors(tmp_path / "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", tensors)
    with pytest.raises(ValueError, match="complete Wan2.1-T2V-14B DiT"):
        classify_dit(path)


def test_scaled_weight_without_its_scale_is_refused(tmp_path):
    """Partial scaling would dequantise some Linears and not others."""
    tensors = _wan_tensors("F8_E4M3", scaled=True)
    del tensors["blocks.7.ffn.0.scale_weight"]
    path = _write_safetensors(tmp_path / "dit.safetensors", tensors)
    with pytest.raises(ValueError, match="has no scale_weight"):
        classify_dit(path)


@pytest.mark.parametrize("count", [2, 5, 7])
def test_only_one_file_or_six_shards_is_accepted(tmp_path, count):
    files = [
        _write_safetensors(tmp_path / f"s{i}.safetensors", {"a.weight": _tensor([4])})
        for i in range(count)
    ]
    with pytest.raises(ValueError, match="one file or six distinct canonical shards"):
        classify_dit(files)


def test_duplicate_tensor_across_shards_is_refused(tmp_path):
    shard = _write_safetensors(tmp_path / "a.safetensors", {"x.weight": _tensor([4])})
    with pytest.raises(ValueError, match="six distinct"):
        classify_dit([shard] * 6)


# --- header reader ------------------------------------------------------------


def test_missing_file_names_the_file_and_directory(tmp_path):
    with pytest.raises(MissingModelFile) as exc:
        read_safetensors_header(tmp_path / "absent.safetensors")
    assert "absent.safetensors" in str(exc.value)
    assert str(tmp_path) in str(exc.value)


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"\x04", "Truncated safetensors length prefix"),
        (struct.pack("<Q", 10) + b"abc", "Truncated safetensors header"),
        (struct.pack("<Q", 0), "Invalid safetensors header length"),
        (struct.pack("<Q", 3) + b"{ x", "Invalid safetensors header JSON"),
        (struct.pack("<Q", 2) + b"{}", "contains no tensors"),
    ],
)
def test_malformed_headers_are_refused(tmp_path, payload, match):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        read_safetensors_header(path)


def test_header_reader_rejects_a_bad_tensor_entry(tmp_path):
    path = _write_safetensors(tmp_path / "bad.safetensors", {"a.weight": {"dtype": "BF16"}})
    with pytest.raises(ValueError, match="Invalid tensor header"):
        read_safetensors_header(path)


# --- lazy folder_paths --------------------------------------------------------


def test_folder_paths_lookup_caches_unavailability(monkeypatch):
    """Pinned None means unavailable and must never be re-imported over."""
    monkeypatch.setattr(paths, "folder_paths", None)
    assert paths._get_folder_paths() is None
    monkeypatch.setattr(paths, "folder_paths", paths._UNSET)
    paths._get_folder_paths()
    assert paths.folder_paths is not paths._UNSET, "the outcome must be cached"


# --- the real files on this machine, when present -----------------------------


@needs_models
@pytest.mark.parametrize("name", ["wan_2.1_vae.safetensors", "Wan2_1_VAE_fp32.safetensors"])
def test_real_wan21_vaes_are_accepted(name):
    path = COMFY_MODELS / "vae" / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    assert resolve_vae(path) == path.resolve()


@needs_models
def test_real_wan22_vae_is_refused_as_the_wrong_architecture():
    """The trap this validation exists for, against the actual file."""
    path = COMFY_MODELS / "vae" / "wan2.2_vae.safetensors"
    if not path.is_file():
        pytest.skip("wan2.2_vae.safetensors not present")
    with pytest.raises(ValueError, match="Wan2.2 VAE"):
        resolve_vae(path)


@needs_models
def test_real_fp16_text_encoder_is_refused():
    path = COMFY_MODELS / "text_encoders" / "umt5_xxl_fp16.safetensors"
    if not path.is_file():
        pytest.skip("umt5_xxl_fp16.safetensors not present")
    with pytest.raises(ValueError, match="not a usable umt5-xxl text encoder"):
        resolve_text_encoder(path)


@needs_models
@pytest.mark.parametrize(
    "relative,expected",
    [
        ("diffusion_models/wan2.2/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors", "scaled_single_file"),
        ("diffusion_models/wan2.1/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", "single_file"),
    ],
)
def test_real_dit_files_classify_by_content(relative, expected):
    path = COMFY_MODELS / relative
    if not path.is_file():
        pytest.skip(f"{relative} not present")
    assert classify_dit(path).kind == expected


def test_hidden_directories_are_not_resolved(tmp_path, monkeypatch):
    """`hf download --local-dir` mirrors real directory names under .cache/.

    That mirror holds only `.metadata` stubs, so resolving it yields a path that
    looks right and contains nothing usable — found live when the umt5 tokenizer
    resolved to .cache/huggingface/download/google/umt5-xxl.
    """
    real = tmp_path / "google" / "umt5-xxl"
    real.mkdir(parents=True)
    (real / "spiece.model").write_bytes(b"real")
    decoy = tmp_path / ".cache" / "huggingface" / "download" / "google" / "umt5-xxl"
    decoy.mkdir(parents=True)
    (decoy / "spiece.model.metadata").write_bytes(b"stub")

    monkeypatch.setattr(paths, "_model_roots", lambda category: (tmp_path,))
    assert paths._resolve("unividx", ("umt5-xxl",), directory=True) == real.resolve()
