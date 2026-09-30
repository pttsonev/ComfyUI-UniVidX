"""Absolute model paths and header-only Wan2.1-T2V-14B source classification.

ComfyUI supplies the model roots; no pack-relative models directory is guessed.
Without ComfyUI, registration is a no-op and explicit absolute paths still work.
Filenames select candidates only. Source kinds come from tensor names, shapes,
dtypes, and scale_weight entries in the safetensors headers, never filenames.
LightX2V-style LoRAs are checked for Wan T2V targets and consistent-rank pairs.
No tensors or runtime dependencies are loaded here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from os import PathLike
from pathlib import Path
import struct
from typing import Literal


_UNSET = object()
folder_paths = _UNSET

CANONICAL_SHARDS = tuple(
    f"diffusion_pytorch_model-{index:05d}-of-00006.safetensors"
    for index in range(1, 7)
)
_VARIANTS = frozenset({"intrinsic", "alpha"})
_TOKENIZER_FILES = frozenset({"spiece.model", "tokenizer.json", "tokenizer_config.json"})

_MAX_HEADER_BYTES = 16 * 1024 * 1024
_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})
_FLOAT_DTYPES = _FP8_DTYPES | {"F16", "BF16", "F32"}

# Architecture signatures from the pinned wan2_1_14b_t2v_dit_config.json and
# WanModel. Requiring all 40 blocks rejects a lone shard, LoRA, 1.3B, or I2V
# checkpoint. Full missing/unexpected-key reporting belongs to the loader.
_WAN_SIGNATURE = {
    "patch_embedding.weight": [5120, 16, 1, 2, 2],
    "text_embedding.0.weight": [5120, 4096],
    "time_embedding.0.weight": [5120, 256],
    "head.head.weight": [64, 5120],
    **{f"blocks.{index}.self_attn.q.weight": [5120, 5120] for index in range(40)},
    **{f"blocks.{index}.ffn.0.weight": [13824, 5120] for index in range(40)},
}

# T2V Linears from the pinned WanModel/config, including the embeddings and
# cross-attention (not just the six targets of UniVidX's modality adapters).
_WAN_LINEAR_SHAPES = {
    "text_embedding.0": (5120, 4096), "text_embedding.2": (5120, 5120),
    "time_embedding.0": (5120, 256), "time_embedding.2": (5120, 5120),
    "time_projection.1": (30720, 5120), "head.head": (64, 5120),
    **{
        f"blocks.{index}.{attention}.{projection}": (5120, 5120)
        for index in range(40) for attention in ("self_attn", "cross_attn")
        for projection in ("q", "k", "v", "o")
    },
    **{f"blocks.{index}.ffn.0": (13824, 5120) for index in range(40)},
    **{f"blocks.{index}.ffn.2": (5120, 13824) for index in range(40)},
}
# LightX2V's extractor also writes direct parameter deltas. They are part of the
# distilled base and must not be silently discarded when the pairs are merged.
_WAN_DELTA_SHAPES = {
    **{f"{name}.weight": shape for name, shape in _WAN_LINEAR_SHAPES.items()},
    **{f"{name}.bias": (shape[0],) for name, shape in _WAN_LINEAR_SHAPES.items()},
    "patch_embedding.weight": (5120, 16, 1, 2, 2), "patch_embedding.bias": (5120,),
    "head.modulation": (1, 2, 5120),
    **{f"blocks.{index}.modulation": (1, 6, 5120) for index in range(40)},
    **{
        f"blocks.{index}.{attention}.{norm}.weight": (5120,)
        for index in range(40) for attention in ("self_attn", "cross_attn")
        for norm in ("norm_q", "norm_k")
    },
    **{
        f"blocks.{index}.norm3.{parameter}": (5120,)
        for index in range(40) for parameter in ("weight", "bias")
    },
}


class MissingModelFile(FileNotFoundError):
    """An absent model file or directory, with its expected placement."""

    def __init__(self, name: str, directory: str | PathLike[str]):
        super().__init__(f"Missing model {name!r}. Put it in {directory}.")


@dataclass(frozen=True)
class WeightSource:
    """Resolved files, raw headers in matching order, and weight-to-scale keys.

    Headers retain dtype, shape, and data_offsets for the loader. scale_weights
    maps each scaled Linear weight name to its scale tensor name; scale values
    themselves are tensor data and are deliberately not read here.
    """

    kind: Literal["canonical_shards", "single_file", "scaled_single_file"]
    paths: tuple[Path, ...]
    headers: tuple[dict, ...]
    scale_weights: dict[str, str]


@dataclass(frozen=True)
class ModelPaths:
    """Every model path required for one selected UniVidX family."""

    dit: WeightSource
    vae: Path
    text_encoder: Path
    checkpoint: Path
    tokenizer: Path


@dataclass(frozen=True)
class LoRASource:
    """Header-validated base updates, keyed by the destination DiT parameter.

    pairs holds (A/down, B/up) keys; deltas holds direct diff tensor keys.
    Per-Linear alpha tensors override a global metadata alpha, when supplied.
    Tensor values are read only by the loader.
    """

    path: Path
    header: dict
    pairs: dict[str, tuple[str, str]]
    deltas: dict[str, str]
    alpha_keys: dict[str, str]
    alpha: float | None


def _get_folder_paths():
    """Cache both successful import and unavailability; None is never retried."""
    global folder_paths
    if folder_paths is _UNSET:
        try:
            import folder_paths as module
        except ImportError:
            module = None
        folder_paths = module
    return folder_paths


def register_model_folder() -> Path | None:
    """Register models/unividx without creating directories or replacing roots."""
    registry = _get_folder_paths()
    if registry is None:
        return None
    directory = Path(os.path.abspath(Path(registry.models_dir) / "unividx"))
    registry.add_model_folder_path("unividx", str(directory))
    return directory


def _model_roots(category: str) -> tuple[Path, ...]:
    registry = _get_folder_paths()
    if registry is None:
        return ()
    if category == "unividx":
        register_model_folder()
    try:
        roots = registry.get_folder_paths(category)
    except KeyError:
        roots = []
    if not roots:
        roots = [Path(registry.models_dir) / category]
    return tuple(Path(os.path.abspath(root)) for root in roots)


def _resolve(
    category: str,
    filename: str | PathLike[str],
    *,
    directory: bool = False,
    allow_directory: bool = False,
) -> Path:
    """Resolve one explicit selection without following model-file or directory links."""
    if filename is None or not os.fspath(filename).strip():
        raise ValueError(f"Select an explicit {category} model path.")
    requested = Path(filename)
    roots = _model_roots(category) if not requested.is_absolute() else ()

    def accepted(path):
        if directory:
            return path.is_dir() and any((path / name).is_file() for name in _TOKENIZER_FILES)
        return path.is_file() or (allow_directory and path.is_dir())

    if requested.is_absolute():
        if accepted(requested):
            return Path(os.path.abspath(requested))
    registry = _get_folder_paths()
    if not requested.is_absolute() and not directory and registry is not None:
        try:
            return Path(registry.get_full_path_or_raise(category, os.fspath(filename)))
        except FileNotFoundError:
            pass
    if not requested.is_absolute() and (directory or allow_directory):
        for root in roots:
            candidate = root / requested
            if candidate.is_dir() and accepted(candidate):
                return Path(os.path.abspath(candidate))
    expected = requested
    if expected.is_absolute():
        raise MissingModelFile(expected.name, expected.parent)
    if roots:
        locations = ", ".join(str(root / expected.parent) for root in roots)
    else:
        locations = (
            f"ComfyUI/models/{category}/{expected.parent} "
            "(folder_paths is unavailable; use an absolute path for standalone resolution)"
        )
    raise MissingModelFile(expected.name, locations)


def tokenizer_choices() -> list[str]:
    """List root-relative tokenizer directories, pruning hidden trees and link cycles."""
    choices = set()
    for root in _model_roots("unividx"):
        ancestors = {root: frozenset()}
        for directory, subdirs, files in os.walk(root, followlinks=True):
            subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
            path = Path(directory)
            lineage = ancestors.pop(path, frozenset())
            try:
                stat = path.stat()
            except OSError:
                subdirs[:] = []
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in lineage:
                subdirs[:] = []
                continue
            for name in subdirs:
                ancestors[path / name] = lineage | {identity}
            if _TOKENIZER_FILES.intersection(files):
                choices.add(path.relative_to(root).as_posix())
    return sorted(choices)


def read_safetensors_header(path: str | PathLike[str]) -> dict:
    """Read the 8-byte length and JSON header only, with no buffered tensor reads."""
    path = Path(os.path.abspath(path))
    try:
        with path.open("rb", buffering=0) as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors length prefix: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= _MAX_HEADER_BYTES:
                raise ValueError(f"Invalid safetensors header length {length}: {path}")
            encoded = stream.read(length)
            if len(encoded) != length:
                raise ValueError(f"Truncated safetensors header: {path}")
    except FileNotFoundError:
        raise MissingModelFile(path.name, path.parent) from None
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"Invalid safetensors header JSON: {path}") from None
    if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
        raise ValueError(f"Safetensors header contains no tensors: {path}")
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"Invalid tensor header {name!r}: {path}")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if (
            not isinstance(tensor.get("dtype"), str)
            or not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"Invalid tensor header {name!r}: {path}")
    return header


# DiffSynth's ModelManager dispatches on an MD5 over the state dict's sorted
# "name:shape" pairs (vendor/univid/src/models/util.py, hash_state_dict_keys) and
# never on the filename. Ignoring that is dangerous here because ComfyUI's own
# model tree ships two look-alikes: umt5_xxl_fp16.safetensors matches NO detector,
# and wan2.2_vae.safetensors matches the Wan2.2 WanVideoVAE38 class registered
# under the very same "wan_video_vae" name — so a name-based allowlist accepts it.
# The first prints "We cannot detect the model type. No models are loaded." and
# then fails incomprehensibly during sampling; the second quietly assembles the
# wrong architecture. Both are a dropdown click away once the loader node exists.
# Hashes are transcribed from the pinned vendor model_config.py; a test re-parses
# that file so a snapshot refresh cannot silently invalidate them.
_TEXT_ENCODER_HASHES = frozenset({"9c8818c2cbea55eca56c7b447df170da"})
_VAE_HASHES = frozenset({
    "1378ea763357eea97acdef78e65d6d96",
    "ccc42284ea13e1ad04693284c7a09be6",
})
_REJECTED_HASHES = {
    "e1de6c02cdac79f8b739f4d3698cd216": (
        "the Wan2.2 VAE (WanVideoVAE38). UniVidX is built on Wan2.1 and needs the Wan2.1 VAE"
    ),
}


def state_dict_key_hash(header: dict) -> str:
    """Reproduce DiffSynth's hash_state_dict_keys from a header alone.

    Header-only, so this costs one short read and never touches tensor data.
    """
    keys = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        keys.append(f"{name}:" + "_".join(str(size) for size in tensor["shape"]))
        keys.append(name)
    keys.sort()
    return hashlib.md5(",".join(keys).encode("utf-8")).hexdigest()


def _require_known_model(path: Path, accepted: frozenset, what: str) -> Path:
    """Refuse a look-alike before DiffSynth fails to detect it, or mis-detects it.

    Sniff non-safetensors names before upstream's extension-based dispatch.
    Every `.safetensors` selection is checked by its state-dict key hash.
    """
    if path.suffix != ".safetensors":
        try:
            read_safetensors_header(path)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"{path.name} is a safetensors file with a non-.safetensors name; "
                "rename it to .safetensors"
            )
        if path.suffix == ".pth":
            with path.open("rb") as stream:
                prefix = stream.read(8)
            if not prefix.startswith((b"PK", b"\x80")):
                raise ValueError(
                    f"{path.name} is not a zip or pickle .pth file; first bytes: {prefix.hex(' ')}."
                )
        return path
    digest = state_dict_key_hash(read_safetensors_header(path))
    if digest in accepted:
        return path
    known = _REJECTED_HASHES.get(digest)
    detail = f" It is {known}." if known else ""
    raise ValueError(
        f"{path.name} is not a usable {what}: its state-dict key hash is {digest}, which "
        f"DiffSynth's detector does not map to the model UniVidX needs.{detail} Expected one "
        f"of: {', '.join(sorted(accepted))}."
    )


def _matches_wan_signature(tensors: dict) -> bool:
    return all(
        tensors.get(name, {}).get("shape") == shape
        for name, shape in _WAN_SIGNATURE.items()
    )


def _classify(paths: tuple[Path, ...], headers: tuple[dict, ...]) -> WeightSource:
    if len(paths) not in (1, 6) or len(set(paths)) != len(paths):
        raise ValueError("A DiT source must contain one file or six distinct canonical shards.")
    tensors = {}
    for header in headers:
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate DiT tensor across shards: {name!r}")
            tensors[name] = tensor
    if not _matches_wan_signature(tensors):
        raise ValueError("Headers do not describe a complete Wan2.1-T2V-14B DiT weight source.")
    for name, tensor in tensors.items():
        if tensor["dtype"] not in _FLOAT_DTYPES:
            raise ValueError(f"Unsupported DiT tensor dtype {tensor['dtype']!r} for {name!r}.")

    scales = {}
    for name, scale in tensors.items():
        if not name.endswith(".scale_weight"):
            continue
        weight_name = name.removesuffix("scale_weight") + "weight"
        weight = tensors.get(weight_name)
        if weight is None or len(weight["shape"]) != 2 or weight["dtype"] not in _FP8_DTYPES:
            raise ValueError(f"Scale {name!r} must name a corresponding FP8 Linear weight.")
        if (
            scale["dtype"] not in {"F16", "BF16", "F32"}
            or len(scale["shape"]) > 2
            or any(
                size not in (1, dim)
                for size, dim in zip(reversed(scale["shape"]), reversed(weight["shape"]))
            )
        ):
            raise ValueError(f"Scale {name!r} cannot broadcast to {weight_name!r}.")
        scales[weight_name] = name

    if len(paths) == 6:
        if scales or any(tensor["dtype"] not in {"BF16", "F32"} for tensor in tensors.values()):
            raise ValueError(
                "Canonical shards must contain unscaled BF16 weights, with optional F32 tensors."
            )
        if tensors["blocks.0.self_attn.q.weight"]["dtype"] != "BF16":
            raise ValueError("Canonical shards must contain BF16 DiT weights.")
        kind = "canonical_shards"
    elif scales:
        missing = [
            name for name, tensor in tensors.items()
            if name.endswith(".weight") and len(tensor["shape"]) == 2
            and tensor["dtype"] in _FP8_DTYPES and name not in scales
        ]
        if missing:
            raise ValueError(f"Scaled DiT weight {missing[0]!r} has no scale_weight tensor.")
        kind = "scaled_single_file"
    else:
        if "scaled_fp8" in tensors:
            raise ValueError("A scaled_fp8 marker without Linear scale_weight tensors is unsupported.")
        kind = "single_file"
    return WeightSource(kind=kind, paths=paths, headers=headers, scale_weights=scales)


def classify_dit(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> WeightSource:
    """Classify one complete file or six shards from headers, regardless of names."""
    if isinstance(paths, (str, PathLike)):
        paths = (paths,)
    resolved = tuple(Path(os.path.abspath(path)) for path in paths)
    headers = tuple(read_safetensors_header(path) for path in resolved)
    return _classify(resolved, headers)


def _canonical_paths(directory: Path) -> tuple[Path, ...]:
    paths = tuple(directory / name for name in CANONICAL_SHARDS)
    for path in paths:
        if not path.is_file():
            raise MissingModelFile(path.name, directory)
    return paths


def resolve_dit(filename: str | PathLike[str]) -> WeightSource:
    """Classify an explicitly selected single file or canonical six-shard set.

    Standard shard filenames only locate siblings after a header proves incomplete;
    a complete single file is classified as such even under a shard filename.
    """
    path = _resolve("diffusion_models", filename, allow_directory=True)
    if path.is_dir():
        return classify_dit(_canonical_paths(path))
    header = read_safetensors_header(path)
    if not _matches_wan_signature(header) and path.name in CANONICAL_SHARDS:
        return classify_dit(_canonical_paths(path.parent))
    return _classify((path,), (header,))


def resolve_vae(filename: str | PathLike[str]) -> Path:
    """Resolve the Wan2.1 VAE, refusing anything DiffSynth would mis-detect."""
    path = _resolve("vae", filename)
    return _require_known_model(path, _VAE_HASHES, "Wan2.1 VAE")


def resolve_text_encoder(filename: str | PathLike[str]) -> Path:
    """Resolve the umt5-xxl encoder, refusing anything DiffSynth would not detect."""
    path = _resolve("text_encoders", filename)
    return _require_known_model(path, _TEXT_ENCODER_HASHES, "umt5-xxl text encoder")


def resolve_checkpoint(variant: str, *, filename: str | PathLike[str]) -> Path:
    """Resolve either UniVidX family checkpoint from the unividx registry roots."""
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown UniVidX variant: {variant!r}")
    return _resolve("unividx", filename)


def _lora_metadata_number(metadata: dict, names: tuple[str, ...]) -> float | None:
    values = []
    for name in names:
        if name not in metadata:
            continue
        try:
            value = float(metadata[name])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid LightX2V metadata {name!r}: expected a number.") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid LightX2V metadata {name!r}: expected a finite non-negative number.")
        values.append(value)
    if len(set(values)) > 1:
        raise ValueError(f"Conflicting LightX2V metadata values for {names}: {values}")
    return values[0] if values else None


def classify_lightx2v(path: str | PathLike[str]) -> LoRASource:
    """Require consistent-rank Wan T2V pairs and account for every tensor in the header.

    Accept PEFT A/B and LightX2V's equivalent down/up naming. This proves
    architectural compatibility, not training provenance. Unrecognised targets,
    inconsistent ranks and extra tensors are refused.
    """
    path = Path(os.path.abspath(path))
    header = read_safetensors_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid LightX2V metadata: {path}")
    alpha = _lora_metadata_number(metadata, ("lora_alpha", "ss_network_alpha", "alpha"))
    metadata_rank = _lora_metadata_number(metadata, ("lora_rank", "ss_network_dim", "rank", "r"))
    ranks = set()
    suffixes = {
        ".lora_A.weight": "A", ".lora_B.weight": "B",
        ".lora_A.default.weight": "A", ".lora_B.default.weight": "B",
        ".lora_down.weight": "A", ".lora_up.weight": "B",
        ".alpha": "alpha", ".diff": "weight", ".diff_b": "bias",
        ".diff_m": "modulation",
    }
    pairs, deltas, alpha_keys = {}, {}, {}
    unexpected = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        key = name
        for prefix in ("diffusion_model.", "base_model.model.", "dit."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        match = next(((suffix, kind) for suffix, kind in suffixes.items() if key.endswith(suffix)), None)
        if match is None:
            unexpected.append(name)
            continue
        suffix, kind = match
        target = key.removesuffix(suffix)
        allowed_dtypes = {"F16", "BF16", "F32"}
        if kind == "alpha":
            allowed_dtypes |= {"I32", "I64"}
        if tensor["dtype"] not in allowed_dtypes:
            raise ValueError(f"Unsupported LightX2V tensor dtype for {name!r}: {path}")
        if kind in {"A", "B", "alpha"}:
            if target not in _WAN_LINEAR_SHAPES:
                unexpected.append(name)
                continue
            weight_name = f"{target}.weight"
            if kind == "alpha":
                if weight_name in alpha_keys or tensor["shape"] not in ([], [1]):
                    raise ValueError(f"Duplicate or non-scalar LightX2V alpha {name!r}: {path}")
                alpha_keys[weight_name] = name
            else:
                pair = pairs.setdefault(weight_name, {})
                if kind in pair:
                    raise ValueError(f"Duplicate LightX2V {kind} for {weight_name!r}: {path}")
                out_features, in_features = _WAN_LINEAR_SHAPES[target]
                shape = tensor["shape"]
                rank = shape[0 if kind == "A" else 1] if len(shape) == 2 else 0
                expected = [rank, in_features] if kind == "A" else [out_features, rank]
                if rank <= 0 or shape != expected:
                    raise ValueError(
                        f"LightX2V-style LoRA {name!r} must have positive rank and shape "
                        f"{expected} for Wan2.1-T2V-14B, got {shape}: {path}"
                    )
                ranks.add(rank)
                pair[kind] = name
        else:
            parameter = f"{target}.{kind}"
            if tuple(tensor["shape"]) != _WAN_DELTA_SHAPES.get(parameter):
                unexpected.append(name)
                continue
            if parameter in deltas:
                raise ValueError(f"Duplicate LightX2V delta for {parameter!r}: {path}")
            deltas[parameter] = name
    missing = [
        f"{target}: missing LoRA {kind}"
        for target, pair in pairs.items() for kind in ("A", "B") if kind not in pair
    ]
    unexpected.extend(alpha_keys[target] for target in alpha_keys.keys() - pairs.keys())
    if missing or unexpected:
        raise ValueError(
            f"Incompatible LightX2V LoRA {path}: missing pairs={missing}; unexpected keys={unexpected}. "
            "Expected LightX2V-style LoRA pairs targeting Wan2.1-T2V-14B Linears."
        )
    if not pairs:
        raise ValueError(f"LightX2V-style LoRA {path} contains zero Wan DiT Linear pairs.")
    if len(ranks) != 1:
        raise ValueError(f"LightX2V-style LoRA must have one consistent rank, got {sorted(ranks)}: {path}")
    rank = next(iter(ranks))
    if metadata_rank is not None and metadata_rank != rank:
        raise ValueError(f"LightX2V metadata rank {metadata_rank} does not match tensor rank {rank}: {path}")
    if pairs.keys() & deltas.keys():
        raise ValueError(
            f"LightX2V supplies both a pair and a direct delta for {sorted(pairs.keys() & deltas.keys())}."
        )
    return LoRASource(
        path=path, header=header, pairs={target: (pair["A"], pair["B"]) for target, pair in pairs.items()},
        deltas=deltas, alpha_keys=alpha_keys, alpha=alpha,
    )


def resolve_lightx2v(filename: str | PathLike[str]) -> LoRASource:
    """Resolve the step-distillation LoRA from ComfyUI's loras folder roots."""
    path = _resolve("loras", filename)
    return classify_lightx2v(path)


def resolve_tokenizer(directory: str | PathLike[str]) -> Path:
    """Resolve an explicit tokenizer directory, such as google/umt5-xxl."""
    return _resolve("unividx", directory, directory=True)


def resolve_models(
    variant: str,
    *,
    dit: str | PathLike[str],
    vae: str | PathLike[str],
    text_encoder: str | PathLike[str],
    checkpoint: str | PathLike[str],
    tokenizer: str | PathLike[str],
) -> ModelPaths:
    """Resolve all paths for one family; explicit absolute paths work offline."""
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown UniVidX variant: {variant!r}")
    return ModelPaths(
        dit=resolve_dit(filename=dit),
        vae=resolve_vae(filename=vae),
        text_encoder=resolve_text_encoder(filename=text_encoder),
        checkpoint=resolve_checkpoint(variant, filename=checkpoint),
        tokenizer=resolve_tokenizer(directory=tokenizer),
    )
