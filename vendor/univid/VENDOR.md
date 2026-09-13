# Vendored upstream: UniVidX

This directory is a **materialised snapshot** of upstream UniVidX, not a git
submodule. `scripts/export_release.sh` builds pack exports with `git archive`,
which turns a gitlink into an empty directory — a submodule would export as
nothing at all. The snapshot is therefore committed in full.

| | |
|---|---|
| Upstream | https://github.com/houyuanchen111/UniVidX |
| Pinned commit | `acf0ef791ae6d0294dc5f37d61a82d9b294185b9` |
| Commit date | 2026-05-15 |
| Snapshot taken | 2026-09-09 |
| Upstream licence | Apache License 2.0 (see `LICENSE` beside this file) |

Upstream ships no `NOTICE` file, so there is none to reproduce; attribution is
carried by `LICENSE` and by this file.

## What is included, and why

- `src/` — the whole package. We reuse upstream's pipelines wholesale: the
  denoising loop, Cross-Modal Self-Attention, per-modality LoRA routing, the
  VAE decode branches. None of it is reimplemented here.
- `configs/wan2_1_14b_t2v_dit_config.json` — the DiT architecture config. Our
  loader reads it by **absolute** path.
- `LICENSE` — upstream's Apache-2.0 text, unmodified.

## What is deliberately excluded

- `assets/` — several hundred MB of demo video and GIFs, needed by nothing at
  inference time.
- `scripts/` — the inference CLIs and the `MODEL_REGISTRY` indirection. Our
  loader imports the pipeline classes directly, so the registry is dead weight.
- `checkpoints/`, `data/`, `models/` — empty placeholder trees upstream; model
  files live under `ComfyUI/models/` and are resolved through ComfyUI's
  `folder_paths`.

## The one thing we do not use

`WanVideoPipeline.from_pretrained` in `src/pipelines/univid_intrinsic.py` and
`src/pipelines/univid_alpha.py` hardcodes two **relative** paths — the DiT
config above, and a six-shard weight pattern under
`models/Wan-AI/Wan2.1-T2V-14B/`. Both resolve against the process working
directory, which is why other wrappers of this model resort to `os.chdir` and
symlinks. `univid/loader.py` replaces that entry point with an absolute-path
equivalent and calls nothing else differently. Everything downstream of model
assembly is upstream's own code.

## Refreshing the snapshot

Upstream is treated as read-only: **never edit files in this directory.** A fix
that appears to belong here belongs either upstream or in `univid/` as a
wrapper-side adaptation. To move to a newer upstream:

```sh
git clone https://github.com/houyuanchen111/UniVidX.git /tmp/univid-refresh
cd /tmp/univid-refresh && git checkout <new-commit>
# from the monorepo root, with <V> = ComfyUI-UniVidX/vendor/univid
rm -rf "<V>/src" "<V>/configs"
mkdir -p "<V>/configs"
cp -r /tmp/univid-refresh/src "<V>/src"
cp /tmp/univid-refresh/configs/*dit*.js*n "<V>/configs/"
cp /tmp/univid-refresh/LICENSE "<V>/LICENSE"
find "<V>" -name __pycache__ -type d -prune -exec rm -rf {} +
```

Then update the commit, date and licence rows in the table above, re-run the
pack's tests, and re-run the Phase 6 parity check from
`docs/agent/plans/F_0.1.0_univid-aov-pack.plan.md` — a change in upstream's
model assembly is exactly what that check exists to catch.

## Licence boundary (binding)

Upstream UniVidX is Apache-2.0 in both code and published weights. The
third-party wrapper `dreamrec/UniVidX_ComfyUI` is **GPL-3.0** and **no code
from it appears in this pack**. Where this pack solves a problem that wrapper
also solved — path resolution, VRAM management, FP8 handling — the
implementation is derived from upstream's own source and from documented file
formats only.
