# ComfyUI-UniVidX

Decompose a video plate into **albedo / irradiance / normal** (intrinsic) or
**composite / matte / foreground / background** (alpha), inside ComfyUI, using
[UniVidX](https://github.com/houyuanchen111/UniVidX) (SIGGRAPH 2026).

Upstream's pipeline runs unmodified. This pack replaces exactly one thing —
model loading — so weights live in `ComfyUI/models/` like every other pack.

## Production recipe (2026-09-13)

Version **0.2.0** defaults to `attention=sage`, `context_window_frames=21`,
`context_stride_frames=16`, `context_blend=cosine`, `rope_precision=float32`,
and loader `num_persistent_param_in_dit=2e9`. `shared_kv=True` and
`tiled_encode=False` are unchanged. Install `sageattention` in ComfyUI's Python
or select `attention=sdpa`; the sampler refuses SageAttention when it is missing.

Measured on an **RTX 5090 with SageAttention 2.2.0**: **189 frames at 704x1248
in 26:17**, **28.9 s/window**, **15.2 GiB peak**. That R2AIN run also used
`distillation=lightx2v`, 4 steps, an empty prompt (CFG 1.0), seed 0, and context
windows enabled. Distillation, step count, resolution, frame count and
`context_enabled` retain their existing defaults; set them for this measured run.

Stride 16 gives a 2-latent overlap; cosine halves its seam penalty at zero cost.
Float32 RoPE saves 0.4 s/window and 1.4 GiB for a 0.49° mean normal difference
(kernel-noise class); `float64` keeps upstream numerics. SageAttention 2.2.0 was
measured at 6x SDPA per attention call. On a 32 GB card, never use cap 4e9 with
2.2.0: it pages. Cap 0 streams the whole DiT (12.6 GiB peak, ~+1 s/window);
raise the cap only with real headroom (96 GB: full residency).

The previous defaults (`sdpa`, stride 12, `triangular`, `float64`, and the
upstream-default residency cap via `-1`) remain selectable.
`-1` still means unset; the persistent
parameter cap remains mutually exclusive with `vram_limit`.

**Historical measurement (2026-09-10):** 189 frames at 704x1248 in 40:11 used
`attention=sage` (SageAttention 1.0.6), `num_persistent_param_in_dit=8e9`,
`distillation=lightx2v` at 4 steps, an empty prompt (CFG 1.0), context windows
21/12, `triangular` blending and `float64` RoPE, with `vram_limit` unset.
The 8e9 cap was withdrawn on 2026-09-12: use **<= 4e9 with SageAttention 1.0.6**
on the 32 GB card at this resolution, or **<= 2e9 with SageAttention 2.2.0**.

## Read this before installing

This is a **14-billion-parameter video diffusion model**. Three limits are
properties of the model, not of this pack, and no wrapper removes them:

- **21 frames at 640×480** is the reference operating point. Other sizes run,
  but they are off the distribution the model was trained on.
- **Roughly 9–10 minutes per 21-frame chunk** on an RTX 5090. A minute of
  24 fps footage is ~90 chunks.
- **There is no relighting.** UniVidX decomposes and stops. A grep of upstream
  for envmap / HDRI / shading / relight finds nothing. Whether
  `albedo × irradiance` reconstructs your plate is an open question this pack
  does not answer — the paper relights through a separate diffusion pass.

You need a CUDA GPU. The loader refuses CPU, and probes a real kernel launch
before allocating anything, because a driver-visible GPU under a mismatched
torch build reports available and then cannot execute.

## Install

```
cd ComfyUI/custom_nodes/ComfyUI-UniVidX
python -m pip install -r requirements.txt
```

Most of those already ship with ComfyUI. **`modelscope` is the one that does
not** — upstream imports it unconditionally in two modules this pack reaches,
even though nothing here ever downloads anything.

## Where the models go

Everything resolves through ComfyUI's `folder_paths`. There are no symlinks, no
junctions, and nothing changes the process working directory.

| What | Where | Notes |
|---|---|---|
| Wan2.1-T2V-14B DiT | `models/diffusion_models/**` | A ComfyUI single-file checkpoint **or** upstream's six-shard set |
| Wan2.1 VAE | `models/vae/` | e.g. `wan_2.1_vae.safetensors` |
| umt5-xxl text encoder | `models/text_encoders/` | `umt5-xxl-enc-bf16.safetensors` |
| UniVidX checkpoints | `models/unividx/univid_{intrinsic,alpha}.safetensors` | ~1.6 GB, from [the model repo](https://huggingface.co/houyuanchen/UniVidX) |
| umt5 tokenizer | `models/unividx/umt5-xxl/` | small |

**Models are identified by content, never by filename.** DiffSynth dispatches on
a hash of state-dict key names, and ComfyUI's own model tree contains two
look-alikes that would otherwise fail far from the cause:

- `umt5_xxl_fp16.safetensors` matches no detector at all — you would get
  "We cannot detect the model type. No models are loaded." and then a crash
  during sampling.
- `wan2.2_vae.safetensors` is the **Wan2.2** VAE. It is registered under the
  same `wan_video_vae` name, so a name-based check accepts it and quietly
  assembles the wrong architecture.

Both are refused up front, in a message that names what the file actually is.
The DiT is checked the same way: all 40 blocks must be present, so a lone
shard, a LoRA, the 1.3B model or an I2V checkpoint is rejected rather than
half-loaded.

## The input encoding contract — read this if you work in linear

Upstream's `src/dataset/` is a placeholder ("Put your Dateset code here"), so
**the training-side transform from source EXR to model tensor is unspecified**.
The only convention that exists at inference is upstream's own video loader:
decode an 8-bit H.264 MP4, divide by 255, scale to `[-1, 1]`. The model's real
input contract is therefore **display-referred**.

This matters because feeding a linear plate straight in produces *plausible but
wrong* output, not an error. So the sampler makes the choice explicit:

- **`display_referred_srgb`** (default) — your IMAGE already matches the
  model's convention; only the range map is applied. This is the reference path.
- **`linear_rec709_to_srgb`** — applies the sRGB encoding transfer first. It
  requires scene-linear **Rec.709** values within `[0, 1]`, and enforces it: an
  encoding transfer changes the tone curve only. It does not rotate primaries,
  so linear ACEScg is simply wrong here, and it does not bound magnitude — a
  linear `4.0` would reach the model as `2.65` against a `[-1, 1]` training
  range, silently.

For a wide-gamut or HDR EXR, use **`Gamut: Load EXR` with
`tonemap_preview=True`** — it decodes the transfer, rotates primaries to
Rec.709, clips, and sRGB-encodes in one step — then select
`display_referred_srgb`, since that output is already encoded.

The setting affects **rgb, albedo, irradiance, fgr and bgr only**. Normal and
matte conditioning are geometry and coverage, not colour, and never pass
through a transfer under either setting.

## Nodes

| Node | Purpose |
|---|---|
| UniVidX: Load Model | Resolve and assemble a variant; caches per resolved settings |
| UniVidX: Select Task | Pick one of the 30 modes |
| UniVidX: Sampler | Validate conditioning and run upstream's pipeline |
| UniVidX: Decode Intrinsic | rgb / albedo / irradiance / normal |
| UniVidX: Decode Alpha | rgb / pha / fgr / bgr |

Sampler behaviours to know:

**`rope_precision` defaults to `float32`**, using float32/complex64 arithmetic in
all loaded intrinsic, alpha and base DiTs. Select `float64` to keep upstream's
float64/complex128 RoPE.
The temporary wrapper caches frequency casts for one
pipeline call, restores upstream afterwards, and reports the actual call count.
Measured 2026-09-13 (RTX 5090, 21 frames at 704x1248, R2AIN, lightx2v 4 steps, sage + shared_kv, persistent cap 4e9, seed 0), float32 vs upstream float64: 0.4 s saved per 34 s window (~1%), peak VRAM 20.0 → 18.6 GiB from removing float64 q/k temporaries, normal angle difference 0.49° mean / 2.0° p99, albedo mean abs 0.002 (0.3% of mean) — the same order as SageAttention's own approximation.

**Interrupt** is checked before every noise prediction and context window;
cancellation takes effect after the current prediction finishes.

**An empty prompt forces CFG to 1.0**, matching upstream's inference scripts,
and says so in the log and in the result. Plate decomposition is exactly this
case; without the rule you would run guidance against a negative prompt with an
empty positive — off-reference, and a second denoiser evaluation per step.

**Frame fitting is reported, not silent.** A clip longer than `num_frames` is
truncated and a shorter one is padded by repeating its last frame; either way
the node says which it did and how many frames were involved.

Decoder outputs always share one resolution and frame count. A modality that
was a *condition* comes back as the resized, frame-fitted tensor actually fed
to the model — never your original IMAGE, which would break alignment for
anything consuming the four outputs together, and never a black placeholder,
which reads downstream as real data.

**TeaCache and `cfg_merge` are not exposed.** They look available upstream but
are dead wiring in UniVidX: `WanVideoUnit_TeaCache` is absent from the
pipeline's unit list, so `tea_cache` is always `None`; and `cfg_merge` chunks a
batch dimension that holds the four modalities rather than a merged
positive/negative pair, so enabling it above CFG 1.0 corrupts results instead
of accelerating them.

## Long clips

The model generates 21 frames. For anything longer, turn on **context windows**
on the sampler: it slices the latent temporally, runs the model per window, and
blends the **noise predictions at every denoising step**. That is a different
thing from chunking a clip and crossfading the finished frames — a crossfade
hides seams while global drift survives underneath it.
Choose `context_blend` per shot: `cosine` (default) halves the stride-16 seam penalty
at zero cost, `triangular` keeps the upstream-style taper across the whole window,
`flat` averages overlaps, and `sharp`
limits the taper to the outermost latent frames.
Pair it with `context_stride_frames`: the default 16 gives a 2-latent (8-frame)
overlap for a 21-frame window,
while a smaller stride gives more overlap for shots that need cleaner seams.

Two practical notes before you set frame counts:

**Frame counts must satisfy `num_frames % 4 == 1`.** Upstream rounds *up*, so a
request for 191 becomes 193 and your plate's last frame is repeated twice. Feed
**189** instead — the nearest valid length below 191, an exact 48-latent fit
with no padding. The node reports any rounding it applied.

**Cost scales with window count, not frame count.** At the default
`context_window_frames=21` / `context_stride_frames=16`, 189 frames is 12 windows,
and every denoising step runs all 12. Widening the stride reduces windows and trades
continuity for time. The sampler reports the window count before it starts, so
you see the cost rather than discovering it.

What this does *not* do: each window still only sees its own span, so it gives
seam-free blending and good local continuity, not global awareness of a
189-frame shot. Anchor-frame conditioning is the known further mitigation and is
deliberately not implemented without a measurement to justify it.

## Speed, and what it costs

A 14B model is slow at shot length. For reference, Wan2.1-**1.3B**-based
decomposers run roughly an order of magnitude faster — that gap is the model,
not this pack, and no wrapper closes it.

Two levers, in the order worth reaching for:

**`step_distill_lora = lightx2v`** on the loader merges a step-distillation LoRA
into the DiT base weights, so 4 steps replace 50. Pairs with `cfg_scale = 1.0`,
which the empty-prompt rule already forces for decomposition. **Read the caveat
before trusting the output:** LightX2V was trained on natural video, not on
synthetic decomposition targets, and normals and alpha mattes are the
highest-risk outputs under distillation — which is what this pack exists to
produce. Treat it as an iteration and measurement lever, not a production
default. It is off by default, and the merge reports how many Linears it
touched; if that number is zero it raises rather than pretending.

**Fewer steps without distillation.** `num_inference_steps` defaults to 50,
upstream's reference. Lowering it is the honest, boring lever.

**`attention = sage`** is the sampler default, for resolution rather than steps. Upstream
only knows flash-attn-3 (not installed) or torch's SDPA, and attention is the
term that grows quadratically with pixels. SageAttention — the INT8-quantised
kernel ComfyUI itself runs — is swapped in for the duration of one call by
replacing the vendored DiT module's `F` with a delegating proxy, so upstream's
layout logic uses the selected kernel, including when `shared_kv` is enabled.
Nothing in `vendor/` is edited. It is an
approximation: A/B it against `sdpa` on a short clip before trusting normals
from it. The run reports **how many attention calls actually took the fast
path**, and a kernel failure falls back to SDPA for the rest of the run with one
logged line — never silently. Needs the `sageattention` package in ComfyUI's
Python; if it is missing the sampler refuses rather than quietly running slow.

**`shared_kv`** is on by default and holds cross-modal keys and values once,
instead of copying them for each modality. Each query row attends to the same
K/V with identical attention math, targeting about a third less activation
VRAM and enough headroom for 1088x1920. It works with both `sdpa` and `sage`;
SageAttention remains an approximation. Turn it off to A/B against upstream's
exact tensor layout. The run reports **Shared K/V attention applied on N of M
attention calls** and, on CUDA, peak allocated and reserved VRAM in GiB so the
saving can be measured on your workflow.

## Memory

**Measured recipe (2026-09-12, RTX 5090 32 GB, 704x1248, 21-frame windows):** `attention=sage`,
`shared_kv` on, `num_persistent_param_in_dit` between 0 and 4e9 with SageAttention 1.0.6 (**<= 2e9 with SageAttention 2.2.0**, whose INT8/FP8 working buffers eat the headroom — at 4e9 it pages, measured 2026-09-13) — per-step time is flat across that
range (35.3 → 34.4 s per window) because streaming the whole DiT costs under 2 s per window, while
6e9 and 8e9 fill the card and the driver pages silently (100% "utilisation" at ~110 W, 5-10x slower,
no error). Peak allocated VRAM at cap 0 is 12.6 GiB, which also fits a 24 GB card. Watch the
`Peak VRAM during sampling` line the sampler logs; if it approaches the card, lower the cap.


If you are hitting OOM, reach for the loader's VRAM controls before reducing
quality. They change residency, not weights — so they cost wall-time and not
fidelity, unlike quantisation or distillation.

## VAE tiling

The `tiled` / `tile_size` / `tile_stride` controls affect the VAE **decode**
by default, and default to upstream's reference values. They are a fine
adjustment, not an OOM fix: the DiT weights and sampling activations dominate
memory, and upstream's conditioning path calls `vae.encode` without forwarding
tiling at all. The encoder does stream temporally, so long clips are safe there
— it is raising *resolution* that hits the untiled spatial path.

Strides must be **smaller** than their tile sizes whenever tiling is on: the
blend border is `size − stride`, and upstream's mask builder crashes on a zero
border — on the decode path, only after denoising has finished. The sampler
refuses that combination up front.

**`tiled_encode`** forwards the same tile settings into that conditioning
encode, in the same latent units. Off by default because tiling blends tile
borders into the conditioning latents; turn it on only when the untiled encode
is what runs out of memory, which the traceback will name.

## Writing passes to EXR

Pipe the decoders into [`ComfyUI-Gamut`](../ComfyUI-Gamut). Colour passes go to
`Gamut: Save EXR` with a real colour declaration. **Normals go to
`Gamut: Save Data EXR`**, which writes named data channels and deliberately
carries no `chromaticities` and no `colorSpace` — a surface normal has neither,
and claiming otherwise would be a lie in the header. `Gamut: Normal Decode`
converts `[0, 1]` codes to signed vectors on the way.

The normal coordinate frame and handedness are **unverified** at time of
writing; the Gamut normal nodes default to no correction so that a measurement
can set them rather than a guess.

## Licence

This pack is MIT (see `LICENSE` and `NOTICE`). Upstream UniVidX is vendored at
a pinned commit under `vendor/univid/` and keeps its own Apache-2.0 licence — see
[`vendor/univid/VENDOR.md`](vendor/univid/VENDOR.md) for the pin, what is
included, and how to refresh it. No code from any GPL-licensed third-party
UniVidX wrapper appears here.
