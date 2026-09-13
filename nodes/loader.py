"""Model handle node over the absolute-path loader."""

if "." in __package__:
    from ..univid import loader
else:
    from univid import loader


_DISTILLATION_TOOLTIP = (
    "LightX2V was trained on natural video, not synthetic decomposition targets. "
    "The third-party pack reports about 22-26 dB PSNR against BF16 and rates Normal "
    "and Alpha mattes as the HIGHEST-RISK outputs. Use with 4 steps and cfg_scale 1.0 "
    "(the sampler's empty-prompt rule already forces CFG 1.0). "
    "An iteration and measurement lever, never a production default. "
    "Changing this rebuilds the model; the merged base cannot be un-merged."
)
_VRAM_TOOLTIP = (
    "vram_limit and num_persistent_param_in_dit are mutually exclusive. "
    "Lowering residency costs WALL TIME, not quality: weights and compute dtype are unchanged. "
    "Try num_persistent_param_in_dit first to trade a little speed for a lot of headroom; "
    "use vram_limit when you need the absolute VRAM floor. T5 already streams from CPU."
)


_AUTO = "auto (discover)"


def _model_choices(category: str, label: str):
    """A dropdown of what is actually on disk, with discovery as the default.

    ComfyUI loaders offer a list, not a text box: a blank field is neither
    discoverable nor obviously working. The list is built at schema time, when
    folder_paths is available inside ComfyUI; outside it — tests, a bare import
    — there is nothing to enumerate, so this degrades to a free-text path rather
    than presenting an empty menu.

    `auto (discover)` keeps the resolver's own search, which identifies models
    by content rather than filename and can therefore pick correctly out of a
    folder holding several look-alikes.
    """
    try:
        if "." in __package__:
            from ..univid import paths
        else:
            from univid import paths
        registry = paths._get_folder_paths()
        if registry is None:
            raise RuntimeError("no registry")
        if category == "unividx":
            paths.register_model_folder()
        names = list(registry.get_filename_list(category))
    except Exception:
        return ("STRING", {
            "default": "",
            "tooltip": f"Path to the {label}. Leave blank to discover it automatically.",
        })
    return ([_AUTO, *names], {
        "default": _AUTO,
        "tooltip": (
            f"Which {label} to use. '{_AUTO}' searches the registered folders and "
            "verifies the file by its content rather than its name."
        ),
    })


class UniVidXLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (["intrinsic", "alpha"], {"default": "intrinsic"}),
                "compute_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "vram_buffer": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "step": 0.1,
                    "tooltip": (
                        "VRAM reserve in GiB; 0.5 is upstream's default. Subtracted from "
                        "vram_limit (or total VRAM when unset). Ignored with a persistent "
                        "parameter cap. " + _VRAM_TOOLTIP
                    ),
                }),
                **{
                    name: _model_choices(category, name)
                    for name, category in (
                        ("dit", "diffusion_models"), ("vae", "vae"),
                        ("text_encoder", "text_encoders"), ("checkpoint", "unividx"),
                    )
                },
                "distillation": (["none", "lightx2v"], {
                    "default": "none", "tooltip": _DISTILLATION_TOOLTIP,
                }),
                "distillation_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": _DISTILLATION_TOOLTIP,
                }),
                "vram_limit": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "step": 0.1,
                    "tooltip": (
                        "CPU streaming budget in GiB, before subtracting vram_buffer. "
                        "0 means unset (upstream uses total VRAM). Lower this for minimum "
                        "residency at greater wall-time cost. " + _VRAM_TOOLTIP
                    ),
                }),
                "num_persistent_param_in_dit": ("INT", {
                    "default": 2_000_000_000, "min": -1, "max": 2 ** 53 - 1, "step": 1,
                    "tooltip": (
                        "2e9 = measured recipe on a 32 GB card with SageAttention 2.2.0 "
                        "(never 4e9 with 2.2.0: it pages); 0 streams the whole DiT "
                        "(12.6 GiB peak, ~+1 s/window); raise only on cards with real headroom "
                        "(96 GB: full residency). Mutually exclusive with vram_limit."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("UNIVIDX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "UniVidX/Models"
    DESCRIPTION = "Load or reuse an intrinsic or alpha model from ComfyUI's model folders."

    def load(
        self, variant="intrinsic", compute_dtype="bfloat16", vram_buffer=0.5,
        dit=_AUTO, vae=_AUTO, text_encoder=_AUTO, checkpoint=_AUTO,
        distillation="none", distillation_strength=1.0,
        vram_limit=0.0, num_persistent_param_in_dit=2_000_000_000,
    ):
        # Both the dropdown's discovery sentinel and an empty text field (the
        # fallback outside ComfyUI) mean "let the resolver find it".
        overrides = {
            name: (None if value.strip() in ("", _AUTO) else value.strip())
            for name, value in (
                ("dit", dit), ("vae", vae), ("text_encoder", text_encoder), ("checkpoint", checkpoint),
            )
            if value is not None
        }
        return (loader.load_model(
            variant, compute_dtype=compute_dtype, vram_buffer=vram_buffer,
            vram_limit=vram_limit, num_persistent_param_in_dit=num_persistent_param_in_dit,
            distillation=distillation, distillation_strength=distillation_strength, **overrides,
        ),)
