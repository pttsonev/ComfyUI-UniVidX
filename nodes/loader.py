"""Model handle node over the absolute-path loader."""

if "." in __package__:
    from ..univid import loader, paths
else:
    from univid import loader, paths


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


def _model_choices(category: str, label: str, *, tokenizer=False, lora=False):
    """Explicit registry choices in ComfyUI, free-text paths on a bare import."""
    options = {
        "tooltip": _DISTILLATION_TOOLTIP if lora else f"Select the {label}; an explicit path is required.",
    }
    if lora:
        options["default"] = "none"
    try:
        registry = paths._get_folder_paths()
        if registry is None:
            raise RuntimeError("no registry")
        if category == "unividx":
            paths.register_model_folder()
        names = paths.tokenizer_choices() if tokenizer else list(registry.get_filename_list(category))
        if category == "unividx" and not tokenizer:
            # The unividx folder is registered without an extension filter and
            # also holds the tokenizer and `hf download` .cache stubs; the
            # checkpoint is only ever loaded as safetensors.
            names = [
                name for name in names
                if name.endswith(".safetensors")
                and not any(part.startswith(".") for part in name.replace("\\", "/").split("/"))
            ]
    except Exception:
        return ("STRING", {"default": "", **options})
    return (["none", *names] if lora else names, options)


class UniVidXLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (["intrinsic", "alpha"], {"default": "intrinsic"}),
                "compute_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                **{
                    name: _model_choices(category, name)
                    for name, category in (
                        ("dit", "diffusion_models"), ("vae", "vae"),
                        ("text_encoder", "text_encoders"), ("checkpoint", "unividx"),
                    )
                },
                "tokenizer": _model_choices("unividx", "tokenizer directory", tokenizer=True),
                "distillation_lora": _model_choices("loras", "LightX2V-style LoRA", lora=True),
                "distillation_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": _DISTILLATION_TOOLTIP,
                }),
                "vram_buffer": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "step": 0.1,
                    "tooltip": (
                        "VRAM reserve in GiB; 0.5 is upstream's default. Subtracted from "
                        "vram_limit (or total VRAM when unset). Ignored with a persistent "
                        "parameter cap. " + _VRAM_TOOLTIP
                    ),
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
        self, variant, compute_dtype, dit, vae, text_encoder, checkpoint, tokenizer,
        distillation_lora="none", distillation_strength=1.0,
        vram_buffer=0.5, vram_limit=0.0, num_persistent_param_in_dit=2_000_000_000,
    ):
        return (loader.load_model(
            variant, compute_dtype=compute_dtype, vram_buffer=vram_buffer,
            dit=dit, vae=vae, text_encoder=text_encoder, checkpoint=checkpoint, tokenizer=tokenizer,
            vram_limit=vram_limit, num_persistent_param_in_dit=num_persistent_param_in_dit,
            distillation_lora=None if distillation_lora == "none" else distillation_lora,
            distillation_strength=distillation_strength,
        ),)
