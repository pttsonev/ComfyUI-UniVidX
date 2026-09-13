"""Decode generated passes and carried conditions into aligned float32 IMAGEs."""

if "." in __package__:
    from ..univid import modes, tensors
else:
    from univid import modes, tensors


def _decode(result, *, family: str, modalities: tuple[str, ...]):
    mode = modes.get_mode(result.mode.name)
    if result.mode.family != mode.family or mode.family != family:
        raise ValueError(f"UniVidX mode {mode.name!r} is not an {family} decoder result.")
    images = []
    for modality in modalities:
        if modality in mode.conditions:
            video = result.conditions.get(modality)
            if video is None:
                raise ValueError(f"UniVidX mode {mode.name!r} is missing carried condition {modality!r}.")
            if video.ndim != 5 or video.shape[0] != 1:
                raise ValueError(
                    f"UniVidX mode {mode.name!r} condition {modality!r} must have shape [1,3,T,H,W]."
                )
            video = video[0]
        else:
            key = modes.RESULT_KEYS["N"] if modality == "normal" else modality
            video = result.outputs.get(key)
            if video is None:
                raise ValueError(f"UniVidX mode {mode.name!r} is missing generated result {key!r}.")
        image = tensors.video_to_image(video)
        if tuple(image.shape) != result.shape:
            raise ValueError(
                f"UniVidX mode {mode.name!r} modality {modality!r} has shape {tuple(image.shape)}; "
                f"expected fitted shape {result.shape}."
            )
        images.append(image)
    return tuple(images)


class UniVidXIntrinsicDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": ("UNIVIDX_RESULT",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb", "albedo", "irradiance", "normal")
    FUNCTION = "decode"
    CATEGORY = "UniVidX/Decode"
    DESCRIPTION = "Return aligned intrinsic passes. Normal output uses [0,1] IMAGE codes."

    def decode(self, result):
        return _decode(result, family="intrinsic", modalities=self.RETURN_NAMES)


class UniVidXAlphaDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": ("UNIVIDX_RESULT",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb", "pha", "fgr", "bgr")
    FUNCTION = "decode"
    CATEGORY = "UniVidX/Decode"
    DESCRIPTION = "Return aligned composite, matte, foreground and background IMAGE passes."

    def decode(self, result):
        return _decode(result, family="alpha", modalities=self.RETURN_NAMES)
