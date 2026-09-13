"""ComfyUI-UniVidX V1 registration; importing the pack needs no runtime dependencies."""

if __package__:
    from .univid import modes
    from .nodes.decoder import UniVidXAlphaDecoder, UniVidXIntrinsicDecoder
    from .nodes.loader import UniVidXLoader
    from .nodes.sampler import UniVidXSampler
    from .nodes.task import UniVidXTask
else:
    # pytest may load a checkout with a hyphenated directory name as __init__.
    from univid import modes
    from nodes.decoder import UniVidXAlphaDecoder, UniVidXIntrinsicDecoder
    from nodes.loader import UniVidXLoader
    from nodes.sampler import UniVidXSampler
    from nodes.task import UniVidXTask


NODE_CLASS_MAPPINGS = {
    "UniVidX_Loader": UniVidXLoader,
    "UniVidX_Task": UniVidXTask,
    "UniVidX_Sampler": UniVidXSampler,
    "UniVidX_IntrinsicDecoder": UniVidXIntrinsicDecoder,
    "UniVidX_AlphaDecoder": UniVidXAlphaDecoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniVidX_Loader": "UniVidX: Load Model",
    "UniVidX_Task": "UniVidX: Select Task",
    "UniVidX_Sampler": "UniVidX: Sampler",
    "UniVidX_IntrinsicDecoder": "UniVidX: Decode Intrinsic",
    "UniVidX_AlphaDecoder": "UniVidX: Decode Alpha",
}

# `modes` is re-exported rather than left as an unused import: it is the one
# module that must import with no runtime dependencies at all, so importing it
# here doubles as a load-time check that the pure layer stayed pure.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "modes"]
