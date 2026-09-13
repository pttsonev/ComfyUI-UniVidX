"""Select an exact upstream mode, carrying its family and required modalities."""

if "." in __package__:
    from ..univid import modes
else:
    from univid import modes


class UniVidXTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(modes.INTRINSIC_MODES + modes.ALPHA_MODES), {
                    "default": "R2AIN",
                    "tooltip": "Intrinsic: R=rgb, A=albedo, I=irradiance, N=normal. "
                    "Alpha: R=rgb, P=matte, F=foreground, B=background. t2 modes use text only.",
                }),
            },
        }

    RETURN_TYPES = ("UNIVIDX_TASK",)
    RETURN_NAMES = ("task",)
    FUNCTION = "select"
    CATEGORY = "UniVidX/Tasks"
    DESCRIPTION = "Choose which modalities to provide and which to generate."

    def select(self, mode="R2AIN"):
        return (modes.get_mode(mode),)
