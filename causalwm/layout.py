"""Stream registry of the causal chain: RGB0/text -> flow -> XYZ pointmap -> future RGB.

Pointmap frame 0 is generated, not supplied as a condition.
"""

from __future__ import annotations

from dataclasses import dataclass

from causalwm.flow_codec import FLOW_CODEC

FULL_SCENE_SCHEMA = "full_scene_flow_pointmap_v1"
INPUT_CONTRACT = "rgb0_text_only"
# Tag of the signed-log pointmap codec (see pointmap_codec.decode_pointmap_log).
POINTMAP_CODEC = "pointmap_camera_jointdepth_signedlog_xy16_z32_v1"
COORDINATE_CONVENTION = "camera_t_xyz_joint_frame0_median_depth"


@dataclass(frozen=True)
class StreamDef:
    name: str
    stream_id: int
    stage: int
    codec: str


STREAMS: tuple[StreamDef, ...] = (
    StreamDef("flow", 0, 0, FLOW_CODEC),
    StreamDef("pointmap", 1, 1, POINTMAP_CODEC),
)
STREAM_NAMES = tuple(stream.name for stream in STREAMS)
NUM_STREAMS = len(STREAMS)


def registry_metadata() -> dict[str, str]:
    """Checkpoint metadata that identifies this registry (strings only, safe for safetensors)."""
    return {
        "robot_cot_schema": FULL_SCENE_SCHEMA,
        "robot_cot_streams": ",".join(f"{s.name}:{s.stream_id}:{s.stage}:{s.codec}" for s in STREAMS),
        "input_contract": INPUT_CONTRACT,
        "pointmap_frame0": "generated",
        "pointmap_coordinate_convention": COORDINATE_CONVENTION,
    }


def arch_from_metadata(metadata: dict[str, str] | None) -> tuple[bool, bool]:
    """(modality_heads, modality_adaln) implied by checkpoint metadata; absent = (False, False)."""
    metadata = metadata or {}
    return metadata.get("robot_cot_modality_heads") == "1", metadata.get("robot_cot_modality_adaln") == "1"
