"""
이 파일은 컴포넌트 간 공유 dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math # pi

@dataclass
class EnvRuntimeSpec:
    num_envs: int
    episode_length_step: int
    max_lookahead_step: int
    hit_window_step: int
    step_dt: float

@dataclass
class PartLength:
    upper_arm: float = 0.230
    lower_arm: float = 0.200
    stick: float = 0.325 + 0.048
    waist: float = 0.520

@dataclass
class RobotSpec:
    part_length: PartLength = field(default_factory=PartLength)

    # 관절 이름과 ik 푼 결과의 인덱스 매칭
    joint_name_to_ik_result_idx: dict = field(default_factory=lambda: {
        "waist_joint":          0,
        "left_shoulder_1":      2,
        "left_shoulder_2":      5,
        "left_elbow":           6,
        "right_shoulder_1":     1,
        "right_shoulder_2":     3,
        "right_elbow":          4,
        "left_wrist":           8,
        "right_wrist":          7,
    })

    # 제어할 관절만
    ctrl_joint_names: list = field(default_factory=lambda: [
        "waist_joint",
        "left_shoulder_1","left_shoulder_2","left_elbow",
        "right_shoulder_1","right_shoulder_2","right_elbow",
        "left_wrist","right_wrist",
    ])

    # wrist link to tip
    tip_offset_left: tuple[float, float, float] = (0.385, 0.0, -0.023)   # [m]
    tip_offset_right: tuple[float, float, float] = (0.385, 0.0, -0.026)  # [m]

    # 관절 제한 범위
    # 실기 관절 가동범위 (deg). 손목은 실측값(-108~35/-108~135)으로 수정 — 기존 -10~90 은 틀림.
    joint_limit: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "waist_joint":          (-89*math.pi/180,    90*math.pi/180),
        "left_shoulder_1":      ( 29*math.pi/180,   175*math.pi/180),
        "left_shoulder_2":      (-59*math.pi/180,    85*math.pi/180),
        "left_elbow":           (  15*math.pi/180,   135.5*math.pi/180),
        "right_shoulder_1":     (  0*math.pi/180,   145*math.pi/180),
        "right_shoulder_2":     (-59*math.pi/180,    85*math.pi/180),
        "right_elbow":          (  15*math.pi/180,   135.5*math.pi/180),
        "left_wrist":           (-29*math.pi/180,  69*math.pi/180),
        "right_wrist":          (-29*math.pi/180,   69*math.pi/180),
    })

    # 로봇 좌표계와 USD 파일의 방향 차이
    joint_usd_dir: dict[str, int] = field(default_factory=lambda: {
        "waist_joint":          +1,
        "left_shoulder_1":      -1,
        "left_shoulder_2":      +1,
        "left_elbow":           +1,
        "right_shoulder_1":     -1,
        "right_shoulder_2":     -1,
        "right_elbow":          -1,
        "left_wrist":           -1,
        "right_wrist":          +1,
    })

@dataclass
class Instrument:
    id: int
    position: tuple[float, float, float]    # 악기의 x, y, z 좌표 (허리 조인트 기준)

@dataclass
class Instruments:
    all: dict[str, Instrument] = field(default_factory=lambda: {
        "snare":   Instrument(1, (-0.100,  0.361,  -0.480)),
        "floor":   Instrument(2, ( 0.232,  0.359,  -0.485)),
        "mid":     Instrument(3, ( 0.216,  0.597,  -0.378)),
        "high":    Instrument(4, (-0.069,  0.607,  -0.321)),
        "hihat":   Instrument(5, (-0.292,  0.493,  -0.224)),
        "ride":    Instrument(6, ( 0.326,  0.644,  -0.146)),
        "crash_r": Instrument(7, ( 0.485,  0.424,  -0.249)),
        "crash_l": Instrument(8, (-0.184,  0.669,  -0.147)),
    })

    drum_noise_scale: float = 0.02
