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
    hit_window_step: int        # 목표 매칭 범위 최댓값 [스텝] (HitDetectorCfg.max_match_window_s / step_dt)
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
    stick_radius: float = 0.0075        # 스틱 캡슐 충돌체 반지름 [m] (팁 = 끝 반구, 스틱 끝에서 이 길이까지)

    # 관절 제한 범위
    joint_limit: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "waist_joint":          (-90*math.pi/180,    90*math.pi/180),
        "left_shoulder_1":      ( 30*math.pi/180,   180*math.pi/180),
        "left_shoulder_2":      (-60*math.pi/180,    90*math.pi/180),
        "left_elbow":           (  0*math.pi/180,   140*math.pi/180),
        "right_shoulder_1":     (  0*math.pi/180,   150*math.pi/180),
        "right_shoulder_2":     (-60*math.pi/180,    90*math.pi/180),
        "right_elbow":          (  0*math.pi/180,   140*math.pi/180),
        "left_wrist":           (-90*math.pi/180,    100*math.pi/180),
        "right_wrist":          (-90*math.pi/180,    100*math.pi/180),
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
    position: tuple[float, float, float]                    # 타면(윗면) 중심 x, y, z 좌표 (허리 조인트 기준)
    radius: float = 0.13                                    # 헤드 반지름 [m]
    thickness: float = 0.015                                # 헤드 원판 두께 [m]
    restitution_range: tuple[float, float] = (0.5, 0.7)     # 반발계수 랜덤 범위
    friction_range: tuple[float, float] = (0.3, 0.5)        # 마찰계수 랜덤 범위 (정지·운동 마찰 같은 값)
    zone: str = "center"                                    # 위치 점수 모양: "center" = 중심일수록 좋음 (북), "edge" = 로봇 쪽 가장자리 안쪽 (심벌)
    edge_margin: float = 0.02                               # zone="edge"일 때 최고점이 가장자리에서 안쪽으로 떨어진 거리 [m]

# 마찰계수 (나무 팁 기준)
MESH_FRICTION = (0.3, 0.5)      # 나무 <-> 메시 헤드
RUBBER_FRICTION = (0.6, 1.0)    # 나무 <-> 고무 패드

# dict 순서 = 드럼 인덱스 (드럼 강체, 접촉 센서 필터, 반발계수·마찰 텐서 모두 이 순서)
@dataclass
class Instruments:
    all: dict[str, Instrument] = field(default_factory=lambda: {
        "snare":   Instrument(1, (-0.100,  0.361,  -0.480), radius=0.152, restitution_range=(0.70, 0.85), friction_range=MESH_FRICTION),     # 메시 스네어 (타이트), PDA120LS 12"
        "floor":   Instrument(2, ( 0.232,  0.359,  -0.485), radius=0.152, restitution_range=(0.50, 0.70), friction_range=MESH_FRICTION),     # 메시 탐 (느슨하면 반발 0.40~0.60), PDA120L 12"
        "mid":     Instrument(3, ( 0.216,  0.597,  -0.378), radius=0.127, restitution_range=(0.50, 0.70), friction_range=MESH_FRICTION),     # 메시 탐, PDA100L 10"
        "high":    Instrument(4, (-0.069,  0.607,  -0.321), radius=0.127, restitution_range=(0.50, 0.70), friction_range=MESH_FRICTION),     # 메시 탐, PDA100L 10"
        "hihat":   Instrument(5, (-0.292,  0.493,  -0.224), radius=0.152, restitution_range=(0.40, 0.60), friction_range=RUBBER_FRICTION, zone="edge"),   # 하이햇 패드 (고무, 닫힘), VH-10 12"
        "ride":    Instrument(6, ( 0.326,  0.644,  -0.146), radius=0.178, restitution_range=(0.30, 0.50), friction_range=RUBBER_FRICTION, zone="edge"),   # 고무 심벌 패드 (보우), 14"
        "crash_r": Instrument(7, ( 0.485,  0.424,  -0.249), radius=0.152, restitution_range=(0.30, 0.50), friction_range=RUBBER_FRICTION, zone="edge"),   # 고무 심벌 패드 (보우), 12"
        "crash_l": Instrument(8, (-0.184,  0.669,  -0.147), radius=0.152, restitution_range=(0.30, 0.50), friction_range=RUBBER_FRICTION, zone="edge"),   # 고무 심벌 패드 (보우), 12"
    })

    drum_noise_scale: float = 0.02
