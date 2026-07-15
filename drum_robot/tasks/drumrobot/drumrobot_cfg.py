from __future__ import annotations
                                                    # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
from isaaclab.assets import ArticulationCfg         # pyright: ignore[reportMissingImports]
from isaaclab.envs import DirectRLEnvCfg            # pyright: ignore[reportMissingImports]
from isaaclab.scene import InteractiveSceneCfg      # pyright: ignore[reportMissingImports]
from isaaclab.sim import SimulationCfg              # pyright: ignore[reportMissingImports]
from isaaclab.utils import configclass              # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils                    # pyright: ignore[reportMissingImports]
from isaaclab.actuators import ImplicitActuatorCfg  # pyright: ignore[reportMissingImports]

import math # pi
from dataclasses import dataclass, field

USD_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/assets/drum_robot/usd/drum_robot.usd"
PRIM_PATH = "/World/envs/env_.*/Robot"   # env_.*를 써야 수백 개의 환경에 복제됩니다.

# sim
SIM_DT = 1.0 / 120.0
DEVICE = "cuda"

@configclass
class DrumRobotEnvCfg(DirectRLEnvCfg):

    """ 기본 환경 설정 """
    decimation: int = 2             # 정책(Policy) 업데이트 한 번당 시뮬레이션 스텝 수
    episode_length_s: float = 5.0   # 에피소드 최대 길이 (초)
    action_space: int = 9           # 로봇 제어 차원
    observation_space: int = 96     # 관측 차원 (+2: 팔별 역할 active/idle)
    state_space: int = 0
    action_scale: float = math.pi        # 기본 관절 지령 각속도 상한
    wrist_action_scale: float = 25.0      # 손목만 상한 크게 (빠른 패시지 ~24 rad/s 커버)

    # 관측값 정규화
    joint_vel_scale: float = 5.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT, 
        device=DEVICE, 
        render_interval=decimation
    )

    # robot (s)
    robot_waist_joint_to_link_z: float = 0.0755
    robot_waist_joint_offset_z: float = 1.0

    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot_waist_joint_offset_z+robot_waist_joint_to_link_z), 
            joint_pos={"(.*)": 0.0}, 
        ),
        prim_path=PRIM_PATH,
        # 실기(motors.json + 데이터시트) 기준 관절별 actuator.
        #   waist: TMotor AK10_9 (gear 9),  arm: TMotor AK70_10 (gear 10),  wrist: Maxon DCX22L+GPX22HP (gear 35)
        #   팔 stiffness/damping은 실기 TMotor 서보(control_gain) → 단순 PD 환산 불가라 위치추종용 값 유지.
        #   effort/velocity_limit는 데이터시트 근사(팔은 T-Motor/CubeMars 공개값, 손목은 Maxon 데이터시트 확정값).
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint"],
                stiffness=200.0,
                damping=10.0,
                effort_limit=48.0,      # AK10_9 peak torque ~48 N·m (근사, 확인 권장)
                velocity_limit=28.0,    # AK10_9 no-load ~28 rad/s (근사)
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_1", "left_shoulder_1",
                    "right_shoulder_2", "left_shoulder_2",
                    "right_elbow", "left_elbow",
                ],
                stiffness=200.0,
                damping=10.0,
                effort_limit=24.0,      # AK70_10 peak torque ~24.8 N·m (current_limit 23.2A와 정합)
                velocity_limit=46.0,    # AK70_10 no-load ~46 rad/s (근사)
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["right_wrist", "left_wrist"],
                stiffness=2.1,          # 실기 CST: kp(60) × gear(35) — 컴플라이언트 (기존 균일 200과 정반대)
                damping=0.035,          # 실기: kd(1) × gear(35)
                effort_limit=2.5,       # GPX22HP 단속 허용토크 2.4 N·m (기어헤드가 병목; 모터 stall 294mNm보다 낮음)
                velocity_limit=30.0,    # 기어 최대 입력 10000rpm / 35 ≈ 30 rad/s (연속 8000rpm→24)
                armature=1.1e-3,        # 반사 로터관성 = I_rotor(8.85 gcm²) × gear²(35²) ≈ 1.08e-3 kg·m²
            ),
        },
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
