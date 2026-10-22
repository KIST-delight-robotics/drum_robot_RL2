from __future__ import annotations
                                                    # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg  # pyright: ignore[reportMissingImports]
from isaaclab.envs import DirectRLEnvCfg            # pyright: ignore[reportMissingImports]
from isaaclab.scene import InteractiveSceneCfg      # pyright: ignore[reportMissingImports]
from isaaclab.sensors import ContactSensorCfg       # pyright: ignore[reportMissingImports]
from isaaclab.sim import SimulationCfg, PhysxCfg    # pyright: ignore[reportMissingImports]
from isaaclab.utils import configclass              # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils                    # pyright: ignore[reportMissingImports]
from isaaclab.actuators import ImplicitActuatorCfg  # pyright: ignore[reportMissingImports]

import os
from dataclasses import dataclass, field

from .components.specs import Instruments

USD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "drum_robot", "usd", "drum_robot.usd")
PRIM_PATH = "/World/envs/env_.*/Robot"   # env_.*를 써야 수백 개의 환경에 복제됩니다.

# sim
SIM_DT = 0.0075     # 7.5ms (물리 133.3Hz) x decimation 2 = 정책 15ms (66.7Hz)
DEVICE = "cuda"

# drum (순서 = 드럼 인덱스)
DRUM_PATHS = [f"/World/envs/env_.*/Drum_{name}" for name in Instruments().all]

def _drums_cfg(waist_z: float) -> RigidObjectCollectionCfg:
    # 클래스 본문 안 comprehension에서는 클래스 변수를 못 읽어서 함수로 분리
    objs = {}
    for name, inst in Instruments().all.items():
        x, y, z = inst.position
        objs[name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Drum_{name}",
            spawn=sim_utils.CylinderCfg(
                radius=inst.radius,
                height=inst.thickness,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),                  # 맞아도 안 움직이는 고정 강체
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0), # 5cm 전부터 접촉 감시 (빠른 타격 터널링 방지)
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    restitution=sum(inst.restitution_range) / 2,        # 임시값, env __init__에서 랜덤으로 덮어씀
                    static_friction=sum(inst.friction_range) / 2,       # 임시값, env __init__에서 랜덤으로 덮어씀
                    dynamic_friction=sum(inst.friction_range) / 2,
                    restitution_combine_mode="max",                     # 스틱과 접촉 시 드럼 값 사용
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            ),
            # 윗면이 타면 위치가 되도록 두께 절반만큼 내림
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z + waist_z - inst.thickness / 2)),
        )
    return RigidObjectCollectionCfg(rigid_objects=objs)

@configclass
class DrumRobotEnvCfg(DirectRLEnvCfg):

    """ 기본 환경 설정 """
    decimation: int = 2             # 정책(Policy) 업데이트 한 번당 시뮬레이션 스텝 수
    episode_length_s: float = 5.0   # 에피소드 최대 길이 (초)
    action_space: int = 9           # 로봇 제어 차원
    observation_space: int = 138    # 관측 차원 (관절 9 + 관절속도 9 + 목표 앞섬 9 + 팁 6 + 드럼 24 + 다음 타격 60 + 손별 다음 음표 18 + 정렬 오차 3)
    state_space: int = 0

    # 행동 -> 관절 목표: 목표 = 직전 목표 + a x scale x step_dt (직전 목표에 누적)
    action_scale: float = 13.0          # 팔·허리 목표 이동 속도 상한 [rad/s]
    wrist_action_scale: float = 70.0    # 손목 목표 이동 속도 상한 [rad/s]

    # 관측값 정규화
    joint_vel_scale: float = 5.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        device=DEVICE,
        render_interval=decimation,
        physx=PhysxCfg(bounce_threshold_velocity=0.1),  # 기본 0.5m/s 이하 충돌은 반발 무시 → 낮춤
        # 장면 기본 재질: 재질이 없는 스틱 캡슐·로봇 링크가 사용
        physics_material=sim_utils.RigidBodyMaterialCfg(
            restitution=0.0,
            restitution_combine_mode="max",     # max(드럼 e, 0) = 드럼 e
            # 마찰
            static_friction=2.0,
            dynamic_friction=2.0,
            friction_combine_mode="min",
        ),
    )

    # robot (s)
    robot_waist_joint_to_link_z: float = 0.0755
    robot_waist_joint_offset_z: float = 1.0

    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=True,      # 로봇 강체에 PhysX 접촉 보고 켜기 (ContactSensor 필요)
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
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint"],
                stiffness=80.0,         # 실기 MIT Kp
                damping=4.0,            # 실기 MIT Kd (상한 5)
                effort_limit_sim=48.0,      # AK10_9 peak torque 약 48 N m
                velocity_limit_sim=28.0,    # AK10_9 no-load 약 28 rad/s
                armature=0.01,          
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_1", "left_shoulder_1",
                    "right_shoulder_2", "left_shoulder_2",
                    "right_elbow", "left_elbow",
                ],
                stiffness=80.0,         # 실기 MIT Kp
                damping=4.0,            # 실기 MIT Kd (상한 5)
                effort_limit_sim=24.0,      # AK70_10 peak torque 약 24.8 N m (current_limit 23.2A 와 정합)
                velocity_limit_sim=46.0,    # AK70_10 no-load 약 46 rad/s
                armature=0.01,          
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["right_wrist", "left_wrist"],
                stiffness=1.4,          # 실기 CST kp(40) x gear(35) / 1000
                damping=0.14,           # 실기 CST kd(4) x gear(35) / 1000
                effort_limit_sim=2.5,       # GPX22HP 단속 허용토크 2.4 N m (기어헤드가 병목)
                velocity_limit_sim=30.0,
                armature=1.1e-3,        
            ),
        },
    )

    # drum
    drums_cfg: RigidObjectCollectionCfg = _drums_cfg(robot_waist_joint_offset_z)

    # 스틱(손목 링크) 접촉 센서: 드럼별 접촉력(force_matrix)은 센서당 prim 1개일 때만 정확해서 좌/우 분리
    contact_L_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/drumrobot_RL_urdf/left_wrist",
        filter_prim_paths_expr=DRUM_PATHS,  # 드럼별 접촉력을 따로 받음
        history_length=decimation,          # 정책 스텝 안의 물리 스텝 모두 기록
        track_contact_points=True,          # 실제 접촉 지점 (팁/몸통 구분용, 마지막 물리 스텝 것만 제공)
        max_contact_data_count_per_prim=16, # 접촉 지점 버퍼 크기
    )
    contact_R_cfg: ContactSensorCfg = contact_L_cfg.replace(
        prim_path="/World/envs/env_.*/Robot/drumrobot_RL_urdf/right_wrist",
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
