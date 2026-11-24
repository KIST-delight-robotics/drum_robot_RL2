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
SIM_DT = 0.005     # 5ms x decimation 3 = 정책 15ms (66.7Hz)
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
    decimation: int = 3             # 정책(Policy) 업데이트 한 번당 시뮬레이션 스텝 수
    episode_length_s: float = 5.0   # 에피소드 최대 길이 (초)
    action_space: int = 9           # 로봇 제어 차원
    observation_space: int = 90    # 관측 차원 (+ 목표 어깨 yaw 속도 θ̇1_d, θ̇2_d 2)
    state_space: int = 0

    # 행동 -> 관절 목표: 목표 = 직전 목표 + a x scale x step_dt (직전 목표에 누적)
    action_scale: float = 13.0          # 팔·허리 목표 이동 속도 상한 [rad/s]
    wrist_action_scale: float = 20.0    # 손목 목표 이동 속도 상한 [rad/s]

    # 제어 지연 [물리 스텝]. 
    action_delay_steps_min: int = 1     
    action_delay_steps_max: int = 1

    # 엔코더 해상도 (관측에만 적용, 보상은 실제 값): 실기 MIT 피드백 한 칸 크기. 0이면 끔
    obs_pos_res: float = 25.0 / 65535       # [rad] TMotor 위치 16비트, ±12.5 rad
    obs_vel_res: float = 100.0 / 4095       # [rad/s] TMotor 속도 12비트, ±50 rad/s
    wrist_pos_res: float = 6.283185 / (35 * 4096)   # [rad] Maxon 엔코더 4096 × 기어 35
    cmd_pos_res: float = 25.0 / 65535       # [rad] TMotor MIT 위치 명령 16비트 (버림). 0이면 끔
    mit_p_limit: float = 12.5               # [rad] motors.json mit_p_limit
    mit_v_limit: float = 50.0               # [rad/s] motors.json mit_v_limit
    real_dir_sign: tuple = (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0)    # motors.json direction_sign (모터 id 0~8)
    real_init_deg: tuple = (10.0, 90.0, 90.0, 0.0, 90.0, 0.0, 90.0, 90.0, 90.0)  # motors.json initial_joint_angle [deg] (모터 id 0~8)

    # 관측값 정규화
    joint_vel_scale: float = 5.0
    obs_vel_noise_std: float = 0.06     # [rad/s] 관측 속도 가우시안 잡음 표준편차 (보상은 실제 값). 0이면 끔
    obs_pos_noise_std: float = 1.5e-4   # [rad] 관측 위치 가우시안 잡음 표준편차 (보상은 실제 값). 0이면 끔
    # waist_dist_std: float = 0.5         # [N m] 허리 외란 토크 표준편차 (평균 0). 0이면 끔
    waist_dist_std: float = 0.2         # [N m] 허리 외란 토크 표준편차 (평균 0). 0이면 끔
    yaw_dist_std: float = 0.05          # [N m] 어깨 yaw 외란 토크 표준편차 (평균 0)
    waist_dist_hold_s: float = 0.5      # [s] 평균 몇 초마다 외란을 새로 뽑을지 (env마다 따로)

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
                # enabled_self_collisions=False,
                enabled_self_collisions=True,       # 자기 충돌 켜기 (교차 타격 때 팔·손목·스틱끼리 부딪힘)
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
                effort_limit_sim=36.0,      # AK10_9 peak torque 약 48 N m
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
                effort_limit_sim=18.0,      # AK70_10 peak torque 약 24.8 N m (current_limit 23.2A 와 정합)
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
