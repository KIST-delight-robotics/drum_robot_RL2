from __future__ import annotations
                                                    # pyright ignore 는 패키지 누락이 아니라 Isaac Sim이 런타임에 import path를 추가하는 구조 때문
from isaaclab.assets import ArticulationCfg         # pyright: ignore[reportMissingImports]
from isaaclab.envs import DirectRLEnvCfg            # pyright: ignore[reportMissingImports]
from isaaclab.scene import InteractiveSceneCfg      # pyright: ignore[reportMissingImports]
from isaaclab.sim import SimulationCfg              # pyright: ignore[reportMissingImports]
from isaaclab.utils import configclass              # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils                    # pyright: ignore[reportMissingImports]
from isaaclab.actuators import ImplicitActuatorCfg  # pyright: ignore[reportMissingImports]

from dataclasses import dataclass, field

USD_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/assets/drum_robot/usd/drum_robot.usd"
PRIM_PATH = "/World/envs/env_.*/Robot"   # env_.* 를 써야 여러 환경에 복제된다

# sim
# step_dt = SIM_DT x decimation = 0.0075 x 2 = 0.015s = 66.667Hz (정책 주기)
# 실기 send_loop 가 200Hz(5ms)이고 정책은 그 정수 분주여야 하므로 200 / 66.667 = 3틱으로 맞춘다.
# 이 값을 바꾸면 action_scale 도 반비례로 바꿔야 한다 (아래 action_scale 주석 참조).
# 주기를 ㄷ바꿀 때는 fresh 가 아니라 resume 으로 학습한다 (타격을 아는 정책에서 출발해야 탐색 폭 변화의 영향이 없다).
SIM_DT = 0.0075
DEVICE = "cuda"

@configclass
class DrumRobotEnvCfg(DirectRLEnvCfg):

    """ 기본 환경 설정 """
    decimation: int = 2             # 정책(Policy) 업데이트 한 번당 시뮬레이션 스텝 수
    episode_length_s: float = 5.0   # 에피소드 최대 길이 (초)
    action_space: int = 9           # 로봇 제어 차원
    observation_space: int = 148    # 관측 차원. K=6 글로벌 lookahead(66) + 팔별 담당 2타(16) + 나머지(66)
    state_space: int = 0

    # MIDI 커리큘럼: 성공률(EMA) 기반 점진 램프. 기본 타격을 익힌 뒤 실제 MIDI 를 섞는다.
    # 세기 게이팅(0.7)보다 먼저 시작하도록 문턱을 어긋나게 둔다.
    midi_success_start: float = 0.65  # 이 성공률(EMA)부터 MIDI 섞기 시작
    midi_success_full: float = 0.90   # 이 성공률에서 final_ratio 도달
    midi_final_ratio: float = 0.65    # 최종 MIDI 비율 (나머지는 랜덤 악보)
    midi_selection_strength: float = 0.5       # MIDI 샘플링 스코어 가중 (0=완전랜덤, 클수록 스코어 선별)

    # 평가(play1.py)용. 기본값이면 학습과 동일. play1.py 가 오버라이드한다
    eval_midi_ratio: float = -1.0   # 0 이상이면 커리큘럼 무시하고 MIDI 비율 고정 (1.0=순수 MIDI)
    eval_midi_file: str = ""        # 지정 시 이 악보 파일만 로드 (.mid 또는 실기 서버 형식 .txt)
    eval_full_song: bool = False    # True 면 파일 전체를 1개 긴 악보로
    eval_slow_factor: float = 1.0   # 악보 시간 배율. 1.0=원곡 템포. 파일 악보에만 적용 (랜덤 악보 무관)

    # 배정 히스테리시스: h0 담당팔을 바꾸려면 상대 팔 점수가 이 비율만큼 더 좋아야 한다.
    #   배정 점수는 스케일 차이가 커서 절대 마진 대신 상대(비율) 마진을 쓴다.
    sched_hysteresis: float = 0.25

    # 교차 배정 페널티 [배율]. 왼팔이 우측(x>0) 드럼, 오른팔이 좌측(x<0) 드럼을 맡을 때
    #   반대편 침범량[m] x 이 값을 추가 이동거리로 과금한다.
    sched_cross_penalty: float = 1.5

    # 지령 각속도 상한 [rad/s]. 물리적으로 의미 있는 값은 action_scale x step_dt = dq_max
    action_scale: float = 13      # 기본 관절 지령 각속도 상한 [rad/s]
    wrist_action_scale: float = 70    # 손목만 상한 크게 (세기 f 밴드 + 빠른 패시지)

    # 제어 지연 [물리 스텝]. 1스텝 = SIM_DT = 7.5ms. 0 이면 지연 없음.
    action_delay_steps_min: int = 1      # 0 이면 지연 없음. min == max 이면 고정 지연
    action_delay_steps_max: int = 2      # env 별로 [min, max] 균등 추첨 (리셋마다)

    # 리바운드 주입. 드럼은 좌표점 그대로 두고, 팁이 헤드 면을 아래로 통과한 스텝에 그 팔의 손목 각속도를 -e 배로 반전한다.
    rebound_enabled: bool = False
    rebound_e_drum: float = 0.7          # 스네어, 탐 3개
    rebound_e_cymbal: float = 0.4        # hihat, ride, crash_r, crash_l
    rebound_e_jitter: float = 0.15        # 학습 DR 폭. 0.1 이면 e 가 [e-0.1, e+0.1] 균등 (리셋마다)
    rebound_plane_offset: float = 0.0     # 헤드 면 높이 = drum_z + 이 값 [m]. 실기 헤드가 시뮬 드럼 원점보다 낮으면 음수
    rebound_xy_radius: float = 0.13       # 면 통과를 인정하는 XY 반경 [m]. 판정기 drum_xy_radius 와 동일값 유지

    # 관측값 정규화
    joint_vel_scale: float = 5.0

    # [sim2real 점검] 손목 속도 obs 를 실기와 같은 유한차분 (q_t - q_{t-1})/step_dt 로 대체할지.
    wrist_vel_finite_diff: bool = False

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
        #   waist: TMotor AK10_9 (gear 9),  arm: TMotor AK70_10 (gear 10),  wrist: Maxon DCX22L+GPX22HP (gear 35)
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint"],
                stiffness=80.0,         # 실기 MIT Kp
                damping=4.0,            # 실기 MIT Kd (상한 5)
                effort_limit=48.0,      # AK10_9 peak torque 약 48 N m
                velocity_limit=28.0,    # AK10_9 no-load 약 28 rad/s
                armature=0.01,          # 반사 로터관성 추정 = I_rotor x gear^2
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_1", "left_shoulder_1",
                    "right_shoulder_2", "left_shoulder_2",
                    "right_elbow", "left_elbow",
                ],
                stiffness=80.0,         # 실기 MIT Kp
                damping=4.0,            # 실기 MIT Kd (상한 5)
                effort_limit=24.0,      # AK70_10 peak torque 약 24.8 N m (current_limit 23.2A 와 정합)
                velocity_limit=46.0,    # AK70_10 no-load 약 46 rad/s
                armature=0.01,          # 반사 로터관성 추정 = I_rotor x gear^2
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["right_wrist", "left_wrist"],
                stiffness=1.4,          # 실기 CST kp(40) x gear(35) / 1000
                damping=0.14,           # 실기 CST kd(4) x gear(35) / 1000
                effort_limit=2.5,       # GPX22HP 단속 허용토크 2.4 N m (기어헤드가 병목)
                velocity_limit=30.0,    # 기어 최대 입력 10000rpm / 35 = 약 30 rad/s
                armature=1.1e-3,        # 반사 로터관성 = I_rotor(8.85 gcm^2) x gear^2(35^2) = 약 1.08e-3 kg m^2
            ),
        },
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
