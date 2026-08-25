from __future__ import annotations
                                                    # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
from isaaclab.assets import ArticulationCfg         # pyright: ignore[reportMissingImports]
from isaaclab.envs import DirectRLEnvCfg            # pyright: ignore[reportMissingImports]
from isaaclab.scene import InteractiveSceneCfg      # pyright: ignore[reportMissingImports]
from isaaclab.sim import SimulationCfg              # pyright: ignore[reportMissingImports]
from isaaclab.utils import configclass              # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils                    # pyright: ignore[reportMissingImports]
from isaaclab.actuators import ImplicitActuatorCfg  # pyright: ignore[reportMissingImports]

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
    observation_space: int = 148    # 관측 차원. K=6 글로벌 lookahead(66) + 팔별 담당 2타(16) + 나머지(66)
    state_space: int = 0

    # MIDI 커리큘럼: 성공률(EMA) 기반 점진 램프 — 기본 타격 익힌 뒤 실제 MIDI(연주가능하게 가공됨) 투입
    # [[midi-mixing-and-density]]. 세기 게이팅(0.8~)보다 먼저 시작하도록 stagger (동시투하 방지).
    midi_success_start: float = 0.65  # 이 성공률(EMA)부터 MIDI 섞기 시작 (성공률이 0.8 부근서 정체 → 문턱 낮춤)
    midi_success_full: float = 0.90   # 이 성공률에서 final_ratio 도달
    midi_final_ratio: float = 0.35    # 최종 MIDI 비율 (일단 낮게 — 나머지 랜덤). 안정되면 ↑
    midi_selection_strength: float = 0.5       # MIDI 샘플링 스코어 가중 (0=완전랜덤, 클수록 스코어 선별)

    # 평가(play1.py)용 — 기본값이면 학습과 동일 (무영향). play1.py가 오버라이드
    eval_midi_ratio: float = -1.0   # >=0이면 커리큘럼 무시하고 MIDI 비율 고정 (1.0=순수 MIDI). 레벨1/2/3
    eval_midi_file: str = ""        # 지정 시 이 MIDI 파일만 로드. 레벨2/3
    eval_full_song: bool = False    # True면 파일 전체를 1개 긴 악보로. 레벨3
    eval_slow_factor: float = 1.0   # RDS slow_factor. 1.0=원곡 템포(실제 연주 목표). slow_factor는 MIDI 경로에만 적용(랜덤 RDS 무관)

    # 배정 히스테리시스: h0 담당팔을 바꾸려면 상대 팔 점수가 이 비율만큼 더 좋아야 함.
    #   배정 점수(travel/w)는 스케일이 60배 차이나서 절대 마진은 부적합 → 상대(비율) 마진.
    #   경계 채터링(초당 2.8회 뒤집힘 → idle 팔 헛스윙)만 차단, 재장전 BIG(1e3) 교대는 그대로 통과.
    sched_hysteresis: float = 0.25

    # 교차 배정 페널티 [배율]. 왼팔이 우측(x>0) 드럼 / 오른팔이 좌측(x<0) 드럼을 맡을 때
    #   '반대편 침범량[m] × 이 값'을 추가 이동거리로 과금.
    #   필요성: arm_score가 현재 팁 위치(travel)만 봐서, 한 번 교차하면 "이미 거기 있으니 travel=0" →
    #   교차가 항상 이기는 자기강화 고리가 생김. 교차량은 드럼 x·팔 정체성만으로 결정되어 이 고리를 끊음.
    #   1.5 근거: 3.0은 과도 — 각 팔이 자기 쪽 전체를 혼자 스윕하게 되어 어깨 방위축 과회전·오버슈트
    #   (miss3_dir 0.249→0.445, wrong_rate 0.037→0.074, shoulder_ang_vel +43%)를 유발. 충돌은 이미
    #   stick_contact 0.080→0.006으로 해결됐고, fresh 학습은 교차를 '깨는' 게 아니라 '예방'이라 1.5로 충분.
    #   재장전 중인 팔의 BIG(1e3)은 여전히 압도 → 연타 교대는 그대로 유지됨.
    sched_cross_penalty: float = 1.5

    action_scale: float = 10.0         # 기본 관절 지령 각속도 상한 [rad/s]. π→10: 팔꿈치 먼-드럼 이동 요구치(~7.5)+PD여유 커버, 실기 velocity_limit(46) 아래. 기존 π는 gross 이동 병목 → high/hihat reach-miss 원인 [[joint-limits-corrected]]
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
