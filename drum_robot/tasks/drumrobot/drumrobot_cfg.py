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
# step_dt = SIM_DT × decimation = 0.0075 × 2 = 0.015s = 66.667Hz (정책 주기)
# ⚠️ 이 값을 바꾸면 action_scale 도 반비례로 바꿔야 한다 (아래 action_scale 주석 참조).
#
# 왜 66.667Hz 인가 (2026-09-01):
#   실기 send_loop 은 200Hz(5ms)이고 정책은 그 정수 분주여야 한다. 200 / 66.667 = 정확히 3틱.
#   60Hz 였을 때는 200/60 = 3.33 이라 실기가 3틱(66.7Hz)으로 근사했고, 그 결과 배포 시
#   Δq_max 가 10% 어긋났다:  sim a×10×(1/60)=0.16667  vs  실기 a×10×0.015=0.15
#   66.667Hz + action_scale 11.1111 로 맞추면 양쪽이 0.16667 로 정확히 일치한다.
#
# 2026-09-01 100Hz(SIM_DT=0.005) 시도 → fresh 학습 실패. 되돌린 뒤 66.667Hz 로 재시도.
#   증상: 212k step 에서 success 0.045. 보상이 60k 부터 정체(0.277 → 270k 0.286).
#         "접근·ready_pose 보상으로 먹고살고 타격은 안 함" 국소최적 = 타격을 배우지 못함.
#         대조: 동일 설정 60Hz 런(2026-08-31_17-26-22)은 130k 에서 0.470, 1M 에서 0.519.
#   기제: phase-1 부트스트랩(impact_vel / wrist_snap)이 success 게이팅이라 첫 타격을 무작위
#         탐색으로 찾아야 점화된다. 스텝별 독립 노이즈는 √dt 로 평균화되므로 위치 탐색 폭이
#         100Hz 에서 77.5% 로 줄어 첫 타격을 못 찾았다. (66.667Hz 는 94.9% 라 위험이 훨씬 작다)
#   ⚠️ 주기를 바꿀 때는 fresh 가 아니라 반드시 resume 으로 할 것. 타격을 이미 아는 정책에서
#      출발하면 탐색 약화가 문제되지 않는다.
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

    # 지령 각속도 상한 [rad/s]. 물리적으로 의미 있는 건 action_scale × step_dt = Δq_max(한 스텝에
    #   목표를 앞으로 미는 각도)이고, 그게 PD 오차 상한 = 토크 상한을 정한다: τ_max = Kp × Δq_max.
    #   그래서 SIM_DT 를 바꾸면 반드시 action_scale 도 반비례로 바꿔야 한다. 안 하면 조용히 약해진다.
    #   Δq_max 는 주기가 바뀌어도 보존해야 하는 값이다 (물리적 능력 그 자체):
    #      팔   11.1111 × 0.015 = 0.166667 rad (9.55°)  → τ_max = 100 × 0.1667 = 16.7 N·m
    #      손목 27.7778 × 0.015 = 0.416667 rad (23.9°)  → 손목 달성 천장 ≈ 3.85 rad/s
    #   주기를 바꿀 때 환산: new_scale = old_scale × (old_step_dt / new_step_dt)
    #      60Hz 기준값은 10.0 / 25.0 이었다. 66.667Hz → × (1/60)/0.015 = ×1.11111
    action_scale: float = 11.1111       # 기본 관절 지령 각속도 상한 [rad/s]. 팔꿈치 먼-드럼 이동 요구치(~7.5)+PD여유 커버 [[joint-limits-corrected]]
    wrist_action_scale: float = 27.7778    # 손목만 상한 크게 (빠른 패시지 커버)

    # 제어 지연 [물리 스텝]. 1스텝 = 1/120초 = 8.3ms. 0이면 지연 없음(기존 동작).
    #   실기 지연 예산 ≈ 7.8ms (2026-08-31 재산정):
    #     상태읽기(recv 100us) 0.1 + 정책추론 0.15 + **파이프라인 5.0** + CAN 9모터 왕복 ~2.5
    #     파이프라인 5ms = tick N에서 계산한 결과가 tick N+1(5ms 뒤)부터 적용되는 구조적 지연.
    #     순서를 바꿔도(한 틱 먼저 깨우기) obs가 5ms 낡아져 총합은 같음 → 제거 불가.
    #   → 따라서 1(8.3ms)은 "추정치의 2배라 보수적"이 아니라 **정확한 중심값**.
    #     이전 주석의 "≈4ms" 추정은 파이프라인 5ms를 빼먹은 값이었음. 0으로 되돌리지 말 것.
    #     DR 범위는 [0,2]가 아니라 **[1,2] 중심**으로 잡을 것.
    #   정수 스텝 지연을 쓰는 이유: 혼합/저역통과 방식은 고주파를 깎아 발진을 숨겨서 판정을 무의미하게 만듦.
    #   판정 지표: joint_vel_l2 급증 = 발진.
    #     현재 게인(Kp=100, armature 포함 J=0.0243)에서 팔꿈치 f_n=10.2Hz → 8.3ms면 위상지연 ~31°.
    #     (구 주석의 56°는 Kp=200·armature 미포함 f_n=18.8Hz 기준이었음)
    #   켜는 순서: clean 모델(success_rate>=0.85) 확보 후 resume fine-tune으로 얹을 것.
    #     게인이 흔들리는 중에 넣으면 발진 원인을 지연/게인으로 갈라낼 수 없음.
    action_delay_steps: int = 0

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
        #   waist: TMotor AK10_9 (gear 9),  arm: TMotor AK70_10 (gear 10),  wrist: Maxon DCX22L+GPX22HP (gear 35)
        #   effort/velocity_limit는 데이터시트 근사(팔은 T-Motor/CubeMars 공개값, 손목은 Maxon 데이터시트 확정값).
        #    waist/arm은 서보 속도모드(control_gain 1000/4000) → MIT 모드(τ=Kp·Δp+Kd·Δv+t_ff)로 전환 예정.
        #      MIT는 PhysX ImplicitActuator와 식이 동일해 Kp/Kd를 그대로 복사 가능. 단 CubeMars 매뉴얼 상한 Kd<=5.
        #      손목은 Maxon EPOS라 MIT 무관 — 실기 control_kp/kd 기준(motors.json이 08-18에 60/1→40/4로 변경됨, sim 미반영).
        actuators={
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint"],
                stiffness=100.0,        # MIT 전환 대비. Kd<=5 상한에서 ζ를 벌기 위해 200→100 (ζ 0.427→0.302, f_n 2.74→1.92Hz)
                damping=5.0,            # ★ MIT 모드 상한 Kd<=5. 기존 10은 실기에 넣을 수 없는 값이었음
                effort_limit=48.0,      # AK10_9 peak torque ~48 N·m (근사, 확인 권장)
                velocity_limit=28.0,    # AK10_9 no-load ~28 rad/s (근사)
                armature=0.01,          # 반사 로터관성 추정 = I_rotor × gear²(81). 링크 J=0.676 대비 1.5%라 영향 미미
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_1", "left_shoulder_1",
                    "right_shoulder_2", "left_shoulder_2",
                    "right_elbow", "left_elbow",
                ],
                stiffness=100.0,        # MIT 전환 대비. 200이면 full-scale 액션이 33.3N·m를 요구해 effort_limit 24를 넘어 포화(감쇠 소실)
                damping=5.0,            # ★ MIT 상한. 어깨 ζ 1.136→0.804, 팔꿈치 ζ 2.270→1.605 (armature 포함 J 기준)
                effort_limit=24.0,      # AK70_10 peak torque ~24.8 N·m (current_limit 23.2A와 정합)
                velocity_limit=46.0,    # AK70_10 no-load ~46 rad/s (근사)
                armature=0.01,          # 반사 로터관성 추정 = I_rotor(~1e-4) × gear²(100).
                                        #   팔꿈치 J 0.0143→0.0243(+70%)로 제일 크게 바뀜: ζ 2.96→2.27, f_n 18.8→14.4Hz
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["right_wrist", "left_wrist"],
                stiffness=1.4,          # 실기 CST: kp(40) × gear(35)/1000. motors.json 8298e3b(08-18) 기준 — 2.1은 옛 kp(60) 값이었음
                damping=0.14,           # 실기: kd(4) × gear(35)/1000. ★0.035→0.14(4배). Kp/Kd비 60→10 → 손목 qd_max 16.7→3.85 rad/s 예상
                effort_limit=2.5,       # GPX22HP 단속 허용토크 2.4 N·m (기어헤드가 병목; 모터 stall 294mNm보다 낮음)
                velocity_limit=30.0,    # 기어 최대 입력 10000rpm / 35 ≈ 30 rad/s (연속 8000rpm→24)
                armature=1.1e-3,        # 반사 로터관성 = I_rotor(8.85 gcm²) × gear²(35²) ≈ 1.08e-3 kg·m²
            ),
        },
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
