"""
[확인용 임시 스크립트] sticking_planner(손 배정) GUI 확인: 배정대로 치는 '대본 연주자'

학습된 모델 없이, 이 스크립트가 sticking_planner 의 손 배정대로 로봇을 움직여서
"이 배정대로 치면 실제로 칠 수 있는가"를 확인 (학습·환경 코드는 건드리지 않음, 확인 후 tests 폴더 삭제)

대본 연주자 (실제 로봇 서버 play_motion_generator 와 같은 구조)
    1) 손 배정     : sticking_planner.schedule() (지금 코드 그대로) -> 팔별 다음 음표 (드럼, 시각)
    2) 허리        : 지금 정렬 보상의 드럼 쌍 표 (식 6) pair_waist[오른팔 다음 드럼, 왼팔 다음 드럼]
    3) IK 기본 자세: 팔별로 직전 위치 -> 다음 드럼의 손별 타격 지점까지 부드럽게 보간, IKSolver 로 관절각
                     타격 지점 = 드럼 윗면 + 1cm, 중심에서 왼손은 왼쪽·오른손은 오른쪽 4cm (같은 드럼을 번갈아 칠 때 겹치지 않게)
    4) 들고 치기   : IK 자세에 손목·팔꿈치 오프셋을 더함 (실제 로봇 state_motion_generator 규칙, 세기 5)
                     대기 +20°, 최고점 +20° + 들기각(T<0.5s 면 40T+10°, 아니면 30°), 타격 순간 -5°
                     친 뒤 다음 음표가 아직 없으면 손목을 대기 +20° 로 올림 (실제 로봇의 타격 후 휴식)
    5) 명령        : 관절 목표를 직접 넣음 (이 스크립트 안에서만 행동 처리를 '절대 관절 각도'로 바꿈)

화면
    곧 칠 드럼 노랑, 타격하면 빨간 공 / 팔별 다음 목표 드럼 위 공 (왼팔 파랑, 오른팔 초록)
터미널
    0.25초마다 env 0: 팔별 다음 목표와 남은 시간, 허리 (목표/실제), 타격
    에피소드마다 누적: 맞힘 비율, 놓침, 잘못 침, 잘못된 접촉, 맞힌 타격 평균 시간 오차,
                      불가능 배정(쌍 표 inf), IK 실패, 교차 배정, 같은 손 연속 간격, 드럼별 손 분포

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/planner_test.py
    패턴 악보로 배정 확인: ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/planner_test.py --score pattern --num_envs 6 --slow 0.03
        env 마다 패턴 하나 (rock, ride, roll, fill, crash, wide 순서로 돌아가며), 음표 시각마다 터미널에 배정 결과 한 줄
    옵션: --num_envs 2, --slow 0.0 (스텝마다 대기 초), --score random|midi|pattern, --pattern all|rock|..., --episodes 0 (0이면 창을 닫을 때까지),
          --wrist_deg 25 (IK 손목 기준각), --lead_ms 30 (손목이 늦게 따라오는 만큼 타격을 미리 시작), --side_cm 4 (손별 타격 지점 좌우 간격)
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="sticking_planner 대본 연주자 GUI 확인")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--slow", type=float, default=0.0, help="스텝마다 대기 시간 [s]")
parser.add_argument("--score", choices=["random", "midi", "pattern"], default="random", help="악보 종류 (pattern = 아래 패턴 악보)")
parser.add_argument("--pattern", default="all", help="--score pattern 일 때: all 또는 rock, ride, roll, fill, crash, wide (env 마다 하나씩 돌아가며)")
parser.add_argument("--episodes", type=int, default=0, help="에피소드 수 (0이면 창을 닫을 때까지)")
parser.add_argument("--wrist_deg", type=float, default=25.0, help="IK 손목 기준각 [deg] (시작 자세와 같음)")
parser.add_argument("--lead_ms", type=float, default=30.0, help="타격 동작을 미리 시작하는 시간 [ms]")
parser.add_argument("--side_cm", type=float, default=4.0, help="손별 타격 지점: 드럼 중심에서 좌우로 [cm]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Sim 실행 후에만 import 가능 ----
import math
import time

import torch                                                    # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils                                # pyright: ignore[reportMissingImports]
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg   # pyright: ignore[reportMissingImports]
from isaaclab.utils import math as math_utils                   # pyright: ignore[reportMissingImports]

import drum_robot.tasks.drumrobot.drumrobot_env as drumrobot_env_module
from drum_robot.tasks.drumrobot.drumrobot_cfg import DrumRobotEnvCfg
from drum_robot.tasks.drumrobot.drumrobot_env import DrumRobotEnv
from drum_robot.tasks.drumrobot.components.visualizer import VisualizerCfg
from drum_robot.tasks.drumrobot.components.specs import Instruments, RobotSpec
from drum_robot.tasks.drumrobot.components.ik_solver import IKSolver
from drum_robot.tasks.drumrobot.components.sticking_planner import StickingPlanner, StickingPlannerCfg

NAMES = list(Instruments().all.keys())
L, R = 0, 1


# ---- 패턴 악보 (--score pattern): (시각 [s], [드럼들]), 100 BPM (8분음표 0.3초, 16분음표 0.15초), 0.45초부터 ----
def _beat(cymbal: str) -> list:
    # 8분음표 심벌 + 2·4박 스네어 (스네어는 심벌과 동시)
    return [(0.45 + 0.3 * i, [cymbal] + (["snare"] if i % 4 == 2 else [])) for i in range(14)]

PATTERNS = {
    "rock": _beat("hihat"),                                                 # 사람이면 오른손 하이햇 + 왼손 스네어 (교차)
    "ride": _beat("ride"),                                                  # 오른손 라이드 + 왼손 스네어 (교차 없음)
    "roll": [(0.45 + 0.15 * i, ["snare"]) for i in range(24)],             # 스네어 16분음표 연타 -> 번갈아 치는지
    "fill": ([(0.45 + 0.15 * i, [("high", "mid", "floor")[i // 4]]) for i in range(12)]    # 탐 필인 (16분음표)
             + [(2.25, ["crash_r"])]
             + [(2.85 + 0.3 * i, [("high", "mid", "floor", "floor")[i]]) for i in range(4)]  # 탐 필인 (8분음표)
             + [(4.05, ["crash_l"])]),
    "crash": ([(0.45, ["crash_l", "snare"])] + [(0.75 + 0.3 * i, ["hihat"]) for i in range(6)]          # 크래시 + 스네어 동시
              + [(2.55, ["crash_r", "snare"])] + [(2.85 + 0.3 * i, ["hihat"]) for i in range(6)]),
    "wide": [(0.45 + 0.3 * i, ["hihat", "floor"] if i % 2 == 0 else ["snare"]) for i in range(14)],     # 왼쪽 끝 + 오른쪽 끝 동시
}
HIT_HEIGHT = 0.01           # IK 기본 자세에서 팁 높이 (드럼 윗면 위) [m], 타격 순간 손목 -5° 로 눌러 닿게 함
SIDE_X = args_cli.side_cm / 100.0   # 손별 타격 지점: 드럼 중심에서 왼손은 왼쪽, 오른손은 오른쪽으로 [m] (실제 로봇 좌표도 손별 약 4cm)
REST_TIME = 0.15            # 친 뒤 다음 음표가 없으면 손목을 대기 각도로 올리는 시간 [s]
WAIST_RATE = 6.0            # 허리 목표가 바뀔 때 따라가는 속도 상한 [rad/s]
CROSS_MARGIN = 0.02         # 오른팔 목표가 왼팔 목표보다 이만큼 이상 왼쪽이면 교차 배정 [m]
STAY_DEG, PRESS_DEG = 20.0, -5.0    # 손목 대기 각도, 타격 순간 눌러 치는 각도 [deg] (실제 로봇)


class WindowClosed(Exception):
    pass


def make_env() -> DrumRobotEnv:
    # 드럼 색·타격 공 표시 켜기 (환경 생성 때 VisualizerCfg()를 부르므로 이 스크립트에서만 바꿔치기)
    drumrobot_env_module.VisualizerCfg = lambda: VisualizerCfg(enable_visualization=True)
    cfg = DrumRobotEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    cfg.viewer.eye = (0.9, -1.1, 1.9)       # 로봇 뒤쪽 위에서 드럼 쪽을 봄
    cfg.viewer.lookat = (0.0, 0.5, 0.6)
    return DrumRobotEnv(cfg)


def smooth(s: torch.Tensor) -> torch.Tensor:
    # 0~1 보간, 양 끝 속도 0 (3차)
    s = s.clamp(0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


class Drummer:
    def __init__(self, env: DrumRobotEnv):
        self.env = env
        self.N, self.dev, self.dt = env.num_envs, env.device, env.step_dt
        self.ar = torch.arange(self.N, device=self.dev)
        names = env.robot_interface.get_ctrl_joint_name()
        spec = RobotSpec()
        self.ik_to_ctrl = torch.tensor([spec.joint_name_to_ik_result_idx[n] for n in names], device=self.dev)
        self.j_waist = names.index("waist_joint")
        self.j_wrist = [names.index("left_wrist"), names.index("right_wrist")]
        self.j_elbow = [names.index("left_elbow"), names.index("right_elbow")]
        self.ik = IKSolver(self.dev)
        self.waist_z = env.cfg.robot_waist_joint_offset_z      # env 좌표 z - 이 값 = IK 좌표 z (허리 관절 원점)
        self.hit_off = torch.tensor([[-SIDE_X, 0.0, HIT_HEIGHT], [SIDE_X, 0.0, HIT_HEIGHT]], device=self.dev)   # (2, 3) 손별 타격 지점 (드럼 윗면 중심 기준)
        self.wrist0 = math.radians(args_cli.wrist_deg)
        self.lead = args_cli.lead_ms / 1000.0

        # 손 배정: sticking_planner 지금 코드 그대로
        self.planner = StickingPlanner(
            cfg=StickingPlannerCfg(step_dt=self.dt),
            drum_names=NAMES,
            num_events=env.rds.cfg.num_hits,
            max_lookahead_step=env.rds.env.max_lookahead_step,
            episode_length_step=env.episode_length_step,
            num_envs=self.N,
            device=self.dev,
        )

        # 팁 속도 계산용 (플래너 입력): 손목 링크 속도 + 각속도 x 손목->팁
        body = env.robot_interface.body_to_idx
        self.wrist_body = [body["left_wrist"], body["right_wrist"]]
        self.tip_off = torch.tensor([spec.tip_offset_left, spec.tip_offset_right], device=self.dev)   # (2, 3)

        # 판정 결과 가로채기 (환경 코드는 그대로)
        self.result = None
        orig = env.hit_detector.step
        def wrapped(*a, **kw):
            self.result = orig(*a, **kw)
            return self.result
        env.hit_detector.step = wrapped

        # 행동 처리 = 절대 관절 각도 (이 스크립트 안에서만)
        def pre(q_des):
            env.actions = q_des.clone()
            env.target_joint_pos = env.robot_interface.clip_and_convert(q_des)
            env._substep = 0                                    # 환경의 원래 함수가 하던 물리 스텝별 접촉 지점 기록 초기화
            env.contact_from_tip.fill_(float("nan"))
        env._pre_physics_step = pre

        # 배정 표시 공 (왼팔 파랑, 오른팔 초록)
        self.markers = None
        if not args_cli.headless:
            self.markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/planner_targets",
                markers={
                    "left": sim_utils.SphereCfg(radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0))),
                    "right": sim_utils.SphereCfg(radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2))),
                },
            ))

    # ---------------- 준비 ----------------
    def drum_top_ik(self) -> torch.Tensor:
        # (N, M, 3) 드럼 윗면 중심, IK 좌표 (허리 관절 원점)
        p = self.env.drum_pos.clone()
        p[..., 2] -= self.waist_z
        return p

    def reset(self, seed: int):
        env = self.env
        torch.manual_seed(seed)
        env.reset()
        if args_cli.score == "midi":
            env.rds.reset(env_ids=self.ar, score_ratio=1.0, selection_strength=0.0)
            env.hit_detector.reset(env_ids=self.ar, rds=env.rds.get_rds())
        if args_cli.score == "pattern":
            # 악보 텐서만 패턴으로 바꾸고 판정 초기화 (환경 코드는 그대로)
            names = list(PATTERNS) if args_cli.pattern == "all" else [args_cli.pattern]
            self.env_pattern = [names[e % len(names)] for e in range(self.N)]
            rds = torch.zeros_like(env.rds.rds)
            for e, name in enumerate(self.env_pattern):
                for t, drums in PATTERNS[name]:
                    for d in drums:
                        rds[e, round(t / self.dt), NAMES.index(d)] = 1
            env.rds.rds = rds
            env.hit_detector.reset(env_ids=self.ar, rds=env.rds.get_rds())
        self.planner.reset()
        self.note_log = []

        start = env.reward_computer.alignment.prev_drum.clone()        # (N, 2) [왼, 오] 시작 자세 드럼
        top = self.drum_top_ik()
        self.seg_from = top[self.ar[:, None], start] + self.hit_off                    # (N, 2, 3) 구간 시작점
        self.w_prev = torch.full((self.N, 2), STAY_DEG, device=self.dev)              # 직전에 명령한 손목 오프셋 [deg]
        self.e_prev = torch.zeros((self.N, 2), device=self.dev)                       # 직전에 명령한 팔꿈치 오프셋 [deg]
        self.seg_from_w = self.w_prev.clone()                                          # 구간 시작 때 손목·팔꿈치 오프셋 (여기서부터 들어 올림)
        self.seg_from_e = self.e_prev.clone()
        self.seg_from_step = torch.zeros((self.N, 2), dtype=torch.long, device=self.dev)
        self.seg_to_drum = start.clone()                               # 팔별 향하는(또는 서 있는) 드럼
        self.seg_to_step = torch.full((self.N, 2), -1, dtype=torch.long, device=self.dev)   # -1 = 배정된 음표 없음
        self.last_note_step = torch.full((self.N, 2), -10000, dtype=torch.long, device=self.dev)
        self.last_hit_step = torch.full((self.N, 2), -10000, dtype=torch.long, device=self.dev)
        self.ik_fail_step = torch.full((self.N,), -10000, dtype=torch.long, device=self.dev)   # 마지막 IK 실패 스텝
        q0 = env.robot_interface.get_joint_pos(env.robot)
        self.waist = q0[:, self.j_waist].clone()
        self.q_base = q0.clone()

    # ---------------- 매 스텝 ----------------
    def tip_velocity(self) -> torch.Tensor:
        d = self.env.robot.data
        v = d.body_lin_vel_w[:, self.wrist_body]                       # (N, 2, 3)
        w = d.body_ang_vel_w[:, self.wrist_body]
        q = d.body_quat_w[:, self.wrist_body]
        r = math_utils.quat_apply(q.reshape(-1, 4), self.tip_off.repeat(self.N, 1)).view(self.N, 2, 3)
        return v + torch.cross(w, r, dim=-1)

    def next_notes(self, ev_arm: torch.Tensor, ev_off: torch.Tensor, steps: torch.Tensor):
        # 팔별 다음 음표: 가장 이른 이벤트 중 그 팔이 맡은 드럼 -> (N, 2) 드럼, (N, 2) 절대 스텝 (-1 없음)
        K = ev_arm.shape[1]
        nd = torch.zeros((self.N, 2), dtype=torch.long, device=self.dev)
        nt = torch.full((self.N, 2), -1, dtype=torch.long, device=self.dev)
        for a in (L, R):
            for k in range(K - 1, -1, -1):                              # 뒤에서부터 덮어써서 가장 이른 것이 남음
                mask = ev_arm[:, k] == a                                # (N, M)
                has = mask.any(dim=-1)
                nd[:, a] = torch.where(has, mask.int().argmax(dim=-1), nd[:, a])
                nt[:, a] = torch.where(has, steps + ev_off[:, k], nt[:, a])
        return nd, nt

    def stroke_offsets(self, steps: torch.Tensor):
        # 팔별 손목·팔꿈치 오프셋 [deg] (실제 로봇 state_motion_generator 규칙, 세기 5)
        #   구간 시작 -> 들기 (지금 오프셋에서 최고점까지) -> 유지 -> 내려치기 (음표 시각에 -5°)
        #   음표가 지나고 새 배정이 없으면 -> 손목을 대기 각도(+20°)로 올리고 팔꿈치 0 (실제 로봇의 타격 후 휴식)
        has = self.seg_to_step >= 0
        after = has & (steps[:, None] > self.seg_to_step)                           # 음표가 지났는데 새 배정 없음
        tau = (self.seg_to_step - steps[:, None]).float() * self.dt - self.lead     # 다음 음표까지 남은 시간 [s]
        u = (steps[:, None] - self.seg_from_step).float() * self.dt                 # 구간 시작 후 지난 시간 [s]
        u_after = (steps[:, None] - self.seg_to_step).float() * self.dt             # 음표가 지난 뒤 시간 [s]
        T = ((self.seg_to_step - self.seg_from_step).float() * self.dt).clamp(min=self.dt)   # 구간 길이 [s]

        lift = STAY_DEG + torch.where(T < 0.5, 40.0 * T + 10.0, torch.full_like(T, 30.0))   # 최고점 [deg]
        e_lift = torch.where(T < 0.2, torch.zeros_like(T), torch.where(T <= 0.5, T * 15.0 / 0.5, torch.full_like(T, 15.0)))
        t_down = torch.minimum(torch.full_like(T, 0.12), 0.5 * T)                   # 내려치는 시간
        t_up = torch.minimum(torch.full_like(T, 0.15), 0.5 * T)                     # 드는 시간

        down = tau <= t_down
        k_down = (tau.clamp(min=0.0) / t_down).pow(2)                               # 1 -> 0, 끝으로 갈수록 빨라짐
        k_up = smooth(u / t_up)
        wrist = torch.where(down, PRESS_DEG + (lift - PRESS_DEG) * k_down,
                torch.where(u <= t_up, self.seg_from_w + (lift - self.seg_from_w) * k_up, lift))
        elbow = torch.where(down, e_lift * k_down,
                torch.where(u <= t_up, self.seg_from_e + (e_lift - self.seg_from_e) * k_up, e_lift))

        # 친 뒤 휴식: -5° -> +20°, 팔꿈치 0
        k_rest = smooth(u_after / REST_TIME)
        wrist = torch.where(after, PRESS_DEG + (STAY_DEG - PRESS_DEG) * k_rest, wrist)
        elbow = torch.where(after, torch.zeros_like(elbow), elbow)
        # 배정이 한 번도 없었으면 대기 자세
        wrist = torch.where(has, wrist, torch.full_like(wrist, STAY_DEG))
        elbow = torch.where(has, elbow, torch.zeros_like(elbow))
        return wrist, elbow

    def step(self, stats: "Stats"):
        env = self.env
        steps = env.steps.clone()
        al = env.reward_computer.alignment

        # 1) 손 배정 (sticking_planner 그대로)
        next_hits = env.rds.get_next_hits(steps)
        tip = env.robot_interface.get_body_pos(env.robot)
        out = self.planner.schedule(steps, next_hits, tip, self.tip_velocity(), env.drum_pos, self.last_hit_step)
        ev_arm, ev_off = out[5], out[6]
        nd, nt = self.next_notes(ev_arm, ev_off, steps)
        self.log_notes(ev_arm, ev_off, steps)

        # 음표 시각에 도착: 같은 손 연속 간격, 드럼별 손 분포 기록
        reached = (self.seg_to_step >= 0) & (steps[:, None] == self.seg_to_step)
        stats.note_reached(reached, self.seg_to_drum, steps, self.last_note_step)
        self.last_note_step = torch.where(reached, steps[:, None].expand(-1, 2), self.last_note_step)

        # 2) 구간 갱신: 새 음표가 배정되면 지금 기본 자세 점에서 새 드럼까지 다시 보간
        top = self.drum_top_ik()
        cur_point = self.base_point(steps, top)
        new_seg = (nt >= 0) & ((nt != self.seg_to_step) | (nd != self.seg_to_drum))
        self.seg_from = torch.where(new_seg[..., None], cur_point, self.seg_from)
        self.seg_from_step = torch.where(new_seg, steps[:, None].expand(-1, 2), self.seg_from_step)
        self.seg_to_drum = torch.where(new_seg, nd, self.seg_to_drum)
        self.seg_to_step = torch.where(new_seg, nt, self.seg_to_step)
        self.seg_from_w = torch.where(new_seg, self.w_prev, self.seg_from_w)    # 지금 손목·팔꿈치 오프셋에서부터 들어 올림
        self.seg_from_e = torch.where(new_seg, self.e_prev, self.seg_from_e)
        point = self.base_point(steps, top)                                 # (N, 2, 3) IK 좌표 팁 목표

        # 3) 허리: 지금 정렬 보상의 드럼 쌍 표 (오른팔 드럼, 왼팔 드럼)
        dr, dl = self.seg_to_drum[:, R], self.seg_to_drum[:, L]
        feasible = torch.isfinite(al.pair_cost[self.ar, dr, dl])
        w_target = torch.where(feasible, al.pair_waist[self.ar, dr, dl], self.waist)
        self.waist = self.waist + (w_target - self.waist).clamp(-WAIST_RATE * self.dt, WAIST_RATE * self.dt)

        # 4) IK 기본 자세
        wr = torch.full((self.N,), self.wrist0, device=self.dev)
        ik_out, err = self.ik.solve_geometric_ik(point[:, R], point[:, L], self.waist, wr, wr)
        ok = err < 0.5
        q_new = ik_out[:, self.ik_to_ctrl]                                  # IK 결과 순서 -> 관절 순서
        self.q_base = torch.where(ok[:, None], q_new, self.q_base)         # 실패하면 직전 자세 유지
        self.ik_fail_step = torch.where(ok, self.ik_fail_step, steps)

        # 5) 들고 치기 오프셋 + 명령
        w_off, e_off = self.stroke_offsets(steps)                           # [deg]
        self.w_prev, self.e_prev = w_off, e_off
        q_des = self.q_base.clone()
        for a in (L, R):
            q_des[:, self.j_wrist[a]] += torch.deg2rad(w_off[:, a])
            q_des[:, self.j_elbow[a]] += torch.deg2rad(e_off[:, a])
        if not simulation_app.is_running():
            raise WindowClosed
        env.step(q_des)
        if args_cli.slow > 0:
            time.sleep(args_cli.slow)

        # 판정 결과 -> 팔별 마지막 타격 시각 (플래너 입력)
        r = self.result
        hit = r.hit_per_arm.any(dim=-1)
        self.last_hit_step = torch.where(hit, env.steps[:, None].expand(-1, 2), self.last_hit_step)

        # 기록: 교차 = 오른팔 목표가 왼팔 목표보다 왼쪽 (로봇 좌표 x, 오른쪽이 +)
        both = (self.seg_to_step >= 0).all(dim=-1)
        cross = both & (top[self.ar, dr, 0] < top[self.ar, dl, 0] - CROSS_MARGIN)
        # 플래너 확인용: 양팔 다음 목표가 같은 드럼인지, 드럼과 시각까지 같은지(같은 음표를 양팔에 배정)
        both_next = (nt >= 0).all(dim=-1)
        same_drum = both_next & (nd[:, L] == nd[:, R])
        same_note = same_drum & (nt[:, L] == nt[:, R])
        stats.step(r, feasible, ok, both, cross, both_next, same_drum, same_note)
        # 잘못 친 타격·잘못된 접촉 분류: 그 팔의 목표 드럼인지, 최근 0.2초 안에 IK 가 실패했는지
        own = torch.arange(len(NAMES), device=self.dev).view(1, 1, -1) == self.seg_to_drum[..., None]   # (N, 2, M)
        recent_fail = ((steps - self.ik_fail_step).float() * self.dt < 0.2).view(-1, 1, 1)
        wa = r.hit_per_arm & r.wrong_hit[:, None, :]                       # 손별 잘못 친 타격
        stats.classify(wa & own, wa & ~own & recent_fail, wa & ~own & ~recent_fail, r.bad_per_arm & own, r.bad_per_arm & ~own)

        if self.markers is not None:
            pos = env.drum_pos[self.ar[:, None], self.seg_to_drum] + env.scene.env_origins[:, None] + torch.tensor([0, 0, 0.04], device=self.dev)
            self.markers.visualize(translations=pos.permute(1, 0, 2).reshape(-1, 3),
                                   marker_indices=torch.cat([torch.zeros(self.N), torch.ones(self.N)]).long())
        return steps, nd, nt

    def log_notes(self, ev_arm: torch.Tensor, ev_off: torch.Tensor, steps: torch.Tensor):
        # 이번 스텝이 음표 시각인 이벤트의 배정 결과: (env, 시각, [(드럼, 팔)], 교차 여부, 팔별 같은 손 직전 간격)
        self.note_log = []
        x = self.env.drum_pos[..., 0]                                        # (N, M) 드럼 x (로봇 좌표, 오른쪽 +)
        for k in range(ev_arm.shape[1]):
            now = (ev_off[:, k] == 0) & (ev_arm[:, k] >= 0).any(dim=-1)
            for e in now.nonzero().flatten().tolist():
                pairs = [(m, int(ev_arm[e, k, m])) for m in range(len(NAMES)) if int(ev_arm[e, k, m]) >= 0]
                drum_of = {a: m for m, a in pairs}
                # 이번 음표를 안 치는 팔은 지금 향하는(또는 쉬는) 드럼 위치로 교차 판단
                xr = float(x[e, drum_of.get(R, int(self.seg_to_drum[e, R]))])
                xl = float(x[e, drum_of.get(L, int(self.seg_to_drum[e, L]))])
                gaps = {a: (int(steps[e]) - int(self.last_note_step[e, a])) * self.dt for _, a in pairs}
                self.note_log.append((e, int(steps[e]), pairs, xr < xl - CROSS_MARGIN, gaps))

    def base_point(self, steps: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        # 팔별 IK 기본 자세 팁 목표: 구간 시작점 -> 향하는 드럼 윗면 + HIT_HEIGHT, 음표 시각에 도착
        to = top[self.ar[:, None], self.seg_to_drum] + self.hit_off             # 손별 타격 지점 (중심에서 좌우로 SIDE_X)
        span = (self.seg_to_step - self.seg_from_step).float()
        s = torch.where(span > 0, (steps[:, None] - self.seg_from_step).float() / span.clamp(min=1.0), torch.ones_like(span))
        s = torch.where(self.seg_to_step >= 0, s, torch.ones_like(s))
        return self.seg_from + (to - self.seg_from) * smooth(s)[..., None]


class Stats:
    # 에피소드를 넘어 누적
    def __init__(self, M: int):
        self.c = {k: 0 for k in ("hits", "matched", "missed", "wrong", "bad", "steps", "infeasible", "ik_fail", "both", "cross",
                                 "both_next", "same_drum", "same_note", "both_hand_hit",
                                 "wrong_same", "wrong_other_ik", "wrong_other", "bad_own", "bad_other")}
        self.abs_err_ms = 0.0
        self.intervals = []                                  # 같은 손 연속 음표 간격 [s]
        self.hand = torch.zeros((M, 2), dtype=torch.long)    # 드럼별 손 [왼, 오]

    def step(self, r, feasible, ok, both, cross, both_next, same_drum, same_note):
        c = self.c
        c["both_next"] += int(both_next.sum()); c["same_drum"] += int(same_drum.sum()); c["same_note"] += int(same_note.sum())
        c["both_hand_hit"] += int((r.hit_per_arm[:, 0] & r.hit_per_arm[:, 1]).sum())     # 같은 스텝에 양손이 같은 드럼을 침
        c["hits"] += int(r.hit_per_arm.sum()); c["matched"] += int(r.matched.sum()); c["missed"] += int(r.missed.sum())
        c["wrong"] += int(r.wrong_hit.sum()); c["bad"] += int(r.bad_contact.sum())
        self.abs_err_ms += float((r.time_error.abs() * r.matched).sum()) * 1000.0
        c["steps"] += feasible.numel(); c["infeasible"] += int((~feasible).sum()); c["ik_fail"] += int((~ok).sum())
        c["both"] += int(both.sum()); c["cross"] += int(cross.sum())

    def classify(self, wrong_same, wrong_other_ik, wrong_other, bad_own, bad_other):
        c = self.c
        c["wrong_same"] += int(wrong_same.sum()); c["wrong_other_ik"] += int(wrong_other_ik.sum()); c["wrong_other"] += int(wrong_other.sum())
        c["bad_own"] += int(bad_own.sum()); c["bad_other"] += int(bad_other.sum())

    def note_reached(self, reached, drum, steps, last_note_step):
        for e, a in reached.nonzero().tolist():
            self.hand[int(drum[e, a]), a] += 1
            if last_note_step[e, a] > -10000:
                self.intervals.append(float(steps[e] - last_note_step[e, a]))

    def report(self, title: str, dt: float):
        c = self.c
        n_judged = c["matched"] + c["missed"]
        print(f"\n[planner] {title}")
        print(f"  맞힘 비율 {c['matched'] / max(n_judged, 1) * 100:.1f}% (맞힘 {c['matched']} / 놓침 {c['missed']}), 잘못 침 {c['wrong']}, "
              f"잘못된 접촉 {c['bad']}, 맞힌 타격 평균 |시간 오차| {self.abs_err_ms / max(c['matched'], 1):.1f} ms")
        print(f"  불가능 배정(쌍 표 inf) {c['infeasible'] / max(c['steps'], 1) * 100:.1f}% 스텝, IK 실패 {c['ik_fail'] / max(c['steps'], 1) * 100:.1f}% 스텝, "
              f"교차 배정 {c['cross'] / max(c['both'], 1) * 100:.1f}% (양팔 모두 목표가 있는 스텝 중)")
        print(f"  양팔 다음 목표가 같은 드럼 {c['same_drum'] / max(c['both_next'], 1) * 100:.1f}% 스텝 (번갈아 치기, 정상), "
              f"같은 음표를 양팔에 배정 {c['same_note']} 스텝 (0이어야 함), 같은 스텝에 양손이 같은 드럼을 침 {c['both_hand_hit']}번")
        print(f"  잘못 침 분류: 자기 목표 드럼을 또 침(튕김·누름·시간 밖) {c['wrong_same']}, 다른 드럼(IK 실패 직후) {c['wrong_other_ik']}, "
              f"다른 드럼(그 외: 이동 중 등) {c['wrong_other']} | 잘못된 접촉(밑면·옆면): 자기 목표 드럼 {c['bad_own']}, 다른 드럼 {c['bad_other']}")
        if self.intervals:
            iv = sorted(x * dt for x in self.intervals)
            bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 99)]
            print(f"  같은 손 연속 간격: 최소 {iv[0]:.3f}s, 중앙 {iv[len(iv) // 2]:.3f}s | "
                  + ", ".join(f"{lo}~{hi if hi < 99 else ''}s {sum(lo <= x < hi for x in iv)}개" for lo, hi in bins))
        print("  드럼별 손 (왼/오): " + ", ".join(f"{n} {int(self.hand[m, 0])}/{int(self.hand[m, 1])}" for m, n in enumerate(NAMES)))


def main():
    env = make_env()
    drummer = Drummer(env)
    stats = Stats(len(NAMES))
    dt = env.step_dt
    print(f"[planner] env {env.num_envs}개, 악보 {args_cli.score}, 손목 기준각 {args_cli.wrist_deg}°, 타격 선행 {args_cli.lead_ms} ms")
    ep = 0
    pattern_sum = {}                                                        # 패턴별 누적: 음표 수, 교차 수, 드럼별 손
    try:
        while args_cli.episodes == 0 or ep < args_cli.episodes:
            drummer.reset(seed=ep)
            print(f"\n========== {ep + 1}번째 에피소드 ==========")
            for k in range(env.episode_length_step - 2):                    # 환경의 자동 리셋 전까지
                steps, nd, nt = drummer.step(stats)
                if args_cli.score == "pattern":
                    for e, s_, pairs, cross, gaps in drummer.note_log:
                        name = drummer.env_pattern[e]
                        drums = "+".join(NAMES[m] for m, _ in pairs)
                        assign = ", ".join(f"{'왼팔' if a == L else '오른팔'} {NAMES[m]}" for m, a in sorted(pairs, key=lambda p: p[1]))
                        gap_s = " ".join(f"({'왼' if a == L else '오'} 직전 {g:.2f}s)" for a, g in gaps.items() if g < 5.0)
                        print(f"  [{name:<5s}] env{e} {s_ * dt:5.2f}s  {drums:<14s} -> {assign:<26s} {'(교차)' if cross else '      '} {gap_s}")
                        ps = pattern_sum.setdefault(name, {"notes": 0, "cross": 0, "hand": torch.zeros((len(NAMES), 2), dtype=torch.long)})
                        ps["notes"] += 1; ps["cross"] += int(cross)
                        for m, a in pairs:
                            ps["hand"][m, a] += 1
                    continue
                if k % round(0.25 / dt) == 0:
                    s0 = int(steps[0]); r = drummer.result
                    parts = []
                    for a, name in ((L, "왼팔"), (R, "오른팔")):
                        if int(nt[0, a]) >= 0:
                            parts.append(f"{name} -> {NAMES[int(nd[0, a])]:<7s} ({(int(nt[0, a]) - s0) * dt:4.2f}s)")
                        else:
                            parts.append(f"{name} -> (배정 없음)    ")
                    q = env.robot_interface.get_joint_pos(env.robot)[0, drummer.j_waist]
                    hit = r.hit_per_arm[0]
                    hit_s = ", ".join(f"{'왼' if a == 0 else '오'} {NAMES[m]}" for a, m in hit.nonzero().tolist())
                    print(f"  {s0 * dt:5.2f}s | {' | '.join(parts)} | 허리 목표 {math.degrees(float(drummer.waist[0])):+6.1f}° "
                          f"실제 {math.degrees(float(q)):+6.1f}°" + (f" | 타격 {hit_s}" if hit_s else ""))
            ep += 1
            stats.report(f"누적 ({ep}번째 에피소드까지, env {env.num_envs}개)", dt)
            for name, ps in pattern_sum.items():
                hand = ", ".join(f"{NAMES[m]} {int(ps['hand'][m, 0])}/{int(ps['hand'][m, 1])}" for m in range(len(NAMES)) if ps["hand"][m].sum() > 0)
                print(f"  패턴 {name:<5s}: 음표 {ps['notes']}개, 교차 {ps['cross']}개 ({ps['cross'] / max(ps['notes'], 1) * 100:.0f}%) | 드럼별 손 (왼/오): {hand}")
    except WindowClosed:
        stats.report(f"창을 닫아서 종료 ({ep}번째 에피소드 진행 중)", dt)

    if not args_cli.headless and simulation_app.is_running():
        print("\n[planner] 완료. 창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            env.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
