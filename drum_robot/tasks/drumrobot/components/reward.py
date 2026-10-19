"""
보상 계산

목표 (점수와 직접 연결)
    맞힌 타격    Σ timing × (w_hit + w_pos·위치 + λ·w_height·높이),  timing = exp(-(시간오차/σ)²)
    잘못 침      -w_wrong × 수     (다른 드럼 / 칠 게 없을 때 침 / 이미 맞힌 목표를 또 침)
    놓침         -w_miss × 수
    잘못된 접촉  -w_bad × 수       (밑면·옆면·몸통, hit detector가 분리)
유도 (목표 보상이 드물어서 방향을 알려줌)
    정렬         w_align × (허리·어깨 yaw 정렬)
    접근         w_prox × (Φ_지금 - Φ_직전),  Φ = -Σ(칠 손) exp(-k_prox·t) × 팁-최고점 수평 거리
정규화 (실제 로봇용)
    행동 변화    -w_action_rate × ‖a_t - a_(t-1)‖²

위치 점수 (0~1, 맞힌 타격을 친 손의 팁 위치)
    최고점은 _aim_point 한 곳에서 정함 (접근 보상 목표와 같음)
    북 (zone="center")  : 1 - (r/R)²                 r = 팁과 드럼 중심 수평 거리
    심벌 (zone="edge")  : 1 - ((s - s0)/w)²           s = 로봇 쪽 방향 위치 (+R 로봇 쪽 가장자리, 0 중심)
                          s0 = R - edge_margin (최고점), w = s0 (중심 쪽) / edge_margin (가장자리 쪽) -> 중심 넘으면 0
높이 점수 (0~1, 손목 기준)
    친 손의 (치기 전 최고 손목 각도 - 드럼에 닿은 순간 손목 각도) / 간격별 목표 각도
    목표 = 30° + 40°·T (T = 같은 손 타격 간격, 0.5s 이상 50°)
    λ: 최근 성공률(맞힘 / (맞힘 + 놓침), 모든 env)이 gate_start -> gate_full 일 때 0 -> 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]

from .specs import EnvRuntimeSpec, Instruments, RobotSpec
from .hit_detector import HitResult


# ============================================================
# 정렬 보상 (허리·어깨 yaw): 다음 타격 드럼 쪽으로 허리와 어깨 yaw를 정렬
#   손별 목표 드럼 = 스티킹 플래너가 배정한 그 손의 다음 음표 (배정 없으면 그 손이 직전에 친 드럼)
#   목표 허리 θ0* = (오른손 드럼, 왼손 드럼) 쌍에서 식 (6) L = w1·cos(q1 + q2) + w2·cos(2π(q0 - q0,min) / (q0,max - q0,min)) 최소
#       θ1* = atan2(y1 - l·sin q0, x1 - l·cos q0) - q0      오른손 목표 드럼 (x1, y1)
#       θ2* = atan2(y2 + l·sin q0, x2 + l·cos q0) - q0      왼손 목표 드럼 (x2, y2)
#   보상 -exp(-k_time·t) × [w_waist·|q0 - θ0*| + w_yaw·(|q1 - θ1*| + |q2 - θ2*|)]
# ============================================================

@dataclass
class AlignmentRewardCfg:
    w_waist: float = 1.0            # 허리 항 가중치
    w_yaw: float = 1.0              # 어깨 yaw 항 가중치
    w1: float = 2.0                 # 식 (6) 첫째 항: 양 어깨 yaw 합이 180°에 가깝게
    w2: float = 1.0                 # 식 (6) 둘째 항: 허리가 범위 가운데에 가깝게
    k_time: float = 3.0             # 시간 가중 exp(-k_time × t)
    waist_step_deg: float = 0.5     # 허리 훑는 간격 [deg]

class AlignmentReward:
    def __init__(
            self,
            device: torch.device | str,
            cfg: AlignmentRewardCfg,
            env: EnvRuntimeSpec,
            joint_names: list,
    ):
        self.device = device
        self.cfg = cfg

        spec = RobotSpec()
        self.l = spec.part_length.waist / 2                     # 허리 축 -> 어깨 [m]
        self.r_lim = spec.joint_limit["right_shoulder_1"]       # 오른쪽 어깨 yaw 제한 [rad]
        self.l_lim = spec.joint_limit["left_shoulder_1"]        # 왼쪽 어깨 yaw 제한 [rad]
        w_lo, w_hi = spec.joint_limit["waist_joint"]
        G = round((w_hi - w_lo) / math.radians(cfg.waist_step_deg)) + 1
        self.waist_grid = torch.linspace(w_lo, w_hi, G, device=device)     # (G,)

        # ctrl 관절 순서에서의 인덱스
        self.j_waist = joint_names.index("waist_joint")
        self.j_r = joint_names.index("right_shoulder_1")
        self.j_l = joint_names.index("left_shoulder_1")

        N, M = env.num_envs, len(Instruments().all)
        self.M = M
        self.pair_waist = torch.zeros((N, M, M), device=device)                     # [env, 오른손 드럼 i, 왼손 드럼 j] -> θ0*
        self.pair_cost = torch.full((N, M, M), float("inf"), device=device)         # -> 최소 L (배정 불가면 inf)
        self.prev_drum = torch.zeros((N, 2), device=device, dtype=torch.long)       # 손별 직전 드럼 [왼손, 오른손]
        self.target_r = torch.zeros(N, device=device, dtype=torch.long)             # 현재 목표 드럼 (오른손)
        self.target_l = torch.zeros(N, device=device, dtype=torch.long)             # 현재 목표 드럼 (왼손)
        self.target_waist = torch.zeros(N, device=device)                           # 현재 목표 허리 θ0*
        self.env_arange = torch.arange(N, device=device)

    def reset(self, env_ids: torch.Tensor, drum_pos: torch.Tensor, init_drum: torch.Tensor):
        # drum_pos:  (N, M, 3) 환경 좌표 (x, y는 IK 좌표와 같음)
        # init_drum: (n, 2) 시작 자세 드럼 [왼손, 오른손]
        xy = drum_pos[env_ids, :, :2]
        for i in range(self.M):             # 오른손 드럼
            for j in range(self.M):         # 왼손 드럼
                waist, cost = self._solve_waist(xy[:, i], xy[:, j])
                self.pair_waist[env_ids, i, j] = waist
                self.pair_cost[env_ids, i, j] = cost

        self.prev_drum[env_ids] = init_drum
        self.target_l[env_ids] = init_drum[:, 0]
        self.target_r[env_ids] = init_drum[:, 1]
        self.target_waist[env_ids] = self.pair_waist[env_ids, init_drum[:, 1], init_drum[:, 0]]

    def compute(
            self,
            joint_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            arm_drum: torch.Tensor,
            arm_t: torch.Tensor,
            arm_valid: torch.Tensor,
            hit_per_arm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # joint_pos:   (N, 9) 로봇 좌표계 관절각 (ctrl 관절 순서)
        # drum_pos:    (N, M, 3)
        # arm_drum:    (N, 2) 플래너가 배정한 손별 다음 음표 드럼 [왼손, 오른손]
        # arm_t:       (N, 2) 손별 다음 음표까지 남은 시간 (0~1, 없으면 1)
        # arm_valid:   (N, 2) 손별 다음 음표 있음
        # hit_per_arm: (N, 2, M) 이번 스텝 타격 [왼손, 오른손]
        ar = self.env_arange

        # 1) 손별 직전 드럼 갱신 (실제로 마지막에 친 드럼, 잘못 친 타격 포함)
        hit_any = hit_per_arm.any(dim=-1)                                           # (N, 2)
        self.prev_drum = torch.where(hit_any, hit_per_arm.int().argmax(dim=-1), self.prev_drum)

        # 2) 목표 드럼 = 플래너가 배정한 손별 다음 음표 (배정 없는 손은 직전 드럼에 머묾)
        self.target_l = torch.where(arm_valid[:, 0], arm_drum[:, 0], self.prev_drum[:, 0])
        self.target_r = torch.where(arm_valid[:, 1], arm_drum[:, 1], self.prev_drum[:, 1])

        # 3) 목표 허리 = 쌍 표. 그 쌍으로 가능한 허리가 없으면(교차 등) 이전 목표 유지
        ok = torch.isfinite(self.pair_cost[ar, self.target_r, self.target_l])
        self.target_waist = torch.where(ok, self.pair_waist[ar, self.target_r, self.target_l], self.target_waist)
        t = arm_t.masked_fill(~arm_valid, 1.0).amin(dim=-1)                         # 시간 가중: 가장 가까운 배정 음표 (없으면 1)

        # 4) 오차: 어깨 yaw 목표는 현재 실제 허리 기준
        q0 = joint_pos[:, self.j_waist]
        q1 = joint_pos[:, self.j_r]
        q2 = joint_pos[:, self.j_l]
        pR = drum_pos[ar, self.target_r, :2]
        pL = drum_pos[ar, self.target_l, :2]
        th1 = torch.atan2(pR[:, 1] - self.l * torch.sin(q0), pR[:, 0] - self.l * torch.cos(q0)) - q0
        th2 = torch.atan2(pL[:, 1] + self.l * torch.sin(q0), pL[:, 0] + self.l * torch.cos(q0)) - q0
        e_waist = self._wrap(q0 - self.target_waist).abs()
        e_yaw = self._wrap(q1 - th1).abs() + self._wrap(q2 - th2).abs()

        reward = -torch.exp(-self.cfg.k_time * t) * (self.cfg.w_waist * e_waist + self.cfg.w_yaw * e_yaw)

        log_terms = {
            "align_waist_err_deg": torch.rad2deg(e_waist),
            "align_yaw_err_deg": torch.rad2deg(e_yaw / 2),     # 양 어깨 평균
        }
        return reward, log_terms

    def _solve_waist(self, pR: torch.Tensor, pL: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pR, pL: (n, 2) 오른손·왼손 목표 드럼 x, y -> (θ0*, 최소 L), 가능한 허리가 없으면 L = inf
        th = self.waist_grid[None, :]                                               # (1, G)
        q1 = self._wrap_yaw(torch.atan2(pR[:, 1:2] - self.l * torch.sin(th), pR[:, 0:1] - self.l * torch.cos(th)) - th)
        q2 = self._wrap_yaw(torch.atan2(pL[:, 1:2] + self.l * torch.sin(th), pL[:, 0:1] + self.l * torch.cos(th)) - th)
        ok = (q1 >= self.r_lim[0]) & (q1 <= self.r_lim[1]) & (q2 >= self.l_lim[0]) & (q2 <= self.l_lim[1])  # (n, G)

        # 허리 범위 = 가능한 첫 허리 ~ 마지막 허리 (실제 로봇 서버와 같은 방식)
        G = th.shape[1]
        idx = torch.arange(G, device=th.device).expand_as(ok)
        lo = torch.where(ok, idx, torch.full_like(idx, G)).min(dim=1).values
        hi = torch.where(ok, idx, torch.full_like(idx, -1)).max(dim=1).values
        valid = hi >= 0
        q0_min = self.waist_grid[lo.clamp(max=G - 1)][:, None]
        span = (self.waist_grid[hi.clamp(min=0)][:, None] - q0_min).clamp(min=1e-6)

        L = self.cfg.w1 * torch.cos(q1 + q2) + self.cfg.w2 * torch.cos(2 * math.pi * (th - q0_min) / span)
        L = torch.where(ok, L, torch.full_like(L, float("inf")))
        k = L.argmin(dim=1)
        L_min = L.gather(1, k[:, None]).squeeze(1)
        return self.waist_grid[k], torch.where(valid, L_min, torch.full_like(L_min, float("inf")))

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        # 각도 차이를 -π ~ π 로
        return torch.remainder(a + math.pi, 2 * math.pi) - math.pi

    @staticmethod
    def _wrap_yaw(a: torch.Tensor) -> torch.Tensor:
        # 어깨 yaw를 -π/2 ~ 3π/2 로 (관절 제한 0~180°가 경계에 걸리지 않게)
        return torch.remainder(a + math.pi / 2, 2 * math.pi) - math.pi / 2


# ============================================================
# 전체 보상
# ============================================================
@dataclass
class RewardCfg:
    # ---- 목표 ----
    w_hit: float = 10.0                     # 오차 0으로 맞힌 타격 1개
    sigma_start_s: float = 0.05             # 시간 허용 폭 σ 시작값 [s] (넓게 시작해야 '가까워짐'을 배움)
    sigma_end_s: float = 0.02               # 시간 허용 폭 σ 최종값 [s]
    sigma_anneal_steps: int = 1_000_000     # σ를 줄이는 기간 [정책 스텝]
    w_wrong: float = 2.0                    # 잘못 친 타격 1개 (w_miss보다 작아야 '안 치기'를 배우지 않음)
    w_miss: float = 5.0                     # 놓친 목표 1개
    w_bad: float = 2.0                      # 잘못된 접촉 1개 (밑면·옆면·몸통, w_miss보다 작게)

    # ---- 타격 품질 보너스 (맞힌 타격에만, timing을 곱함) ----
    w_pos: float = 2.0                      # 위치 점수(0~1)의 크기
    w_height: float = 3.0                   # 손목 들기 점수(0~1)의 크기
    lift_T: tuple = (0.0, 0.5)              # 같은 손 타격 간격 [s] (이 사이 선형, 밖은 끝값)
    lift_deg: tuple = (30.0, 50.0)          # 간격별 목표 손목 들기 각도 [deg]
    height_window_s: float = 0.5            # 치기 전 최고 각도를 찾는 최근 시간 [s]
    gate_start: float = 0.3                 # 성공률이 이 값부터 높이 보너스를 켜기 시작
    gate_full: float = 0.5                  # 이 성공률에서 높이 보너스 100%
    gate_alpha: float = 0.01                # 성공률 평균 갱신 비율 (작을수록 천천히)

    # ---- 유도 ----
    w_align: float = 0.2                    # 정렬 보상 전체 배율
    align: AlignmentRewardCfg = field(default_factory=AlignmentRewardCfg)
    w_prox: float = 10.0                    # 접근 보상 배율
    k_prox: float = 3.0                     # 접근 시간 가중 exp(-k_prox × t)

    # ---- 정규화 ----
    w_action_rate: float = 0.01             # 행동 변화량 (명령 떨림)


class RewardComputer:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RewardCfg,
            env: EnvRuntimeSpec,
            joint_names: list,
            num_actions: int,
    ):
        self.device = device
        self.cfg = cfg
        self.dt = env.step_dt
        N = env.num_envs

        insts = list(Instruments().all.values())
        self.radius = torch.tensor([inst.radius for inst in insts], device=device)              # (M,) 패드 반지름
        self.is_edge = torch.tensor([inst.zone == "edge" for inst in insts], device=device)     # (M,) 심벌형
        self.edge_margin = torch.tensor([inst.edge_margin for inst in insts], device=device)    # (M,)
        self.edge_peak = self.radius - self.edge_margin                                        # (M,) 심벌 최고점: 중심에서 로봇 쪽으로 떨어진 거리
        self.j_wrist = [joint_names.index("left_wrist"), joint_names.index("right_wrist")]     # ctrl 관절 순서에서 [왼손, 오른손] 손목

        self.alignment = AlignmentReward(device, cfg.align, env, joint_names)
        self.prev_actions = torch.zeros((N, num_actions), device=device)                       # 직전 행동
        W = round(cfg.height_window_s / env.step_dt)
        self.wrist_buf = torch.zeros((N, W, 2), device=device)                                # 최근 W스텝 손별 손목 각도 (순환, 로봇 좌표 + = 위)
        self.since_hit = torch.zeros((N, 2), device=device, dtype=torch.long)                  # 손별 직전 타격 이후 스텝 수
        self.m_ema = torch.zeros((), device=device)                                            # 성공률 평균용: 맞힘
        self.x_ema = torch.ones((), device=device)                                             # 성공률 평균용: 놓침
                                                                                               # (놓침 1개 분량에서 시작: 학습 초반 우연한 맞힘 몇 번에 λ가 켜지지 않게)
        self.success_rate = torch.zeros((), device=device)
        self.env_arange = torch.arange(N, device=device)
        self.dt_norm = 1.0 / max(env.max_lookahead_step - 1, 1)                                # 한 스텝 동안 남은 시간 t(0~1)가 줄어드는 양
        self.prev_tip_xy = torch.zeros((N, 2, 2), device=device)                               # 직전 스텝 손별 팁 x, y
        self.prev_tip_ok = torch.zeros(N, device=device, dtype=torch.bool)                     # 리셋 직후 첫 스텝엔 직전 값 없음

    def reset(self, env_ids: torch.Tensor, drum_pos: torch.Tensor, init_drum: torch.Tensor):
        self.alignment.reset(env_ids=env_ids, drum_pos=drum_pos, init_drum=init_drum)
        self.prev_actions[env_ids] = 0.0
        self.wrist_buf[env_ids] = 0.0
        self.since_hit[env_ids] = 0
        self.prev_tip_ok[env_ids] = False                       # 리셋 직후 첫 스텝은 접근 보상 0
        # 성공률 평균(m_ema, x_ema)은 학습 전체 기준이라 리셋하지 않음

    def compute(
            self,
            result: HitResult,
            joint_pos: torch.Tensor,
            tip_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            arm_drum: torch.Tensor,
            arm_t: torch.Tensor,
            arm_valid: torch.Tensor,
            actions: torch.Tensor,
            steps: torch.Tensor,
            global_step: int,
    ) -> tuple[torch.Tensor, dict]:
        # result:      타격 판정 + 악보 매칭 결과
        # joint_pos:   (N, 9) 로봇 좌표계 관절각
        # tip_pos:     (N, 2, 3) 손별 스틱 팁 위치 (env 좌표) [왼손, 오른손]
        # drum_pos:    (N, M, 3) 드럼 윗면 중심 (env 좌표)
        # arm_drum:    (N, 2) 플래너가 배정한 손별 다음 음표 드럼 [왼손, 오른손]
        # arm_t:       (N, 2) 손별 다음 음표까지 남은 시간 (0~1, 없으면 1)
        # arm_valid:   (N, 2) 손별 다음 음표 있음
        # actions:     (N, A) 이번 스텝 행동 (-1 ~ 1)
        # steps:       (N,) 에피소드 안 현재 스텝
        # global_step: 학습 전체 정책 스텝 수 (σ 커리큘럼용)
        cfg = self.cfg
        self._record_wrist(joint_pos, steps)
        gate = self._update_gate(result)                                                        # λ (0~1, 모든 env 공통)

        # ---- 목표 ----
        sigma = self._sigma(global_step)
        timing = torch.exp(-(result.time_error / sigma) ** 2)
        timing = torch.where(result.matched, timing, torch.zeros_like(timing))                  # (N, M) 안 맞힌 드럼은 0
        pos = self._position_score(tip_pos, drum_pos, result.hit_per_arm)                       # (N, M) 0~1
        height, lift = self._height_score(joint_pos, result.hit_per_arm, steps)                 # (N, M) 0~1, (N, 2) 내려친 폭
        hit_arm = result.hit_per_arm.any(dim=-1)                                                # (N, 2) 이번에 친 손

        r_hit = cfg.w_hit * timing.sum(-1)
        r_pos = cfg.w_pos * (timing * pos).sum(-1)
        r_height = gate * cfg.w_height * (timing * height).sum(-1)
        r_wrong = -cfg.w_wrong * result.wrong_hit.float().sum(-1)
        r_miss = -cfg.w_miss * result.missed.float().sum(-1)
        r_bad = -cfg.w_bad * result.bad_contact.float().sum(-1)

        # ---- 유도 ----
        align, align_log = self.alignment.compute(
            joint_pos=joint_pos,
            drum_pos=drum_pos,
            arm_drum=arm_drum,
            arm_t=arm_t,
            arm_valid=arm_valid,
            hit_per_arm=result.hit_per_arm,
        )
        r_align = cfg.w_align * align
        prox, prox_log = self._proximity(tip_pos, drum_pos, arm_drum, arm_t, arm_valid)
        r_prox = cfg.w_prox * prox

        # ---- 정규화 ----
        r_action_rate = -cfg.w_action_rate * (actions - self.prev_actions).square().sum(-1)
        self.prev_actions = actions.clone()

        # 높이 점수를 계산한 뒤에, 이번에 친 손은 스트로크 기록을 새로 시작
        self.since_hit = torch.where(result.hit_per_arm.any(dim=-1), torch.zeros_like(self.since_hit), self.since_hit + 1)

        reward = r_hit + r_pos + r_height + r_wrong + r_miss + r_bad + r_align + r_prox + r_action_rate

        log_terms = {
            "rew_total": reward,
            "rew_hit": r_hit,
            "rew_pos": r_pos,
            "rew_height": r_height,
            "wrist_lift_deg": torch.stack(((torch.rad2deg(lift) * hit_arm).sum(-1), hit_arm.float().sum(-1)), dim=-1),   # 친 순간들의 평균 내려친 폭 [deg]
            "rew_wrong": r_wrong,
            "rew_miss": r_miss,
            "rew_bad": r_bad,
            "rew_align": r_align,
            "rew_prox": r_prox,
            "rew_action_rate": r_action_rate,
            "hit_sigma_ms": torch.full_like(reward, sigma * 1000.0),
            "height_gate": torch.full_like(reward, float(gate)),
            "success_rate_ema": torch.full_like(reward, float(self.success_rate)),
            **align_log,
            **prox_log,
        }
        return reward, log_terms

    # ---------------- 항별 계산 ----------------
    def _sigma(self, global_step: int) -> float:
        # σ: 시작값 -> 최종값 선형으로 줄임
        p = min(global_step / max(self.cfg.sigma_anneal_steps, 1), 1.0)
        return self.cfg.sigma_start_s + (self.cfg.sigma_end_s - self.cfg.sigma_start_s) * p

    def _aim_point(self, drum_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 드럼별 최고점 (위치 점수 1점 = 접근 보상 목표): 북은 중심, 심벌은 중심에서 로봇 쪽으로 edge_peak
        c = drum_pos[..., :2]
        u = -c / c.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        off = torch.where(self.is_edge, self.edge_peak, torch.zeros_like(self.edge_peak))
        return c + off[..., None] * u, u

    def _proximity(
            self,
            tip_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            arm_drum: torch.Tensor,
            arm_t: torch.Tensor,
            arm_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # 접근 보상:
        #   손마다 자기 다음 음표(플래너 배정)로, 자기 남은 시간 t에 맞춰 당김
        ar = self.env_arange
        goal, _ = self._aim_point(drum_pos)                                           # (N, M, 2) 위치 점수와 같은 최고점
        goal = torch.stack((goal[ar, arm_drum[:, 0]], goal[ar, arm_drum[:, 1]]), dim=1)   # (N, 2, 2) [왼손, 오른손]
        tip = tip_pos[..., :2]
        d_now = (tip - goal).norm(dim=-1)                                             # (N, 2) 지금 거리
        d_prev = (self.prev_tip_xy - goal).norm(dim=-1)                               # 직전 팁 위치 ~ 같은 목표
        t = arm_t                                                                     # (N, 2) 손별 다음 음표까지 남은 시간 (0~1)
        w_now = torch.exp(-self.cfg.k_prox * t)
        w_prev = torch.exp(-self.cfg.k_prox * (t + self.dt_norm))                     # 한 스텝 전 같은 음표의 가중치
        on = arm_valid & self.prev_tip_ok[:, None]                                    # 다음 음표가 있는 손만, 리셋 직후 제외
        r = torch.where(on, w_prev * d_prev - w_now * d_now, torch.zeros_like(d_now)).sum(-1)

        self.prev_tip_xy = tip.clone()
        self.prev_tip_ok[:] = True

        at_hit = arm_valid & (t == 0)                                                 # 로그: 치는 순간 거리 [cm]
        log = {"prox_dist_cm": torch.stack(((100 * d_now * at_hit).sum(-1), at_hit.float().sum(-1)), dim=-1)}
        return r, log

    def _position_score(self, tip_pos: torch.Tensor, drum_pos: torch.Tensor, hit_per_arm: torch.Tensor) -> torch.Tensor:
        # 친 손의 팁 위치로 0~1 점수 (최고점 = _aim_point: 북은 중심, 심벌은 로봇 쪽 가장자리 안쪽)
        aim, u = self._aim_point(drum_pos)                                                     # (N, M, 2) 최고점, 로봇 방향
        d = tip_pos[:, :, None, :2] - aim[:, None]                                             # (N, 2, M, 2) 팁 - 최고점
        R = self.radius
        f_center = (1 - (d.norm(dim=-1) / R) ** 2).clamp(min=0)                               # 북: 최고점(=중심) 1, 가장자리 0

        e = (d * u[:, None]).sum(dim=-1)                                                       # 최고점에서 로봇 쪽 +, 중심 쪽 -
        w = torch.where(e < 0, self.edge_peak.expand_as(e), self.edge_margin.expand_as(e))     # 중심 쪽 폭 = edge_peak -> 중심에서 0
        f_edge = (1 - (e / w) ** 2).clamp(min=0)                                               # 심벌: 중심 넘으면 0

        f = torch.where(self.is_edge, f_edge, f_center)                                        # (N, 2, M)
        return torch.where(hit_per_arm, f, torch.zeros_like(f)).amax(dim=1)                    # (N, M) 그 드럼을 친 손 기준

    def _record_wrist(self, joint_pos: torch.Tensor, steps: torch.Tensor):
        W = self.wrist_buf.shape[1]
        self.wrist_buf[self.env_arange, steps % W] = joint_pos[:, self.j_wrist]               # (N, 2) 손목 각도 [rad]

    def _height_score(self, joint_pos: torch.Tensor, hit_per_arm: torch.Tensor, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 친 손의 (치기 전 최고 손목 각도 - 지금(닿은 순간) 손목 각도) / 간격별 목표 각도
        W = self.wrist_buf.shape[1]
        age = (steps[:, None] - torch.arange(W, device=self.device)[None]) % W                # (N, W) 몇 스텝 전 기록인지
        valid = age[:, :, None] <= self.since_hit[:, None, :]                                 # (N, W, 2) 직전 타격 이후 기록만
        peak = torch.where(valid, self.wrist_buf, torch.full_like(self.wrist_buf, -math.inf)).amax(dim=1)   # (N, 2) 치기 전 최고 각도 (직전 타격 이후)
        lift = peak - joint_pos[:, self.j_wrist]                                               # (N, 2) 치기 전 최고 각도 - 닿은 순간 각도 = 내려친 폭 [rad]
        T = (self.since_hit.float() + 1) * self.dt                                             # 손별 타격 간격 [s]
        g = (lift / torch.deg2rad(self._target_lift(T))).clamp(0, 1)                           # (N, 2)
        score = torch.where(hit_per_arm, g[..., None], torch.zeros_like(g[..., None])).amax(dim=1)   # (N, M) 친 손 기준
        return score, lift

    def _target_lift(self, T: torch.Tensor) -> torch.Tensor:
        # 간격 T -> 목표 손목 들기 각도 [deg] (lift_T 사이 선형, 밖은 끝값)
        (t0, t1), (d0, d1) = self.cfg.lift_T, self.cfg.lift_deg
        return d0 + (d1 - d0) * ((T - t0) / (t1 - t0)).clamp(0, 1)

    def _update_gate(self, result: HitResult) -> torch.Tensor:
        # 높이 보너스 램프 λ: 최근 성공률이 gate_start -> gate_full 일 때 0 -> 1
        a = self.cfg.gate_alpha
        self.m_ema = (1 - a) * self.m_ema + a * result.matched.float().sum()
        self.x_ema = (1 - a) * self.x_ema + a * result.missed.float().sum()
        self.success_rate = self.m_ema / (self.m_ema + self.x_ema + 1e-6)
        span = max(self.cfg.gate_full - self.cfg.gate_start, 1e-6)
        return ((self.success_rate - self.cfg.gate_start) / span).clamp(0, 1)
