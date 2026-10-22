"""
보상 계산 (허리·어깨 정렬 + 도달, 타격 없이 학습하는 실험)

정렬    w_align × (허리·어깨 yaw 정렬)
        손별 목표 드럼 = 스티킹 플래너가 배정한 그 손의 다음 음표 (배정 없으면 그 손이 직전에 친 드럼)
        직전 드럼은 가상 타격(음표 시각에 담당 손이 쳤다고 간주)으로 갱신 (drumrobot_env._virtual_hits)
행동 변화 (모든 관절)
        -w_act_rate × Σ_j (a_j,t - a_j,t-1)²   (행동 = 목표 속도라서 변화량 = 가속, 고정 관절은 행동이 0이라 0)
도달    -w_reach × Σ ((d - d_th) / σ)²   (d > d_th 인 손만, 두 손 모두: 쉬는 손은 직전에 친 드럼)
        d = 어깨 pitch 관절(위팔 시작점) -> 그 손이 칠 드럼 직선거리, d_th = reach_ratio × 팔 최대 길이 (상완 + 하완 + 스틱)
        σ = reach_scale (문턱부터 실제로 닿는 한계 약 0.76 m 까지 여유) -> 무차원, 한계에 닿으면 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]

from .specs import EnvRuntimeSpec, Instruments, RobotSpec


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
    w_yaw: float = 3.0              # 어깨 yaw 항 가중치
    w1: float = 2.0                 # 식 (6) 첫째 항: 양 어깨 yaw 합이 180°에 가깝게
    w2: float = 1.0                 # 식 (6) 둘째 항: 허리가 범위 가운데에 가깝게
    k_time: float = 0.0             # 시간 가중 exp(-k_time × t)
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

        # 4) 오차: 어깨 yaw 목표는 현재 실제 허리 기준 (관측에도 같은 함수를 씀)
        err = self.signed_errors(joint_pos, drum_pos)                              # (N, 3) 목표 - 현재
        e_waist = err[:, 0].abs()
        e_yaw = err[:, 1].abs() + err[:, 2].abs()

        reward = -torch.exp(-self.cfg.k_time * t) * (self.cfg.w_waist * e_waist + self.cfg.w_yaw * e_yaw)

        log_terms = {
            "align_waist_err_deg": torch.rad2deg(e_waist),
            "align_yaw_err_deg": torch.rad2deg(e_yaw / 2),     # 양 어깨 평균
        }
        return reward, log_terms

    def signed_errors(self, joint_pos: torch.Tensor, drum_pos: torch.Tensor) -> torch.Tensor:
        # 지금 목표(target_waist, target_r, target_l) 기준 정렬 오차 (목표 - 현재) [rad], -π ~ π
        # -> (N, 3) [허리, 오른 yaw, 왼 yaw]
        ar = self.env_arange
        q0 = joint_pos[:, self.j_waist]
        q1 = joint_pos[:, self.j_r]
        q2 = joint_pos[:, self.j_l]
        pR = drum_pos[ar, self.target_r, :2]
        pL = drum_pos[ar, self.target_l, :2]
        th1 = torch.atan2(pR[:, 1] - self.l * torch.sin(q0), pR[:, 0] - self.l * torch.cos(q0)) - q0
        th2 = torch.atan2(pL[:, 1] + self.l * torch.sin(q0), pL[:, 0] + self.l * torch.cos(q0)) - q0
        return torch.stack([self._wrap(self.target_waist - q0),
                            self._wrap(th1 - q1),
                            self._wrap(th2 - q2)], dim=1)

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
# 전체 보상 (정렬 + 도달)
# ============================================================
@dataclass
class RewardCfg:
    w_align: float = 0.2                    # 정렬 보상 전체 배율
    align: AlignmentRewardCfg = field(default_factory=AlignmentRewardCfg)
    w_act_rate: float = 0.1                 # 행동 변화(가속) 비용, 모든 관절에 같은 가중치: 클수록 관절이 서서히 출발·정지
                                            # (관절 하나당 w_act_rate / w_align = 0.5: 허리만 쓰던 값과 같음)
    w_reach: float = 0.3                    # 도달 벌: -w_reach × (넘은 양 / σ)², 실제로 닿는 한계(넘은 양 = σ)에서의 벌
    reach_ratio: float = 0.75               # 문턱 d_th = reach_ratio × 팔 최대 길이 (상완 + 하완 + 스틱)
    reach_scale: float = 0.16               # 무차원화 기준 σ [m]: 문턱 0.602 m -> 실제로 닿는 한계 약 0.76 m 까지의 여유


class RewardComputer:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RewardCfg,
            env: EnvRuntimeSpec,
            joint_names: list,
    ):
        self.cfg = cfg
        self.alignment = AlignmentReward(device, cfg.align, env, joint_names)
        self.prev_a = torch.zeros((env.num_envs, len(joint_names)), device=device)  # 직전 스텝 행동 (행동 변화 비용용)
        pl = RobotSpec().part_length
        L_max = pl.upper_arm + pl.lower_arm + pl.stick                             # 팔 최대 길이 [m] (0.803)
        self.d_th = cfg.reach_ratio * L_max                                         # 도달 문턱 [m] (0.75 × 0.803 = 0.602)
        self.reach_scale = cfg.reach_scale                                          # 무차원화 기준 σ [m] (문턱 -> 실제 한계 여유, 0.16)

    def reset(self, env_ids: torch.Tensor, drum_pos: torch.Tensor, init_drum: torch.Tensor):
        self.alignment.reset(env_ids=env_ids, drum_pos=drum_pos, init_drum=init_drum)
        self.prev_a[env_ids] = 0.0

    def compute(
            self,
            hit_per_arm: torch.Tensor,
            joint_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            arm_drum: torch.Tensor,
            arm_t: torch.Tensor,
            arm_valid: torch.Tensor,
            actions: torch.Tensor,
            shoulder_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # hit_per_arm: (N, 2, M) 이번 스텝 타격 (가상 타격) -> 손별 직전 드럼 갱신용
        # joint_pos:   (N, 9) 로봇 좌표계 관절각
        # drum_pos:    (N, M, 3) 드럼 윗면 중심 (env 좌표)
        # arm_drum:    (N, 2) 플래너가 배정한 손별 다음 음표 드럼 [왼손, 오른손]
        # arm_t:       (N, 2) 손별 다음 음표까지 남은 시간 (0~1, 없으면 1)
        # arm_valid:   (N, 2) 손별 다음 음표 있음
        # actions:     (N, 9) 이번 스텝 행동 (-1 ~ 1, = 관절 목표 속도 / 배율)
        # shoulder_pos:(N, 2, 3) 어깨 pitch 관절(위팔 시작점) 위치 [왼손, 오른손] (env 좌표)
        align, align_log = self.alignment.compute(
            joint_pos=joint_pos,
            drum_pos=drum_pos,
            arm_drum=arm_drum,
            arm_t=arm_t,
            arm_valid=arm_valid,
            hit_per_arm=hit_per_arm,
        )
        r_align = self.cfg.w_align * align

        # 행동 변화 비용 (모든 관절): 행동 = 목표 속도라서 스텝 사이 변화 = 가속 -> 갑자기 출발·정지하면 비쌈
        r_act_rate = -self.cfg.w_act_rate * ((actions - self.prev_a) ** 2).sum(dim=-1)
        self.prev_a = actions.clone()

        # 도달 벌: 어깨 -> 목표 드럼 직선거리가 문턱 d_th를 넘은 양을 여유 σ로 나눈 값의 제곱 (두 손 모두, 쉬는 손은 직전에 친 드럼)
        al = self.alignment
        target = torch.stack([al.target_l, al.target_r], dim=1)                     # (N, 2) 정렬 항이 방금 정한 목표 드럼 [왼손, 오른손]
        hit_pos = drum_pos[al.env_arange[:, None], target]                           # (N, 2, 3) 그 드럼의 타격점
        dist = (hit_pos - shoulder_pos).norm(dim=-1)                                 # (N, 2) √(Δx² + Δy² + Δz²)
        over = (dist - self.d_th).clamp(min=0.0)                                     # 문턱 안이면 0
        r_reach = -self.cfg.w_reach * ((over / self.reach_scale) ** 2).sum(dim=-1)

        reward = r_align + r_act_rate + r_reach
        log_terms = {
            "rew_total": reward,
            "rew_align": r_align,
            "rew_act_rate": r_act_rate,
            "rew_reach": r_reach,
            **align_log,
        }
        return reward, log_terms
