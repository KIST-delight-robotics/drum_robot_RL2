"""
보상 계산 (허리·어깨 정렬, 타격 없이 학습하는 실험)

정렬    w_align × (허리·어깨 yaw 정렬)
        손별 목표 드럼 = 스티킹 플래너가 배정한 그 손의 다음 음표 (배정 없으면 그 손이 직전에 친 드럼)
        직전 드럼은 가상 타격(음표 시각에 담당 손이 쳤다고 간주)으로 갱신 (drumrobot_env._virtual_hits)
        목표 허리 = (두 팔이 드럼을 향할 수 있음 ∩ 두 손 다 d ≤ d_th) 인 허리 중 식 (6) L 최소
                    교집합이 없으면 방향 가능한 허리 중 넘은 거리² 합 최소 (별도의 도달 벌은 없음)
                    단, 이전 목표 허리가 새 쌍에서도 교집합 안이면 바꾸지 않고 유지
행동 변화 (모든 관절)
        -w_act_rate × Σ_j (a_j,t - a_j,t-1)²   (행동 = 목표 속도라서 변화량 = 가속, 고정 관절은 행동이 0이라 0)
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
#       후보 허리 = 방향 가능 (q1, q2 관절 범위 안) ∩ 두 손 다 어깨 -> 드럼 거리 ≤ d_th, q0,min ~ q0,max = 그 범위의 양 끝
#       이전 목표 허리가 새 쌍의 후보 허리 안이면 θ0* 유지 (도달 가능하면 허리를 움직이지 않음)
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
    reach_ratio: float = 0.9       # 거리 문턱 d_th = reach_ratio × 팔 최대 길이 (상완 + 하완 + 스틱)

class AlignmentReward:
    def __init__(
            self,
            device: torch.device | str,
            cfg: AlignmentRewardCfg,
            env: EnvRuntimeSpec,
            joint_names: list,
            shoulder_z: float,
    ):
        self.device = device
        self.cfg = cfg

        spec = RobotSpec()
        self.l = spec.part_length.waist / 2                     # 허리 축 -> 어깨 [m]
        pl = spec.part_length
        self.d_th = cfg.reach_ratio * (pl.upper_arm + pl.lower_arm + pl.stick)     # d_threshold = 팔 최대 길이 × 비율
        self.z_s = shoulder_z                                   # 어깨 높이 [m] (env 좌표, 어깨 pitch 관절)
        self.r_lim = spec.joint_limit["right_shoulder_1"]       # 오른쪽 어깨 yaw 제한 [rad]
        self.l_lim = spec.joint_limit["left_shoulder_1"]        # 왼쪽 어깨 yaw 제한 [rad]
        w_lo, w_hi = spec.joint_limit["waist_joint"]
        G = round((w_hi - w_lo) / math.radians(cfg.waist_step_deg)) + 1 # 허리 갯수
        self.waist_grid = torch.linspace(w_lo, w_hi, G, device=device)     # (G,)

        # ctrl 관절 순서에서의 인덱스
        self.j_waist = joint_names.index("waist_joint")
        self.j_r = joint_names.index("right_shoulder_1")
        self.j_l = joint_names.index("left_shoulder_1")

        N, M = env.num_envs, len(Instruments().all)
        self.M = M
        self.pair_waist = torch.zeros((N, M, M), device=device)                     # [env, 오른손 드럼 i, 왼손 드럼 j] -> θ0*
        self.pair_cost = torch.full((N, M, M), float("inf"), device=device)         # -> 고른 허리의 점수 (방향부터 불가면 inf)
        self.prev_drum = torch.zeros((N, 2), device=device, dtype=torch.long)       # 손별 직전 드럼 [왼손, 오른손] -> 드럼이 배정이 되지 않으면 직전 드럼으로 목표 유지
        self.target_r = torch.zeros(N, device=device, dtype=torch.long)             # 현재 목표 드럼 (오른손)
        self.target_l = torch.zeros(N, device=device, dtype=torch.long)             # 현재 목표 드럼 (왼손)
        self.target_waist = torch.zeros(N, device=device)                           # 현재 목표 허리 θ0*
        self.env_arange = torch.arange(N, device=device)

    def reset(self, env_ids: torch.Tensor, drum_pos: torch.Tensor, init_drum: torch.Tensor):
        # drum_pos:  (N, M, 3) 환경 좌표 (x, y는 IK 좌표와 같음)
        # init_drum: (n, 2) 시작 자세 드럼 [왼손, 오른손]
        p = drum_pos[env_ids]                # (n, M, 3) 높이차까지 쓰므로 x, y, z
        for i in range(self.M):             # 오른손 드럼
            for j in range(self.M):         # 왼손 드럼
                waist, cost = self._solve_waist(p[:, i], p[:, j])
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

        # 3) 목표 허리: 이전 목표 허리가 새 쌍에서도 B 안(방향 가능 ∩ 두 손 다 d ≤ d_th)이면 유지
        #    B 밖이면 쌍 표, 표도 불가(교차 등)면 이전 목표 유지
        pR = drum_pos[ar, self.target_r]                                            # (N, 3)
        pL = drum_pos[ar, self.target_l]
        keep = self._reach_check(pR, pL, self.target_waist[:, None])[3].squeeze(1)  # (N,) 이전 목표 ∈ B
        ok = torch.isfinite(self.pair_cost[ar, self.target_r, self.target_l])
        table = torch.where(ok, self.pair_waist[ar, self.target_r, self.target_l], self.target_waist)
        self.target_waist = torch.where(keep, self.target_waist, table)
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

    def _reach_check(self, pR: torch.Tensor, pL: torch.Tensor, th: torch.Tensor):
        # pR, pL: (n, 3) 오른손·왼손 목표 드럼 x, y, z
        # th:     허리 각도, (1, G) = 표 만들 때 후보 전체 / (n, 1) = 매 스텝 직전 목표 허리
        # -> q1, q2 (필요한 어깨 yaw), ok_yaw (방향 가능), ok_all (교집합 B), dR, dL (어깨 -> 드럼 거리)
        q1 = self._wrap_yaw(torch.atan2(pR[:, 1:2] - self.l * torch.sin(th), pR[:, 0:1] - self.l * torch.cos(th)) - th)
        q2 = self._wrap_yaw(torch.atan2(pL[:, 1:2] + self.l * torch.sin(th), pL[:, 0:1] + self.l * torch.cos(th)) - th)
        ok_yaw = (q1 >= self.r_lim[0]) & (q1 <= self.r_lim[1]) & (q2 >= self.l_lim[0]) & (q2 <= self.l_lim[1])

        # 어깨 -> 드럼 거리 (어깨: 허리 축에서 l, 높이 z_s)
        dR = torch.sqrt((pR[:, 0:1] - self.l * torch.cos(th)) ** 2 + (pR[:, 1:2] - self.l * torch.sin(th)) ** 2
                        + (pR[:, 2:3] - self.z_s) ** 2)
        dL = torch.sqrt((pL[:, 0:1] + self.l * torch.cos(th)) ** 2 + (pL[:, 1:2] + self.l * torch.sin(th)) ** 2
                        + (pL[:, 2:3] - self.z_s) ** 2)
        ok_all = ok_yaw & (dR <= self.d_th) & (dL <= self.d_th)
        return q1, q2, ok_yaw, ok_all, dR, dL

    def _solve_waist(self, pR: torch.Tensor, pL: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pR, pL: (n, 3) 오른손·왼손 목표 드럼 x, y, z -> (θ0*, 점수), 방향부터 가능한 허리가 없으면 점수 = inf
        # 목표 허리 = 방향 가능 ∩ 두 손 다 거리 ≤ d_th (교집합 B) 중 L 최소. B가 없으면 방향 가능한 허리 중 넘은 거리² 합 최소
        th = self.waist_grid[None, :]                                               # (1, G)
        q1, q2, ok_yaw, ok_all, dR, dL = self._reach_check(pR, pL, th)              # (n, G) 방향 가능, 교집합 B, 거리
        has_all = ok_all.any(dim=1, keepdim=True)
        ok = torch.where(has_all, ok_all, ok_yaw)                                   # B가 없으면 방향 가능 범위

        # 허리 범위 = 가능한 첫 허리 ~ 마지막 허리 (실제 로봇 서버와 같은 방식)
        G = th.shape[1]
        idx = torch.arange(G, device=th.device).expand_as(ok)
        lo = torch.where(ok, idx, torch.full_like(idx, G)).min(dim=1).values
        hi = torch.where(ok, idx, torch.full_like(idx, -1)).max(dim=1).values
        valid = hi >= 0
        q0_min = self.waist_grid[lo.clamp(max=G - 1)][:, None]
        span = (self.waist_grid[hi.clamp(min=0)][:, None] - q0_min).clamp(min=1e-6)

        L = self.cfg.w1 * torch.cos(q1 + q2) + self.cfg.w2 * torch.cos(2 * math.pi * (th - q0_min) / span)
        over2 = (dR - self.d_th).clamp(min=0.0) ** 2 + (dL - self.d_th).clamp(min=0.0) ** 2   # 넘은 거리² 합
        score = torch.where(has_all, L, over2)                                      # B가 있으면 L, 없으면 덜 넘는 곳
        score = torch.where(ok, score, torch.full_like(score, float("inf")))
        k = score.argmin(dim=1)
        s_min = score.gather(1, k[:, None]).squeeze(1)
        return self.waist_grid[k], torch.where(valid, s_min, torch.full_like(s_min, float("inf")))

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        # 각도 차이를 -π ~ π 로
        return torch.remainder(a + math.pi, 2 * math.pi) - math.pi

    @staticmethod
    def _wrap_yaw(a: torch.Tensor) -> torch.Tensor:
        # 어깨 yaw를 -π/2 ~ 3π/2 로 (관절 제한 0~180°가 경계에 걸리지 않게)
        return torch.remainder(a + math.pi / 2, 2 * math.pi) - math.pi / 2


# ============================================================
# 전체 보상 (정렬)
# ============================================================
@dataclass
class RewardCfg:
    w_align: float = 0.2                    # 정렬 보상 전체 배율
    align: AlignmentRewardCfg = field(default_factory=AlignmentRewardCfg)
    w_act_rate: float = 0.1                 # 행동 변화(가속) 비용, 모든 관절에 같은 가중치: 클수록 관절이 서서히 출발·정지
                                            # (관절 하나당 w_act_rate / w_align = 0.5: 허리만 쓰던 값과 같음)


class RewardComputer:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RewardCfg,
            env: EnvRuntimeSpec,
            joint_names: list,
            shoulder_z: float,
    ):
        self.cfg = cfg
        self.alignment = AlignmentReward(device, cfg.align, env, joint_names, shoulder_z)
        self.prev_a = torch.zeros((env.num_envs, len(joint_names)), device=device)  # 직전 스텝 행동 (행동 변화 비용용)

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
    ) -> tuple[torch.Tensor, dict]:
        # hit_per_arm: (N, 2, M) 이번 스텝 타격 (가상 타격) -> 손별 직전 드럼 갱신용
        # joint_pos:   (N, 9) 로봇 좌표계 관절각
        # drum_pos:    (N, M, 3) 드럼 윗면 중심 (env 좌표)
        # arm_drum:    (N, 2) 플래너가 배정한 손별 다음 음표 드럼 [왼손, 오른손]
        # arm_t:       (N, 2) 손별 다음 음표까지 남은 시간 (0~1, 없으면 1)
        # arm_valid:   (N, 2) 손별 다음 음표 있음
        # actions:     (N, 9) 이번 스텝 행동 (-1 ~ 1, = 관절 목표 속도 / 배율)
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

        reward = r_align + r_act_rate
        log_terms = {
            "rew_total": reward,
            "rew_align": r_align,
            "rew_act_rate": r_act_rate,
            **align_log,
        }
        return reward, log_terms
