"""
최적화 기반 스티킹 스케줄러 (기존 env._compute_arm_schedule 대체 후보, 아직 env 에 연결하지 않음)

문제
  결정 변수: 앞으로 K 개 이벤트(타격이 있는 스텝)마다 비트 b_k 하나
    단일타: b=0 왼팔, b=1 오른팔
    동시 2타: b=0 (왼팔 drum0, 오른팔 drum1), b=1 그 반대. drum0 = 드럼 인덱스 큰 쪽 (기존 스케줄러와 같은 정렬)
  목적함수: J(b) = sum_k gamma^k [ lambda_m * C_mech,k + lambda_r * C_role,k ]
  풀이: 2^K 조합 전수 탐색 (K=6 이면 64 개). env 축과 조합 축을 한꺼번에 벡터 연산.
        조합마다 팔별 예측 상태(위치, 속도, 마지막 타격 시각과 드럼)를 따로 들고 이벤트 순서대로 갱신한다

C_mech = w_d * C_d + w_v * C_v + w_rearm * C_rearm   (치는 팔마다, 동시 2타는 두 팔 합)
  C_d     = |p_target - p_tip| / d_ref
  C_v     = |v_tip - v_req|^2 / v_ref^2,  v_req = (p_target - p_tip) / max(dt, eps)
  C_rearm = max(0, 1 - (t_i - t_prev) / T_rearm)^2   소프트 페널티. 짧은 간격도 금지하지는 않는다
C_role  = H[팔, 드럼]   오른손잡이 크로스핸드 기본 손 선호 (실험용 휴리스틱, 통계값 아님)

미래 상태 가정
  첫 이동: 실제 현재 스틱 끝 위치와 속도 (tip_vel, hit_detector 의 필터된 유한차분. 저장소에 FK/Jacobian 이 없어 이걸 쓴다)
  후보가 어떤 팔에 타격을 배정하면, 그 팔은 타격 시각에 드럼 중심에 있고 속도 0 이라고 가정한다
  치지 않는 팔은 마지막 예측 상태를 유지한다 (이동 시작 시각도 그대로)

배정 확정
  이벤트 식별 = 절대 스텝 (steps + ev_off). lookahead 배열 인덱스는 매 스텝 밀리므로 쓰지 않는다
  타격까지 commit_lead_s 이하가 된 이벤트는 그때의 비트를 commit 표에 적고, 이후 탐색에서 그 비트로 고정한다
  동시 2타는 비트 하나라 두 팔이 같이 확정된다. reset(env_ids) 가 표를 지운다
  지나간 스텝의 항목은 다시 읽히지 않는다 (탐색은 현재 이후 이벤트만 본다)
"""

from __future__ import annotations

import torch    # pyright: ignore[reportMissingImports]
from dataclasses import dataclass, field


@dataclass
class StickingPlannerCfg:
    step_dt: float = 0.015          # 정책 주기 [s]
    gamma: float = 1.0              # 미래 이벤트 할인율
    lambda_m: float = 1.0           # C_mech 배율
    lambda_r: float = 1.0           # C_role 배율

    # C_mech
    w_d: float = 0.69431
    d_ref: float = 0.30             # 이 거리에서 C_d = 1 [m]
    w_v: float = 7.74303
    v_ref: float = 2.0              # 속도 차이가 이 값이면 C_v = 1 [m/s]
    w_rearm: float = 14.66093
    t_rearm_s: float = 0.33795         # 재타격 기준 간격 [s]. 기존 t_rearm(7 스텝 = 0.105s)은 판정상 최소 간격이라 그보다 넓게 둔다.
                                    #   이 값과 w_rearm 이 교대로 바뀌는 간격을 정한다 (기본값에서 약 120ms 는 교대, 195ms 는 한 손)
    dt_eps_s: float = 0.015         # 남은 시간 하한 [s] (0 나누기 방지, 1 스텝)

    # C_role: 악기별 (왼손 비용, 오른손 비용). 프로젝트 악기 이름 기준
    role_cost: dict = field(default_factory=lambda: {
        "hihat":   (4.0,    0.0),
        "snare":   (0.0,    2.9124),
        "ride":    (2.358,  0.0),
        "high":    (0.0,    1.9517),
        "mid":     (2.4203, 0.0),
        "floor":   (3.0,    0.0),
        "crash_r": (0.1148, 0.0),
        "crash_l": (1.3407, 0.0),
    },)

    commit_lead_s: float = 0.26061     # 타격까지 이 시간 이하면 배정 확정 [s]
    num_notes: int = 2              # per_arm_* 의 팔별 노트 칸 수 (env 의 sched_notes)


class StickingPlanner:
    TERMS = ("d", "v", "rearm", "role")

    def __init__(
            self,
            cfg: StickingPlannerCfg,
            drum_names: list,           # drum_pos 의 드럼 순서 (Instruments().all 의 키 순서)
            num_events: int,            # K
            max_lookahead_step: int,    # 정규화 시간 1.0 = max_lookahead_step - 1 스텝 (기존과 같은 환산)
            episode_length_step: int,
            num_envs: int,
            device: torch.device | str,
    ):
        self.cfg = cfg
        self.K = num_events
        self.M = len(drum_names)
        self.L1 = float(max(max_lookahead_step - 1, 1))
        self.T = episode_length_step
        self.device = torch.device(device)

        missing = [n for n in drum_names if n not in cfg.role_cost]
        if missing:
            raise ValueError(f"role_cost 에 없는 악기: {missing}")
        self.role = torch.tensor([[cfg.role_cost[n][a] for n in drum_names] for a in range(2)],
                                 dtype=torch.float32, device=self.device)        # (2, M)

        C = 1 << num_events
        c = torch.arange(C, device=self.device).unsqueeze(1)
        k = torch.arange(num_events, device=self.device).unsqueeze(0)
        self.bits = ((c >> k) & 1).bool()                                         # (C, K) 조합 c 의 k 번째 비트

        # 확정 표 (N, T+1). 값 = 확정 비트 (0, 1), 미확정 -1. 마지막 열은 쓰지 않는 쓰레기 칸
        self.commit = torch.full((num_envs, episode_length_step + 1), -1, dtype=torch.int8, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.commit[env_ids] = -1

    def _events(self, next_hits: torch.Tensor, drum_pos: torch.Tensor):
        N, K, M = next_hits.shape[0], self.K, self.M
        multihot = next_hits[:, :, :M] > 0.5
        valid = next_hits[:, :, M + 1] > 0.5                                      # (N,K)
        t_step = next_hits[:, :, M] * self.L1                                     # (N,K) 이벤트까지 남은 스텝
        ar = torch.arange(M, device=self.device).view(1, 1, M).expand(N, K, M)
        top2 = torch.topk(torch.where(multihot, ar, torch.full_like(ar, -1)), k=2, dim=-1).values
        i0 = top2[..., 0].clamp(min=0)
        i1 = top2[..., 1].clamp(min=0)
        two = (top2[..., 1] >= 0) & valid
        p0 = torch.gather(drum_pos, 1, i0.unsqueeze(-1).expand(N, K, 3))
        p1 = torch.gather(drum_pos, 1, i1.unsqueeze(-1).expand(N, K, 3))
        return valid, two, t_step, i0, i1, p0, p1

    def solve(
            self,
            next_hits: torch.Tensor,        # (N, K, M+3)
            tip_pos: torch.Tensor,          # (N, 2, 3)
            tip_vel: torch.Tensor,          # (N, 2, 3) [m/s]
            drum_pos: torch.Tensor,         # (N, M, 3)
            since_hit_step: torch.Tensor,   # (N, 2) 팔별 마지막 타격 후 지난 스텝
            fixed_bits: torch.Tensor | None = None,   # (N, K) int (0, 1, 미확정 -1)
    ) -> tuple[torch.Tensor, dict]:
        """반환: best_bits (N,K) bool, 최적 조합의 항목별 비용 {이름: (N,)}"""
        cfg = self.cfg
        N, K = next_hits.shape[0], self.K
        C = self.bits.shape[0]
        dev = self.device
        valid, two, t_step, i0, i1, _, _ = self._events(next_hits, drum_pos)

        # 조합별 예측 상태 (N, C, 2, ...)
        pos = tip_pos.unsqueeze(1).expand(N, C, 2, 3).clone()
        vel = tip_vel.unsqueeze(1).expand(N, C, 2, 3).clone()
        t_move = torch.zeros((N, C, 2), device=dev)                               # 그 상태가 성립한 시각 [step, now=0]
        t_last = (-since_hit_step.float()).unsqueeze(1).expand(N, C, 2).clone()   # 마지막 타격 시각 [step]
        last_drum = torch.full((N, C, 2), -1, dtype=torch.long, device=dev)       # 마지막 타격 드럼 (예측)
        cost = {name: torch.zeros((N, C), device=dev) for name in self.TERMS}
        n_ar = torch.arange(N, device=dev).view(N, 1).expand(N, C)

        for k in range(K):
            b = self.bits[:, k].view(1, C)                                        # (1,C)
            v_k = valid[:, k].view(N, 1)
            two_k = two[:, k].view(N, 1)
            e0 = i0[:, k].view(N, 1).expand(N, C)
            e1 = i1[:, k].view(N, 1).expand(N, C)
            none = torch.full_like(e0, -1)
            drum_l = torch.where(two_k, torch.where(b, e1, e0), torch.where(b, none, e0))
            drum_r = torch.where(two_k, torch.where(b, e0, e1), torch.where(b, e0, none))
            drum_l = torch.where(v_k, drum_l, none)
            drum_r = torch.where(v_k, drum_r, none)
            t_k = t_step[:, k].view(N, 1)
            g = cfg.gamma ** k

            for a, drum_a in ((0, drum_l), (1, drum_r)):
                hit = drum_a >= 0                                                 # (N,C)
                d_idx = drum_a.clamp(min=0)
                tgt = drum_pos[n_ar, d_idx]                                       # (N,C,3) 드럼 중심
                delta = tgt - pos[:, :, a]
                dt_s = ((t_k - t_move[:, :, a]) * cfg.step_dt).clamp(min=cfg.dt_eps_s)
                v_req = delta / dt_s.unsqueeze(-1)
                c_d = torch.linalg.norm(delta, dim=-1) / cfg.d_ref
                c_v = ((vel[:, :, a] - v_req) ** 2).sum(dim=-1) / (cfg.v_ref ** 2)
                gap_s = (t_k - t_last[:, :, a]) * cfg.step_dt
                c_r = (1.0 - gap_s / cfg.t_rearm_s).clamp(min=0.0) ** 2
                hf = hit.float() * g
                cost["d"] += hf * cfg.lambda_m * cfg.w_d * c_d
                cost["v"] += hf * cfg.lambda_m * cfg.w_v * c_v
                cost["rearm"] += hf * cfg.lambda_m * cfg.w_rearm * c_r
                cost["role"] += hf * cfg.lambda_r * self.role[a][d_idx]

                # 치는 팔 상태 갱신: 타격 시각에 드럼 중심, 속도 0
                h3 = hit.unsqueeze(-1)
                pos[:, :, a] = torch.where(h3, tgt, pos[:, :, a])
                vel[:, :, a] = torch.where(h3, torch.zeros_like(tgt), vel[:, :, a])
                t_move[:, :, a] = torch.where(hit, t_k.expand(N, C), t_move[:, :, a])
                t_last[:, :, a] = torch.where(hit, t_k.expand(N, C), t_last[:, :, a])
                last_drum[:, :, a] = torch.where(hit, drum_a, last_drum[:, :, a])

        total = sum(cost.values())                                                # (N,C)
        bits_l = self.bits.unsqueeze(0)                                           # (1,C,K)
        # 무효 이벤트의 비트는 0 으로만 둔다 (같은 배정이 여러 조합으로 겹치지 않게)
        banned = (bits_l & ~valid.unsqueeze(1)).any(dim=-1)                       # (N,C)
        if fixed_bits is not None:
            fb = fixed_bits.long().unsqueeze(1)                                   # (N,1,K)
            banned = banned | ((fb >= 0) & (fb != bits_l.long())).any(dim=-1)
        total = torch.where(banned, torch.full_like(total, float("inf")), total)

        # 비용이 같으면 조합 번호가 작은 쪽 (= 앞쪽 이벤트부터 왼팔/분할 A 우선)
        best = total.argmin(dim=1)                                                # (N,)
        n_idx = torch.arange(N, device=dev)
        breakdown = {name: v[n_idx, best] for name, v in cost.items()}
        breakdown["total"] = total[n_idx, best]
        return self.bits[best], breakdown

    def schedule(
            self,
            steps: torch.Tensor,            # (N,) 현재 스텝
            next_hits: torch.Tensor,        # (N, K, M+3)
            tip_pos: torch.Tensor,          # (N, 2, 3)
            tip_vel: torch.Tensor,          # (N, 2, 3)
            drum_pos: torch.Tensor,         # (N, M, 3)
            last_hit_step: torch.Tensor,    # (N, 2) 팔별 마지막 타격 스텝 (env.last_hit_step, 안 쳤으면 -10000)
    ):
        """기존 _compute_arm_schedule 과 같은 출력:
        use_left_h0 (N,), use_case1_h0 (N,), per_arm_pos (N,2,NOTES,3), per_arm_time (N,2,NOTES),
        per_arm_valid (N,2,NOTES), ev_arm (N,K,M) int8, ev_off (N,K) long  (+ 로그용 breakdown)"""
        cfg = self.cfg
        N, K, M, NOTES = next_hits.shape[0], self.K, self.M, cfg.num_notes
        dev = self.device
        valid, two, t_step, i0, i1, p0, p1 = self._events(next_hits, drum_pos)
        ev_off = t_step.round().long()                                            # (N,K)

        # 이벤트 식별 = 절대 스텝. 범위 밖이나 무효 이벤트는 쓰레기 칸(T)
        abs_step = steps.long().unsqueeze(1) + ev_off
        slot = torch.where(valid & (abs_step < self.T), abs_step, torch.full_like(abs_step, self.T))
        fixed = self.commit.gather(1, slot).long()                                # (N,K)
        fixed = torch.where(slot < self.T, fixed, torch.full_like(fixed, -1))

        since = (steps.long().unsqueeze(1) - last_hit_step.long()).clamp(min=0)
        bits, breakdown = self.solve(next_hits, tip_pos, tip_vel, drum_pos, since, fixed)

        # 확정: 다가오는 첫 번째 이벤트, 또는 타격까지 commit_lead_s 이하인 이벤트 (아직 미확정이면)
        first = torch.zeros_like(valid)
        first[:, 0] = True                                                        # next_hits는 가까운 순서라 0번이 첫 음표
        near = ev_off.float() * cfg.step_dt <= cfg.commit_lead_s
        to_commit = (slot < self.T) & (fixed < 0) & (first | near)
        cur = self.commit.gather(1, slot)
        self.commit.scatter_(1, slot, torch.where(to_commit, bits.to(torch.int8), cur))
        self.commit[:, self.T] = -1

        # h0 배정 (기존 의미: 단일이면 왼팔 담당인가, 2타면 왼팔이 drum0 인가. 해당 없으면 True)
        b0 = bits[:, 0]
        one0 = valid[:, 0] & ~two[:, 0]
        use_left_h0 = torch.where(one0, ~b0, torch.ones_like(b0))
        use_case1_h0 = torch.where(two[:, 0], ~b0, torch.ones_like(b0))

        # 팔별 타격 여부와 드럼
        l_hit = valid & (two | ~bits)
        r_hit = valid & (two | bits)
        l_drum = torch.where(two & bits, i1, i0)
        r_drum = torch.where(two & ~bits, i1, i0)
        l_pos = torch.where((two & bits).unsqueeze(-1), p1, p0)
        r_pos = torch.where((two & ~bits).unsqueeze(-1), p1, p0)
        tn = next_hits[:, :, M]

        per_arm_pos = drum_pos.new_zeros((N, 2, NOTES, 3))
        per_arm_time = tn.new_ones((N, 2, NOTES))
        per_arm_valid = torch.zeros((N, 2, NOTES), dtype=torch.bool, device=dev)
        for a, hit_a, pos_a in ((0, l_hit, l_pos), (1, r_hit, r_pos)):
            rank = torch.cumsum(hit_a.long(), dim=1) - 1                          # (N,K) 그 팔의 몇 번째 노트
            for s in range(NOTES):
                sel = hit_a & (rank == s)                                         # (N,K) 많아야 하나
                any_s = sel.any(dim=1)
                per_arm_pos[:, a, s] = (sel.unsqueeze(-1).float() * pos_a).sum(dim=1)
                per_arm_time[:, a, s] = torch.where(any_s, (sel.float() * tn).sum(dim=1), per_arm_time[:, a, s])
                per_arm_valid[:, a, s] = any_s

        # 이벤트별 드럼 담당 팔 (0 왼, 1 오, -1 없음)
        ev_arm = torch.full((N, K, M), -1, dtype=torch.int8, device=dev)
        ar = torch.arange(M, device=dev).view(1, 1, M)
        ev_arm = torch.where(r_hit.unsqueeze(-1) & (ar == r_drum.unsqueeze(-1)), torch.ones_like(ev_arm), ev_arm)
        ev_arm = torch.where(l_hit.unsqueeze(-1) & (ar == l_drum.unsqueeze(-1)), torch.zeros_like(ev_arm), ev_arm)

        return use_left_h0, use_case1_h0, per_arm_pos, per_arm_time, per_arm_valid, ev_arm, ev_off, breakdown
