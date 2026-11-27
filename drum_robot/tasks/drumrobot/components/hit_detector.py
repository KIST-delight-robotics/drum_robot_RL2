"""
타격 감지 + 악보 매칭 클래스
"""

from __future__ import annotations

from dataclasses import dataclass
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]

from .specs import EnvRuntimeSpec, Instruments, RobotSpec

@dataclass
class HitDetectorCfg:
    contact_force_eps: float = 1e-3     # 이보다 크면 접촉 중 [N]
    min_hit_force: float = 1.0          # 타격으로 볼 최소 접촉력 [N]
                                        # (얹어 두기 0.1~0.4N, 튕겨서 재접촉 0.5~0.7N, 실제 타격 3~5N - 타격 테스트 측정값)
    max_match_window_s: float = 0.15    # 목표 매칭 범위 최댓값 [s]

    # 정상 타격 = 위에서 친 것 (아니면 잘못된 접촉: 밑면·옆면 -> 매칭 안 함)
    min_up_ratio: float = 0.5           # 위에서 침: 스틱이 받은 힘의 z / 크기 (0.5 = 수직에서 60° 이내)

    # 팁/몸통 구분 (기록만, 판정·보상에는 안 씀)
    # 팁 = 스틱 캡슐 끝 반구: 접촉 지점이 스틱 끝에서 stick_radius + tip_contact_tol 이내
    tip_contact_tol: float = 0.003      # 여유 [m] (센서가 접촉점이 여러 개면 평균 위치를 줌)

    # 얹어 두기: 같은 드럼에 연속으로 이 시간보다 오래 닿아 있으면 "대고 있음" (정상 타격의 짧은 접촉·튕김 재접촉은 통과)
    press_grace_s: float = 0.02         # [s] 물리 스텝 4개

    # 손목 내려친 폭 (보상용): 같은 손 음표 간격 T -> 실기 내려치는 시간 T_down = min(0.5T, wrist_down_max_s) (실기 state_motion_generator.cpp HIT_TO_HIT)
    wrist_T_min_s: float = 0.1          # T 하한 [s] (옛 보상 f03b27b 과 같음)
    wrist_T_max_s: float = 1.0          # T 상한 [s] (그 손이 직전에 맞힌 음표가 없으면 이 값)
    wrist_down_max_s: float = 0.25      # 내려치는 시간 상한 [s] (T >= 0.5 s 이면 이 값)

@dataclass
class HitResult:
    hit_per_arm: torch.Tensor   # (N, 2, M) bool   이번 스텝 정상 타격 (손별, 시각화·로그·보상용)
    tip_per_arm: torch.Tensor   # (N, 2, M) bool   정상 타격 중 팁(끝 반구)으로 친 것 (기록용)
    bad_per_arm: torch.Tensor   # (N, 2, M) bool   이번 스텝 잘못된 접촉 (밑면·옆면)
    bad_contact: torch.Tensor   # (N, M)    bool   잘못된 접촉 (손 구분 없음)
    hit_force: torch.Tensor     # (N, M)    float  타격 순간 접촉력 [N] (타격 없으면 0)
    contact: torch.Tensor       # (N, 2, M) bool   현재(마지막 물리 스텝) 접촉 상태
    matched: torch.Tensor       # (N, M)    bool   이번 타격이 목표와 새로 매칭됨
    time_error: torch.Tensor    # (N, M)    float  친 시각 - 목표 시각 [s] (+면 늦음, matched일 때만 유효)
    wrong_hit: torch.Tensor     # (N, M)    bool   매칭 범위 밖 타격 or 이미 맞힌 목표를 또 친 타격 or 담당이 아닌 손으로 친 타격
    wrong_arm: torch.Tensor     # (N, M)    bool   잘못 침 중 담당이 아닌 손으로 친 것 (기록용)
    missed: torch.Tensor        # (N, M)    bool   뒤쪽 매칭 범위가 끝날 때까지 못 맞힌 목표
    press: torch.Tensor         # (N, 2, M) float  얹어 두기: 이번 정책 스텝 중 연속 접촉이 press_grace_s 를 넘은 물리 스텝 비율 (0~1)
    impact_speed: torch.Tensor  # (N, 2, M) float  정상 타격의 내려치는 속도 [m/s] (접촉 시작 물리 스텝 직전 스틱 끝 수직 속도, 아래 +, 타격 없으면 0)
    matched_speed: torch.Tensor # (N, M)    float  새로 맞힌 음표를 친 손의 내려치는 속도 [m/s] (matched일 때만 유효)
    matched_wrist_vel: torch.Tensor # (N, M) float 새로 맞힌 음표를 친 손의 접촉 직전 손목 각속도 [rad/s] (내려치는 방향 +, matched일 때만 유효)
    matched_grade: torch.Tensor # (N, M)    long   새로 맞힌 음표의 악보 세기 1~7 (matched일 때만 유효)
    matched_wrist_drop: torch.Tensor # (N, M) float 새로 맞힌 음표를 친 손의 내려친 폭 [rad] = 손목 각도(접촉 T_down 전) - 손목 각도(접촉 직전) (matched일 때만 유효)
    matched_T: torch.Tensor     # (N, M)    float  새로 맞힌 음표의 같은 손 음표 간격 T [s] (wrist_T_min_s ~ wrist_T_max_s, matched일 때만 유효)

class HitDetector:
    def __init__(
            self,
            device: torch.device | str,
            cfg: HitDetectorCfg,
            env: EnvRuntimeSpec,
    ):
        self.device = device
        self.cfg = cfg
        self.env = env

        N, T = env.num_envs, env.episode_length_step
        M = len(Instruments().all)
        self.max_window = env.hit_window_step       # 매칭 범위 최댓값 [스텝]
        self.tip_len = RobotSpec().stick_radius + cfg.tip_contact_tol    # 스틱 끝에서 이 길이 안에서 닿으면 팁 [m]

        self.prev_contact = torch.zeros((N, 2, M), device=device, dtype=torch.bool)       # 직전 물리 스텝 접촉 상태
        self.contact_time = torch.zeros((N, 2, M), device=device, dtype=torch.float32)    # 손별·드럼별 연속 접촉 시간 [s] (얹어 두기 판정)
        self.target_grade = torch.zeros((N, T, M), device=device, dtype=torch.int8)       # 목표별 악보 세기 1~7 (0 = 없음)
        self.target = torch.zeros((N, T, M), device=device, dtype=torch.bool)             # 목표 (악보)
        self.matched_target = torch.zeros((N, T, M), device=device, dtype=torch.bool)     # 이미 맞힌 목표
        self.win_early = torch.zeros((N, T, M), device=device, dtype=torch.float32)       # 목표별 앞쪽(일찍) 매칭 범위 [스텝]
        self.win_late = torch.zeros((N, T, M), device=device, dtype=torch.float32)        # 목표별 뒤쪽(늦게) 매칭 범위 [스텝]
        self.target_arm = torch.full((N, T + 1, M), -1, device=device, dtype=torch.int8)  # 목표별 담당 손 (0 왼손, 1 오른손, -1 미배정), 마지막 열은 버리는 칸
        self.arm_idx = torch.tensor([0, 1], device=device).view(1, 2, 1)                 # 손 번호 [왼손, 오른손]
        self.env_arange = torch.arange(N, device=device)
        self.wrist_hist = None                                                            # (N, H, 2) 손목 각도 기록 [rad], 물리 스텝 단위, 시간 순 (첫 step 에서 만듦: H = 내려치는 시간 상한 + 정책 스텝 하나)
        self.wrist_fresh = torch.ones(N, device=device, dtype=torch.bool)                # 리셋 직후 -> 다음 step 에서 기록을 그 첫 각도로 채움 (폭 0에서 시작)
        self.last_note_step = torch.full((N, 2), -10 ** 6, device=device, dtype=torch.long)   # 손별 직전에 맞힌 음표의 악보 스텝 [왼손, 오른손] (없으면 아주 과거 -> T 상한)

    def reset(self, env_ids: torch.Tensor, rds: torch.Tensor):
        # rds: (N, T, M) 전체 악보. env_ids 행만 사용 (rds.reset 이후에 호출)
        target = rds[env_ids] > 0
        early, late = self._compute_match_window(target)
        self.target[env_ids] = target
        self.target_grade[env_ids] = rds[env_ids].to(torch.int8)                        # 악보 값 = 세기 1~7
        self.win_early[env_ids] = early
        self.win_late[env_ids] = late
        self.matched_target[env_ids] = False
        self.prev_contact[env_ids] = False
        self.contact_time[env_ids] = 0.0
        self.target_arm[env_ids] = -1
        self.wrist_fresh[env_ids] = True
        self.last_note_step[env_ids] = -10 ** 6

    def set_target_arm(self, steps: torch.Tensor, ev_arm: torch.Tensor, ev_off: torch.Tensor):
        # 플래너 배정 -> 목표별 담당 손. 확정 전 배정은 매 스텝 덮어쓰고, 확정 뒤엔 같은 값이 다시 써짐
        # steps:  (N,) 현재 스텝
        # ev_arm: (N, K, M) 0 왼손 / 1 오른손 / -1 없음
        # ev_off: (N, K) 이벤트까지 남은 스텝
        N, K, M = ev_arm.shape
        T = self.target_arm.shape[1] - 1
        abs_step = steps.long()[:, None] + ev_off                                        # (N, K) 이벤트의 절대 스텝
        has = (ev_arm >= 0).any(dim=-1) & (abs_step < T)                                 # (N, K) 실제 이벤트
        slot = torch.where(has, abs_step, torch.full_like(abs_step, T))                  # 없는 이벤트는 버리는 칸
        n = self.env_arange.view(N, 1, 1).expand(N, K, M)
        s = slot.unsqueeze(-1).expand(N, K, M)
        m = torch.arange(M, device=self.device).view(1, 1, M).expand(N, K, M)
        cur = self.target_arm[n, s, m]
        self.target_arm[n, s, m] = torch.where(ev_arm >= 0, ev_arm, cur)                # 배정된 드럼만 갱신

    # def step(self, contact_force: torch.Tensor, contact_from_tip: torch.Tensor, steps: torch.Tensor) -> HitResult:
    def step(self, contact_force: torch.Tensor, contact_from_tip: torch.Tensor, steps: torch.Tensor,
             # tip_vz_pre: torch.Tensor | None = None) -> HitResult:
             # tip_vz_pre: torch.Tensor | None = None, wrist_vel_pre: torch.Tensor | None = None) -> HitResult:
             tip_vz_pre: torch.Tensor | None = None, wrist_vel_pre: torch.Tensor | None = None,
             wrist_pos_pre: torch.Tensor | None = None) -> HitResult:
        # tip_vz_pre:       (N, D, 2) 물리 스텝별 그 물리 스텝 직전 스틱 끝 수직 속도 [m/s] (None 이면 내려치는 속도 0)
        # wrist_vel_pre:    (N, D, 2) 물리 스텝별 그 물리 스텝 직전 손목 각속도 [rad/s], 내려치는 방향 + (None 이면 0)
        # wrist_pos_pre:    (N, D, 2) 물리 스텝별 그 물리 스텝 직전 손목 각도 [rad], 로봇 좌표계 드는 쪽 + (None 이면 내려친 폭 0)
        # contact_force:    (N, D, 2, M, 3) 물리 스텝별 스틱이 드럼에서 받은 힘 벡터 [N], 시간 순 (0이 가장 이른 물리 스텝)
        # contact_from_tip: (N, D, 2, M) 물리 스텝별 실제 접촉 지점이 스틱 끝에서 떨어진 거리 [m] (접촉 없으면 NaN)
        # steps:            (N,) 현재 스텝. 이번 정책 스텝 구간 = (steps - 1, steps]
        N, D, _, M = contact_force.shape[:4]
        T = self.env.episode_length_step
        W = self.max_window
        dt = self.env.step_dt
        ar = self.env_arange

        # ---- 1) 타격 이벤트: 물리 스텝 단위로 접촉이 새로 시작된 순간 + 충분한 세기 ----
        f_norm = contact_force.norm(dim=-1)                                     # (N, D, 2, M)
        c = f_norm > self.cfg.contact_force_eps
        prev = torch.cat([self.prev_contact.unsqueeze(1), c[:, :-1]], dim=1)   # 각 물리 스텝의 직전 접촉 상태
        onset = c & ~prev                                                       # 이 물리 스텝에 접촉 시작
        peak_force = f_norm.amax(dim=1)                                         # (N, 2, M)
        # hit_any = onset.any(dim=1) & (peak_force > self.cfg.min_hit_force)      # (N, 2, M) 정상·잘못된 접촉 모두
        hit_any = onset.any(dim=1)
        first = torch.argmax(onset.int(), dim=1)                                # 처음 접촉이 시작된 물리 스텝
        self.prev_contact = c[:, -1]                                            # 마지막 물리 스텝 접촉 상태

        # ---- 얹어 두기: 물리 스텝마다 연속 접촉 시간 누적 (떨어지면 0) -> press_grace_s 를 넘은 물리 스텝 비율 ----
        dt_phys = dt / D                                                        # 물리 스텝 [s]
        over = torch.zeros_like(c)                                              # (N, D, 2, M) 이 물리 스텝에 대고 있음
        for d in range(D):
            self.contact_time = torch.where(c[:, d], self.contact_time + dt_phys, torch.zeros_like(self.contact_time))
            over[:, d] = self.contact_time > self.cfg.press_grace_s
        press = over.float().mean(dim=1)                                        # (N, 2, M) 0~1

        # ---- 정상 타격 = 위에서 친 것. 아니면 잘못된 접촉 (매칭하지 않아서 목표가 남음) ----
        k = f_norm.argmax(dim=1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        f_peak = contact_force.gather(1, k).squeeze(1)                          # (N, 2, M, 3) 최대 힘이 난 물리 스텝의 힘
        from_above = f_peak[..., 2] >= self.cfg.min_up_ratio * peak_force       # 드럼이 스틱을 위로 밀어냄
        hit_per_arm = hit_any & from_above                                      # (N, 2, M) 정상 타격 (팁/몸통 상관없이)
        bad_per_arm = hit_any & ~from_above                                     # (N, 2, M) 잘못된 접촉 (밑면·옆면)

        # 팁/몸통 구분 (기록용): 접촉이 시작된 물리 스텝의 실제 접촉 지점이 스틱 끝 반구 안인지
        from_tip = contact_from_tip.gather(1, first.unsqueeze(1)).squeeze(1)    # (N, 2, M)
        tip_per_arm = hit_per_arm & (from_tip <= self.tip_len)                  # NaN(접촉 지점 없음)이면 False

        # 내려치는 속도: 접촉이 시작된 물리 스텝 직전의 스틱 끝 수직 속도 (아래 +) [m/s], 정상 타격 아니면 0
        if tip_vz_pre is not None:
            vz = tip_vz_pre[:, :, :, None].expand(-1, -1, -1, M)                # (N, D, 2, M)
            impact_speed = -vz.gather(1, first.unsqueeze(1)).squeeze(1) * hit_per_arm   # (N, 2, M)
        else:
            impact_speed = torch.zeros_like(peak_force)

        # 손목 각속도: 접촉이 시작된 물리 스텝 직전의 손목 각속도 (내려치는 방향 +) [rad/s], 정상 타격 아니면 0
        if wrist_vel_pre is not None:
            wv = wrist_vel_pre[:, :, :, None].expand(-1, -1, -1, M)             # (N, D, 2, M)
            wrist_vel = wv.gather(1, first.unsqueeze(1)).squeeze(1) * hit_per_arm       # (N, 2, M)
        else:
            wrist_vel = torch.zeros_like(peak_force)

        # 손목 각도 기록: 물리 스텝 단위로 뒤에 붙임 -> 이번 정책 스텝의 D개 = 마지막 [H-D, H)
        if wrist_pos_pre is not None:
            if self.wrist_hist is None:
                H = round(self.cfg.wrist_down_max_s / dt_phys) + D
                self.wrist_hist = wrist_pos_pre[:, :1].expand(-1, H, -1).clone()
            H = self.wrist_hist.shape[1]
            fresh = self.wrist_fresh
            self.wrist_hist[fresh] = wrist_pos_pre[fresh, :1].expand(-1, H, -1)        # 리셋 직후: 첫 각도로 채움 (폭 0)
            self.wrist_fresh[:] = False
            self.wrist_hist = torch.cat([self.wrist_hist[:, D:], wrist_pos_pre], dim=1)
            hist_arm = self.wrist_hist.permute(0, 2, 1)                                 # (N, 2, H)
            idx_c = H - D + first                                                       # (N, 2, M) 접촉이 시작된 물리 스텝의 기록 위치
            theta_c = hist_arm.gather(2, idx_c)                                         # (N, 2, M) 접촉 직전 손목 각도

        # 타격 시각 [스텝, 연속값] = 접촉이 시작된 물리 스텝 구간의 가운데 (D=2이면 steps-0.75 또는 steps-0.25)
        t_arm = (steps - 1).view(N, 1, 1).float() + (first.float() + 0.5) / D          # (N, 2, M) 손별
        hit_force = (peak_force * hit_per_arm).amax(dim=1)                              # (N, M)

        # ---- 2) 매칭: 같은 드럼 목표 k의 범위 [k - 앞쪽, k + 뒤쪽] 안 타격 중 담당 손으로 친 것만 맞힘 ----
        # 범위끼리 맞닿기만 하고 겹치지 않아서 후보는 최대 1개
        # 범위 안이지만 담당이 아닌 손으로 친 타격 -> 잘못 침 (목표는 남아서 담당 손이 아직 칠 수 있음)
        inf = torch.full_like(t_arm, float("inf"))
        matched = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        duplicate = torch.zeros_like(matched)
        wrong_arm = torch.zeros_like(matched)
        time_error = torch.zeros((N, M), device=self.device)
        matched_speed = torch.zeros((N, M), device=self.device)
        matched_wrist_vel = torch.zeros((N, M), device=self.device)
        matched_grade = torch.zeros((N, M), device=self.device, dtype=torch.long)
        matched_wrist_drop = torch.zeros((N, M), device=self.device)
        matched_T = torch.zeros((N, M), device=self.device)
        remaining = hit_per_arm.clone()                                                # (N, 2, M) 손별로 매칭
        for d in range(-W - 1, W + 1):                         # 목표 후보 시점 k = steps + d
            k = steps + d
            valid = ((k >= 0) & (k < T)).view(N, 1, 1)
            kc = k.clamp(0, T - 1)
            kf = kc.float().view(N, 1, 1)
            in_window = (t_arm >= kf - self.win_early[ar, kc][:, None]) & (t_arm <= kf + self.win_late[ar, kc][:, None])
            m = remaining & valid & self.target[ar, kc][:, None] & in_window            # (N, 2, M) 이 목표 범위 안 타격
            owner = self.target_arm[ar, kc][:, None].long()                             # (N, 1, M) 담당 손
            arm_ok = (owner < 0) | (owner == self.arm_idx)                              # 미배정이면 아무 손
            ok_hit = (m & arm_ok).any(dim=1)                                            # (N, M) 담당 손으로 침
            done = self.matched_target[ar, kc]
            new = ok_hit & ~done                               # 새로 맞힘
            duplicate |= ok_hit & done                         # 이미 맞힌 목표를 또 침 -> 잘못 침
            wrong_arm |= (m & ~arm_ok).any(dim=1)              # 담당이 아닌 손으로 침 -> 잘못 침
            hit_t = torch.where(m & arm_ok, t_arm, inf).amin(dim=1)                     # (N, M) 담당 손 타격 시각
            self.matched_target[ar, kc] = done | new
            matched |= new
            time_error = torch.where(new, (hit_t - kc.float()[:, None]) * dt, time_error)   # +면 늦음 [s]
            spd = torch.where(m & arm_ok, impact_speed, torch.full_like(impact_speed, -float("inf"))).amax(dim=1)   # (N, M) 담당 손 내려치는 속도
            matched_speed = torch.where(new, spd, matched_speed)
            wsp = torch.where(m & arm_ok, wrist_vel, torch.full_like(wrist_vel, -float("inf"))).amax(dim=1)         # (N, M) 담당 손 손목 각속도
            matched_wrist_vel = torch.where(new, wsp, matched_wrist_vel)
            matched_grade = torch.where(new, self.target_grade[ar, kc].long(), matched_grade)   # 맞힌 음표의 악보 세기
            if wrist_pos_pre is not None:
                # 내려친 폭: 같은 손 음표 간격 T (이번 음표 - 그 손이 직전에 맞힌 음표, 악보 시각) -> T_down 전 각도 - 접촉 직전 각도
                T_arm = ((kc[:, None] - self.last_note_step).float() * dt).clamp(self.cfg.wrist_T_min_s, self.cfg.wrist_T_max_s)   # (N, 2) [s]
                T_down = torch.clamp(0.5 * T_arm, max=self.cfg.wrist_down_max_s)                                 # (N, 2) 실기 내려치는 시간 [s]
                lag = torch.round(T_down / dt_phys).long()                                                         # (N, 2) [물리 스텝]
                drop = hist_arm.gather(2, (idx_c - lag[..., None]).clamp(min=0)) - theta_c                         # (N, 2, M) 내려친 폭 [rad]
                dsel = torch.where(m & arm_ok, drop, -inf)
                a_best = dsel.argmax(dim=1, keepdim=True)                                                          # (N, 1, M) 담당 손 (둘 다면 폭 큰 손)
                matched_wrist_drop = torch.where(new, dsel.gather(1, a_best).squeeze(1), matched_wrist_drop)
                matched_T = torch.where(new, T_arm[..., None].expand_as(drop).gather(1, a_best).squeeze(1), matched_T)
                new_arm = (m & arm_ok & new[:, None, :]).any(dim=-1)                                               # (N, 2) 이번에 음표를 새로 맞힌 손
                self.last_note_step = torch.where(new_arm, kc[:, None], self.last_note_step)
            remaining &= ~m
        wrong_hit = remaining.any(dim=1) | duplicate | wrong_arm   # 어떤 범위에도 안 든 타격 + 중복 타격 + 다른 손 타격

        # ---- 3) 놓침: 뒤쪽 범위 끝(k + 뒤쪽)이 이번 정책 스텝 구간 안에 들어온 못 맞힌 목표 ----
        missed = torch.zeros_like(matched)
        s_lo = (steps - 1).float().unsqueeze(-1)
        s_hi = steps.float().unsqueeze(-1)
        for d in range(-W - 1, 1):
            k = steps + d
            valid = (k >= 0).unsqueeze(-1)
            kc = k.clamp(0, T - 1)
            end = kc.float().unsqueeze(-1) + self.win_late[ar, kc]
            missed |= valid & self.target[ar, kc] & ~self.matched_target[ar, kc] & (end > s_lo) & (end <= s_hi)

        return HitResult(
            hit_per_arm=hit_per_arm,
            tip_per_arm=tip_per_arm,
            bad_per_arm=bad_per_arm,
            bad_contact=bad_per_arm.any(dim=1),
            hit_force=hit_force,
            contact=c[:, -1],
            matched=matched,
            time_error=time_error,
            wrong_hit=wrong_hit,
            wrong_arm=wrong_arm,
            missed=missed,
            press=press,
            impact_speed=impact_speed,
            matched_speed=matched_speed,
            matched_wrist_vel=matched_wrist_vel,
            matched_grade=matched_grade,
            matched_wrist_drop=matched_wrist_drop,
            matched_T=matched_T,
        )

    def _compute_match_window(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # target: (n, T, M) bool -> 목표별 (앞쪽, 뒤쪽) 매칭 범위 [스텝]
        # 앞쪽 = min(최대 범위, 이전 목표까지 간격 / 2), 뒤쪽 = min(최대 범위, 다음 목표까지 간격 / 2)
        n, T, M = target.shape
        W = float(self.max_window)
        BIG = 10 ** 6
        idx = torch.arange(T, device=target.device).view(1, T, 1).expand(n, T, M)

        # 같은 드럼 이전 목표 시점 (자기 제외)
        last = torch.cummax(torch.where(target, idx, torch.full_like(idx, -BIG)), dim=1).values
        prev_t = torch.full_like(last, -BIG)
        prev_t[:, 1:] = last[:, :-1]

        # 같은 드럼 다음 목표 시점 (자기 제외)
        first = torch.where(target, idx, torch.full_like(idx, BIG))
        first = torch.flip(torch.cummin(torch.flip(first, dims=[1]), dim=1).values, dims=[1])
        next_t = torch.full_like(first, BIG)
        next_t[:, :-1] = first[:, 1:]

        early = torch.clamp((idx - prev_t).float() / 2, max=W)
        late = torch.clamp((next_t - idx).float() / 2, max=W)
        zero = torch.zeros_like(early)
        return torch.where(target, early, zero), torch.where(target, late, zero)
