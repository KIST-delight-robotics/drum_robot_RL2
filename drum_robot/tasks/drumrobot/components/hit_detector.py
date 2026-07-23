"""
타격 감지 클래스
"""

from __future__ import annotations
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]
from dataclasses import dataclass

from .specs import EnvRuntimeSpec, Instruments

@dataclass
class HitDetectorCfg:
    alpha: float = 0.6  # 속도 계산 필터

    # 타격 판정
    drum_xy_radius: float = 0.13
    drum_z_range: float = 0.07
    min_impact_velocity: float = 0.2
    rearm_height: float = 0.18
    hit_window_step: int = 3

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

        instruments = Instruments()
        self.num_drums = len(instruments.all)

        self._alloc_buffers()
        self.env_arange = torch.arange(self.env.num_envs, device=self.device)

    def detect_hit(
            self,
            tip_pos: torch.Tensor,
            drum_pos: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 팁 속도
        prev_tip_pos = self.tip_pos
        prev_tip_vel = self.tip_vel
        tip_vel = self._compute_tip_velocity(tip_pos=tip_pos, prev_tip_pos=prev_tip_pos, prev_tip_vel=prev_tip_vel)

        # 팁 드럼 거리 계산
        dist_xy_sq, diff_z = self._compute_tip_drum_dist_sq(tip_pos, drum_pos)

        # 접촉 확인
        contact_mask = self._check_contact_drum(dist_xy_sq, diff_z)

        # 타격 판정
        hit_per_arm = self._check_hitting(
            contact_mask=contact_mask,
            hit_armed=self.hit_armed,
            tip_vel=tip_vel,
            diff_z=diff_z,
        )   # (N, 2, M)

        # 양팔 중 하나라도 해당 drum을 strike하면 hit
        hit_mask = torch.any(hit_per_arm, dim=1)   # (N, M)

        # re-arm 확인
        prev_hit_armed = self.hit_armed
        hit_armed = self._check_rearm(prev_hit_armed, contact_mask, diff_z)

        # 기록
        self.tip_pos = tip_pos
        self.prev_tip_pos = prev_tip_pos
        self.tip_vel = tip_vel
        self.hit_armed = hit_armed

        return (
            hit_mask,
            tip_vel, prev_tip_pos,
            prev_hit_armed, hit_per_arm,
        )
    
    def get_result(
            self,
            hit_mask: torch.Tensor,
            steps: torch.Tensor,
            rds: torch.Tensor,
            rds_visit: torch.Tensor,
            rds_visit_vel: torch.Tensor,
            rds_visit_wrist: torch.Tensor,
            rds_visit_elbow_apex: torch.Tensor,
            rds_T: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 현재 ±W 윈도우 타겟
        window_target_union = self._window_target_union(rds, steps)  # (N, M)

        # 잘못친 타격: 윈도우 밖 타격
        wrong_hit = hit_mask & (~window_target_union)

        # 유효 타격(윈도우 안 타격) → 그 타겟 done (재타 downward 억제).
        # 윈도우 밖 오타/미스는 done 안 켜서 다시 칠 수 있음
        valid_hit = hit_mask & window_target_union
        self.target_done = self.target_done | valid_hit

        # 윈도우 끝났을 때 타격 성공 확인 (+ 성공 타격의 속도/손목각속도 + 그 타겟의 목표 velocity). 끝난 타겟은 done 해제
        success, missed_target, time_error, success_impact_vel, success_wrist_ang_vel, target_velocity, success_elbow_apex, target_T = self._finalize_target_outcomes(
            rds=rds,
            rds_visit=rds_visit,
            rds_visit_vel=rds_visit_vel,
            rds_visit_wrist=rds_visit_wrist,
            rds_visit_elbow_apex=rds_visit_elbow_apex,
            rds_T=rds_T,
            steps=steps,
        )

        return success, wrong_hit, missed_target, time_error, success_impact_vel, success_wrist_ang_vel, self.target_done, target_velocity, success_elbow_apex, target_T
    
    def reset(self, env_ids: torch.Tensor, tip_pos: torch.Tensor):
        # 팁 위치/속도 리셋
        self.tip_pos[env_ids] = tip_pos[env_ids]
        self.prev_tip_pos[env_ids] = tip_pos[env_ids]
        self.tip_vel[env_ids] = 0.0

        # 타격 상태 버퍼 리셋
        self.hit_armed[env_ids] = False # 초기 상태는 준비 안됨. 이 후 스텝에서 갱신
        self.target_done[env_ids] = False

    def get_value_for_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tip_pos, self.hit_armed
    
    def _alloc_buffers(self):
        N = self.env.num_envs
        M = self.num_drums

        # 로봇의 tip 위치를 저장할 텐서
        self.tip_pos = torch.zeros((N, 2, 3), device=self.device)
        self.prev_tip_pos = torch.zeros((N, 2, 3), device=self.device)

        # 로봇의 tip 속도 저장할 텐서
        self.tip_vel = torch.zeros((N, 2, 3), device=self.device)

        # 타격 상태 버퍼
        self.hit_armed = torch.ones((N, 2, M), device=self.device, dtype=torch.bool)

        # 타겟 완료 버퍼: 타겟 윈도우 안에서 유효 타격하면 True (재타 downward 억제용)
        self.target_done = torch.zeros((N, M), device=self.device, dtype=torch.bool)
    
    def _compute_tip_velocity(self, tip_pos: torch.Tensor, prev_tip_pos: torch.Tensor, prev_tip_vel: torch.Tensor) -> torch.Tensor:
        tip_vel = (tip_pos - prev_tip_pos) / self.env.step_dt

        tip_vel_f = (1 - self.cfg.alpha) * tip_vel + self.cfg.alpha * prev_tip_vel

        return tip_vel_f
    
    def _compute_tip_drum_dist_sq(self, tip_pos: torch.Tensor, drum_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tip_pos: (N, 2, 3), drum_pos: (N, M, 3)
        # N: num env, M: num drum

        diff_xy = tip_pos[:, :, None, 0:2] - drum_pos[:, None, :, 0:2] # (N, 2, M, 2)

        dist_xy_sq = torch.sum(diff_xy * diff_xy, dim=-1)    # (N, 2, M)

        diff_z = tip_pos[:, :, None, 2] - drum_pos[:, None, :, 2] # (N, 2, M)

        return dist_xy_sq, diff_z
    
    def _check_contact_drum(self, dist_xy_sq: torch.Tensor, diff_z: torch.Tensor) -> torch.Tensor:
        # N: num env, M: num drum
        # xy 범위 확인
        radius_sq = self.cfg.drum_xy_radius ** 2
        in_xy_range = dist_xy_sq <= radius_sq   # (N, 2, M)

        # z 높이 확인
        in_z_range = (diff_z <= self.cfg.drum_z_range) & (diff_z >= 0.0)    # (N, 2, M)

        contact_mask = in_xy_range & in_z_range

        return contact_mask

    def _check_hitting(self, contact_mask: torch.Tensor, hit_armed: torch.Tensor, tip_vel: torch.Tensor, diff_z: torch.Tensor) -> torch.Tensor:
        # contact_mask: (N, 2, M)
        # hit_armed:    (N, 2, M)
        # tip_vel:      (N, 2, 3)
        # diff_z:       (N, 2, M)

        # 아래 방향 속도 조건
        tip_vel_z = tip_vel[:, :, 2].unsqueeze(-1)   # (N, 2, 1)
        is_downward = tip_vel_z < (-self.cfg.min_impact_velocity)   # (N, 2, 1), broadcast 가능

        # strike candidate
        hit_per_arm = contact_mask & hit_armed & is_downward   

        return hit_per_arm

    def _window_target_union(self, rds: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        # 현재 step ±W 안에 각 드럼이 타겟인지 (N, M)
        N = self.env.num_envs
        T = self.env.episode_length_step
        M = self.num_drums
        W = self.cfg.hit_window_step

        window_target_union = torch.zeros((N, M), device=self.device, dtype=torch.bool)

        for offset in range(-W, W + 1):
            cand_steps = steps + offset
            valid = (cand_steps >= 0) & (cand_steps < T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            target_mask = rds[self.env_arange, cand_steps_clamped, :] > 0.5     # (N, M)
            target_mask &= valid.unsqueeze(-1)

            window_target_union |= target_mask

        return window_target_union

    def _finalize_target_outcomes(self, rds: torch.Tensor, rds_visit: torch.Tensor, rds_visit_vel: torch.Tensor, rds_visit_wrist: torch.Tensor, rds_visit_elbow_apex: torch.Tensor, rds_T: torch.Tensor, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # rds(값=velocity), rds_visit, rds_visit_vel, rds_visit_wrist, rds_visit_elbow_apex: (N, T, M)
        # steps: (N,)

        N = self.env.num_envs
        T = self.env.episode_length_step
        M = self.num_drums
        W = self.cfg.hit_window_step

        success = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        time_error = torch.full((N, M), -1, device=self.device, dtype=torch.int64)
        success_impact_vel = torch.zeros((N, M), device=self.device, dtype=torch.float32)
        success_wrist_ang_vel = torch.zeros((N, M), device=self.device, dtype=torch.float32)
        success_elbow_apex = torch.zeros((N, M), device=self.device, dtype=torch.float32)

        # 윈도우 왼쪽 끝 스텝
        window_end_step = steps - W
        window_valid = (window_end_step >= 0) & (window_end_step < T)
        window_end_step = window_end_step.clamp(0, T - 1)

        target_vel_raw = rds[self.env_arange, window_end_step, :].to(torch.float32)   # (N, M) 그 타겟의 MIDI velocity
        target_mask = (target_vel_raw > 0.5)
        target_mask &= window_valid.unsqueeze(-1)     # (N, M)

        # 윈도우가 끝난 타겟은 done 해제 → 다음 타겟 주기에 다시 유도 가능
        self.target_done = self.target_done & (~target_mask)

        offsets = self._get_hit_window_offsets(W)
        for offset in offsets:
            cand_steps = window_end_step + offset
            cand_valid = (cand_steps >= 0) & (cand_steps < T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            hit_mask = rds_visit[self.env_arange, cand_steps_clamped, :] > 0.5  # (N, M)

            match_mask = hit_mask & cand_valid.unsqueeze(-1) & target_mask & (~success)

            # 성공으로 매칭된 시간의 타격 속도/손목각속도/팔꿈치리프트apex 기록 (타겟당 첫 매칭 1회)
            vel_at_cand = rds_visit_vel[self.env_arange, cand_steps_clamped, :]     # (N, M)
            wrist_at_cand = rds_visit_wrist[self.env_arange, cand_steps_clamped, :] # (N, M)
            elbow_at_cand = rds_visit_elbow_apex[self.env_arange, cand_steps_clamped, :]  # (N, M)
            success_impact_vel = torch.where(match_mask, vel_at_cand, success_impact_vel)
            success_wrist_ang_vel = torch.where(match_mask, wrist_at_cand, success_wrist_ang_vel)
            success_elbow_apex = torch.where(match_mask, elbow_at_cand, success_elbow_apex)

            success |= match_mask

            time_error[match_mask] = abs(offset)  # step 차이

        missed_target = target_mask & (~success)

        # 마감된 타겟의 목표 velocity (성공/미스 무관, target_mask 셀만). 보상에서 목표 세기로 변환
        target_velocity = torch.where(target_mask, target_vel_raw, torch.zeros_like(target_vel_raw))

        # 마감된 타겟의 간격 T[s] (손목/팔꿈치 목표 T-종속 계산용)
        target_T_raw = rds_T[self.env_arange, window_end_step, :]                      # (N, M)
        target_T = torch.where(target_mask, target_T_raw, torch.zeros_like(target_T_raw))

        return success, missed_target, time_error, success_impact_vel, success_wrist_ang_vel, target_velocity, success_elbow_apex, target_T

    def _get_hit_window_offsets(self, W: int) -> list:
        # offset 순서: [0, -1, +1, -2, +2, ...] 마스크가 "가장 작은 |offset|부터 매칭"되도록
        offsets = [0]
        for i in range(1, W + 1):
            offsets.append(-i)
            offsets.append(i)

        return offsets

    def _check_rearm(self, prev_hit_armed: torch.Tensor, contact_mask: torch.Tensor, diff_z: torch.Tensor) -> torch.Tensor:
        """
        준비

        Args:
            prev_hit_armed: (N, 2, M) # 이전 준비 상태
            contact_mask:   (N, 2, M) # 접촉 여부
            diff_z:         (N, 2, M) # z 거리

        Returns:
            next_hit_armed: (N, 2, M)
        """
        next_hit_armed = prev_hit_armed.clone()

        # 충분히 벗어나고 올라가면 rearm
        rearm_mask = diff_z > self.cfg.rearm_height
        next_hit_armed[rearm_mask] = True

        # 접촉 중이면 disarm
        contact_expanded = contact_mask.any(dim=2, keepdim=True)  # (N,2,1)
        next_hit_armed = next_hit_armed.masked_fill(contact_expanded, False)

        return next_hit_armed