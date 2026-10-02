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
    # 재장전(rearm) 문턱 [m]. diff_z = tip_z - drum_z 기준이라 드럼마다 절대 높이는 다르다 (밴드 상한 0.07 + 마진 0.03)
    rearm_height: float = 0.10

    # 스텝 단위이므로 제어주기를 바꾸면 같이 바꿔야 한다 (초로 자동 변환되지 않음)
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
        contact_mask, in_xy_range, in_z_range = self._check_contact_drum(dist_xy_sq, diff_z)

        # [진단] miss 3분할용 순수 기하 방문 기록 (팔 무관, 중재/하강/armed 조건 없음)
        #   xy_in_any : 어느 팔이든 이 드럼 xy반경(0.13m) 안에 들어옴
        #   xyz_in_any: 어느 팔이든 xy반경 AND z밴드(0~0.07m) 동시 충족 (하강여부 무관)
        xy_in_any = in_xy_range.any(dim=1)                     # (N, M)
        xyz_in_any = (in_xy_range & in_z_range).any(dim=1)     # (N, M)

        # 타격 판정
        hit_per_arm = self._check_hitting(
            contact_mask=contact_mask,
            hit_armed=self.hit_armed,
            tip_vel=tip_vel,
            diff_z=diff_z,
        )   # (N, 2, M)

        # 양팔 중 하나라도 해당 drum을 strike하면 hit
        hit_mask = torch.any(hit_per_arm, dim=1)   # (N, M)

        # [진단] miss3_dir 원인 4분할 — "기하는 충족했는데 어느 게이트에 막혔나"
        _geo = in_xy_range & in_z_range                                  # (N,2,M) 기하 충족 (중재 전)
        _vz = tip_vel[:, :, 2].unsqueeze(-1)                             # (N,2,1)
        _going = _vz < 0.0                                               # 하강 방향
        _enough = _vz < (-self.cfg.min_impact_velocity)                  # 0.2 m/s 이상
        _armed = self.hit_armed                                          # 이번 판정에 쓰인 값
        self._st_armed = (_geo & ~_armed).any(dim=1)                                   # (N,M) 재장전 안 됨
        self._st_down  = (_geo &  _armed & ~_going).any(dim=1)                         # 상승 중 통과
        self._st_slow  = (_geo &  _armed &  _going & ~_enough).any(dim=1)              # 하강인데 느림
        self._st_arb   = (_geo &  _armed &  _enough & ~contact_mask).any(dim=1)        # 최근접 중재에 밀림

        # [진단] reach/zpos 세분화 — 이번 스텝 값 (팔 무관, 양팔 중 더 가까운 쪽)
        _zr = self.cfg.drum_z_range
        self._st_dxy = torch.sqrt(dist_xy_sq.clamp(min=0.0)).min(dim=1).values         # (N,M) XY 거리 [m]
        self._st_above = (in_xy_range & (diff_z > _zr)).any(dim=1)                     # 밴드보다 위
        self._st_below = (in_xy_range & (diff_z < 0.0)).any(dim=1)                     # 드럼면보다 아래
        _err = torch.where(diff_z > _zr, diff_z - _zr,
                           torch.where(diff_z < 0.0, -diff_z, torch.zeros_like(diff_z)))
        self._st_zerr = torch.where(in_xy_range, _err, torch.full_like(_err, 1.0e3)).min(dim=1).values

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
            xy_in_any, xyz_in_any,
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
            rds_visit_xy: torch.Tensor,
            rds_visit_xyz: torch.Tensor,
            rds_T: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
        # 현재 ±W 윈도우 타겟
        window_target_union = self._window_target_union(rds, steps)  # (N, M)

        # 잘못친 타격: 윈도우 밖 타격
        wrong_hit = hit_mask & (~window_target_union)

        # 유효 타격(윈도우 안 타격) → 그 타겟 done (재타 downward 억제).
        # 윈도우 밖 오타/미스는 done 안 켜서 다시 칠 수 있음
        valid_hit = hit_mask & window_target_union
        self.target_done = self.target_done | valid_hit

        # [진단] 원인 귀속 누적 — ★ ±W 판정 창 안에 그 드럼이 타겟일 때만 누적한다.
        #   창 밖까지 누적하면 "직전에 그 드럼을 친 직후" 상태가 섞여 원인이 포화된다.
        _iw = window_target_union                                        # (N,M)
        self.blk_armed |= self._st_armed & _iw
        self.blk_down  |= self._st_down  & _iw
        self.blk_slow  |= self._st_slow  & _iw
        self.blk_arb   |= self._st_arb   & _iw
        self.near_dxy = torch.where(_iw, torch.minimum(self.near_dxy, self._st_dxy), self.near_dxy)
        self.zp_above |= self._st_above & _iw
        self.zp_below |= self._st_below & _iw
        self.zp_gap = torch.where(_iw, torch.minimum(self.zp_gap, self._st_zerr), self.zp_gap)

        # 윈도우 끝났을 때 타격 성공 확인 (+ 성공 타격의 속도/손목각속도 + 그 타겟의 목표 velocity). 끝난 타겟은 done 해제
        (success, missed_target, time_error, success_impact_vel, success_wrist_ang_vel, target_velocity,
         success_elbow_apex, target_T, time_error_signed, miss_reach, miss_zpos, miss_dir,
         dir_armed, dir_down, dir_slow, dir_arb,
         reach_near, reach_mid, reach_far,
         zpos_above, zpos_below, zpos_both, zpos_gap) = self._finalize_target_outcomes(
            rds=rds,
            rds_visit=rds_visit,
            rds_visit_vel=rds_visit_vel,
            rds_visit_wrist=rds_visit_wrist,
            rds_visit_elbow_apex=rds_visit_elbow_apex,
            rds_visit_xy=rds_visit_xy,
            rds_visit_xyz=rds_visit_xyz,
            rds_T=rds_T,
            steps=steps,
        )

        return (success, wrong_hit, missed_target, time_error, success_impact_vel, success_wrist_ang_vel,
                self.target_done, target_velocity, success_elbow_apex, target_T, time_error_signed,
                miss_reach, miss_zpos, miss_dir,
                dir_armed, dir_down, dir_slow, dir_arb,
                reach_near, reach_mid, reach_far,
                zpos_above, zpos_below, zpos_both, zpos_gap)

    def reset(self, env_ids: torch.Tensor, tip_pos: torch.Tensor):
        # 팁 위치/속도 리셋
        self.tip_pos[env_ids] = tip_pos[env_ids]
        self.prev_tip_pos[env_ids] = tip_pos[env_ids]
        self.tip_vel[env_ids] = 0.0

        # 타격 상태 버퍼 리셋
        self.hit_armed[env_ids] = False # 초기 상태는 준비 안됨. 이 후 스텝에서 갱신
        self.target_done[env_ids] = False

        # [진단] miss3_dir 원인 플래그 리셋
        self.blk_armed[env_ids] = False
        self.blk_down[env_ids] = False
        self.blk_slow[env_ids] = False
        self.blk_arb[env_ids] = False
        self.near_dxy[env_ids] = 1.0e3
        self.zp_above[env_ids] = False
        self.zp_below[env_ids] = False
        self.zp_gap[env_ids] = 1.0e3

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

        # [진단] miss3_dir 원인 4분할 플래그 (N,M bool = 64KB/개). 타겟 마감 시 클리어됨
        self.blk_armed = torch.zeros((N, M), device=self.device, dtype=torch.bool)  # 재장전 안 됨
        self.blk_down  = torch.zeros((N, M), device=self.device, dtype=torch.bool)  # 상승 중 통과
        self.blk_slow  = torch.zeros((N, M), device=self.device, dtype=torch.bool)  # 하강인데 느림
        self.blk_arb   = torch.zeros((N, M), device=self.device, dtype=torch.bool)  # 최근접 중재에 밀림

        # [진단] miss3_reach / miss3_zpos 세분화 — 두 조각이 전체 miss 의 3/4 인데 기전이 없었다.
        #   near_dxy : 타겟 구간 중 팁이 그 드럼에 가장 가까웠던 XY 거리 [m]. reach miss 를
        #              '아예 안 감(먼 거리)' vs '문턱에서 놓침(0.13 근처)' 으로 가른다.
        #   zp_above / zp_below : XY반경 안에 있던 동안 z 가 밴드 위였나 / 아래였나.
        #              zpos miss 의 처방이 정반대다 (못 내려옴 vs 뚫고 지나감).
        #   zp_gap   : XY반경 안에서의 최소 |z 오차| [m]. 근접 실패인지 한참 빗나갔는지.
        BIG = 1.0e3
        self.near_dxy = torch.full((N, M), BIG, device=self.device, dtype=torch.float32)
        self.zp_above = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        self.zp_below = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        self.zp_gap   = torch.full((N, M), BIG, device=self.device, dtype=torch.float32)

        # detect_hit 이 매 스텝 채우는 '이번 스텝 상태'. get_result 가 창 게이팅해서 위 버퍼에 누적한다.
        #   런타임 순서는 detect_hit → get_result 로 보장되지만, 단독 호출 시 AttributeError 방지용 초기화.
        _z = lambda: torch.zeros((N, M), device=self.device, dtype=torch.bool)
        self._st_armed, self._st_down, self._st_slow, self._st_arb = _z(), _z(), _z(), _z()
        self._st_above, self._st_below = _z(), _z()
        self._st_dxy  = torch.full((N, M), BIG, device=self.device, dtype=torch.float32)
        self._st_zerr = torch.full((N, M), BIG, device=self.device, dtype=torch.float32)
    
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

        contact_mask = in_xy_range & in_z_range   # (N, 2, M)

        # 최근접 드럼 중재: 한 스틱(팔)이 여러 드럼 반경에 동시에 들어도 '가장 가까운 드럼 하나'만 접촉 인정.
        # 반경(0.13) 겹침 + 드럼 perturbation으로 한 번의 타격이 인접 드럼까지 타격으로 세지는 것(거짓 성공) 방지.
        # 진짜 동시타는 스틱 2개가 각자 최근접을 잡으므로 그대로 인정됨.
        M = contact_mask.shape[-1]
        dist_sq = dist_xy_sq + diff_z * diff_z                              # (N, 2, M) 3D 근사거리
        dist_masked = torch.where(contact_mask, dist_sq, torch.full_like(dist_sq, float("inf")))
        nearest = dist_masked.argmin(dim=-1, keepdim=True)                 # (N, 2, 1) 접촉 드럼 중 최근접
        has_contact = contact_mask.any(dim=-1, keepdim=True)               # (N, 2, 1) 접촉이 하나라도 있나
        arange = torch.arange(M, device=contact_mask.device).view(1, 1, M)
        contact_mask = (arange == nearest) & has_contact                   # 최근접 드럼만 True (접촉 없으면 전부 False)

        return contact_mask, in_xy_range, in_z_range

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

    def _finalize_target_outcomes(self, rds: torch.Tensor, rds_visit: torch.Tensor, rds_visit_vel: torch.Tensor, rds_visit_wrist: torch.Tensor, rds_visit_elbow_apex: torch.Tensor, rds_visit_xy: torch.Tensor, rds_visit_xyz: torch.Tensor, rds_T: torch.Tensor, steps: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # rds(값=velocity), rds_visit, rds_visit_vel, rds_visit_wrist, rds_visit_elbow_apex: (N, T, M)
        # steps: (N,)

        N = self.env.num_envs
        T = self.env.episode_length_step
        M = self.num_drums
        W = self.cfg.hit_window_step

        success = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        time_error = torch.full((N, M), -1, device=self.device, dtype=torch.int64)
        # [진단] 부호 있는 시간오차 [스텝]. 보상은 abs 를 쓰므로 조기/지연을 구분할 수 없었다.
        #   offset > 0 = 타겟 시각보다 늦게 침  /  < 0 = 일찍 침.
        time_error_signed = torch.full((N, M), -99, device=self.device, dtype=torch.int64)
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

        # [진단] miss 3분할용: 타겟 윈도우 동안 팁이 xy반경/xyz박스에 한 번이라도 들어왔나 (OR)
        xy_ever = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        xyz_ever = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        offsets = self._get_hit_window_offsets(W)
        for offset in offsets:
            cand_steps = window_end_step + offset
            cand_valid = (cand_steps >= 0) & (cand_steps < T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            hit_mask = rds_visit[self.env_arange, cand_steps_clamped, :] > 0.5  # (N, M)

            # 순수 기하 방문 (target_mask/valid로 게이팅해 마감 타겟 셀만 집계)
            cv = cand_valid.unsqueeze(-1) & target_mask
            xy_ever |= (rds_visit_xy[self.env_arange, cand_steps_clamped, :] > 0.5) & cv
            xyz_ever |= (rds_visit_xyz[self.env_arange, cand_steps_clamped, :] > 0.5) & cv

            match_mask = hit_mask & cand_valid.unsqueeze(-1) & target_mask & (~success)

            # 성공으로 매칭된 시간의 타격 속도/손목각속도/팔꿈치리프트apex 기록 (타겟당 첫 매칭 1회)
            vel_at_cand = rds_visit_vel[self.env_arange, cand_steps_clamped, :]     # (N, M)
            wrist_at_cand = rds_visit_wrist[self.env_arange, cand_steps_clamped, :] # (N, M)
            elbow_at_cand = rds_visit_elbow_apex[self.env_arange, cand_steps_clamped, :]  # (N, M)
            success_impact_vel = torch.where(match_mask, vel_at_cand, success_impact_vel)
            success_wrist_ang_vel = torch.where(match_mask, wrist_at_cand, success_wrist_ang_vel)
            success_elbow_apex = torch.where(match_mask, elbow_at_cand, success_elbow_apex)

            success |= match_mask

            time_error[match_mask] = abs(offset)  # step 차이 (보상용 — 건드리지 않음)
            time_error_signed[match_mask] = offset  # [진단] 부호 유지

        missed_target = target_mask & (~success)

        # [진단] miss 3분할 (상호배타·전수):
        #   ① reach : xy반경에 한 번도 못 옴          → 진짜 리치/이동시간
        #   ② zpos  : xy는 왔는데 z밴드 동시충족 없음  → 접근 각도/타점높이
        #   ③ dir   : xyz박스는 왔는데 타격 미등록      → 하강X/armed X/중재밀림 = 접근 방향/동역학
        miss_reach = missed_target & (~xy_ever)
        miss_zpos = missed_target & xy_ever & (~xyz_ever)
        miss_dir = missed_target & xyz_ever

        # [진단] miss_dir 원인 4분할 — detect_hit 이 누적한 게이트 플래그를 miss_dir 셀로 한정해 읽는다.
        dir_armed = miss_dir & self.blk_armed
        dir_down  = miss_dir & self.blk_down
        dir_slow  = miss_dir & self.blk_slow
        dir_arb   = miss_dir & self.blk_arb

        # [진단] reach 세분화 — 최근접 XY 거리로 3분할 (판정 반경 0.13m 기준)
        #   near : 0.13 ~ 0.20  문턱에서 놓침 → 타이밍·정밀도
        #   mid  : 0.20 ~ 0.35  중간
        #   far  : 0.35 이상    아예 안 감 → 능력 부족 또는 배정 문제
        reach_near = miss_reach & (self.near_dxy < 0.20)
        reach_mid  = miss_reach & (self.near_dxy >= 0.20) & (self.near_dxy < 0.35)
        reach_far  = miss_reach & (self.near_dxy >= 0.35)

        # [진단] zpos 세분화 — 방향(위/아래)과 근접도. 처방이 정반대라 방향이 핵심.
        zpos_above = miss_zpos & self.zp_above & (~self.zp_below)
        zpos_below = miss_zpos & self.zp_below & (~self.zp_above)
        zpos_both  = miss_zpos & self.zp_above & self.zp_below
        zpos_gap   = torch.where(miss_zpos, self.zp_gap.clamp(max=1.0), torch.zeros_like(self.zp_gap))

        # 마감된 타겟의 플래그는 여기서 클리어 — 안 하면 에피소드 내내 누적돼 다음 타겟으로 샌다
        _closing = success | missed_target
        self.blk_armed &= ~_closing
        self.blk_down  &= ~_closing
        self.blk_slow  &= ~_closing
        self.blk_arb   &= ~_closing
        self.near_dxy = torch.where(_closing, torch.full_like(self.near_dxy, 1.0e3), self.near_dxy)
        self.zp_above &= ~_closing
        self.zp_below &= ~_closing
        self.zp_gap = torch.where(_closing, torch.full_like(self.zp_gap, 1.0e3), self.zp_gap)

        # 마감된 타겟의 목표 velocity (성공/미스 무관, target_mask 셀만). 보상에서 목표 세기로 변환
        target_velocity = torch.where(target_mask, target_vel_raw, torch.zeros_like(target_vel_raw))

        # 마감된 타겟의 간격 T[s] (손목/팔꿈치 목표 T-종속 계산용)
        target_T_raw = rds_T[self.env_arange, window_end_step, :]                      # (N, M)
        target_T = torch.where(target_mask, target_T_raw, torch.zeros_like(target_T_raw))

        return (success, missed_target, time_error, success_impact_vel, success_wrist_ang_vel,
                target_velocity, success_elbow_apex, target_T, time_error_signed,
                miss_reach, miss_zpos, miss_dir,
                dir_armed, dir_down, dir_slow, dir_arb,
                reach_near, reach_mid, reach_far,
                zpos_above, zpos_below, zpos_both, zpos_gap)

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

        # 접촉 중이면 disarm —> 접촉 중이면 disarm. 접촉이 끝나도 바로 rearm 되지 않도록, 충분히 올라가야 rearm.
        next_hit_armed = next_hit_armed & (~contact_mask)         # (N,2,M) 닿은 드럼만 disarm

        return next_hit_armed