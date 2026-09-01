"""
보상 클래스
"""

from __future__ import annotations

import torch    # pyright: ignore[reportMissingImports]  — Isaac Sim이 런타임에 import path 추가
from dataclasses import dataclass

from .specs import EnvRuntimeSpec, Instruments

# === 모듈 레벨 jit 함수들 ===
@torch.jit.script
def _goal_terms(
        success: torch.Tensor,
        wrong_hit: torch.Tensor,
        missed_target: torch.Tensor,
        time_error: torch.Tensor,
        w_drum_success: torch.Tensor,
        k_accuracy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    success_reward = (success.float() * w_drum_success).sum(dim=-1)
    wrong_cost   = wrong_hit.float().sum(dim=-1)
    missed_cost  = missed_target.float().sum(dim=-1)

    valid_mask = time_error >= 0
    time_accuracy_reward = torch.exp(-k_accuracy * time_error)
    time_accuracy_reward[~valid_mask] = 0
    time_accuracy_reward = time_accuracy_reward.sum(dim=-1)

    return success_reward, wrong_cost, missed_cost, time_accuracy_reward

@torch.jit.script
def compute_arm_target_assignment(
        target_mask: torch.Tensor,        # (N, M)  현재 타격 h0
        drum_pos: torch.Tensor,     # (N, M, 3)
        left_tip_pos: torch.Tensor, # (N, 3)
        right_tip_pos: torch.Tensor,# (N, 3)
        hit_xy_radius: float,       # crowd 판정용 타격영역 XY 반경
        hit_z_range: float,         # crowd 판정용 타격영역 Z 범위
        ready_height: float,        # 노는 팔 준비 높이 (드럼 위, off_target 밴드보다 위)
        k_ready: float,             # ready-pose 보상 감쇠 계수
        aim_offset: float,          # 팔별 조준 오프셋 [m] (왼팔 −x / 오른팔 +x)
        assigned_use_left: torch.Tensor,  # (N,) bool 스케줄러(reachability)가 정한 h0 담당(왼팔=True)
        assigned_use_case1: torch.Tensor, # (N,) bool 동시타 case2 배정(왼팔→idx0?). 스케줄러가 양팔 분배
        idle_tgt_pos: torch.Tensor,   # (N, 2, 3) 스케줄러가 각 팔에 준 첫 노트 위치 (per_arm_pos[:,:,0,:])
        idle_tgt_valid: torch.Tensor, # (N, 2) bool 그 노트가 유효한가 (per_arm_valid[:,:,0])
        idle_tgt_time: torch.Tensor,  # (N, 2) 그 노트까지 정규화 시간 (per_arm_time[:,:,0], 1=멀거나 없음)
        last_drum_pos: torch.Tensor,  # (N, 2, 3) 각 팔이 마지막으로 친 드럼 위치 (배정 없을 때 대기점)
        k_ready_urgency: float,       # 긴급도 감쇠
        rest_margin_x: float,         # 휴식 시 미드라인 허용 여유 [m]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = target_mask.device
    N, M = target_mask.shape

    # target index 추출 (최대 2개)
    masked_idx = torch.where(
        target_mask,
        torch.arange(M, device=device).expand(N, M),
        -1,
    )

    values, indices = torch.topk(masked_idx, k=2, dim=1, largest=True)  # (N, 2) 타겟 인덱스 상위 2개

    idx0 = values[:, 0].unsqueeze(1)    # (N, 1)
    idx0 = torch.where(idx0 >= 0, idx0, torch.zeros_like(idx0))  # -1 → 0 (인덱싱 오류 방지)

    idx1 = values[:, 1].unsqueeze(1)
    idx1 = torch.where(idx1 >= 0, idx1, torch.zeros_like(idx1))  # -1 → 0 (인덱싱 오류 방지)

    # --------------------

    # 팔별 조준 오프셋: 왼팔은 드럼 중심 −x, 오른팔은 +x (각자 어깨 쪽)
    aim_off = torch.zeros_like(drum_pos)
    aim_off[:, :, 0] = aim_offset
    aim_l = drum_pos - aim_off                      # (N, M, 3) 왼팔 조준점
    aim_r = drum_pos + aim_off                      # (N, M, 3) 오른팔 조준점

    # XYZ 거리 (각 팔의 조준점 기준)
    diff_l = left_tip_pos[:, None, :] - aim_l       # (N, M, 3)
    diff_r = right_tip_pos[:, None, :] - aim_r

    dist_l = torch.sum(diff_l * diff_l, dim=-1)     # (N, M)
    dist_r = torch.sum(diff_r * diff_r, dim=-1)

    # XY 제곱거리 (crowd 판정용)
    diff_xy_l = diff_l[:, :, 0:2]
    diff_xy_r = diff_r[:, :, 0:2]
    dist_xy_l = torch.sum(diff_xy_l * diff_xy_l, dim=-1)   # (N, M)
    dist_xy_r = torch.sum(diff_xy_r * diff_xy_r, dim=-1)

    d_l0 = dist_l.gather(1, idx0).squeeze(1)   # (N,) 각 팔 → h0 3D 제곱거리
    d_r0 = dist_r.gather(1, idx0).squeeze(1)

    d_l1 = dist_l.gather(1, idx1).squeeze(1)   # 각 팔 → 동시타 둘째 드럼
    d_r1 = dist_r.gather(1, idx1).squeeze(1)

    # crowd 판정용: 각 팔의 h0(타겟)까지 XY 거리 / Z 차이
    dxy_l0 = dist_xy_l.gather(1, idx0).squeeze(1)
    dxy_r0 = dist_xy_r.gather(1, idx0).squeeze(1)
    dz_l0 = diff_l[:, :, 2].gather(1, idx0).squeeze(1)
    dz_r0 = diff_r[:, :, 2].gather(1, idx0).squeeze(1)

    # --------------------

    target_count = target_mask.float().sum(dim=1)
    cost = torch.zeros(N, device=device)
    crowd_pen = torch.zeros(N, device=device)

    # --------------------
    # case 1: 현재 타격 1개 (+ 다음 타격 h1을 노는 팔에 배정)
    # --------------------
    one_mask = (target_count == 1)

    # h0 배정: 스케줄러(reachability)가 정한 담당 팔을 그대로 사용.
    # 현재위치+이동가용시간+재장전을 forward-scan으로 반영 → 빠른 클러스터는 양팔 분배됨.
    use_left = assigned_use_left
    # 접근/진척 보상용 비용은 담당 팔의 팁-h0 거리 (팁이 실제로 드럼에 닿아야 하므로)
    one_cost = torch.where(use_left, d_l0, d_r0)
    cost = torch.where(one_mask, one_cost, cost)

    # crowd: 배정 안 된(노는) 팔이 타겟 h0의 타격영역에 들어오면 페널티 (시점 무관)
    free_dxy = torch.where(use_left, dxy_r0, dxy_l0)   # 노는 팔의 h0까지 XY 제곱거리
    free_dz  = torch.where(use_left, dz_r0,  dz_l0)    # 노는 팔의 h0까지 Z 차이
    free_in_zone = (
        (free_dxy <= hit_xy_radius ** 2)
        & (free_dz >= 0.0)
        & (free_dz <= hit_z_range)
    )
    crowd_pen = torch.where(one_mask, free_in_zone.float(), crowd_pen)

    # --------------------
    # case 2: 타겟 2개
    # --------------------
    two_mask = (target_count >= 2)

    # 배정 2가지: case1 = 왼팔→idx0 / 오른팔→idx1,  case2 = swap
    cost_case1 = (d_l0 + d_r1) / 2
    cost_case2 = (d_l1 + d_r0) / 2

    # 동시타 비용은 두 배정 중 싼 쪽(관용적). 스케줄러 배정(assigned_use_case1)은 obs로만 전달됨.
    #   ⚠️ where(assigned_use_case1, ...)로 강제해봤으나 실패: useA에 히스테리시스가 없어 채터링이
    #      보상에 유입 → wrong_rate 2배(0.033→0.067), miss3_dir 0.302→0.422, 성공률 0.84→0.795.
    #      강제하려면 (1) useA 히스테리시스 (2) 부분 강제(w_follow) 가 선행돼야 함.
    two_cost = torch.minimum(cost_case1, cost_case2)
    cost = torch.where(two_mask, two_cost, cost)

    # --------------------
    # case 3: 타겟 0개
    # --------------------
    #
    # 나중에 필요하면 작성
    #

    # --------------------
    # 팔별 역할 (관측 신호용) + 노는 팔 ready-pose 보상
    # --------------------
    active_arm_l = torch.zeros(N, device=device)
    active_arm_r = torch.zeros(N, device=device)
    # case 1: h0 담당 팔만 활성
    active_arm_l = torch.where(one_mask, use_left.float(), active_arm_l)
    active_arm_r = torch.where(one_mask, (~use_left).float(), active_arm_r)
    # case 2: 양팔 모두 활성
    active_arm_l = torch.where(two_mask, torch.ones_like(active_arm_l), active_arm_l)
    active_arm_r = torch.where(two_mask, torch.ones_like(active_arm_r), active_arm_r)
    active_arm = torch.stack([active_arm_l, active_arm_r], dim=1)   # (N, 2)  1=활성, 0=노는팔

    # ready-pose: case 1의 노는 팔을 "드럼 위 ready_height"에 3D로 고정.
    #   기준점은 스케줄러 배정 하나만 사용 (팔꿈치 거리 휴리스틱 폐기 — 재장전 교대를 몰라서 스케줄러와 어긋났음).
    #   배정 없음(=lookahead 2초 안에 자기 노트 없음, 상대팔 연타 구간) → 마지막으로 친 드럼 위에서 대기.
    #   폴백은 반드시 '움직이지 않는' 기준이어야 함. h0/h1로 폴백하면 음표마다 기준이 떠서
    #      노는 팔이 치는 팔과 동기해 위아래로 스윙함 (관측된 버그).
    idle_tip = torch.where(use_left.unsqueeze(-1), right_tip_pos, left_tip_pos)          # (N,3)
    sched_p  = torch.where(use_left.unsqueeze(-1), idle_tgt_pos[:, 1], idle_tgt_pos[:, 0])   # (N,3)
    last_p   = torch.where(use_left.unsqueeze(-1), last_drum_pos[:, 1], last_drum_pos[:, 0]) # (N,3)
    has_tgt  = torch.where(use_left, idle_tgt_valid[:, 1], idle_tgt_valid[:, 0])          # (N,)
    home_p   = torch.where(has_tgt.unsqueeze(-1), sched_p, last_p)                        # (N,3)
    # 대기점도 같은 조준 오프셋 (쉬는 팔의 자기 어깨 쪽). use_left=True → 쉬는 팔은 오른팔(+x)
    home_off = torch.zeros_like(home_p)
    home_off[:, 0] = torch.where(use_left, aim_offset, -aim_offset)
    home_p = home_p + home_off

    # ① 긴급도: 노트가 임박할수록 1. 멀면 0 → 휴식 x 로 수렴
    t_idle = torch.where(use_left, idle_tgt_time[:, 1], idle_tgt_time[:, 0])              # (N,)
    urgency = torch.exp(-k_ready_urgency * t_idle)                                        # (N,) 0~1
    # ② 휴식 x: 미드라인을 rest_margin_x 이상 넘지 못하게. sign 으로 양팔을 같은 식으로 처리
    sign = torch.where(use_left, torch.ones_like(t_idle), -torch.ones_like(t_idle))       # 쉬는팔 오른=+1
    own_x = sign * home_p[:, 0]                                                           # >0 = 자기 쪽
    rest_x = sign * torch.clamp(own_x, min=-rest_margin_x)
    home_x = rest_x * (1.0 - urgency) + home_p[:, 0] * urgency
    home_p = torch.cat([home_x.unsqueeze(-1), home_p[:, 1:3]], dim=-1)                    # y·z 는 그대로

    d_xy = idle_tip[:, 0:2] - home_p[:, 0:2]
    d_z  = idle_tip[:, 2] - (home_p[:, 2] + ready_height)
    ready_sq = torch.sum(d_xy * d_xy, dim=-1) + d_z * d_z
    #   동시타(case 2)만 제외 — 그때는 '쉬는 팔' 개념이 없음. 타겟 0개(case 3)는 폴백이 정의돼 있어 그대로 작동.
    ready_pose_reward = torch.where(two_mask, torch.zeros(N, device=device), torch.exp(-k_ready * ready_sq))

    return cost, crowd_pen, active_arm, ready_pose_reward

@torch.jit.script
def _proximity_terms(
        left_tip_pos: torch.Tensor,
        right_tip_pos: torch.Tensor,
        prev_left_tip_pos: torch.Tensor,
        prev_right_tip_pos: torch.Tensor,
        drum_pos: torch.Tensor,
        next_hits: torch.Tensor,
        k_time_to_hit: float,
        hit_xy_radius: float,
        hit_z_range: float,
        ready_height: float,
        k_ready: float,
        aim_offset: float,
        assigned_use_left: torch.Tensor,
        assigned_use_case1: torch.Tensor,
        idle_tgt_pos: torch.Tensor,
        idle_tgt_valid: torch.Tensor,
        idle_tgt_time: torch.Tensor,
        last_drum_pos: torch.Tensor,
        k_ready_urgency: float,
        rest_margin_x: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = drum_pos.shape[1]

    nearest_target_mask = next_hits[:, 0, :M] > 0.5   # h0
    first_time = next_hits[:, 0, M]

    curr_cost, crowd_pen, active_arm, ready_pose_reward = compute_arm_target_assignment(
        target_mask=nearest_target_mask,
        drum_pos=drum_pos,
        left_tip_pos=left_tip_pos,
        right_tip_pos=right_tip_pos,
        hit_xy_radius=hit_xy_radius,
        hit_z_range=hit_z_range,
        ready_height=ready_height,
        k_ready=k_ready,
        aim_offset=aim_offset,
        assigned_use_left=assigned_use_left,
        assigned_use_case1=assigned_use_case1,
        idle_tgt_pos=idle_tgt_pos,
        idle_tgt_valid=idle_tgt_valid,
        idle_tgt_time=idle_tgt_time,
        last_drum_pos=last_drum_pos,
        k_ready_urgency=k_ready_urgency,
        rest_margin_x=rest_margin_x,
    )
    # prev-position 비용은 진척(progress) 계산용. 배정(assigned_use_left)은 동일하게 주입.
    prev_cost, _, _, _ = compute_arm_target_assignment(
        target_mask=nearest_target_mask,
        drum_pos=drum_pos,
        left_tip_pos=prev_left_tip_pos,
        right_tip_pos=prev_right_tip_pos,
        hit_xy_radius=hit_xy_radius,
        hit_z_range=hit_z_range,
        ready_height=ready_height,
        k_ready=k_ready,
        aim_offset=aim_offset,
        assigned_use_left=assigned_use_left,
        assigned_use_case1=assigned_use_case1,
        idle_tgt_pos=idle_tgt_pos,
        idle_tgt_valid=idle_tgt_valid,
        idle_tgt_time=idle_tgt_time,
        last_drum_pos=last_drum_pos,
        k_ready_urgency=k_ready_urgency,
        rest_margin_x=rest_margin_x,
    )

    proximity_cost = torch.exp(-k_time_to_hit * first_time) * curr_cost
    progress_reward = prev_cost - curr_cost

    return proximity_cost, progress_reward, crowd_pen, active_arm, ready_pose_reward

@torch.jit.script
def _self_collision_penalty(
        elbow_l: torch.Tensor,  # (N, 3) 왼쪽 전완 시작 (elbow)
        wrist_l: torch.Tensor,  # (N, 3) 왼쪽 전완 끝 (wrist)
        elbow_r: torch.Tensor,  # (N, 3) 오른쪽 전완 시작 (elbow)
        wrist_r: torch.Tensor,  # (N, 3) 오른쪽 전완 끝 (wrist)
        threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # penalty, dist, s, t
    # 두 전완(elbow→wrist)을 선분으로 보고 선분 간 최소 거리 계산
    d1 = wrist_l - elbow_l  # (N, 3)
    d2 = wrist_r - elbow_r  # (N, 3)
    r  = elbow_l - elbow_r  # (N, 3)

    a = (d1 * d1).sum(-1)  # (N,)
    e = (d2 * d2).sum(-1)
    b = (d1 * d2).sum(-1)
    c = (d1 * r ).sum(-1)
    f = (d2 * r ).sum(-1)

    eps = 1e-8
    denom = a * e - b * b

    s0 = torch.clamp((b * f - c * e) / (denom + eps), 0.0, 1.0)
    t0 = (b * s0 + f) / (e + eps)

    # t0가 [0,1] 밖으로 나간 경우 각 끝점에 대해 s 재계산
    s_at_t0 = torch.clamp(-c / (a + eps), 0.0, 1.0)
    s_at_t1 = torch.clamp((b - c) / (a + eps), 0.0, 1.0)

    s = torch.where(t0 < 0.0, s_at_t0, torch.where(t0 > 1.0, s_at_t1, s0))
    t = torch.clamp(t0, 0.0, 1.0)

    closest_l = elbow_l + s.unsqueeze(-1) * d1  # (N, 3)
    closest_r = elbow_r + t.unsqueeze(-1) * d2  # (N, 3)
    diff = closest_l - closest_r
    dist = torch.sqrt((diff * diff).sum(-1) + eps)  # (N,)

    return torch.clamp(threshold - dist, min=0.0), dist, s, t  # penalty, 거리, 선분파라미터(0=시작,1=끝)

@torch.jit.script
def _tip_position_penalties(
        left_tip_pos: torch.Tensor,
        right_tip_pos: torch.Tensor,
        drum_pos: torch.Tensor,
        target_mask: torch.Tensor,  # (N, M) 현재 타격 타겟 (여기서 제외)
        x_limit: float,
        y_limit_l: float,
        y_limit_h: float,
        z_limit: float,
        drum_xy_margin: float,
        drum_z_margin: float,
        off_target_xy_radius: float,
        off_target_z_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tip_limit_pen = (
        (left_tip_pos[:, 0] > x_limit)
        | (left_tip_pos[:, 0] < -x_limit)
        | (left_tip_pos[:, 1] < y_limit_l)
        | (left_tip_pos[:, 1] > y_limit_h)
        | (left_tip_pos[:, 2] < z_limit)
        | (right_tip_pos[:, 0] > x_limit)
        | (right_tip_pos[:, 0] < -x_limit)
        | (right_tip_pos[:, 1] < y_limit_l)
        | (right_tip_pos[:, 1] > y_limit_h)
        | (right_tip_pos[:, 2] < z_limit)
    ).float()  # bool → float로 미리 캐스팅 (jit가 dtype 추론 안정적)

    diff_xy_l = left_tip_pos[:, None, 0:2] - drum_pos[:, :, 0:2]
    diff_xy_r = right_tip_pos[:, None, 0:2] - drum_pos[:, :, 0:2]
    diff_z_l = left_tip_pos[:, None, 2] - drum_pos[:, :, 2]
    diff_z_r = right_tip_pos[:, None, 2] - drum_pos[:, :, 2]

    dist_xy_l = torch.sum(diff_xy_l * diff_xy_l, dim=-1)
    dist_xy_r = torch.sum(diff_xy_r * diff_xy_r, dim=-1)

    in_xy_l = dist_xy_l <= drum_xy_margin ** 2
    in_xy_r = dist_xy_r <= drum_xy_margin ** 2

    under_drum_l = in_xy_l & (diff_z_l < -drum_z_margin)
    under_drum_r = in_xy_r & (diff_z_r < -drum_z_margin)

    under_drum_pen = under_drum_l.float().sum(dim=-1) + under_drum_r.float().sum(dim=-1)

    # off-target: 비타겟 드럼의 회피영역(XY 반경 & 드럼면 위 Z밴드)에 팁이 들어간 (팔, 드럼) 쌍 개수
    non_target = ~target_mask   # (N, M)

    in_hit_xy_l = dist_xy_l <= off_target_xy_radius ** 2
    in_hit_xy_r = dist_xy_r <= off_target_xy_radius ** 2

    in_hit_z_l = (diff_z_l >= 0.0) & (diff_z_l <= off_target_z_range)
    in_hit_z_r = (diff_z_r >= 0.0) & (diff_z_r <= off_target_z_range)

    zone_l = in_hit_xy_l & in_hit_z_l   # (N, M) 왼팔이 각 드럼 회피영역 안
    zone_r = in_hit_xy_r & in_hit_z_r

    off_target_l = zone_l & non_target
    off_target_r = zone_r & non_target

    off_target_pen = off_target_l.float().sum(dim=-1) + off_target_r.float().sum(dim=-1)

    return tip_limit_pen, under_drum_pen, off_target_pen

@torch.jit.script
def _global_penalties(
        joint_vel: torch.Tensor,
        joint_torque: torch.Tensor,
        action: torch.Tensor,
        robot_pos: torch.Tensor,
        joint_low: torch.Tensor,
        joint_high: torch.Tensor,
        limit_margin: float,
        limit_joint_mask: torch.Tensor,
        joint_vel_weight: torch.Tensor,
        action_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_l2 = torch.sum(action_weight * action * action, dim=-1)
    joint_vel_l2 = torch.sum(joint_vel_weight * joint_vel * joint_vel, dim=-1)

    # 기계적 일률 |τ·q̇| [W] — 무거운 관절 가속만 비싸고 자세 유지는 공짜. 손목은 τ가 작아 자동으로 저렴
    power_l1 = torch.sum(torch.abs(joint_torque * joint_vel), dim=-1)

    low_v = torch.clamp((joint_low + limit_margin) - robot_pos, min=0.0)
    high_v = torch.clamp(robot_pos - (joint_high - limit_margin), min=0.0)
    limit_pen = torch.sum((low_v * low_v + high_v * high_v) * limit_joint_mask, dim=-1)  # 손목=0 (손목은 wrist_limit_pen에서 전담)

    return action_l2, joint_vel_l2, limit_pen, power_l1

@torch.jit.script
def compute_rewards(
        success_reward: torch.Tensor,
        wrong_cost: torch.Tensor,
        missed_cost: torch.Tensor,
        time_accuracy_reward: torch.Tensor,

        proximity_cost: torch.Tensor,
        progress_reward: torch.Tensor,

        action_l2: torch.Tensor,
        joint_vel_l2: torch.Tensor,
        power_l1: torch.Tensor,
        limit_pen: torch.Tensor,
        tip_limit_pen: torch.Tensor,
        under_drum_pen: torch.Tensor,

        w_success: float,
        w_wrong: float,
        w_miss: float,
        w_time_accuracy: float,

        w_progress: float,
        w_proximity: float,

        w_action: float,
        w_joint_vel: float,
        w_power: float,
        w_limit: float,
        w_tip_limit: float,
        w_under_drum: float,
        w_self_collision: float,
        self_col_pen: torch.Tensor,
        impact_vel_reward: torch.Tensor,
        w_impact_vel: float,
        wrist_snap_reward: torch.Tensor,
        w_wrist_snap: float,
        intensity_reward: torch.Tensor,
        w_intensity: float,
        off_target_pen: torch.Tensor,
        w_off_target: float,
        crowd_pen: torch.Tensor,
        w_crowd: float,
        waist_vel_pen: torch.Tensor,
        w_waist: float,
        ready_pose_reward: torch.Tensor,
        w_ready: float,
) -> torch.Tensor:
    reward = (
        w_success * success_reward
        - w_wrong * wrong_cost
        - w_miss * missed_cost
        + w_time_accuracy * time_accuracy_reward

        - w_proximity * proximity_cost
        + w_progress * progress_reward

        + w_impact_vel * impact_vel_reward
        + w_wrist_snap * wrist_snap_reward
        + w_intensity * intensity_reward

        - w_action * action_l2
        - w_joint_vel * joint_vel_l2
        - w_power * power_l1
        - w_limit * limit_pen
        - w_tip_limit * tip_limit_pen
        - w_under_drum * under_drum_pen
        - w_self_collision * self_col_pen
        - w_off_target * off_target_pen
        - w_crowd * crowd_pen
        - w_waist * waist_vel_pen
        + w_ready * ready_pose_reward
    )
    return reward

# === 클래스는 jit 함수 호출해서 사용 ===
@dataclass
class RewardComputerCfg:
    # ══════════ 판정 파라미터 ══════════
    limit_margin: float = 0.10          # 관절한계 제동 시작거리 [rad] — 비손목 7관절
    wrist_limit_margin: float = 0.0    # 손목 전용 — 25rad/s라 1스텝 24° 이동, 팔의 2배 필요
    k_accuracy: float = 1.0             # 박자 정확도 exp 감쇠 (time_error = 정수 스텝수)
    k_time_to_hit: float = 3.0          # proximity 시간 감쇠 (타격 임박할수록 거리비용 ↑)

    # ══════════ 노는 팔 준비자세 ══════════
    w_ready: float = 0.6                # 0.3→0.6: 대기점까지 이동비용(균일 power)이 대기이득보다 커서 정책이
                                        #   40cm 떨어진 곳에 park했음. hover-hacking은 쉬는 팔 한정 적용이라 불가
    ready_height: float = 0.18          # 다음 드럼 위 준비높이 [m] — off_target 밴드 0.12보다 위
    k_ready: float = 10.0               # 3D 제곱거리 exp 감쇠
    stick_len: float = 0.373            # 손목 편차를 팁 호길이[m]로 환산 (위치 오차와 같은 단위로 맞춤)

    # 팔별 조준 오프셋 [m] — 각 팔이 드럼 중심에서 '자기 어깨 쪽'으로 비껴 조준.
    #   문제: 양팔이 같은 좌표를 조준하면 위상 반대(한쪽 상승/한쪽 하강)일 때 두 스틱 선분이 x축에서 맞닿음.
    #         측정상 접촉은 100% 비타격 스텝(stick_close_move 0.020 / stick_close_hit 0.000)이고 (s,t)=(1,0) 패턴.
    #   ±aim_offset이면 두 선분의 x구간이 2·aim_offset 벌어져 관통이 기하학적으로 불가.
    #   hit_detector는 드럼 중심 기준 그대로(인정 반경 0.13) → 4cm는 그 31%라 성공률 영향 없음.
    #   부수 이득: 오프셋이 항상 자기 어깨 쪽이라 크로스 배정 시 도달거리가 aim_offset 만큼 줄어듦.
    #   mid↔ride 간격이 0.120m(off_target 반경 0.13 안쪽)라 그 쌍은 부담이 약간 늘 수 있음 → 성적 하락 시 0.03으로
    aim_offset: float = 0.04

    # 대기점 긴급도 가중 + 휴식 x 제한 — 쉬는 팔이 반대편으로 '미리, 오래' 건너가는 것 차단.
    #   측정: 좌측 92% 곡에서 home_crossed 0.714 (학습 균형분포는 0.146) → ready_pose가 크로스를 명령 중이었음.
    #   시간 가중이 없어 노트가 1.9초 뒤여도 즉시 건너가 계속 머묾 → 조준 오프셋의 x분리 전제가 붕괴.
    #   ① urgency = exp(−k·t) 로 임박할 때만 노트 위치로. ② 그 전엔 x를 자기 쪽으로 제한.
    #   y·z는 손대지 않음 → '드럼면+ready_height' 안전성(off_target 0.12 / crowd 0.07 위) 그대로 유지.
    k_ready_urgency: float = 3.0      # proximity의 k_time_to_hit와 동일 스케일. ↓면 더 일찍 출발
    rest_margin_x: float = 0.02       # 휴식 시 미드라인을 넘어도 되는 여유 [m]

    # 노는 팔 접근 보상 (prev_d − curr_d). 0.5→0: 순효과 ≈0(−0.002)에 idle_power와 상충해 스톱앤고 떨림 유발.
    # 되돌림 신호 = miss3_reach 상승. 코드는 env에 남아 있어 값만 되돌리면 복원.
    w_idle_approach: float = 0.0

    # ══════════ 팁 위치 ══════════
    hit_xy_radius: float = 0.13         # 타격영역 반경 [m] — hit_detector와 동일값 유지
    hit_z_range: float = 0.07           # 타격영역 높이 [m] — 〃

    off_target_xy_radius: float = 0.13  # 비타겟 회피영역 반경 [m]
    off_target_z_range: float = 0.12    # 회피영역 높이 [m] — hit 밴드(0.07)보다 넓게 = 마진
    w_off_target: float = 0.3           # 비타겟 드럼 회피영역 침범
    w_crowd: float = 0.5                # 단일 타겟에 양팔 몰림 (ready와 한 쌍)

    x_limit: float = 0.5                # 팁 작업공간 상자 [m] — |x| ≤ 0.5
    y_limit_l: float = 0.2              #                        0.2 ≤ y ≤ 0.8
    y_limit_h: float = 0.8
    z_limit: float = -0.6               #                        z ≥ −0.6
    drum_xy_margin: float = 0.15        # under_drum 판정 수평범위 [m]
    drum_z_margin: float = 0.05         # under_drum 판정 침투깊이 [m]

    # ══════════ 목표 (타격 성공) ══════════
    w_success: float = 5.5              # 타격 성공 — ±3스텝(±50ms) 창 안이면 이진 만점
    w_wrong: float = 3.0                # 다른 드럼 오타
    w_miss: float = 3.0                 # 타겟 놓침
    w_time_accuracy: float = 3.0        # 창 안에서의 정확도 등급 exp(−|offset|)
    w_progress: float = 4.0             # 타겟에 가까워진 만큼 (prev_d − curr_d)
    w_proximity: float = 1.5            # 타겟까지 거리 (감점 형태지만 실질은 끌어당김)

    # ══════════ phase-1 부트스트랩 — "빠르게 내리쳐라" ══════════
    # 상한 없는 단조증가라 intensity("목표만큼")와 정면 충돌 → ramp_i만큼 크로스페이드로 소거.
    # 지우면 안 됨: 둘 다 없앤 런은 세기 게이트가 닫힌 구간에서 하강 신호가 0이 되어 성공률 0.04에서 정체.
    w_impact_vel: float = 3.0
    impact_vel_max: float = 1.0         # 정규화 기준 [m/s]
    w_wrist_snap: float = 3.0
    wrist_snap_max: float = 12.0        # 정규화 기준 [rad/s]

    # ══════════ 세기 매칭 ══════════
    # 목표는 고정수가 아니라 타격간격 T의 함수 (compute() 참고). 고정목표(7/11/15)는 f가 물리 천장을
    # 넘어 mf≈f로 붕괴했음 → 되돌리지 말 것. [[t-dependent-intensity-targets]]
    # 아래 6개 vel_* + k_imp_match 는 미사용 (T-종속 공식으로 대체됨. 필드 삭제는 별도 정리 예정)
    vel_imp_min: float = 0.3
    vel_imp_max: float = 1.5
    vel_wrist_min: float = 3.0
    vel_wrist_max: float = 15.0
    vel_elbow_min: float = 0.15
    vel_elbow_max: float = 0.40
    k_imp_match: float = 0.5
    k_wrist_match: float = 0.15     # 0.05→0.15: 반값폭 3.7→2.15 rad/s. 실측 계급간격 2.9(3.45/6.30/9.25)보다
                                    #   허용폭이 넓어 p·mf·f가 겹쳤음 → 조여서 세기 차등 강제
    k_elbow_match: float = 30.0     # 팔꿈치 리프트 오차² 감쇠 — 반값폭 0.15 rad
    w_wrist_match: float = 5.0      # 내부 배율: 손목ω = 세기 주력(임팩트)
    w_elbow_match: float = 2.0      # 내부 배율: 팔꿈치 리프트 = 형태(윈드업)
    w_intensity_max: float = 2.0    # 5.0→2.0: 타격당 최대 35 → 14. sparse 보상이라 35는 advantage 스파이크가
                                    #   목표항(타격당 ~4.7)의 6배 → KL 초과·LR 하락·policy loss 발산 유발했음
    w_intensity_success_start: float = 0.7   # 램프 B 시작 성공률(EMA) — 타격을 먼저 익힌 뒤 세기
    w_intensity_success_full: float = 0.75   # 램프 B 완료 = impact_vel/wrist_snap 완전 소거

    # ══════════ 에너지 ══════════
    # action/joint_vel은 정규화용 (에피소드 합 1 미만). 손목·팔꿈치는 ×0.001 = 사실상 면제 —
    # 스냅/굽힘은 이 관절들이 담당하므로 포화해도 되게 둠. 떨림 억제 항이 아님(Σa²는 부호 뒤집기를 안 벌함).
    w_action: float = 0.0005            # 명령 크기 Σ a² — ±1 clamp 밖으로 새는 것 방지
    w_wrist_action_scale: float = 0.001
    w_elbow_action_scale: float = 0.001
    w_joint_vel: float = 0.0001         # 실제 관절속도 Σ q̇² — 명령 안 한 관성/중력 운동까지 포착
    w_wrist_vel_scale: float = 0.001
    w_elbow_vel_scale: float = 0.001

    # 기계적 일률 |τ·q̇| — 무거운 관절 급가속만 비싸고 자세유지는 공짜(q̇≈0).
    w_power_max: float = 0.05           # 0.1→0.05: 몸통 회전을 싸게 만들어 어깨 과꺾임 해소 [[shoulder-overbend-waist]]
    w_power_success_start: float = 0.6  # 램프 A 시작 성공률(EMA)
    w_power_success_full: float = 0.77  # 램프 A 완료
    w_power_ema_beta: float = 0.9       # 성공률 EMA 평활 (갱신 100스텝 → 시정수 ≈1000스텝)

    # 노는 팔 전용 추가 일률 (헛스윙 표적). 마스크 (1−active_arm) → 치는 팔 0, 동시타 0, 허리 제외.
    # 과거 idle_still(팁 속도 직접 억제)은 이동까지 벌해 붕괴. 이건 (a)일률=정지 무료 (b)배정 히스테리시스
    #    안정 후 (c)램프 B 게이팅 — 세 조건이 다름. [[idle-arm-swing-solved]]
    w_idle_power_max: float = 0.1      # 0.4→0.1: ready_pose 0.264(=home에서 36cm)로 정책이 못 맞추는 요구였고
                                       #   순보상의 63%를 차지하는 최대 페널티였음 (miss3_reach 0.352 동반)

    # ══════════ 자세 ══════════
    # 몸통 비틀림 = |sh1_L + sh1_R − 180°|. 두 shoulder_1은 같은 기준(+x)이라 각각 90°면 그 팔이 몸통 정면
    # → 합 180°면 두 팔이 몸통 기준 좌우대칭. 드럼 위치·좌우분류·허리 부호규약이 전부 불필요.
    # 이전 '드럼 x 기준 허리 정렬'을 대체함. 그건 몸중심 기준이라, 어깨가 중심에서 26cm 밖인 이 로봇에서는
    #    중앙쪽 드럼(crash_l 등)에 대해 정반대를 가리켰음.
    # 실측(2026-08 프로브): 허리 1rad = 몸통 1rad, sideways = sh1 − 90°, 허리 δ 회전 → 합편차 −2δ
    #    ⇒ 허리는 합편차의 절반만 돌면 대칭 회복. 회전 방향은 정책이 스스로 찾음.
    w_waist_twist_max: float = 2.0        # 게이팅 없음 — 편차 22°에서 0.72/타격 (성공 5.5의 13%)
    waist_twist_deadzone: float = 0.175   # 10° 이내 허용 (팔이 안쪽으로 뻗는 건 정상)
    waist_twist_scale: float = 0.35       # 초과분 20°에서 만점
    w_waist_max: float = 0.0              # 0.05→0: 허리 속도 페널티 제거 = 어깨 과꺾임의 뿌리 [[shoulder-overbend-waist]]

    # ══════════ 한계 · 충돌 ══════════
    w_limit: float = 1.0                  # 관절한계 (비손목 7관절, margin 0.10)
    w_wrist_limit: float = 1.5            # 손목 전담 — limit_pen은 마스크로 손목 제외되므로 손목의 유일한 한계 신호
    w_tip_limit: float = 0.15             # 작업공간 상자 이탈 (양팔 OR 이진)
    w_under_drum: float = 0.1             # 드럼면 아래 침투
    w_self_collision: float = 0.5         # 전완끼리 근접
    self_collision_threshold: float = 0.10  # 전완(elbow→wrist) 선분거리 임계 [m]

    # 스틱 '근접(5cm)' 페널티 재도입 금지: 같은드럼 교대연타의 필수 근접까지 벌해서 반대편 접근 →
    #    상시 X자 자세를 학습시켰음. X자는 다음 드럼 이동을 막아 miss를 부름. [[stick-collision-x-lesson]]
    # 접촉-한정(2.5cm)은 올바른 기술 안에서 만족 가능해 X 재구조화 압력이 없음. 철회 신호 = hit_sh_frac↑ / forearm_dist↓
    w_stick_contact_max: float = 4.0        # 램프 B 최대치 (완전접촉 0.1/step, 이벤트당 ~0.4)
    stick_contact_threshold: float = 0.025  # 진짜 접촉 스케일 [m] = 스틱 두께

    # 접근 밴드 — 접촉 '전에' 밀어내는 항. 위 접촉항은 2.5cm 밖에서 gradient가 0이라 접근을 못 막음.
    #    과거 5cm 근접 페널티가 X자를 만든 건 '정상 타격의 필연적 근접'을 벌했기 때문.
    #      지금은 stick_close_hit = 0.000 (타격 시 2.5cm 이내 0회)이고, 아래 strike_phase 게이트로
    #      하강 구간 전체를 면제하므로 그 메커니즘이 구조적으로 발생할 수 없음.
    stick_gap_range: float = 0.08     # 이 거리부터 선형 증가 [m]. 실측 hit_stick_dist=0.452 라 여유 5.6배
    w_stick_gap_max: float = 3.0      # 램프 B 최대치
    # 하강(타격) 면제 조건. 0.15m/0.5m/s 는 과했음 — strike_phase 0.345 로 34.5% 스텝이 무력화됐고,
    #   접촉이 일어나는 중앙-고지대가 그 안에 들어갔음. 타격 시 실측 간격 0.452m 라 면제는 최소만 필요.
    gap_exempt_radius: float = 0.08   # 이 반경 안 + 아래 속도 이상 하강일 때만 면제
    gap_exempt_vz: float = 1.0

    elbow_min_bend: float = 0.35        # 최소 굽힘 [rad] ≈20° — 팔 쫙 펴고 치는 자세 방지 (0°=편, 140°=굽힘)
    w_elbow_bend: float = 1.5           # 부족분 선형 벌점 (양팔 합)


    update_interval: int = 100          # 악기별 가중치 + 게이트 램프 갱신 주기 [step]

class RewardComputer:
    def __init__(
            self, device: torch.device | str,
            cfg: RewardComputerCfg,
            env: EnvRuntimeSpec,
            ctrl_joint_names: list,
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.env = env
        instruments = Instruments()
        self.drum_names = list(instruments.all.keys())

        # 관절 속도 페널티 가중치 (손목·팔꿈치 완화 → 원위 관절로 자연스럽게 침). ctrl_joint_names는 resolve된 순서
        def _vel_scale(name: str) -> float:
            if "wrist" in name: return self.cfg.w_wrist_vel_scale
            if "elbow" in name: return self.cfg.w_elbow_vel_scale
            return 1.0
        vel_w = [_vel_scale(name) for name in ctrl_joint_names]
        self.joint_vel_weight = torch.tensor(
            vel_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # action 페널티 가중치 (손목·팔꿈치 완화). 원위 관절 스냅/굽힘 위해 세금 감면
        def _act_scale(name: str) -> float:
            if "wrist" in name: return self.cfg.w_wrist_action_scale
            if "elbow" in name: return self.cfg.w_elbow_action_scale
            return 1.0
        act_w = [_act_scale(name) for name in ctrl_joint_names]
        self.action_weight = torch.tensor(
            act_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # 손목 관절 인덱스 (각속도 로깅용). hit_per_arm 순서(0=왼,1=오)와 맞춤
        name_to_idx = {name: i for i, name in enumerate(ctrl_joint_names)}
        self.wrist_ids = torch.tensor(
            [name_to_idx["left_wrist"], name_to_idx["right_wrist"]],
            device=self.device, dtype=torch.long,
        )

        # 팔별 관절 인덱스 [왼4, 오4] = 어깨1·어깨2·팔꿈치·손목. 노는 팔 전용 일률 페널티용 (허리는 공용이라 제외)
        self.arm_joint_ids = torch.tensor([
            [name_to_idx["left_shoulder_1"],  name_to_idx["left_shoulder_2"],
             name_to_idx["left_elbow"],       name_to_idx["left_wrist"]],
            [name_to_idx["right_shoulder_1"], name_to_idx["right_shoulder_2"],
             name_to_idx["right_elbow"],      name_to_idx["right_wrist"]],
        ], device=self.device, dtype=torch.long)   # (2, 4)

        # 일반 limit_pen에서 손목 제외용 마스크 (비손목=1, 손목=0). 손목은 wrist_limit_pen에서 전담
        self.limit_joint_mask = torch.ones((1, len(ctrl_joint_names)), device=self.device)
        self.limit_joint_mask[0, self.wrist_ids] = 0.0

        # 팔꿈치 각속도 인덱스 [왼,오] (진단 로그용 — 세기↑ 때 팔꿈치가 실제로 스윙에 참여하는지 관찰). 목표 아님
        self.elbow_ids = torch.tensor(
            [name_to_idx["left_elbow"], name_to_idx["right_elbow"]],
            device=self.device, dtype=torch.long,
        )

        # 어깨1(방위축) 인덱스 [왼,오] — 몸통 비틀림 = (sh1_L + sh1_R) − 180° 계산용
        self.sh1_ids = torch.tensor(
            [name_to_idx["left_shoulder_1"], name_to_idx["right_shoulder_1"]],
            device=self.device, dtype=torch.long,
        )

        # 허리/어깨 각속도 인덱스 (진단 로그용 — 무거운 관절 움직임 모니터)
        self.waist_ids = torch.tensor(
            [i for i, n in enumerate(ctrl_joint_names) if "waist" in n],
            device=self.device, dtype=torch.long,
        )
        self.shoulder_ids = torch.tensor(
            [i for i, n in enumerate(ctrl_joint_names) if "shoulder" in n],
            device=self.device, dtype=torch.long,
        )

        # 게이팅 실효 가중치 — 전부 0에서 시작해 success_ema에 따라 램프인 (갱신은 compute() 말미)
        self.w_power_eff = 0.0         # 램프 A (0.60 → 0.77)
        self.w_waist_eff = 0.0         # 램프 A — 단 w_waist_max=0 이라 항상 0
        self.w_intensity_eff = 0.0     # 램프 B (0.70 → 0.75)
        self.w_stick_eff = 0.0         # 램프 B
        self.w_stick_gap_eff = 0.0     # 램프 B (접근 밴드)
        self.w_idle_power_eff = 0.0    # 램프 B
        # 이 둘만 반대 방향 — 램프 B가 켜지는 만큼 소거 (크로스페이드)
        self.w_impact_vel_eff = self.cfg.w_impact_vel
        self.w_wrist_snap_eff = self.cfg.w_wrist_snap
        # 체크포인트에 저장되지 않음 → resume 시 0에서 재시작, 게이트가 전부 닫힌 뒤 ~2000스텝간 재상승
        self.success_ema = 0.0

        # 악기별 보상 가중치
        self.update_counter = 0

        M = len(instruments.all)
        self.w_drum_success = torch.ones((1, M), device=self.device)
        self.n_success = torch.zeros((1, M), device=self.device)
        self.n_hit = torch.zeros((1, M), device=self.device)

    def compute(
            self,
            success: torch.Tensor,
            wrong_hit: torch.Tensor,
            missed_target: torch.Tensor,
            time_error: torch.Tensor,
            tip_pos: torch.Tensor,
            prev_tip_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            next_hits: torch.Tensor,
            tip_vel: torch.Tensor,
            robot_vel: torch.Tensor,
            robot_torque: torch.Tensor,
            actions: torch.Tensor,
            robot_pos: torch.Tensor,
            joint_low: torch.Tensor,
            joint_high: torch.Tensor,
            wrist_pos: torch.Tensor,    # (N, 2, 3)
            elbow_pos: torch.Tensor,    # (N, 2, 3)
            hit_per_arm: torch.Tensor,  # (N, 2, M)
            success_impact_vel: torch.Tensor,  # (N, M) 성공한 타격의 하강 속도
            success_wrist_ang_vel: torch.Tensor,  # (N, M) 성공한 타격의 손목 각속도
            target_velocity: torch.Tensor,  # (N, M) 마감 타겟의 목표 MIDI velocity(0~127) → 세기 매칭
            success_elbow_apex: torch.Tensor,  # (N, M) 성공 타격의 팔꿈치 리프트 apex [rad]
            target_T: torch.Tensor,  # (N, M) 마감 타겟의 간격 T[s] → 손목/팔꿈치 목표 T-종속 계산
            assigned_use_left: torch.Tensor,  # (N,) bool 스케줄러(reachability)가 정한 h0 담당(왼팔=True)
            assigned_use_case1: torch.Tensor, # (N,) bool 동시타 case2 배정(왼팔→idx0?)
            idle_tgt_pos: torch.Tensor,   # (N,2,3) 스케줄러가 각 팔에 준 첫 노트 (per_arm_pos[:,:,0,:])
            idle_tgt_valid: torch.Tensor, # (N,2) bool 그 노트 유효 (per_arm_valid[:,:,0])
            idle_tgt_time: torch.Tensor,  # (N,2) 그 노트까지 정규화 시간 (per_arm_time[:,:,0])
            last_drum_pos: torch.Tensor,  # (N,2,3) 각 팔이 마지막으로 친 드럼 (배정 없을 때 대기점)
    ) -> tuple[torch.Tensor, dict, dict, torch.Tensor]:  # reward, log, rate_log, active_arm(N,2)
        # goal
        success_reward, wrong_cost, missed_cost, time_accuracy_reward = _goal_terms(
            success=success,
            wrong_hit=wrong_hit,
            missed_target=missed_target,
            time_error=time_error,
            w_drum_success=self.w_drum_success,
            k_accuracy=self.cfg.k_accuracy,
        )

        # proximity / progress / arm assignment / crowd
        proximity_cost, progress_reward, crowd_pen, active_arm, ready_pose_reward = _proximity_terms(
            left_tip_pos=tip_pos[:, 0, :],
            right_tip_pos=tip_pos[:, 1, :],
            prev_left_tip_pos=prev_tip_pos[:, 0, :],
            prev_right_tip_pos=prev_tip_pos[:, 1, :],
            drum_pos=drum_pos,
            next_hits=next_hits,
            k_time_to_hit=self.cfg.k_time_to_hit,
            hit_xy_radius=self.cfg.hit_xy_radius,
            hit_z_range=self.cfg.hit_z_range,
            ready_height=self.cfg.ready_height,
            k_ready=self.cfg.k_ready,
            aim_offset=self.cfg.aim_offset,
            assigned_use_left=assigned_use_left,
            assigned_use_case1=assigned_use_case1,
            idle_tgt_pos=idle_tgt_pos,
            idle_tgt_valid=idle_tgt_valid,
            idle_tgt_time=idle_tgt_time,
            last_drum_pos=last_drum_pos,
            k_ready_urgency=self.cfg.k_ready_urgency,
            rest_margin_x=self.cfg.rest_margin_x,
        )

        # 쉬는 팔 손목을 가동범위 중앙으로 — 드럼은 손목 주도라 '양방향 스냅 여유'가 타격력을 결정.
        #   ready_pose에 곱하기 = ready_sq에 손목항 더하기와 동일 (e^{-k(a+b)} = e^{-ka}·e^{-kb}) → jit 시그니처 불변.
        #   목표가 '중앙'이라 스냅 회전 방향(부호)을 몰라도 안전. 한계 park만 벌하고 스윙 구간(±20°)은 거의 무료.
        #   필요 스윙폭 2L(T) = 최대 60°, 가동범위 99° → 중앙(19.5°)만이 양쪽 19.5° 여유 확보.
        w_ang = robot_pos[:, self.wrist_ids]                                  # (N,2) 손목각
        w_lo  = joint_low[:,  self.wrist_ids]                                 # (1,2)
        w_hi  = joint_high[:, self.wrist_ids]
        w_rng = (w_hi - w_lo).clamp(min=1e-6)
        w_frac = (w_ang - w_lo) / w_rng                                       # (N,2) 0=하한, 1=상한
        w_dev = w_ang - 0.5 * (w_lo + w_hi)                                   # (N,2) 중앙 대비 편차
        idle_dev = torch.where(assigned_use_left, w_dev[:, 1], w_dev[:, 0])   # (N,) 왼팔이 침→오른 손목
        d_wrist = self.cfg.stick_len * idle_dev                               # 호길이[m]
        at_home = ready_pose_reward                                           # (N,) 위치만 (손목항 적용 전) — idle_power 게이트
        ready_pose_reward = ready_pose_reward * torch.exp(-self.cfg.k_ready * d_wrist * d_wrist)
        idle_wrist_frac = torch.where(assigned_use_left, w_frac[:, 1], w_frac[:, 0])   # (N,) 로그용

        # off-target 페널티에서 제외할 현재 타겟 마스크 (h0)
        M = drum_pos.shape[1]
        target_mask = next_hits[:, 0, :M] > 0.5

        # tip position penalties
        tip_limit_pen, under_drum_pen, off_target_pen = _tip_position_penalties(
            left_tip_pos=tip_pos[:, 0, :],
            right_tip_pos=tip_pos[:, 1, :],
            drum_pos=drum_pos,
            target_mask=target_mask,
            x_limit=self.cfg.x_limit,
            y_limit_l=self.cfg.y_limit_l,
            y_limit_h=self.cfg.y_limit_h,
            z_limit=self.cfg.z_limit,
            drum_xy_margin=self.cfg.drum_xy_margin,
            drum_z_margin=self.cfg.drum_z_margin,
            off_target_xy_radius=self.cfg.off_target_xy_radius,
            off_target_z_range=self.cfg.off_target_z_range,
        )

        # global penalties
        action_l2, joint_vel_l2, limit_pen, power_l1 = _global_penalties(
            joint_vel=robot_vel,
            joint_torque=robot_torque,
            action=actions,
            robot_pos=robot_pos,
            joint_low=joint_low,
            joint_high=joint_high,
            limit_margin=self.cfg.limit_margin,
            limit_joint_mask=self.limit_joint_mask,
            joint_vel_weight=self.joint_vel_weight,
            action_weight=self.action_weight,
        )

        # self collision penalty (전완 선분: elbow → wrist)
        self_col_pen, forearm_dist, _, _ = _self_collision_penalty(
            elbow_l=elbow_pos[:, 0, :],
            wrist_l=wrist_pos[:, 0, :],
            elbow_r=elbow_pos[:, 1, :],
            wrist_r=wrist_pos[:, 1, :],
            threshold=self.cfg.self_collision_threshold,
        )

        # 스틱(wrist→tip) 선분거리 — threshold=0 이라 거리만 측정, 페널티는 아래 접촉-한정 항에서 별도 계산
        _, stick_dist, stick_s, stick_t = _self_collision_penalty(
            elbow_l=wrist_pos[:, 0, :],   # 왼 스틱 시작 = 왼 wrist
            wrist_l=tip_pos[:, 0, :],     # 왼 스틱 끝 = 왼 tip
            elbow_r=wrist_pos[:, 1, :],   # 오 스틱 시작 = 오 wrist
            wrist_r=tip_pos[:, 1, :],     # 오 스틱 끝 = 오 tip
            threshold=0.0,                 # 페널티 0 — 거리만 사용
        )

        # phase-1 부트스트랩 — 성공 타격의 하강속도/손목ω 자체에 보상 (상한 없는 단조증가). 램프 B에 반비례 소거
        impact_vel_reward = (success_impact_vel / (self.cfg.impact_vel_max + 1e-8)).sum(dim=1)  # (N,)
        wrist_snap_reward = (success_wrist_ang_vel / (self.cfg.wrist_snap_max + 1e-8)).sum(dim=1)  # (N,)

        # 세기 매칭 — 목표는 세기(i)와 타격간격(T)의 함수. 실기 state_motion_generator.cpp 이식
        d2r = 3.141592653589793 / 180.0
        u = target_velocity / 127.0                                          # (N, M) 0~1
        i = 1.0 + 6.0 * u                                                    # MIDI velocity → 실기 intensity 1~7
        T = target_T.clamp(0.1, 1.0)                                         # (N, M) 간격[s]

        # 손목ω 목표 = 2·L(T)·f_w(i) / T_hit  — 스트로크 왕복거리 ÷ 허용시간 (주력 항)
        f_w = torch.where(i < 5.0, 0.25 * i - 0.25, i / 3.0 - 2.0 / 3.0)
        L   = torch.where(T < 0.5, (40.0 * T + 10.0) * d2r, torch.full_like(T, 30.0 * d2r))
        T_hit = (0.3 * T).clamp(max=0.2)
        w_wrist_tgt = (2.0 * L * f_w / T_hit).clamp(min=0.0, max=30.0)       # (N, M) rad/s

        # 팔꿈치 리프트 목표 = lift_mag(T)·f_e(i) — 형태 항. i≤3이면 0 = 여린 타격은 팔 안 듦
        f_e = torch.where(i <= 3.0, torch.zeros_like(i), 0.1 * i + 0.5)
        lift_mag = torch.where(T < 0.2, torch.zeros_like(T),
                     torch.where(T <= 0.5, T * (15.0 * d2r) / 0.5, torch.full_like(T, 15.0 * d2r)))
        elbow_tgt = lift_mag * f_e                                           # (N, M) rad

        err_wrist = success_wrist_ang_vel - w_wrist_tgt
        err_elbow = success_elbow_apex    - elbow_tgt
        r_wrist = torch.exp(-self.cfg.k_wrist_match * err_wrist * err_wrist)  # (N, M) 1=정확
        r_elbow = torch.exp(-self.cfg.k_elbow_match * err_elbow * err_elbow)
        intensity_reward = (success.float() * (
            self.cfg.w_wrist_match * r_wrist + self.cfg.w_elbow_match * r_elbow
        )).sum(dim=1)  # (N,)

        # 허리 속도 페널티 (현재 w_waist_max=0 이라 미적용)
        waist_vel = robot_vel[:, self.waist_ids]                          # (N, n_waist)
        waist_vel_pen = torch.sum(waist_vel * waist_vel, dim=-1)          # (N,)

        # 팔꿈치 최소 굽힘 — min_bend 미만이면 부족분만큼 벌점 (양팔 합)
        elbow_angle_now = robot_pos[:, self.elbow_ids]                    # (N, 2) [0=편, 140°=굽힘]
        elbow_bend_pen = torch.clamp(self.cfg.elbow_min_bend - elbow_angle_now, min=0.0).sum(dim=-1)  # (N,)

        reward = compute_rewards(
            success_reward=success_reward,
            wrong_cost=wrong_cost,
            missed_cost=missed_cost,
            time_accuracy_reward=time_accuracy_reward,
            
            proximity_cost=proximity_cost,
            progress_reward=progress_reward,

            action_l2=action_l2,
            joint_vel_l2=joint_vel_l2,
            power_l1=power_l1,
            limit_pen=limit_pen,
            tip_limit_pen=tip_limit_pen,
            under_drum_pen=under_drum_pen,

            w_success=self.cfg.w_success,
            w_wrong=self.cfg.w_wrong,
            w_miss=self.cfg.w_miss,
            w_time_accuracy=self.cfg.w_time_accuracy,

            w_progress=self.cfg.w_progress,
            w_proximity=self.cfg.w_proximity,

            w_action=self.cfg.w_action,
            w_joint_vel=self.cfg.w_joint_vel,
            w_power=self.w_power_eff,
            w_limit=self.cfg.w_limit,
            w_tip_limit=self.cfg.w_tip_limit,
            w_under_drum=self.cfg.w_under_drum,
            w_self_collision=self.cfg.w_self_collision,
            self_col_pen=self_col_pen,
            impact_vel_reward=impact_vel_reward,
            w_impact_vel=self.w_impact_vel_eff,
            wrist_snap_reward=wrist_snap_reward,
            w_wrist_snap=self.w_wrist_snap_eff,
            intensity_reward=intensity_reward,
            w_intensity=self.w_intensity_eff,
            off_target_pen=off_target_pen,
            w_off_target=self.cfg.w_off_target,
            crowd_pen=crowd_pen,
            w_crowd=self.cfg.w_crowd,
            waist_vel_pen=waist_vel_pen,
            w_waist=self.w_waist_eff,
            ready_pose_reward=ready_pose_reward,
            w_ready=self.cfg.w_ready,
        )

        # ── 이하 jit 밖 적용 (compute_rewards 시그니처 고정용) ──
        reward = reward - self.cfg.w_elbow_bend * elbow_bend_pen

        # 손목 한계 — 명령은 clip_and_convert가 이미 자르므로, 이 항이 막는 건 '한계에 붙어사는 것'.
        # margin이 팔의 2배인 이유: 손목 25rad/s면 1스텝에 24° 이동 → 팔과 같은 margin은 사후 통보가 됨
        w_pos = robot_pos[:, self.wrist_ids]                        # (N, 2)
        w_lo  = joint_low[:,  self.wrist_ids]                       # (1, 2)
        w_hi  = joint_high[:, self.wrist_ids]
        wm = self.cfg.wrist_limit_margin
        lo_v = torch.clamp((w_lo + wm) - w_pos, min=0.0)
        hi_v = torch.clamp(w_pos - (w_hi - wm), min=0.0)
        wrist_limit_pen = torch.sum(lo_v * lo_v + hi_v * hi_v, dim=-1)   # (N,)
        reward = reward - self.cfg.w_wrist_limit * wrist_limit_pen

        # 스틱 접촉 — 2.5cm 미만 '진짜 접촉'만. 임계를 키우면 X자 자세 재발 [[stick-collision-x-lesson]]
        stick_contact_pen = torch.clamp(self.cfg.stick_contact_threshold - stick_dist, min=0.0)
        reward = reward - self.w_stick_eff * stick_contact_pen

        # 접근 밴드 — 하강(타격) 구간은 면제하고 이동/대기 중에만 6cm부터 밀어냄.
        #   strike_phase = 어느 팁이든 '드럼 15cm 이내 + 0.5m/s 이상 하강' → 접근·타격·리바운드 전 구간 커버.
        #   접촉항(2.5cm)은 상시 유지 — 이건 그 앞단에 기울기를 만드는 항.
        tip_xy = tip_pos[:, :, None, 0:2]                                    # (N,2,1,2)
        drum_xy = drum_pos[:, None, :, 0:2]                                  # (N,1,M,2)
        d2_td = ((tip_xy - drum_xy) ** 2).sum(dim=-1)                        # (N,2,M)
        near_drum = d2_td.min(dim=-1).values < (self.cfg.gap_exempt_radius ** 2)   # (N,2)
        descending = tip_vel[:, :, 2] < -self.cfg.gap_exempt_vz                    # (N,2)
        strike_phase = (near_drum & descending).any(dim=1)                   # (N,)
        gap_gate = 1.0 - strike_phase.float()
        stick_gap_pen = torch.clamp(self.cfg.stick_gap_range - stick_dist, min=0.0) * gap_gate
        reward = reward - self.w_stick_gap_eff * stick_gap_pen

        # ===== [진단] 스틱 접촉이 '어느 상황'에서 나는지 3분할 + 대기점 겹침 =====
        #   처방이 상황마다 완전히 다름: 동시타→스케줄러 분배 / 타격→대기점·연타간격 / 이동→CROSS
        #   ⚠️ 원인 모르고 임계를 키우면 X자 버그 재발. 반드시 재보고 처방할 것.
        _close = (stick_dist < self.cfg.stick_contact_threshold).float()          # (N,)
        _tgt_n = target_mask.float().sum(dim=-1)                                  # (N,) h0 타겟 개수
        _two = (_tgt_n >= 2.0)
        _hit = hit_per_arm.any(dim=2).any(dim=1)
        _f_hit = (_hit & ~_two).float()
        # 대기점이 상대 팔의 타격 드럼 위에 겹치는가 (crowd는 0~7cm만 봐서 ready 18cm를 놓침)
        _sched = torch.where(assigned_use_left.unsqueeze(-1), idle_tgt_pos[:, 1], idle_tgt_pos[:, 0])
        _last  = torch.where(assigned_use_left.unsqueeze(-1), last_drum_pos[:, 1], last_drum_pos[:, 0])
        _has   = torch.where(assigned_use_left, idle_tgt_valid[:, 1], idle_tgt_valid[:, 0])
        _home  = torch.where(_has.unsqueeze(-1), _sched, _last)                   # (N,3)
        _f_one = (_tgt_n == 1.0).float()
        # 쉬는 팔의 대기점이 '자기 반대편'인가 — ready_pose가 능동적으로 크로스를 명령하는지 직접 측정.
        #   use_left=True → 쉬는 팔은 오른팔 → home_x < 0 이면 크로스.
        #   시간 가중이 없어 노트가 1.9초 뒤여도 지금 건너가므로, 이 비율이 곧 크로스 노출 시간.
        _idle_is_right = assigned_use_left
        _home_x = _home[:, 0]
        _crossed = torch.where(_idle_is_right, _home_x < 0.0, _home_x > 0.0).float()
        _home_own = torch.where(_idle_is_right, _home_x, -_home_x)   # >0 = 자기 쪽, <0 = 크로스 깊이
        # 접촉 '위치' — s,t로 최근접점을 복원. x가 좌측(<−0.1)에 몰리면 오른팔 크로스가 근본 원인이고,
        #   0 근처 분산이면 중앙 드럼 교대 → 접근 밴드로 해결 가능. 처방이 갈리는 지점.
        _cl_L = wrist_pos[:, 0] + stick_s.unsqueeze(-1) * (tip_pos[:, 0] - wrist_pos[:, 0])   # (N,3)
        _cl_R = wrist_pos[:, 1] + stick_t.unsqueeze(-1) * (tip_pos[:, 1] - wrist_pos[:, 1])
        _cl_mid = 0.5 * (_cl_L + _cl_R)                                                       # (N,3)

        # 몸통 비틀림 — 타격 순간 |sh1_L + sh1_R − 180°|. 합 180° = 두 팔이 몸통 기준 좌우대칭.
        # 성공률 게이팅 없음(처음부터 최대치): any_hit 게이트로 충분하고, 자세는 리치와 함께 배워야 하므로.
        #   안 치는 스텝은 0 → '자세만 다듬기' 우회로 없음. 최악 자세로 쳐도 5.5−2.0 > −3.0 이라 회피 유인 없음
        sh1_sum = robot_pos[:, self.sh1_ids].sum(dim=-1)                        # (N,) sh1_L + sh1_R [rad]
        twist = (sh1_sum - 3.141592653589793).abs()                             # (N,) |합 − 180°|
        twist_ex = torch.clamp(twist - self.cfg.waist_twist_deadzone, min=0.0)  # deadzone 초과분
        twist_n = (twist_ex / self.cfg.waist_twist_scale).clamp(max=1.0)        # 0~1 정규화·포화
        any_hit_f = hit_per_arm.any(dim=2).any(dim=1).float()                   # (N,) 타격 순간만
        waist_twist_pen = twist_n * twist_n * any_hit_f                         # (N,)
        reward = reward - self.cfg.w_waist_twist_max * waist_twist_pen

        # 노는 팔 일률 — 치는 팔은 마스크 0(세게 치는 데 영향 없음), 허리는 공용이라 제외.
        #   × at_home: 대기점에 도착한 뒤에만 과금 → 이동 중에는 무료라 선행 포지셔닝을 막지 않음.
        #   게임 불가: 집에 정지하면 q̇≈0 이라 게이트가 열려도 |τq̇|≈0. "집에 가서 가만히"가 유일한 최적해.
        arm_pw = (robot_torque[:, self.arm_joint_ids] * robot_vel[:, self.arm_joint_ids]).abs()  # (N,2,4)
        arm_power = arm_pw.sum(dim=-1)                                          # (N,2) 팔별 기계일률
        idle_power = ((1.0 - active_arm) * arm_power).sum(dim=-1) * at_home     # (N,) 노는 팔 몫 × 도착도
        reward = reward - self.w_idle_power_eff * idle_power

        num_hit_drum = success.float() + missed_target.float()

        num_hit = num_hit_drum.float().sum(dim=-1)
        num_success = success.float().sum(dim=-1)
        num_wrong = wrong_hit.float().sum(dim=-1)
        num_missed = missed_target.float().sum(dim=-1)

        log_terms = {
            "reward": reward,
            "proximity": proximity_cost,
            "progress(x100)": progress_reward * 100,
            "joint_vel_l2": joint_vel_l2,
            "power_l1": power_l1,
            "w_power_eff(x1000)": torch.full_like(reward, self.w_power_eff * 1000.0),
            "waist_ang_vel": robot_vel[:, self.waist_ids].abs().mean(dim=-1),
            "shoulder_ang_vel": robot_vel[:, self.shoulder_ids].abs().mean(dim=-1),
            "elbow_ang_vel": robot_vel[:, self.elbow_ids].abs().mean(dim=-1),
            "ready_pose": ready_pose_reward,
            "idle_wrist_frac": idle_wrist_frac,        # 쉬는 팔 손목 위치 (0=하한, 0.5=중앙, 1=상한)
            "wrist_limit_pen(x100)": wrist_limit_pen * 100,
            "waist_twist_pen": waist_twist_pen,
            "idle_power": idle_power,
            "crowd_pen": crowd_pen,
            "self_collision_pen": self_col_pen,
            "forearm_dist": forearm_dist,
            "stick_dist": stick_dist,
            "stick_gap_pen(x100)": stick_gap_pen * 100,
            "intensity_reward": intensity_reward,
            "w_intensity_eff": torch.full_like(reward, self.w_intensity_eff),
        }

        # 실제 타격 순간만 뽑아 평균 (hit이 난 팔만)


        rate_log_terms = {
            "success_rate": torch.stack([num_success, num_hit], dim=-1),
            "wrong_rate": torch.stack([num_wrong, num_hit], dim=-1),
            "miss_rate": torch.stack([num_missed, num_hit], dim=-1),
            # 스틱 접촉률: 임계(2.5cm) 미만인 스텝 비율 → 0으로 수렴해야 성공
            "stick_contact": torch.stack([(stick_dist < self.cfg.stick_contact_threshold).float(), torch.ones_like(stick_dist)], dim=-1),
            # 스틱 접촉 3분할 — 각 상황 '안에서의' 접촉 비율 (분모가 그 상황의 스텝 수)
            "stick_close_hit":  torch.stack([_close * _f_hit, _f_hit], dim=-1),   # 단일 타격 순간
            # 쉬는 팔 대기점이 상대 팔 타격 드럼 위(수평 13cm 이내)에 있는 비율
            # 대기점이 반대편인 비율 — 0.5↑ 면 ready_pose가 크로스를 명령 중 (가설 확정)
            "home_crossed": torch.stack([_crossed * _f_one, _f_one], dim=-1),
            # 대기점의 '자기 쪽' 부호거리 [m] — 음수면 크로스, 크기가 침범 깊이
            "home_own_x": torch.stack([_home_own * _f_one, _f_one], dim=-1),
            # 접촉 '부위' — 선분 파라미터 (0=손목, 1=팁). (1,0)/(0,1) = 팁↔자루 교차 = 위상반대 스트로크
            # 실제 접촉(스틱 두께 14mm) — 2.5cm '근접'과 구분
            "stick_lt_14mm": torch.stack([(stick_dist < 0.014).float(), torch.ones_like(stick_dist)], dim=-1),
            # 거리 분포 꼬리 — 밴드 폭을 넓힐 여지가 있는지 판단
            "stick_lt_60mm": torch.stack([(stick_dist < 0.060).float(), torch.ones_like(stick_dist)], dim=-1),
            # 타격 스텝의 스틱 간격 — 접근 밴드가 정상 주법을 건드리는지 확인 (밴드 0.06보다 커야 안전)
            "hit_stick_dist": torch.stack([stick_dist * _f_hit, _f_hit], dim=-1),
            # 하강 구간 면제 비율 (너무 크면 밴드가 무력화)
            "strike_phase": torch.stack([strike_phase.float(), torch.ones_like(stick_dist)], dim=-1),
            # 접촉 지점 좌표 — x가 좌측 편중이면 크로스 배정, 0 근처면 중앙 드럼 교대
            "close_x": torch.stack([_cl_mid[:, 0] * _close, _close], dim=-1),
            "close_z": torch.stack([_cl_mid[:, 2] * _close, _close], dim=-1),
            # 타격 순간 손목이 가동범위 어디인가 (0/1에 붙으면 스냅이 한계에서 끝남 = 세기 상한 원인)
            "hit_wrist_frac": torch.stack([(w_frac * hit_per_arm.any(dim=2).float()).sum(dim=1),
                                           hit_per_arm.any(dim=2).float().sum(dim=1)], dim=-1),
        }

        for i, name in enumerate(self.drum_names):
            rate_log_terms[f"{name}_success_rate"] = torch.stack([success[:, i], num_hit_drum[:, i]], dim=-1)

        # 세기(dynamics) 진단: 성공 타격을 목표 velocity 구간(p≤48 / mf≤90 / f>90, 실측 terciles)으로 나눠
        #   달성 충격속도·손목ω 기록 → soft<mid<loud로 분리되면 다이내믹 학습된 것. + 목표 매칭 정확도(0~1).
        succ_f = success.float()
        match = 0.5 * (r_wrist + r_elbow)                                      # (N, M) 1=목표 정확 매칭 (손목ω + 팔꿈치 리프트)
        rate_log_terms["intensity_match"] = torch.stack([(match * succ_f).sum(dim=-1), succ_f.sum(dim=-1)], dim=-1)
        for lo, hi, tag in ((0.5, 48.5, "p"), (48.5, 90.5, "mf"), (90.5, 1e9, "f")):
            bm = (succ_f * ((target_velocity > lo) & (target_velocity <= hi)).float())  # (N, M) 그 세기구간 성공
            cnt = bm.sum(dim=-1)
            rate_log_terms[f"dyn_wrist_{tag}"]  = torch.stack([(success_wrist_ang_vel * bm).sum(dim=-1), cnt], dim=-1)
            rate_log_terms[f"dyn_elbow_{tag}"]  = torch.stack([(success_elbow_apex * bm).sum(dim=-1), cnt], dim=-1)

        return reward, log_terms, rate_log_terms, active_arm
    
    def update_difficulty_weights(
            self,
            success: torch.Tensor,
            missed_target: torch.Tensor,
    ):
        self.update_counter += 1

        # 통계 누적
        self.n_success += success.float().sum(dim=0)
        self.n_hit += (
            success.float().sum(dim=0)
            + missed_target.float().sum(dim=0)
        )

        # 일정 횟수마다 업데이트
        if self.update_counter % self.cfg.update_interval == 0:
            eps = 1e-6
            success_ratio = self.n_success / (self.n_hit + eps)

            # 어려운 드럼일수록 큰 가중치
            self.w_drum_success = 1.0 / (success_ratio + eps)

            # 평균 1로 정규화
            self.w_drum_success /= self.w_drum_success.mean()

            # 전체 성공률 EMA로 일률 페널티 게이팅 (헛스윙은 후반 현상 → 나중에 켬)
            overall_succ = (self.n_success.sum() / (self.n_hit.sum() + eps)).item()
            beta = self.cfg.w_power_ema_beta
            self.success_ema = beta * self.success_ema + (1.0 - beta) * overall_succ
            denom = max(self.cfg.w_power_success_full - self.cfg.w_power_success_start, 1e-6)
            ramp = (self.success_ema - self.cfg.w_power_success_start) / denom
            ramp = min(max(ramp, 0.0), 1.0)
            self.w_power_eff = self.cfg.w_power_max * ramp
            self.w_waist_eff = self.cfg.w_waist_max * ramp   # 전용 허리 페널티도 동일 게이팅

            # 세기 매칭은 별도 램프 — 성공률 0.7부터 시작 (타격 먼저 익힌 뒤 세기). [[intensity-teaching-design]]
            denom_i = max(self.cfg.w_intensity_success_full - self.cfg.w_intensity_success_start, 1e-6)
            ramp_i = (self.success_ema - self.cfg.w_intensity_success_start) / denom_i
            ramp_i = min(max(ramp_i, 0.0), 1.0)
            self.w_intensity_eff = self.cfg.w_intensity_max * ramp_i
            # 접촉-한정 스틱 페널티도 동일 램프 (스킬 굳은 뒤에만 켬 → 초기 탐색/분지 선택 영향 0)
            self.w_stick_eff = self.cfg.w_stick_contact_max * ramp_i
            self.w_stick_gap_eff = self.cfg.w_stick_gap_max * ramp_i
            self.w_idle_power_eff = self.cfg.w_idle_power_max * ramp_i   # 노는 팔 일률 (스킬 굳은 뒤에만)
            # 크로스페이드: 세기 매칭 켜지는 만큼 무조건 "세게" 페이드아웃 (phase-1 결정적 타격 → 목표 매칭)
            self.w_impact_vel_eff = self.cfg.w_impact_vel * (1.0 - ramp_i)
            self.w_wrist_snap_eff = self.cfg.w_wrist_snap * (1.0 - ramp_i)

            # 통계 초기화
            self.n_success.zero_()
            self.n_hit.zero_()