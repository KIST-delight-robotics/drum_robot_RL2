"""
보상 클래스
"""

from __future__ import annotations

import torch    # pyright: ignore[reportMissingImports]  (Isaac Sim이 런타임에 import path 추가)
from dataclasses import dataclass

from .specs import EnvRuntimeSpec, Instruments

# 팔 상태 번호. RewardComputer.compute 맨 앞에서 관측에 이미 있는 값으로만 정한다 (정책이 같은 판단을 할 수 있게).
#   스트로크: 자기 노트까지 stroke_s 이하, 또는 스틱 끝이 드럼 근처에서 빠르게 내려가는 중. 당기지 않음, 정지 벌점 없음
#   준비 이동: 자기 노트까지 prep_start_s 이하. 노트 위로 당김, 정지 벌점 없음
#   노는 팔: 맡은 노트가 있고 prep_start_s 보다 멂. 다음 노트 기준 휴식 위치로 당기고, 도착하면 정지 벌점
#   자유 팔: 맡은 노트가 없음. 스틱 끝에서 수평으로 가장 가까운 드럼 기준 휴식 위치로 당기고, 도착하면 정지 벌점
ARM_STROKE = 0
ARM_PREP = 1
ARM_IDLE = 2
ARM_FREE = 3

# 모듈 레벨 jit 함수들
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
        ready_height: float,        # 대기점 높이 (드럼 위)
        aim_offset: float,          # 팔별 조준 오프셋 [m] (왼팔 -x, 오른팔 +x)
        assigned_use_left: torch.Tensor,  # (N,) bool 스케줄러가 정한 h0 담당 (왼팔=True)
        assigned_use_case1: torch.Tensor, # (N,) bool 동시타 배정 (왼팔이 idx0 담당이면 True)
        idle_tgt_pos: torch.Tensor,   # (N, 2, 3) 스케줄러가 각 팔에 준 첫 노트 위치 (per_arm_pos[:,:,0,:])
        idle_tgt_valid: torch.Tensor, # (N, 2) bool 그 노트가 유효한가 (per_arm_valid[:,:,0])
        pull_note: torch.Tensor,      # (N, 2) bool 준비 이동 상태인 팔 (대기점 = 노트 위)
        rest_margin_x: float,         # 휴식 위치가 미드라인을 넘어도 되는 여유 [m]
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
    idx0 = torch.where(idx0 >= 0, idx0, torch.zeros_like(idx0))  # -1 이면 0 (인덱싱 오류 방지)

    idx1 = values[:, 1].unsqueeze(1)
    idx1 = torch.where(idx1 >= 0, idx1, torch.zeros_like(idx1))  # -1 이면 0 (인덱싱 오류 방지)

    # --------------------

    # 팔별 조준 오프셋: 왼팔은 드럼 중심 -x, 오른팔은 +x (각자 어깨 쪽)
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

    d_l0 = dist_l.gather(1, idx0).squeeze(1)   # (N,) 각 팔에서 h0 까지 3D 제곱거리
    d_r0 = dist_r.gather(1, idx0).squeeze(1)

    d_l1 = dist_l.gather(1, idx1).squeeze(1)   # 각 팔에서 동시타 둘째 드럼까지
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
    # case 1: 현재 타격 1개 (다음 타격 h1 은 노는 팔에 배정)
    # --------------------
    one_mask = (target_count == 1)

    # h0 배정: 스케줄러가 정한 담당 팔을 그대로 사용
    use_left = assigned_use_left
    # 접근/진척 보상용 비용은 담당 팔의 팁-h0 거리
    one_cost = torch.where(use_left, d_l0, d_r0)
    cost = torch.where(one_mask, one_cost, cost)

    # crowd: 배정 안 된(노는) 팔이 타겟 h0 의 타격영역에 들어오면 페널티
    free_dxy = torch.where(use_left, dxy_r0, dxy_l0)   # 노는 팔의 h0 까지 XY 제곱거리
    free_dz  = torch.where(use_left, dz_r0,  dz_l0)    # 노는 팔의 h0 까지 Z 차이
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

    # 배정 2가지: case1 = 왼팔 idx0 / 오른팔 idx1,  case2 = swap
    cost_case1 = (d_l0 + d_r1) / 2
    cost_case2 = (d_l1 + d_r0) / 2

    # 동시타 비용은 두 배정 중 싼 쪽. 스케줄러 배정(assigned_use_case1)은 obs 로만 전달됨
    two_cost = torch.minimum(cost_case1, cost_case2)
    cost = torch.where(two_mask, two_cost, cost)

    # --------------------
    # case 3: 타겟 0개
    # --------------------
    #
    # 나중에 필요하면 작성
    #

    # --------------------
    # 다음 노트 담당 팔 (관측의 역할 칸에 들어가는 값). 팔 상태 판단과는 별개이며 팔 상태는 RewardComputer.compute 맨 앞에서 정한다
    # --------------------
    active_arm_l = torch.zeros(N, device=device)
    active_arm_r = torch.zeros(N, device=device)
    # case 1: h0 담당 팔만 활성
    active_arm_l = torch.where(one_mask, use_left.float(), active_arm_l)
    active_arm_r = torch.where(one_mask, (~use_left).float(), active_arm_r)
    # case 2: 양팔 모두 활성
    active_arm_l = torch.where(two_mask, torch.ones_like(active_arm_l), active_arm_l)
    active_arm_r = torch.where(two_mask, torch.ones_like(active_arm_r), active_arm_r)
    active_arm = torch.stack([active_arm_l, active_arm_r], dim=1)   # (N, 2)  1=다음 노트 담당

    # 대기점 위치 (팔별 [m]). 관측에 있는 값(팔별 노트 칸, 스틱 끝, 드럼 위치)만으로 정한다.
    #   기준 드럼: 맡은 노트가 있으면 그 노트, 없으면 스틱 끝에서 수평으로 가장 가까운 드럼.
    #   준비 이동(pull_note)이면 기준 드럼 위 ready_height.
    #   그 외는 같은 기준의 휴식 위치: 높이와 앞뒤는 기준과 같고, x 만 미드라인을 rest_margin_x 이상 넘지 못하게 자기 쪽으로 제한
    #   (반대편에서 기다리지 않게 해 스틱 충돌 방지).
    tips = torch.stack([left_tip_pos, right_tip_pos], dim=1)                              # (N,2,3)
    d2_near = ((tips[:, :, None, 0:2] - drum_pos[:, None, :, 0:2]) ** 2).sum(dim=-1)      # (N,2,M)
    near_p = torch.gather(drum_pos, 1, d2_near.argmin(dim=-1).unsqueeze(-1).expand(-1, -1, 3))   # (N,2,3)
    base_p = torch.where(idle_tgt_valid.unsqueeze(-1), idle_tgt_pos, near_p)              # (N,2,3)
    home_list: list[torch.Tensor] = []
    for a in range(2):
        sgn = -1.0 if a == 0 else 1.0                                                     # 자기 쪽 x 방향 (왼팔 -x, 오른팔 +x)
        note_p = base_p[:, a]                                                             # (N,3)
        note_x = note_p[:, 0] + sgn * aim_offset                                          # 조준 오프셋 (자기 어깨 쪽)
        rest_x = sgn * torch.clamp(sgn * note_x, min=-rest_margin_x)
        home_x = torch.where(pull_note[:, a], note_x, rest_x)
        home_list.append(torch.stack([home_x, note_p[:, 1], note_p[:, 2] + ready_height], dim=-1))
    home_pos = torch.stack(home_list, dim=1)                                              # (N,2,3)

    return cost, crowd_pen, active_arm, home_pos

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
        aim_offset: float,
        assigned_use_left: torch.Tensor,
        assigned_use_case1: torch.Tensor,
        idle_tgt_pos: torch.Tensor,
        idle_tgt_valid: torch.Tensor,
        pull_note: torch.Tensor,
        rest_margin_x: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = drum_pos.shape[1]

    nearest_target_mask = next_hits[:, 0, :M] > 0.5   # h0
    first_time = next_hits[:, 0, M]

    curr_cost, crowd_pen, active_arm, home_pos = compute_arm_target_assignment(
        target_mask=nearest_target_mask,
        drum_pos=drum_pos,
        left_tip_pos=left_tip_pos,
        right_tip_pos=right_tip_pos,
        hit_xy_radius=hit_xy_radius,
        hit_z_range=hit_z_range,
        ready_height=ready_height,
        aim_offset=aim_offset,
        assigned_use_left=assigned_use_left,
        assigned_use_case1=assigned_use_case1,
        idle_tgt_pos=idle_tgt_pos,
        idle_tgt_valid=idle_tgt_valid,
        pull_note=pull_note,
        rest_margin_x=rest_margin_x,
    )
    # prev-position 비용은 진척(progress) 계산용. 배정(assigned_use_left)은 동일하게 주입
    prev_cost, _, _, _ = compute_arm_target_assignment(
        target_mask=nearest_target_mask,
        drum_pos=drum_pos,
        left_tip_pos=prev_left_tip_pos,
        right_tip_pos=prev_right_tip_pos,
        hit_xy_radius=hit_xy_radius,
        hit_z_range=hit_z_range,
        ready_height=ready_height,
        aim_offset=aim_offset,
        assigned_use_left=assigned_use_left,
        assigned_use_case1=assigned_use_case1,
        idle_tgt_pos=idle_tgt_pos,
        idle_tgt_valid=idle_tgt_valid,
        pull_note=pull_note,
        rest_margin_x=rest_margin_x,
    )

    proximity_cost = torch.exp(-k_time_to_hit * first_time) * curr_cost
    progress_reward = prev_cost - curr_cost

    return proximity_cost, progress_reward, crowd_pen, active_arm, home_pos   # home_pos (N,2,3) 팔별 대기점 위치 [m]

@torch.jit.script
def _self_collision_penalty(
        elbow_l: torch.Tensor,  # (N, 3) 왼쪽 전완 시작 (elbow)
        wrist_l: torch.Tensor,  # (N, 3) 왼쪽 전완 끝 (wrist)
        elbow_r: torch.Tensor,  # (N, 3) 오른쪽 전완 시작 (elbow)
        wrist_r: torch.Tensor,  # (N, 3) 오른쪽 전완 끝 (wrist)
        threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # penalty, dist, s, t
    # 두 전완(elbow에서 wrist)을 선분으로 보고 선분 간 최소 거리 계산
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

    # t0 가 [0,1] 밖으로 나간 경우 각 끝점에 대해 s 재계산
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
    ).float()  # bool 을 float 로 미리 캐스팅 (jit 의 dtype 추론 안정)

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

    # off-target: 비타겟 드럼의 회피영역(XY 반경, 드럼면 위 Z 밴드)에 팁이 들어간 (팔, 드럼) 쌍 개수
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

    # 기계적 일률 |토크 x 각속도| [W]. 자세 유지는 각속도가 0 이라 비용 0, 손목은 토크가 작아 비용이 작다
    power_l1 = torch.sum(torch.abs(joint_torque * joint_vel), dim=-1)

    low_v = torch.clamp((joint_low + limit_margin) - robot_pos, min=0.0)
    high_v = torch.clamp(robot_pos - (joint_high - limit_margin), min=0.0)
    limit_pen = torch.sum((low_v * low_v + high_v * high_v) * limit_joint_mask, dim=-1)  # 손목은 마스크 0 (wrist_limit_pen 에서 전담)

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

# 클래스는 jit 함수를 호출해서 사용
@dataclass
class RewardComputerCfg:
    # ---------- 판정 파라미터 ----------
    limit_margin: float = 0.10          # 관절한계 제동 시작거리 [rad], 비손목 7관절
    wrist_limit_margin: float = 0.0     # 손목 전용 제동 시작거리 [rad]
    k_accuracy: float = 1.0             # 박자 정확도 exp 감쇠 (time_error = 정수 스텝수)
    k_time_to_hit: float = 3.0          # proximity 시간 감쇠 (타격 임박할수록 거리비용 증가)

    # ---------- 대기점 ----------
    # 대기점 보상: 준비 이동, 노는 팔, 자유 팔 상태에서 대기점으로 당긴다 (스트로크 중에는 끔).
    #   모양은 원뿔: w x max(0, 1 - 거리 / ready_cone_m). 대기점 바로 옆에서도 1cm 당 같은 양이 줄어 작은 흔들림에도 힘이 남는다
    #   거리 = 스틱 끝과 대기점 거리, 손목이 가동범위 가운데에서 벗어난 각도를 스틱 끝 거리(stick_len 곱)로 환산해 합친 것
    w_ready: float = 0.6                # 대기점 보상 시작 계수 (노는 팔 램프 시작 전)
    w_ready_max: float = 1.2            # 노는 팔 램프 완료 후 계수 (성공률 0.3 에서 0.6)
    ready_cone_m: float = 0.5           # 원뿔 반경 [m]. 이 거리에서 보상 0
    ready_height: float = 0.18          # 대기점 높이 [m], 드럼 위 (off_target 밴드 0.12 보다 위)
    stick_len: float = 0.373            # 손목 편차를 팁 호길이 [m] 로 환산 (위치 오차와 같은 단위)

    # 팔별 조준 오프셋 [m]: 각 팔이 드럼 중심에서 자기 어깨 쪽으로 비껴 조준
    aim_offset: float = 0.04

    # 노는 자리 x 제한: 노는 팔이 반대편으로 건너가 머무는 것을 막는다 (노트 위로 가는 시점은 env 의 prep_start_s)
    rest_margin_x: float = 0.02       # 노는 자리가 미드라인을 넘어도 되는 여유 [m]

    # 노는 팔 접근 보상 (prev_d - curr_d). 0 이면 미사용. 계산은 env 쪽에 있다
    w_idle_approach: float = 0.0

    # ---------- 팁 위치 ----------
    hit_xy_radius: float = 0.13         # 타격영역 반경 [m], hit_detector 와 동일값 유지
    hit_z_range: float = 0.07           # 타격영역 높이 [m], hit_detector 와 동일값 유지

    off_target_xy_radius: float = 0.13  # 비타겟 회피영역 반경 [m]
    off_target_z_range: float = 0.12    # 회피영역 높이 [m], hit 밴드(0.07)보다 넓게
    w_off_target: float = 0.3           # 비타겟 드럼 회피영역 침범
    w_crowd: float = 0.5                # 단일 타겟에 양팔 몰림 (ready 와 한 쌍)

    x_limit: float = 0.5                # 팁 작업공간 상자 [m]: |x| <= 0.5
    y_limit_l: float = 0.2              #                       0.2 <= y <= 0.8
    y_limit_h: float = 0.8
    z_limit: float = -0.6               #                       z >= -0.6
    drum_xy_margin: float = 0.15        # under_drum 판정 수평범위 [m]
    drum_z_margin: float = 0.05         # under_drum 판정 침투깊이 [m]

    # ---------- 목표 (타격 성공) ----------
    w_success: float = 5.5              # 타격 성공. 판정 창 안이면 이진 만점 (담당 팔이 친 경우만)
    w_wrong: float = 4.0                # 오타: 판정 창 밖 타격, 또는 담당 아닌 팔이 타겟을 친 경우
    w_miss: float = 3.0                 # 타겟 놓침
    w_time_accuracy: float = 3.0        # 창 안에서의 정확도 등급 exp(-|offset|)
    w_progress: float = 4.0             # 타겟에 가까워진 만큼 (prev_d - curr_d)
    w_proximity: float = 1.5            # 타겟까지 거리 (감점 형태지만 실질은 끌어당김)

    # ---------- phase-1 부트스트랩: 빠르게 내리치기 ----------
    # 상한 없는 단조증가 항. 세기 매칭(ramp_i)이 켜지는 만큼 크로스페이드로 소거된다.
    # 세기 게이트가 닫힌 초기 구간의 유일한 하강 신호이므로 제거하면 안 된다.
    w_impact_vel: float = 3.0
    impact_vel_max: float = 1.0         # 정규화 기준 [m/s]
    w_wrist_snap: float = 3.0
    wrist_snap_max: float = 12.0        # 정규화 기준 [rad/s]

    # ---------- 세기 매칭 ----------
    # 목표는 고정수가 아니라 타격간격 T 의 함수 (compute() 참고).
    # 아래 vel_* 6개와 k_imp_match 는 미사용 (T-종속 공식으로 대체됨)
    vel_imp_min: float = 0.3
    vel_imp_max: float = 1.5
    vel_wrist_min: float = 3.0
    vel_wrist_max: float = 15.0
    vel_elbow_min: float = 0.15
    vel_elbow_max: float = 0.40
    k_imp_match: float = 0.5
    k_wrist_match: float = 0.15     # 손목 각속도 오차 제곱 감쇠. 반값폭 약 2.15 rad/s
    k_elbow_match: float = 30.0     # 팔꿈치 리프트 오차 제곱 감쇠. 반값폭 0.15 rad
    w_wrist_match: float = 5.0      # 내부 배율: 손목 각속도 (임팩트 주력 항)
    w_elbow_match: float = 2.0      # 내부 배율: 팔꿈치 리프트 (형태 항)
    w_intensity_max: float = 2.0    # 세기 매칭 최대 가중치 (타격당 최대 14)
    w_intensity_success_start: float = 0.7   # 램프 B 시작 성공률(EMA). 타격을 먼저 익힌 뒤 세기
    w_intensity_success_full: float = 0.75   # 램프 B 완료 = impact_vel/wrist_snap 완전 소거

    # ---------- 에너지 ----------
    # action/joint_vel 은 정규화용 소량 항. 손목과 팔꿈치는 x0.001 로 사실상 면제 (스냅/굽힘 담당 관절)
    w_action: float = 0.0005            # 명령 크기 제곱합. clamp 밖으로 새는 것 방지
    w_wrist_action_scale: float = 0.001
    w_elbow_action_scale: float = 0.001
    w_joint_vel: float = 0.0001         # 실제 관절속도 제곱합. 명령 안 한 관성/중력 운동까지 포착
    w_wrist_vel_scale: float = 0.001
    w_elbow_vel_scale: float = 0.001

    # 기계적 일률 |토크 x 각속도|. 무거운 관절 급가속만 비싸고 자세 유지는 비용 0.
    w_power_floor: float = 0.0005        # 램프 시작 전 상시값
    w_power_max: float = 0.05           # 램프 A 최대치
    w_power_success_start: float = 0.6  # 램프 A 시작 성공률(EMA)
    w_power_success_full: float = 0.77  # 램프 A 완료
    w_power_ema_beta: float = 0.9       # 성공률 EMA 평활 (갱신 100스텝, 시정수 약 1000스텝)

    # 팔 상태 판정 (compute 맨 앞). 관측의 팔별 다음 노트까지 남은 시간으로 정한다
    prep_start_s: float = 0.46            # 자기 노트까지 이 시간 이하면 준비 이동 (대기점이 노트 위로 바뀜) [초]
    stroke_s: float = 0.25                # 자기 노트까지 이 시간 이하면 스트로크 (대기점으로 당기지 않음) [초]

    # 노는 팔 정지 항. 노는 팔과 자유 팔의 움직임
    idle_near_m: float = 0.1             # 대기점에서 이 거리 안이면 정지 벌점 완전히 켬 [m]
    idle_far_m: float = 0.20              # 대기점에서 이 거리 밖이면 정지 벌점 끔 [m]
    w_idle_still_max: float = 0.001       # 팔 항 제곱 부분, 램프 완료 후 계수
    w_idle_arm_l1_max: float = 0.3        # 팔 항 속도 크기 비례 부분 [1 rad/s 당 스텝마다], 관절별 합. 어깨 0.2 rad/s 흔들림 1초에 약 2.7
    w_idle_still_success_start: float = 0.3
    w_idle_still_success_full: float = 0.6
    idle_vel_sat: float = 9.0             # 팔 항 관절별 각속도 제곱 상한 (3 rad/s). 비례 부분은 이 제곱근에서 자른다
    # 손목 항. 비례 계수는 같은 스틱 끝 흔들림이면 어깨와 비슷한 벌이 되도록 팔 항 0.2 를 스틱 길이 비율로 맞춘 값
    w_idle_wrist_floor: float = 0.00005       # 제곱 부분 램프 시작 전 상시값
    w_idle_wrist_max: float = 0.04            # 제곱 부분 램프 완료 후 계수
    w_idle_wrist_l1_floor: float = 0.0003    # 비례 부분 램프 시작 전 상시값 (최대값 대비 비율은 제곱 부분과 동일)
    w_idle_wrist_l1_max: float = 0.24         # 비례 부분 램프 완료 후 계수. 손목 0.3 rad/s 흔들림 1초에 약 4.8
    idle_wrist_vel_sat: float = 100.0         # 손목 각속도 제곱 상한 (10 rad/s). 절댓값 항은 이 제곱근에서 자른다

    # ---------- 자세 ----------
    # 몸통 비틀림 = |sh1_L + sh1_R - 180deg|. 두 shoulder_1 은 같은 기준축이라 각각 90deg 면 그 팔이 몸통 정면,
    w_waist_twist_max: float = 2.0        # 게이팅 없음. 타격 순간에만 적용
    waist_twist_deadzone: float = 0.175   # 10deg 이내 허용 (팔이 안쪽으로 뻗는 건 정상)
    waist_twist_scale: float = 0.35       # 초과분 20deg 에서 만점
    w_waist_max: float = 0.0              # 허리 속도 페널티. 0 이면 미적용

    # ---------- 한계, 충돌 ----------
    w_limit: float = 1.0                  # 관절한계 (비손목 7관절, margin 0.10)
    w_wrist_limit: float = 1.5            # 손목 전담. limit_pen 은 마스크로 손목 제외
    w_tip_limit: float = 0.15             # 작업공간 상자 이탈 (양팔 OR 이진)
    w_under_drum: float = 0.1             # 드럼면 아래 침투
    w_self_collision: float = 0.5         # 전완끼리 근접
    self_collision_threshold: float = 0.10  # 전완(elbow에서 wrist) 선분거리 임계 [m]

    # 스틱 접촉 페널티. 임계는 스틱 두께 수준의 진짜 접촉만 벌한다.
    # 임계를 근접 수준(5cm)으로 키우면 같은 드럼 교대연타에 필수인 근접까지 벌해서 교차 자세를 유도하므로 금지.
    w_stick_contact_max: float = 4.0        # 램프 B 최대치
    stick_contact_threshold: float = 0.025  # 접촉 임계 [m] = 스틱 두께

    # 접근 밴드: 접촉 전에 밀어내는 항. 접촉항은 임계 밖에서 gradient 가 0 이라 접근을 못 막는다.
    # 하강(타격) 구간은 strike_phase 게이트로 면제한다.
    stick_gap_range: float = 0.08     # 이 거리부터 선형 증가 [m]
    w_stick_gap_max: float = 3.0      # 램프 B 최대치
    # 하강(타격) 면제 조건: 이 반경 안에서 아래 속도 이상으로 하강 중일 때만
    gap_exempt_radius: float = 0.08
    gap_exempt_vz: float = 1.0

    elbow_min_bend: float = 0.35        # 최소 굽힘 [rad] (약 20deg). 팔 쫙 펴고 치는 자세 방지 (0=편, 140deg=굽힘)
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

        # 관절 속도 페널티 가중치 (손목, 팔꿈치 완화). ctrl_joint_names 는 resolve 된 순서
        def _vel_scale(name: str) -> float:
            if "wrist" in name: return self.cfg.w_wrist_vel_scale
            if "elbow" in name: return self.cfg.w_elbow_vel_scale
            return 1.0
        vel_w = [_vel_scale(name) for name in ctrl_joint_names]
        self.joint_vel_weight = torch.tensor(
            vel_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # action 페널티 가중치 (손목, 팔꿈치 완화)
        def _act_scale(name: str) -> float:
            if "wrist" in name: return self.cfg.w_wrist_action_scale
            if "elbow" in name: return self.cfg.w_elbow_action_scale
            return 1.0
        act_w = [_act_scale(name) for name in ctrl_joint_names]
        self.action_weight = torch.tensor(
            act_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # 손목 관절 인덱스. hit_per_arm 순서(0=왼, 1=오)와 맞춤
        name_to_idx = {name: i for i, name in enumerate(ctrl_joint_names)}
        self.wrist_ids = torch.tensor(
            [name_to_idx["left_wrist"], name_to_idx["right_wrist"]],
            device=self.device, dtype=torch.long,
        )

        # 팔별 관절 인덱스 [왼4, 오4] = 어깨1, 어깨2, 팔꿈치, 손목. 노는 팔 일률 페널티용 (허리는 공용이라 제외)
        self.arm_joint_ids = torch.tensor([
            [name_to_idx["left_shoulder_1"],  name_to_idx["left_shoulder_2"],
             name_to_idx["left_elbow"],       name_to_idx["left_wrist"]],
            [name_to_idx["right_shoulder_1"], name_to_idx["right_shoulder_2"],
             name_to_idx["right_elbow"],      name_to_idx["right_wrist"]],
        ], device=self.device, dtype=torch.long)   # (2, 4)

        # 일반 limit_pen 에서 손목 제외용 마스크 (비손목=1, 손목=0). 손목은 wrist_limit_pen 에서 전담
        self.limit_joint_mask = torch.ones((1, len(ctrl_joint_names)), device=self.device)
        self.limit_joint_mask[0, self.wrist_ids] = 0.0

        # 팔꿈치 관절 인덱스 [왼, 오]
        self.elbow_ids = torch.tensor(
            [name_to_idx["left_elbow"], name_to_idx["right_elbow"]],
            device=self.device, dtype=torch.long,
        )

        # 어깨1(방위축) 인덱스 [왼, 오]. 몸통 비틀림 계산용
        self.sh1_ids = torch.tensor(
            [name_to_idx["left_shoulder_1"], name_to_idx["right_shoulder_1"]],
            device=self.device, dtype=torch.long,
        )

        # 허리/어깨 관절 인덱스
        self.waist_ids = torch.tensor(
            [i for i, n in enumerate(ctrl_joint_names) if "waist" in n],
            device=self.device, dtype=torch.long,
        )
        self.shoulder_ids = torch.tensor(
            [i for i, n in enumerate(ctrl_joint_names) if "shoulder" in n],
            device=self.device, dtype=torch.long,
        )

        # 게이팅 실효 가중치. 전부 0 에서 시작해 success_ema 에 따라 램프인 (갱신은 update_difficulty_weights)
        self.w_power_eff = self.cfg.w_power_floor   # 램프 A: floor 에서 시작, 성공률 0.60 에서 0.77 사이에 max 로
        self.w_waist_eff = 0.0         # 램프 A (w_waist_max=0 이면 항상 0)
        self.w_intensity_eff = 0.0     # 램프 B (성공률 0.70 에서 0.75)
        self.w_stick_eff = 0.0         # 램프 B
        self.w_stick_gap_eff = 0.0     # 램프 B (접근 밴드)
        self.w_idle_still_eff = 0.0    # 노는 팔 정지 항 (성공률 0.3 에서 0.6)
        self.w_idle_wrist_eff = self.cfg.w_idle_wrist_floor   # 노는 팔 손목 제곱 항: floor 에서 시작, 같은 램프로 max 까지
        self.w_idle_wrist_l1_eff = self.cfg.w_idle_wrist_l1_floor   # 노는 팔 손목 절댓값 항: 위와 같음
        self.w_idle_arm_l1_eff = 0.0   # 노는 팔 팔 항 속도 비례 부분 (같은 램프)
        self.w_ready_eff = self.cfg.w_ready   # 대기점 보상 계수: w_ready 에서 시작, 같은 램프로 w_ready_max 까지
        # 이 둘은 반대 방향: 램프 B 가 켜지는 만큼 소거 (크로스페이드)
        self.w_impact_vel_eff = self.cfg.w_impact_vel
        self.w_wrist_snap_eff = self.cfg.w_wrist_snap
        # 체크포인트에 저장되지 않으므로 resume 시 0 에서 다시 올라간다
        self.success_ema = 0.0
        # 램프 래치: 한 번 올라간 램프는 성공률이 떨어져도 내려가지 않는다 (보상 함수가 성공률을 따라 흔들리지 않게)
        self.ramp_a_latch = 0.0   # 일률, 허리
        self.ramp_b_latch = 0.0   # 세기, 스틱, 부트스트랩 크로스페이드
        self.ramp_s_latch = 0.0   # 노는 팔 정지 항

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
            target_velocity: torch.Tensor,  # (N, M) 마감 타겟의 목표 MIDI velocity (0~127)
            success_elbow_apex: torch.Tensor,  # (N, M) 성공 타격의 팔꿈치 리프트 apex [rad]
            target_T: torch.Tensor,  # (N, M) 마감 타겟의 간격 T [s]
            assigned_use_left: torch.Tensor,  # (N,) bool 스케줄러가 정한 h0 담당 (왼팔=True)
            assigned_use_case1: torch.Tensor, # (N,) bool 동시타 배정 (왼팔이 idx0 담당이면 True)
            idle_tgt_pos: torch.Tensor,   # (N,2,3) 스케줄러가 각 팔에 준 첫 노트 (per_arm_pos[:,:,0,:])
            idle_tgt_valid: torch.Tensor, # (N,2) bool 그 노트 유효 (per_arm_valid[:,:,0])
            idle_tgt_time: torch.Tensor,  # (N,2) 그 노트까지 정규화 시간 (per_arm_time[:,:,0], 관측의 팔별 노트 칸과 같은 값)
    ) -> tuple[torch.Tensor, dict, dict, torch.Tensor, torch.Tensor]:  # reward, log, rate_log, active_arm(N,2), arm_state(N,2)
        # goal
        success_reward, wrong_cost, missed_cost, time_accuracy_reward = _goal_terms(
            success=success,
            wrong_hit=wrong_hit,
            missed_target=missed_target,
            time_error=time_error,
            w_drum_success=self.w_drum_success,
            k_accuracy=self.cfg.k_accuracy,
        )

        # 팔 상태 판정 (N,2). 관측에 이미 있는 값으로만 정한다: 팔별 다음 노트 유효 여부와 남은 시간(팔별 노트 칸),
        #   스틱 끝 위치와 드럼 위치, 스틱 끝 속도(관측의 관절 각도와 각속도로 정해짐).
        #   남은 시간은 직전 스텝 관측 값이라 한 스텝 늦지만, 정책이 실제로 본 값과 같게 하려고 그대로 쓴다
        t_left = idle_tgt_time * float(self.env.max_lookahead_step - 1) * self.env.step_dt          # (N,2) [초]
        d2_td = ((tip_pos[:, :, None, 0:2] - drum_pos[:, None, :, 0:2]) ** 2).sum(dim=-1)          # (N,2,M)
        #   내려치는 중: 드럼 근처에서 빠르게 하강 (스틱 간격 면제와 같은 기준). 늦게 치는 팔도 이 조건에 걸린다
        striking = ((d2_td.min(dim=-1).values < self.cfg.gap_exempt_radius ** 2)
                    & (tip_vel[:, :, 2] < -self.cfg.gap_exempt_vz))                                 # (N,2)
        arm_state = torch.full_like(idle_tgt_time, ARM_FREE, dtype=torch.long)
        arm_state = torch.where(idle_tgt_valid, torch.full_like(arm_state, ARM_IDLE), arm_state)
        arm_state = torch.where(idle_tgt_valid & (t_left <= self.cfg.prep_start_s),
                                torch.full_like(arm_state, ARM_PREP), arm_state)
        arm_state = torch.where((idle_tgt_valid & (t_left <= self.cfg.stroke_s)) | striking,
                                torch.full_like(arm_state, ARM_STROKE), arm_state)
        pull_note = arm_state == ARM_PREP                                          # (N,2) 대기점 = 노트 위
        work = (arm_state == ARM_STROKE) | (arm_state == ARM_PREP)                 # (N,2) 다음 노트를 향해 일하는 팔

        # proximity / progress / arm assignment / crowd
        proximity_cost, progress_reward, crowd_pen, active_arm, home_pos = _proximity_terms(
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
            aim_offset=self.cfg.aim_offset,
            assigned_use_left=assigned_use_left,
            assigned_use_case1=assigned_use_case1,
            idle_tgt_pos=idle_tgt_pos,
            idle_tgt_valid=idle_tgt_valid,
            pull_note=pull_note,
            rest_margin_x=self.cfg.rest_margin_x,
        )

        # 대기점 보상 (원뿔): 스트로크가 아닌 팔(준비 이동, 노는 팔, 자유 팔)마다 max(0, 1 - 거리 / ready_cone_m) 를 더한다. 계수는 w_ready_eff.
        #   거리에는 손목이 가동범위 중앙에서 벗어난 각도를 스틱 끝 호길이로 환산해 합친다. 중앙이어야 양방향 스냅 여유가 생긴다.
        #   자유 팔은 노는 팔 램프(ramp_s_latch)만큼만 당긴다. 처음부터 당기면 타격을 배우기 전에 대기점에 떠 있는 쪽으로 굳는다
        home_dist = torch.linalg.norm(tip_pos - home_pos, dim=-1)            # (N,2) 대기점까지 거리 [m]
        w_ang = robot_pos[:, self.wrist_ids]                                  # (N,2) 손목각
        w_lo  = joint_low[:,  self.wrist_ids]                                 # (1,2)
        w_hi  = joint_high[:, self.wrist_ids]
        w_rng = (w_hi - w_lo).clamp(min=1e-6)
        w_frac = (w_ang - w_lo) / w_rng                                       # (N,2) 0=하한, 1=상한
        w_dev = w_ang - 0.5 * (w_lo + w_hi)                                   # (N,2) 중앙 대비 편차
        d_wrist = self.cfg.stick_len * w_dev                                  # (N,2) 호길이 [m]
        ready_d = torch.sqrt(home_dist * home_dist + d_wrist * d_wrist)       # (N,2) 대기 자세까지 거리 [m]
        ready_arm = (1.0 - ready_d / self.cfg.ready_cone_m).clamp(min=0.0)    # (N,2) 원뿔 0~1
        pull_on = (((arm_state == ARM_PREP) | (arm_state == ARM_IDLE)).float()
                   + (arm_state == ARM_FREE).float() * self.ramp_s_latch)     # (N,2) 대기점으로 당기는 정도
        ready_pose_reward = (pull_on * ready_arm).sum(dim=-1)                 # (N,) 당기는 상태의 팔마다 받는다

        # 노는 팔 스위치 (N,2, 0~1): 정지 벌점과 노는 팔 로그가 쓴다. 준비 이동과 스트로크는 0
        #   노는 팔과 자유 팔: 대기점 idle_near_m 안이면 1, idle_far_m 밖이면 0, 사이는 거리 비례 (옮겨 가는 이동은 과금하지 않음)
        near = ((self.cfg.idle_far_m - home_dist)
                / max(self.cfg.idle_far_m - self.cfg.idle_near_m, 1e-6)).clamp(0.0, 1.0)              # (N,2)
        resting_state = (arm_state == ARM_IDLE) | (arm_state == ARM_FREE)                              # (N,2)
        idle_gate = torch.where(resting_state, near, torch.zeros_like(near))                          # (N,2)
        n_idle = idle_gate.sum(dim=-1)                                        # (N,)
        idle_wrist_frac = torch.where(n_idle > 0.0, (idle_gate * w_frac).sum(dim=-1) / n_idle.clamp(min=1e-6),
                                      torch.full_like(n_idle, 0.5))           # (N,) 로그용, 노는 팔 평균

        # off-target 페널티에서 제외할 현재 타겟 마스크 (h0)
        M = drum_pos.shape[1]
        target_mask = next_hits[:, 0, :M] > 0.5

        # 다음 노트로 다가가게 하는 두 항(거리 비용, 다가간 만큼 보상)은 그 노트 담당 팔이 준비 이동이나 스트로크일 때만 켠다.
        #   노트가 멀어 담당 팔이 노는 팔 상태인데 다가가라고 하면, 노는 자리에 있으라는 대기점 보상, 정지 벌점과 반대가 된다.
        #   동시 2타는 두 팔 중 하나라도 일하는 상태면 켠다
        two_mask = target_mask.float().sum(dim=-1) >= 2.0                     # (N,)
        owner_work = torch.where(two_mask, work.any(dim=1),
                                 torch.where(assigned_use_left, work[:, 0], work[:, 1])).float()   # (N,)
        proximity_cost = proximity_cost * owner_work
        progress_reward = progress_reward * owner_work

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

        # self collision penalty (전완 선분: elbow 에서 wrist)
        self_col_pen, forearm_dist, _, _ = _self_collision_penalty(
            elbow_l=elbow_pos[:, 0, :],
            wrist_l=wrist_pos[:, 0, :],
            elbow_r=elbow_pos[:, 1, :],
            wrist_r=wrist_pos[:, 1, :],
            threshold=self.cfg.self_collision_threshold,
        )

        # 스틱(wrist 에서 tip) 선분거리. threshold=0 이라 거리만 측정, 페널티는 아래 접촉항에서 별도 계산
        _, stick_dist, _, _ = _self_collision_penalty(
            elbow_l=wrist_pos[:, 0, :],   # 왼 스틱 시작 = 왼 wrist
            wrist_l=tip_pos[:, 0, :],     # 왼 스틱 끝 = 왼 tip
            elbow_r=wrist_pos[:, 1, :],   # 오 스틱 시작 = 오 wrist
            wrist_r=tip_pos[:, 1, :],     # 오 스틱 끝 = 오 tip
            threshold=0.0,                 # 페널티 0, 거리만 사용
        )

        # phase-1 부트스트랩: 성공 타격의 하강속도/손목 각속도 자체에 보상 (상한 없음). 램프 B 에 반비례 소거
        impact_vel_reward = (success_impact_vel / (self.cfg.impact_vel_max + 1e-8)).sum(dim=1)  # (N,)
        wrist_snap_reward = (success_wrist_ang_vel / (self.cfg.wrist_snap_max + 1e-8)).sum(dim=1)  # (N,)

        # 세기 매칭: 목표는 세기(i)와 타격간격(T)의 함수. 실기 state_motion_generator.cpp 와 같은 공식
        d2r = 3.141592653589793 / 180.0
        u = target_velocity / 127.0                                          # (N, M) 0~1
        i = 1.0 + 6.0 * u                                                    # MIDI velocity 를 실기 intensity 1~7 로
        T = target_T.clamp(0.1, 1.0)                                         # (N, M) 간격 [s]

        # 손목 각속도 목표 = 2 * L(T) * f_w(i) / T_hit  (스트로크 왕복거리 / 허용시간, 주력 항)
        f_w = torch.where(i < 5.0, 0.25 * i - 0.25, i / 3.0 - 2.0 / 3.0)
        L   = torch.where(T < 0.5, (40.0 * T + 10.0) * d2r, torch.full_like(T, 30.0 * d2r))
        T_hit = (0.3 * T).clamp(max=0.2)
        w_wrist_tgt = (2.0 * L * f_w / T_hit).clamp(min=0.0, max=30.0)       # (N, M) rad/s

        # 팔꿈치 리프트 목표 = lift_mag(T) * f_e(i)  (형태 항). i <= 3 이면 0 = 여린 타격은 팔을 들지 않음
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

        # 허리 속도 페널티 (w_waist_max=0 이면 미적용)
        waist_vel = robot_vel[:, self.waist_ids]                          # (N, n_waist)
        waist_vel_pen = torch.sum(waist_vel * waist_vel, dim=-1)          # (N,)

        # 팔꿈치 최소 굽힘: min_bend 미만이면 부족분만큼 벌점 (양팔 합)
        elbow_angle_now = robot_pos[:, self.elbow_ids]                    # (N, 2) 0=편, 140deg=굽힘
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
            w_ready=self.w_ready_eff,
        )

        # 이하 jit 밖에서 적용 (compute_rewards 시그니처 고정)
        reward = reward - self.cfg.w_elbow_bend * elbow_bend_pen

        # 손목 한계: 명령은 clip_and_convert 가 이미 자르므로, 이 항은 한계에 붙어 있는 자세를 벌한다
        w_pos = robot_pos[:, self.wrist_ids]                        # (N, 2)
        w_lo  = joint_low[:,  self.wrist_ids]                       # (1, 2)
        w_hi  = joint_high[:, self.wrist_ids]
        wm = self.cfg.wrist_limit_margin
        lo_v = torch.clamp((w_lo + wm) - w_pos, min=0.0)
        hi_v = torch.clamp(w_pos - (w_hi - wm), min=0.0)
        wrist_limit_pen = torch.sum(lo_v * lo_v + hi_v * hi_v, dim=-1)   # (N,)
        reward = reward - self.cfg.w_wrist_limit * wrist_limit_pen

        # 스틱 접촉: 임계(스틱 두께) 미만의 진짜 접촉만
        stick_contact_pen = torch.clamp(self.cfg.stick_contact_threshold - stick_dist, min=0.0)
        reward = reward - self.w_stick_eff * stick_contact_pen

        # 접근 밴드: 하강(타격) 구간은 면제하고 이동/대기 중에만 stick_gap_range 부터 밀어냄.
        #   strike_phase = 어느 팁이든 드럼 근처에서 일정 속도 이상 하강 중. 접촉항은 상시 유지
        tip_xy = tip_pos[:, :, None, 0:2]                                    # (N,2,1,2)
        drum_xy = drum_pos[:, None, :, 0:2]                                  # (N,1,M,2)
        d2_td = ((tip_xy - drum_xy) ** 2).sum(dim=-1)                        # (N,2,M)
        near_drum = d2_td.min(dim=-1).values < (self.cfg.gap_exempt_radius ** 2)   # (N,2)
        descending = tip_vel[:, :, 2] < -self.cfg.gap_exempt_vz                    # (N,2)
        strike_phase = (near_drum & descending).any(dim=1)                   # (N,)
        gap_gate = 1.0 - strike_phase.float()
        stick_gap_pen = torch.clamp(self.cfg.stick_gap_range - stick_dist, min=0.0) * gap_gate
        reward = reward - self.w_stick_gap_eff * stick_gap_pen

        # [진단] 노는 팔(맡은 노트 있음)의 대기점이 자기 반대편인지 (대기점이 크로스를 명령하는지 측정). 분모 = 노는 팔 수
        #   대기점 = 위 home_pos (보상이 실제로 당기는 위치). 왼팔(0)은 x > 0, 오른팔(1)은 x < 0 이면 반대편
        _hx = home_pos[:, :, 0]                                                    # (N,2)
        _crossed = torch.stack([_hx[:, 0] > 0.0, _hx[:, 1] < 0.0], dim=1).float()  # (N,2)
        _idle_note = (arm_state == ARM_IDLE).float()                               # (N,2)

        # 몸통 비틀림: 타격 순간 |sh1_L + sh1_R - 180deg|. 합 180deg 가 두 팔의 좌우대칭.
        # 타격 순간에만 적용하므로 안 치는 스텝은 0
        sh1_sum = robot_pos[:, self.sh1_ids].sum(dim=-1)                        # (N,) sh1_L + sh1_R [rad]
        twist = (sh1_sum - 3.141592653589793).abs()                             # (N,) |합 - 180deg|
        twist_ex = torch.clamp(twist - self.cfg.waist_twist_deadzone, min=0.0)  # deadzone 초과분
        twist_n = (twist_ex / self.cfg.waist_twist_scale).clamp(max=1.0)        # 0~1 정규화, 포화
        any_hit_f = hit_per_arm.any(dim=2).any(dim=1).float()                   # (N,) 타격 순간만
        waist_twist_pen = twist_n * twist_n * any_hit_f                         # (N,)
        reward = reward - self.cfg.w_waist_twist_max * waist_twist_pen

        # 노는 팔 정지 항: 대기점에 도착한 노는 팔과 드럼 구역 밖의 자유 팔은 움직일 이유가 없으므로, 움직이는 만큼 낸다.
        #   팔 항: 어깨1, 어깨2, 팔꿈치. 제곱 부분 = 관절별 idle_vel_sat 에서 자른 각속도 제곱의 평균,
        #          비례 부분 = 관절별 각속도 크기(3 rad/s 에서 자름)의 합
        #   손목 항: 제곱 부분과 비례 부분. 둘 다 idle_wrist_vel_sat 기준으로 자른다
        #   관절 각속도는 몸통 기준이라 허리 회전은 들어가지 않는다
        arm_v = robot_vel[:, self.arm_joint_ids]                                                       # (N,2,4) 어깨1, 어깨2, 팔꿈치, 손목
        v2_arm = (arm_v[..., :3] ** 2).clamp(max=self.cfg.idle_vel_sat).mean(dim=-1)                   # (N,2)
        v1_arm = arm_v[..., :3].abs().clamp(max=self.cfg.idle_vel_sat ** 0.5).sum(dim=-1)              # (N,2)
        v2_wrist = (arm_v[..., 3] ** 2).clamp(max=self.cfg.idle_wrist_vel_sat)                         # (N,2)
        v1_wrist = arm_v[..., 3].abs().clamp(max=self.cfg.idle_wrist_vel_sat ** 0.5)                   # (N,2)
        #   과금 여부는 위의 노는 팔 스위치 idle_gate 를 그대로 쓴다
        idle_still = (idle_gate * v2_arm).sum(dim=-1)                                                  # (N,)
        idle_arm_l1 = (idle_gate * v1_arm).sum(dim=-1)                                                 # (N,)
        idle_wrist = (idle_gate * v2_wrist).sum(dim=-1)                                                # (N,)
        idle_wrist_l1 = (idle_gate * v1_wrist).sum(dim=-1)                                             # (N,)
        reward = (reward - self.w_idle_still_eff * idle_still - self.w_idle_arm_l1_eff * idle_arm_l1
                  - self.w_idle_wrist_eff * idle_wrist - self.w_idle_wrist_l1_eff * idle_wrist_l1)

        num_hit_drum = success.float() + missed_target.float()

        num_hit = num_hit_drum.float().sum(dim=-1)
        num_success = success.float().sum(dim=-1)
        num_wrong = wrong_hit.float().sum(dim=-1)
        num_missed = missed_target.float().sum(dim=-1)

        log_terms = {
            "reward": reward,
            "proximity": proximity_cost,
            "progress(x100)": progress_reward * 100,
            "power_l1": power_l1,
            "w_power_eff(x1000)": torch.full_like(reward, self.w_power_eff * 1000.0),
            "ready_pose": ready_pose_reward,
            "idle_wrist_frac": idle_wrist_frac,        # 쉬는 팔 손목 위치 (0=하한, 0.5=중앙, 1=상한)
            "wrist_limit_pen(x100)": wrist_limit_pen * 100,
            "waist_twist_pen": waist_twist_pen,
            "idle_still": idle_still,                  # 노는 팔 정지 팔 항 제곱 부분 원값 (x w_idle_still_eff 이 차감량)
            "w_idle_still_eff(x1000)": torch.full_like(reward, self.w_idle_still_eff * 1000.0),
            "idle_arm_l1": idle_arm_l1,                # 노는 팔 정지 팔 항 비례 부분 원값 (x w_idle_arm_l1_eff 이 차감량)
            "w_idle_arm_l1_eff(x1000)": torch.full_like(reward, self.w_idle_arm_l1_eff * 1000.0),
            "w_ready_eff(x1000)": torch.full_like(reward, self.w_ready_eff * 1000.0),
            "idle_wrist": idle_wrist,                  # 노는 팔 손목 제곱 항 원값 (x w_idle_wrist_eff 이 차감량)
            "w_idle_wrist_eff(x1000)": torch.full_like(reward, self.w_idle_wrist_eff * 1000.0),
            "idle_wrist_l1": idle_wrist_l1,            # 노는 팔 손목 절댓값 항 원값 (x w_idle_wrist_l1_eff 이 차감량)
            "w_idle_wrist_l1_eff(x1000)": torch.full_like(reward, self.w_idle_wrist_l1_eff * 1000.0),
            "idle_gate": idle_gate.mean(dim=-1),       # 과금 게이트 평균 (1=양팔 모두 과금 구간)
            "crowd_pen": crowd_pen,
            "self_collision_pen": self_col_pen,
            "forearm_dist": forearm_dist,
            "stick_dist": stick_dist,
            "stick_gap_pen(x100)": stick_gap_pen * 100,
            "intensity_reward": intensity_reward,
            "w_intensity_eff": torch.full_like(reward, self.w_intensity_eff),
            "elbow_bend_pen": elbow_bend_pen,      # 최소 굽힘 부족분 (양팔 합)
            "off_target_pen": off_target_pen,      # 비타겟 드럼 회피영역 침범
            "tip_limit_pen":  tip_limit_pen,       # 작업공간 상자 이탈
            "under_drum_pen": under_drum_pen,      # 드럼면 아래 침투
        }

        # 실제 타격 순간만 뽑아 평균 (hit 이 난 팔만)


        rate_log_terms = {
            "success_rate": torch.stack([num_success, num_hit], dim=-1),
            "wrong_rate": torch.stack([num_wrong, num_hit], dim=-1),
            "miss_rate": torch.stack([num_missed, num_hit], dim=-1),
            # 스틱 접촉률: 임계 미만인 스텝 비율
            "stick_contact": torch.stack([(stick_dist < self.cfg.stick_contact_threshold).float(), torch.ones_like(stick_dist)], dim=-1),
            # 쉬는 팔 대기점이 반대편인 비율
            "home_crossed": torch.stack([(_crossed * _idle_note).sum(dim=-1), _idle_note.sum(dim=-1)], dim=-1),
            # 스틱 거리 분포 꼬리
            "stick_lt_60mm": torch.stack([(stick_dist < 0.060).float(), torch.ones_like(stick_dist)], dim=-1),
            # 하강 구간 면제 비율
            "strike_phase": torch.stack([strike_phase.float(), torch.ones_like(stick_dist)], dim=-1),
            # 타격 순간 손목이 가동범위 어디인가 (0 이나 1 에 붙으면 스냅이 한계에서 끝남)
            "hit_wrist_frac": torch.stack([(w_frac * hit_per_arm.any(dim=2).float()).sum(dim=1),
                                           hit_per_arm.any(dim=2).float().sum(dim=1)], dim=-1),
        }

        for i, name in enumerate(self.drum_names):
            rate_log_terms[f"{name}_success_rate"] = torch.stack([success[:, i], num_hit_drum[:, i]], dim=-1)

        # 세기 진단: 성공 타격을 목표 velocity 구간(p, mf, f)으로 나눠 팔꿈치 리프트 기록 + 목표 매칭 정확도(0~1)
        succ_f = success.float()
        match = 0.5 * (r_wrist + r_elbow)                                      # (N, M) 1=목표 정확 매칭 (손목 + 팔꿈치)
        rate_log_terms["intensity_match"] = torch.stack([(match * succ_f).sum(dim=-1), succ_f.sum(dim=-1)], dim=-1)
        for lo, hi, tag in ((0.5, 48.5, "p"), (48.5, 90.5, "mf"), (90.5, 1e9, "f")):
            bm = (succ_f * ((target_velocity > lo) & (target_velocity <= hi)).float())  # (N, M) 그 세기구간 성공
            cnt = bm.sum(dim=-1)
            rate_log_terms[f"dyn_elbow_{tag}"]  = torch.stack([(success_elbow_apex * bm).sum(dim=-1), cnt], dim=-1)

        # 가만히 있어야 할 팔 진단. 노는 팔 스위치가 완전히 켜진 팔이고, 이번 스텝에 스스로 친 팔은 뺀다.
        #   그래서 이번 스텝 타격이 있으면 다른 팔이 친 것이다
        #   손목 액션 0.1 은 정상상태에서 손목 약 1 rad/s 에 해당한다. 속도에 비해 액션이 작으면 몸 흔들림으로 생긴 움직임
        hit_self = hit_per_arm.any(dim=2)                                      # (N,2) 이번 스텝 그 팔이 침
        resting = ((idle_gate >= 0.999) & (~hit_self)).float()                 # (N,2) 1=가만히 있어야 할 팔
        notgt = resting * (~idle_tgt_valid).float()                            # (N,2) 그중 맡은 노트가 아예 없는 팔
        w_abs = robot_vel[:, self.wrist_ids].abs()                             # (N,2) 손목 각속도 크기 [rad/s]
        w_act = actions[:, self.wrist_ids].abs()                               # (N,2) 손목 액션 크기 [0~1]
        vz_abs = tip_vel[:, :, 2].abs()                                        # (N,2) 스틱 끝 수직 속도 크기 [m/s]
        hit_now = hit_per_arm.any(dim=2).any(dim=1).float().unsqueeze(1)       # (N,1) 이번 스텝 타격 발생
        rest_hit = resting * hit_now                                           # (N,2) 다른 팔이 친 순간의 가만히 있어야 할 팔

        def _pair(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:      # (N,2) 두 팔 합을 (분자, 분모) 로
            return torch.stack([num.sum(dim=1), den.sum(dim=1)], dim=-1)

        rate_log_terms["rest_frac"]             = _pair(resting, torch.ones_like(resting))
        rate_log_terms["rest_wrist"]            = _pair(resting * w_abs, resting)
        rate_log_terms["rest_wrist_act"]        = _pair(resting * w_act, resting)
        rate_log_terms["rest_wrist_at_hit"]     = _pair(rest_hit * w_abs, rest_hit)
        rate_log_terms["rest_vz_at_hit"]        = _pair(rest_hit * vz_abs, rest_hit)
        rate_log_terms["rest_wrist_act_at_hit"] = _pair(rest_hit * w_act, rest_hit)
        rate_log_terms["rest_wrist_notgt"]      = _pair(notgt * w_abs, notgt)

        # 팔 상태 비율 (분모 = 팔 수 x 스텝). 상태 판단이 의도대로 나오는지 확인용
        _ones2 = torch.ones_like(idle_gate)
        for _tag, _code in (("stroke", ARM_STROKE), ("prep", ARM_PREP), ("idle", ARM_IDLE), ("free", ARM_FREE)):
            rate_log_terms[f"state_{_tag}"] = _pair((arm_state == _code).float(), _ones2)

        return reward, log_terms, rate_log_terms, active_arm, arm_state

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

            # 전체 성공률 EMA 로 일률 페널티 게이팅 (램프 A)
            overall_succ = (self.n_success.sum() / (self.n_hit.sum() + eps)).item()
            beta = self.cfg.w_power_ema_beta
            self.success_ema = beta * self.success_ema + (1.0 - beta) * overall_succ
            denom = max(self.cfg.w_power_success_full - self.cfg.w_power_success_start, 1e-6)
            ramp = (self.success_ema - self.cfg.w_power_success_start) / denom
            ramp = min(max(ramp, 0.0), 1.0)
            self.ramp_a_latch = max(self.ramp_a_latch, ramp)
            self.w_power_eff = self.cfg.w_power_floor + (self.cfg.w_power_max - self.cfg.w_power_floor) * self.ramp_a_latch
            self.w_waist_eff = self.cfg.w_waist_max * self.ramp_a_latch   # 허리 페널티도 동일 게이팅

            # 세기 매칭은 별도 램프 (램프 B). 타격을 먼저 익힌 뒤 세기
            denom_i = max(self.cfg.w_intensity_success_full - self.cfg.w_intensity_success_start, 1e-6)
            ramp_i = (self.success_ema - self.cfg.w_intensity_success_start) / denom_i
            ramp_i = min(max(ramp_i, 0.0), 1.0)
            self.ramp_b_latch = max(self.ramp_b_latch, ramp_i)
            self.w_intensity_eff = self.cfg.w_intensity_max * self.ramp_b_latch
            # 스틱 페널티도 같은 램프 (스킬이 굳은 뒤에만 켜서 초기 탐색에 영향 없음)
            self.w_stick_eff = self.cfg.w_stick_contact_max * self.ramp_b_latch
            self.w_stick_gap_eff = self.cfg.w_stick_gap_max * self.ramp_b_latch
            # 노는 팔 정지 항: 타격을 찾은 뒤(0.3)부터 켜서 0.6 에서 최대
            denom_s = max(self.cfg.w_idle_still_success_full - self.cfg.w_idle_still_success_start, 1e-6)
            ramp_s = (self.success_ema - self.cfg.w_idle_still_success_start) / denom_s
            self.ramp_s_latch = max(self.ramp_s_latch, min(max(ramp_s, 0.0), 1.0))
            self.w_idle_still_eff = self.cfg.w_idle_still_max * self.ramp_s_latch
            self.w_idle_arm_l1_eff = self.cfg.w_idle_arm_l1_max * self.ramp_s_latch
            self.w_ready_eff = self.cfg.w_ready + (self.cfg.w_ready_max - self.cfg.w_ready) * self.ramp_s_latch
            self.w_idle_wrist_eff = self.cfg.w_idle_wrist_floor + (self.cfg.w_idle_wrist_max - self.cfg.w_idle_wrist_floor) * self.ramp_s_latch
            self.w_idle_wrist_l1_eff = (self.cfg.w_idle_wrist_l1_floor
                                        + (self.cfg.w_idle_wrist_l1_max - self.cfg.w_idle_wrist_l1_floor) * self.ramp_s_latch)
            # 크로스페이드: 세기 매칭이 켜지는 만큼 부트스트랩 항을 페이드아웃
            self.w_impact_vel_eff = self.cfg.w_impact_vel * (1.0 - self.ramp_b_latch)
            self.w_wrist_snap_eff = self.cfg.w_wrist_snap * (1.0 - self.ramp_b_latch)

            # 통계 초기화
            self.n_success.zero_()
            self.n_hit.zero_()
