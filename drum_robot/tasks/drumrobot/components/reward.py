"""
보상 클래스
"""

from __future__ import annotations
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]
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
        second_target_mask: torch.Tensor, # (N, M)  다음 타격 h1 (노는 팔 배정용)
        drum_pos: torch.Tensor,     # (N, M, 3)
        left_tip_pos: torch.Tensor, # (N, 3)
        right_tip_pos: torch.Tensor,# (N, 3)
        tip_vel: torch.Tensor,      # (N, 2, 3)
        hit_armed: torch.Tensor,    # (N, 2, M)
        k_align: float,
        w_free_arm: float,          # 노는 팔의 h1 이동 가중치
        hit_xy_radius: float,       # crowd 판정용 타격영역 XY 반경
        hit_z_range: float,         # crowd 판정용 타격영역 Z 범위
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = target_mask.device
    N, M = target_mask.shape

    # target index 추출 (최대 2개)
    masked_idx = torch.where(
        target_mask,
        torch.arange(M, device=device).expand(N, M),
        -1,
    )

    values, indices = torch.topk(masked_idx, k=2, dim=1, largest=True)  # (N, 2)   # 입력 받은 tensor의 상위 k개의 index와 value를 반환함.

    idx0 = values[:, 0].unsqueeze(1)    # (N, 1)
    idx0 = torch.where(idx0 >= 0, idx0, torch.zeros_like(idx0)) # 타겟이 없는 경우 인덱싱 오류 방지

    idx1 = values[:, 1].unsqueeze(1)
    idx1 = torch.where(idx1 >= 0, idx1, torch.zeros_like(idx1)) # 타겟이 2개가 아닌 경우 인덱싱 오류 방지

    # h1(다음 타격) 인덱스 추출 (노는 팔 배정용)
    masked_idx_h1 = torch.where(
        second_target_mask,
        torch.arange(M, device=device).expand(N, M),
        -1,
    )
    val_h1, _ = torch.topk(masked_idx_h1, k=1, dim=1)   # (N, 1)
    idx_h1 = torch.where(val_h1 >= 0, val_h1, torch.zeros_like(val_h1))
    h1_exist = second_target_mask.any(dim=1)            # (N,)

    # --------------------

    # XYZ 거리
    diff_l = left_tip_pos[:, None, :] - drum_pos    # (N, M, 3)
    diff_r = right_tip_pos[:, None, :] - drum_pos

    dist_l = torch.sum(diff_l * diff_l, dim=-1)     # (N, M)
    dist_r = torch.sum(diff_r * diff_r, dim=-1)

    # XY 제곱거리 (정렬 게이트 + 노는 팔 h1 이동용)
    diff_xy_l = diff_l[:, :, 0:2]
    diff_xy_r = diff_r[:, :, 0:2]
    dist_xy_l = torch.sum(diff_xy_l * diff_xy_l, dim=-1)   # (N, M)
    dist_xy_r = torch.sum(diff_xy_r * diff_xy_r, dim=-1)

    d_l0 = dist_l.gather(1, idx0).squeeze(1)   # (N,)  # gather(input, dim, index): 지정한 dim 축에서 index 위치의 값을 가져오기
    d_r0 = dist_r.gather(1, idx0).squeeze(1)

    d_l1 = dist_l.gather(1, idx1).squeeze(1)
    d_r1 = dist_r.gather(1, idx1).squeeze(1)

    # 각 팔에서 h1(다음 타격)까지 XY 거리 (노는 팔은 높이 자유, XY 기둥으로만 유도 → off_target 밴드와 줄다리기 방지)
    d_l_h1 = dist_xy_l.gather(1, idx_h1).squeeze(1)
    d_r_h1 = dist_xy_r.gather(1, idx_h1).squeeze(1)

    # crowd 판정용: 각 팔의 h0(타겟)까지 XY 거리 / Z 차이
    dxy_l0 = dist_xy_l.gather(1, idx0).squeeze(1)
    dxy_r0 = dist_xy_r.gather(1, idx0).squeeze(1)
    dz_l0 = diff_l[:, :, 2].gather(1, idx0).squeeze(1)
    dz_r0 = diff_r[:, :, 2].gather(1, idx0).squeeze(1)

    # --------------------

    # 정렬 게이트 (타겟 위=1, 멀수록 0). downward 보상에만 사용
    align_l = torch.exp(-k_align * dist_xy_l)   # (N, M)
    align_r = torch.exp(-k_align * dist_xy_r)

    a_l0 = align_l.gather(1, idx0).squeeze(1)   # (N,)
    a_r0 = align_r.gather(1, idx0).squeeze(1)
    a_l1 = align_l.gather(1, idx1).squeeze(1)
    a_r1 = align_r.gather(1, idx1).squeeze(1)

    # --------------------

    down_vel_l = torch.clamp(-tip_vel[:, 0, 2], 0.0, 2.0)   # (N,)
    down_vel_r = torch.clamp(-tip_vel[:, 1, 2], 0.0, 2.0)

    up_vel_l = torch.clamp( tip_vel[:, 0, 2], 0.0, 2.0)     # (N,)
    up_vel_r = torch.clamp( tip_vel[:, 1, 2], 0.0, 2.0)

    hit_armed_l = hit_armed[:, 0, :]    # (N, M)
    hit_armed_r = hit_armed[:, 1, :]

    h_l0 = hit_armed_l.gather(1, idx0).squeeze(1)      # (N,)
    h_r0 = hit_armed_r.gather(1, idx0).squeeze(1)

    h_l1 = hit_armed_l.gather(1, idx1).squeeze(1)
    h_r1 = hit_armed_r.gather(1, idx1).squeeze(1)

    # --------------------

    target_count = target_mask.float().sum(dim=1)
    cost = torch.zeros(N, device=device)
    upward_reward = torch.zeros(N, device=device)
    downward_reward = torch.zeros(N, device=device)
    crowd_pen = torch.zeros(N, device=device)

    # --------------------
    # case 1: 현재 타격 1개 (+ 다음 타격 h1을 노는 팔에 배정)
    # --------------------
    one_mask = (target_count == 1)

    # 노는 팔의 h1 이동 비용 (h1 없으면 0 → 기존 단일 배정과 동일)
    free_r_h1 = d_r_h1 * h1_exist.float()   # 오른팔이 노는 경우
    free_l_h1 = d_l_h1 * h1_exist.float()   # 왼팔이 노는 경우

    # 두 팔을 {h0, h1}에 최적 매칭 (case 2와 동일 원리, 순차 타격 버전)
    cost_A = d_l0 + w_free_arm * free_r_h1   # 왼팔→h0, 오른팔→h1
    cost_B = d_r0 + w_free_arm * free_l_h1   # 오른팔→h0, 왼팔→h1

    use_left = cost_A <= cost_B              # 왼팔이 h0 담당
    one_cost = torch.where(use_left, cost_A, cost_B)
    cost = torch.where(one_mask, one_cost, cost)

    # up/down(타격)은 h0 담당 팔만. 노는 팔은 cost(이동)로만 h1 유도.
    one_upward_reward = torch.where(
        use_left,
        (~h_l0).float() * up_vel_l,
        (~h_r0).float() * up_vel_r,
    )

    one_downward_reward = torch.where(
        use_left,
        h_l0.float() * down_vel_l * a_l0,
        h_r0.float() * down_vel_r * a_r0,
    )

    upward_reward = torch.where(one_mask, one_upward_reward, upward_reward)
    downward_reward = torch.where(one_mask, one_downward_reward, downward_reward)

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

    # assignment 2가지
    cost_case1 = (d_l0 + d_r1) / 2
    cost_case2 = (d_l1 + d_r0) / 2

    two_cost = torch.minimum(cost_case1, cost_case2)
    cost = torch.where(two_mask, two_cost, cost)

    use_case1 = cost_case1 <= cost_case2

    two_upward_reward = torch.where(
        use_case1,
        (~h_l0).float() * up_vel_l + (~h_r1).float() * up_vel_r,
        (~h_l1).float() * up_vel_l + (~h_r0).float() * up_vel_r,
    ) / 2

    two_downward_reward = torch.where(
        use_case1,
        h_l0.float() * down_vel_l * a_l0 + h_r1.float() * down_vel_r * a_r1,
        h_l1.float() * down_vel_l * a_l1 + h_r0.float() * down_vel_r * a_r0,
    ) / 2
    
    upward_reward = torch.where(two_mask, two_upward_reward, upward_reward)
    downward_reward = torch.where(two_mask, two_downward_reward, downward_reward)
    
    # --------------------
    # case 3: 타겟 0개
    # --------------------
    #
    # 나중에 필요하면 작성
    #

    return cost, upward_reward, downward_reward, crowd_pen

@torch.jit.script
def _proximity_terms(
        left_tip_pos: torch.Tensor,
        right_tip_pos: torch.Tensor,
        prev_left_tip_pos: torch.Tensor,
        prev_right_tip_pos: torch.Tensor,
        drum_pos: torch.Tensor,
        next_hits: torch.Tensor,
        tip_vel: torch.Tensor,
        hit_armed: torch.Tensor,
        target_done: torch.Tensor,
        k_time_to_hit: float,
        k_align: float,
        w_free_arm: float,
        hit_xy_radius: float,
        hit_z_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = drum_pos.shape[1]

    nearest_target_mask = next_hits[:, 0, :M] > 0.5   # h0
    h1_mask = next_hits[:, 1, :M] > 0.5
    h2_mask = next_hits[:, 2, :M] > 0.5
    first_time = next_hits[:, 0, M]

    # 노는 팔 타겟: h1이 h0와 같은 드럼이면(같은 팔이 연속으로 쳐야 함) h2로 대체
    h1_same_as_h0 = (h1_mask == nearest_target_mask).all(dim=1)   # (N,)
    second_target_mask = torch.where(h1_same_as_h0.unsqueeze(1), h2_mask, h1_mask)

    curr_cost, upward_reward, downward_reward, crowd_pen = compute_arm_target_assignment(
        target_mask=nearest_target_mask,
        second_target_mask=second_target_mask,
        drum_pos=drum_pos,
        left_tip_pos=left_tip_pos,
        right_tip_pos=right_tip_pos,
        tip_vel=tip_vel,
        hit_armed=hit_armed,
        k_align=k_align,
        w_free_arm=w_free_arm,
        hit_xy_radius=hit_xy_radius,
        hit_z_range=hit_z_range,
    )
    prev_cost, _, _, _ = compute_arm_target_assignment(
        target_mask=nearest_target_mask,
        second_target_mask=second_target_mask,
        drum_pos=drum_pos,
        left_tip_pos=prev_left_tip_pos,
        right_tip_pos=prev_right_tip_pos,
        tip_vel=tip_vel,
        hit_armed=hit_armed,
        k_align=k_align,
        w_free_arm=w_free_arm,
        hit_xy_radius=hit_xy_radius,
        hit_z_range=hit_z_range,
    )

    proximity_cost = torch.exp(-k_time_to_hit * first_time) * curr_cost
    progress_reward = prev_cost - curr_cost

    # 배정된 타겟(h0)이 이미 유효 타격 완료(done)면 downward 억제 → 재타 유도 제거
    done_nearest = (target_done & nearest_target_mask).any(dim=1)   # (N,)
    downward_reward = downward_reward * (~done_nearest).float()

    return proximity_cost, progress_reward, upward_reward, downward_reward, crowd_pen

@torch.jit.script
def _self_collision_penalty(
        elbow_l: torch.Tensor,  # (N, 3) 왼쪽 전완 시작 (elbow)
        wrist_l: torch.Tensor,  # (N, 3) 왼쪽 전완 끝 (wrist)
        elbow_r: torch.Tensor,  # (N, 3) 오른쪽 전완 시작 (elbow)
        wrist_r: torch.Tensor,  # (N, 3) 오른쪽 전완 끝 (wrist)
        threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:  # (penalty (N,), dist (N,))
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

    return torch.clamp(threshold - dist, min=0.0), dist  # penalty, 실제 거리

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

    # off-target 침범 페널티
    # 타겟이 아닌 드럼의 회피영역(XY off_target_xy_radius & Z 0~off_target_z_range 위)에 팁이 들어가면 페널티
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
        joint_vel_weight: torch.Tensor,
        action_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_l2 = torch.sum(action_weight * action * action, dim=-1)
    joint_vel_l2 = torch.sum(joint_vel_weight * joint_vel * joint_vel, dim=-1)

    # 기계적 일률 |τ·q̇| (W). 무게 인지적(무거운 팔 가속=고일률) + 유지 자세는 공짜(q̇≈0).
    # 손목은 가벼워 물리적으로 저렴하므로 별도 배율 없이 균일 합산
    power_l1 = torch.sum(torch.abs(joint_torque * joint_vel), dim=-1)

    low_v = torch.clamp((joint_low + limit_margin) - robot_pos, min=0.0)
    high_v = torch.clamp(robot_pos - (joint_high - limit_margin), min=0.0)
    limit_pen = torch.sum(low_v * low_v + high_v * high_v, dim=-1)

    return action_l2, joint_vel_l2, limit_pen, power_l1

@torch.jit.script
def compute_rewards(
        success_reward: torch.Tensor,
        wrong_cost: torch.Tensor,
        missed_cost: torch.Tensor,
        time_accuracy_reward: torch.Tensor,

        proximity_cost: torch.Tensor,
        progress_reward: torch.Tensor,

        upward_reward: torch.Tensor,
        downward_reward: torch.Tensor,

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

        w_upward: float,
        w_downward: float,

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
        off_target_pen: torch.Tensor,
        w_off_target: float,
        crowd_pen: torch.Tensor,
        w_crowd: float,
) -> torch.Tensor:
    reward = (
        w_success * success_reward
        - w_wrong * wrong_cost
        - w_miss * missed_cost
        + w_time_accuracy * time_accuracy_reward

        - w_proximity * proximity_cost
        + w_progress * progress_reward

        + w_upward * upward_reward
        + w_downward * downward_reward

        + w_impact_vel * impact_vel_reward
        + w_wrist_snap * wrist_snap_reward

        - w_action * action_l2
        - w_joint_vel * joint_vel_l2
        - w_power * power_l1
        - w_limit * limit_pen
        - w_tip_limit * tip_limit_pen
        - w_under_drum * under_drum_pen
        - w_self_collision * self_col_pen
        - w_off_target * off_target_pen
        - w_crowd * crowd_pen
    )
    return reward

# === 클래스는 jit 함수 호출해서 사용 ===
@dataclass
class RewardComputerCfg:
    # 하이퍼 파라미터
    limit_margin: float = 0.08
    k_accuracy: float = 1.0
    k_time_to_hit: float = 3.0
    k_align: float = 40.0   # downward 보상 XY 정렬 게이트 (XY 제곱거리 기준, 13cm≈0.51)
    w_free_arm: float = 0.5 # 노는 팔을 다음 타격(h1)으로 유도하는 이동 가중치

    # 타격영역 판정 (hit_detector와 동일). crowd_pen 등에 사용
    hit_xy_radius: float = 0.13
    hit_z_range: float = 0.07

    # off-target 전용 (비타겟 드럼 회피 유도). hit 밴드와 분리 — 나중에 넓혀 회피 마진 확보 가능
    off_target_xy_radius: float = 0.13   # 우선 hit과 동일
    off_target_z_range: float = 0.12     # hit 밴드(0.07)보다 넓게 → 비타겟 회피 마진
    w_off_target: float = 0.3
    w_crowd: float = 0.5          # 단일 타겟에 양팔 협공 페널티

    # 팁이 가면 안되는 범위
    x_limit: float = 0.5
    y_limit_l: float = 0.2
    y_limit_h: float = 0.8
    z_limit: float = -0.6
    drum_xy_margin: float = 0.15
    drum_z_margin: float = 0.05

    # 가중치
    w_success: float = 5.0
    w_wrong: float = 3.0 #1.5
    w_miss: float = 2.0
    w_time_accuracy: float = 5.0

    w_progress: float = 4.0
    w_proximity: float = 1.5

    w_upward: float = 0 #0.4
    w_downward: float = 0 #0.5

    w_impact_vel: float = 3.0
    impact_vel_max: float = 1.0

    # 손목 스냅 보상 (성공 타격 시 손목 각속도 rad/s에 비례). 팔 대신 손목 사용 유도
    w_wrist_snap: float = 3.0
    wrist_snap_max: float = 12.0   # 정규화 기준 (실측 느긋 7~13 rad/s의 중간)

    w_action: float = 0.0003
    w_wrist_action_scale: float = 0.001  # 손목 action 페널티 배율 (손목 스냅 위해 완화)
    w_joint_vel: float = 0.0001
    w_wrist_vel_scale: float = 0.001  # 손목 관절 속도 페널티 배율 (스윙 위해 완화)
    # 기계적 일률 |τ·q̇| 페널티. 헛스윙은 후반 현상이므로 성공률 게이팅으로 나중에 켬 (초기 탐색 방해 방지)
    w_power_max: float = 0.005          # 램프 최대치 (성공률 full 이상에서 도달)
    w_power_success_start: float = 0.6  # 이 성공률(EMA)부터 페널티 켜기 시작
    w_power_success_full: float = 0.85  # 이 성공률(EMA)에서 최대치 도달
    w_power_ema_beta: float = 0.9       # 성공률 EMA 평활 계수
    w_limit: float = 0.5
    w_tip_limit: float = 0.15
    w_under_drum: float = 0.1 #0.08
    w_self_collision: float = 0.5
    self_collision_threshold: float = 0.10  # 스틱 간 거리 임계값 [m]

    # 악기별 가중치 업데이트 주기 (step)
    update_interval: int = 100

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

        # 관절 속도 페널티 가중치 (손목만 완화). ctrl_joint_names는 resolve된 순서
        vel_w = [self.cfg.w_wrist_vel_scale if "wrist" in name else 1.0
                 for name in ctrl_joint_names]
        self.joint_vel_weight = torch.tensor(
            vel_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # action 페널티 가중치 (손목만 완화). 손목 스냅 위해 손목 action에 세금 감면
        act_w = [self.cfg.w_wrist_action_scale if "wrist" in name else 1.0
                 for name in ctrl_joint_names]
        self.action_weight = torch.tensor(
            act_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # 손목 관절 인덱스 (각속도 로깅용). hit_per_arm 순서(0=왼,1=오)와 맞춤
        name_to_idx = {name: i for i, name in enumerate(ctrl_joint_names)}
        self.wrist_ids = torch.tensor(
            [name_to_idx["left_wrist"], name_to_idx["right_wrist"]],
            device=self.device, dtype=torch.long,
        )

        # 일률 페널티 성공률 게이팅 상태 (초기엔 꺼짐 → 성공률 오르면 램프인)
        self.w_power_eff = 0.0
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
            prev_hit_armed: torch.Tensor,
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
            target_done: torch.Tensor,  # (N, M) 유효 타격 완료된 타겟 (재타 downward 억제)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        proximity_cost, progress_reward, upward_reward, downward_reward, crowd_pen = _proximity_terms(
            left_tip_pos=tip_pos[:, 0, :],
            right_tip_pos=tip_pos[:, 1, :],
            prev_left_tip_pos=prev_tip_pos[:, 0, :],
            prev_right_tip_pos=prev_tip_pos[:, 1, :],
            drum_pos=drum_pos,
            next_hits=next_hits,
            tip_vel=tip_vel,
            hit_armed=prev_hit_armed,
            target_done=target_done,
            k_time_to_hit=self.cfg.k_time_to_hit,
            k_align=self.cfg.k_align,
            w_free_arm=self.cfg.w_free_arm,
            hit_xy_radius=self.cfg.hit_xy_radius,
            hit_z_range=self.cfg.hit_z_range,
        )

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
            joint_vel_weight=self.joint_vel_weight,
            action_weight=self.action_weight,
        )

        # self collision penalty (전완 선분: elbow → wrist)
        self_col_pen, forearm_dist = _self_collision_penalty(
            elbow_l=elbow_pos[:, 0, :],
            wrist_l=wrist_pos[:, 0, :],
            elbow_r=elbow_pos[:, 1, :],
            wrist_r=wrist_pos[:, 1, :],
            threshold=self.cfg.self_collision_threshold,
        )

        # impact velocity: 성공한 타격의 속도에만 보상 (타겟당 1회). 연타 유인 제거
        impact_vel_reward = (success_impact_vel / (self.cfg.impact_vel_max + 1e-8)).sum(dim=1)  # (N,)

        # wrist snap: 성공한 타격의 손목 각속도에 보상 (타겟당 1회). 팔 대신 손목 사용 유도
        wrist_snap_reward = (success_wrist_ang_vel / (self.cfg.wrist_snap_max + 1e-8)).sum(dim=1)  # (N,)

        reward = compute_rewards(
            success_reward=success_reward,
            wrong_cost=wrong_cost,
            missed_cost=missed_cost,
            time_accuracy_reward=time_accuracy_reward,
            
            proximity_cost=proximity_cost,
            progress_reward=progress_reward,

            upward_reward=upward_reward,
            downward_reward=downward_reward,

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

            w_upward=self.cfg.w_upward,
            w_downward=self.cfg.w_downward,
            
            w_action=self.cfg.w_action,
            w_joint_vel=self.cfg.w_joint_vel,
            w_power=self.w_power_eff,
            w_limit=self.cfg.w_limit,
            w_tip_limit=self.cfg.w_tip_limit,
            w_under_drum=self.cfg.w_under_drum,
            w_self_collision=self.cfg.w_self_collision,
            self_col_pen=self_col_pen,
            impact_vel_reward=impact_vel_reward,
            w_impact_vel=self.cfg.w_impact_vel,
            wrist_snap_reward=wrist_snap_reward,
            w_wrist_snap=self.cfg.w_wrist_snap,
            off_target_pen=off_target_pen,
            w_off_target=self.cfg.w_off_target,
            crowd_pen=crowd_pen,
            w_crowd=self.cfg.w_crowd,
        )

        num_hit_drum = success.float() + missed_target.float()

        num_hit = num_hit_drum.float().sum(dim=-1)
        num_success = success.float().sum(dim=-1)
        num_wrong = wrong_hit.float().sum(dim=-1)
        num_missed = missed_target.float().sum(dim=-1)

        log_terms = {
            "reward": reward,
            "proximity": proximity_cost,
            "progress(x100)": progress_reward * 100,
            "upward": upward_reward,
            "downward": downward_reward,
            "action_l2": action_l2,
            "joint_vel_l2": joint_vel_l2,
            "power_l1": power_l1,
            "w_power_eff(x1000)": torch.full_like(reward, self.w_power_eff * 1000.0),
            "limit_pen(x100)": limit_pen * 100,
            "tip_limit_pen": tip_limit_pen,
            "under_drum_pen": under_drum_pen,
            "off_target_pen": off_target_pen,
            "crowd_pen": crowd_pen,
            "self_collision_pen": self_col_pen,
            "forearm_dist": forearm_dist,
            "impact_vel": impact_vel_reward,
            "wrist_snap": wrist_snap_reward,
        }

        # 실제 타격 순간만 뽑아 평균 (hit이 난 팔만)
        hit_any_per_arm = hit_per_arm.any(dim=2)                           # (N, 2)
        hit_arm_count = hit_any_per_arm.float().sum(dim=1)                 # (N,)

        # 타격 순간 팁 속도 (m/s)
        down_vel_ms = torch.clamp(-tip_vel[:, :, 2], 0.0)                 # (N, 2)
        vel_sum = (hit_any_per_arm.float() * down_vel_ms).sum(dim=1)      # (N,)

        # 타격 순간 손목 각속도 (rad/s). wrist_ids는 [왼,오] 순서라 팔 축과 일치
        wrist_vel_abs = robot_vel[:, self.wrist_ids].abs()                # (N, 2)
        wrist_vel_sum = (hit_any_per_arm.float() * wrist_vel_abs).sum(dim=1)  # (N,)

        # 드럼별 타격 속도 (m/s): 팔의 하강 속도를 그 팔이 친 드럼에 배분
        vel_per_drum = (hit_per_arm.float() * down_vel_ms[:, :, None]).sum(dim=1)  # (N, M)
        hit_per_drum = hit_per_arm.float().sum(dim=1)                             # (N, M)

        rate_log_terms = {
            "success_rate": torch.stack([num_success, num_hit], dim=-1),
            "wrong_rate": torch.stack([num_wrong, num_hit], dim=-1),
            "miss_rate": torch.stack([num_missed, num_hit], dim=-1),
            "avg_impact_vel_ms": torch.stack([vel_sum, hit_arm_count], dim=-1),
            "hit_wrist_ang_vel": torch.stack([wrist_vel_sum, hit_arm_count], dim=-1),
        }

        for i, name in enumerate(self.drum_names):
            rate_log_terms[f"{name}_success_rate"] = torch.stack([success[:, i], num_hit_drum[:, i]], dim=-1)
            rate_log_terms[f"{name}_wrong_rate"]   = torch.stack([wrong_hit[:, i], num_hit_drum[:, i]], dim=-1)
            rate_log_terms[f"{name}_miss_rate"]    = torch.stack([missed_target[:, i], num_hit_drum[:, i]], dim=-1)
            rate_log_terms[f"{name}_impact_vel"]   = torch.stack([vel_per_drum[:, i], hit_per_drum[:, i]], dim=-1)

        return reward, log_terms, rate_log_terms
    
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

            # 통계 초기화
            self.n_success.zero_()
            self.n_hit.zero_()