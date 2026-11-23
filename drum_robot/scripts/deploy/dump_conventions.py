"""
실기 배포에 필요한 "규약"을 Isaac 안에서 찍어 낸다. export_policy.py 의 필수 입력이다.

코드만 봐서는 알 수 없는 값들:
  - obs/action 관절 순서: robot.find_joints() 가 preserve_order 없이 호출되므로 USD articulation 순서가
    이긴다. specs.py 의 ctrl_joint_names 목록 순서도, URDF 선언 순서도 아니다.
  - shoulder_z: drumrobot_env.py __init__ 에서 바디 위치로 런타임 계산 (정렬 reach 판정에 쓴다)
  - max_lead: joint_effort_limits / joint_stiffness (Isaac 이 실제로 적용한 값)
  - frozen_ids: 정렬 전용 학습에서 행동 0 으로 고정한 관절

기존 학습 코드는 건드리지 않는다. 환경을 만들어 읽기만 한다.

사용법 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/deploy/dump_conventions.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --headless \
        --out drum_robot/deploy_export/conventions.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher   # type: ignore[arg-type]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="DrumRobot-drum_robot-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default="drum_robot/deploy_export/conventions.json")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
from pathlib import Path

import gymnasium as gym    # type: ignore[arg-type]
import torch               # type: ignore[arg-type]

import drum_robot          # gym.register 실행용  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config   # type: ignore[arg-type]


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    ri = u.robot_interface
    robot = u.robot

    # ---- 관절 순서 (핵심) ----
    resolved = list(ri.ctrl_joint_names)                 # articulation 순서로 재정렬된 이름
    resolved_ids = [int(i) for i in ri.ctrl_joint_ids]
    requested = list(ri.robot_spec.ctrl_joint_names)     # specs.py 목록 순서
    name_to_motor = dict(ri.robot_spec.joint_name_to_ik_result_idx)
    motor_by_obs = [int(name_to_motor[n]) for n in resolved]

    lo = ri.joint_low.flatten().tolist()
    hi = ri.joint_high.flatten().tolist()
    usd_dir = ri.dir_tensor.flatten().tolist()

    # ---- 행동 규약 ----
    z_off = float(env_cfg.robot_waist_joint_offset_z)
    max_lead = u.max_lead.flatten().tolist()                     # (9,) obs 순서, effort/stiffness
    action_scale_vec = u.action_scale_vec.flatten().tolist()
    frozen_names = [resolved[i] for i in u.frozen_ids]

    # ---- 정렬 상수: 어깨 높이 (env 프레임 -> 허리 기준) ----
    al = u.reward_computer.alignment
    shoulder_z_env = float(al.z_s)
    shoulder_z_waist = shoulder_z_env - z_off

    # ---- 리셋 자세 1회 관측 (실기 solve_fk 대조용) ----
    env.reset()
    jpos = ri.get_joint_pos(robot).detach().cpu()[0].tolist()
    tip = ri.get_body_pos(robot).detach().cpu()[0].tolist()                    # (2,3) [왼, 오른] env 프레임
    wrist_ids = [ri.body_to_idx["left_wrist"], ri.body_to_idx["right_wrist"]]
    wrist = (robot.data.body_pos_w[0, wrist_ids] - u.scene.env_origins[0]).detach().cpu().tolist()
    drum = u.drum_pos.detach().cpu()[0].tolist()                               # (8,3) env 프레임 (노이즈 포함)

    payload = {
        "_comment": "dump_conventions.py 가 Isaac 안에서 생성. export_policy.py 의 필수 입력.",
        "task": args_cli.task,

        "joint_order_resolved": resolved,
        "joint_order_requested": requested,
        "joint_order_matches_requested": resolved == requested,
        "ctrl_joint_ids_in_articulation": resolved_ids,
        "motor_id_by_obs_index": motor_by_obs,
        "all_joint_names_in_articulation": [str(n) for n in robot.data.joint_names],
        "all_body_names_in_articulation": [str(n) for n in robot.data.body_names],

        "joint_limit_low_resolved_order": lo,
        "joint_limit_high_resolved_order": hi,
        "joint_usd_dir_resolved_order": usd_dir,

        "max_lead_obs_order": max_lead,
        "action_scale_obs_order": action_scale_vec,
        "frozen_joint_names": frozen_names,
        "frozen_obs_index": [int(i) for i in u.frozen_ids],

        "shoulder_z_env_frame_m": shoulder_z_env,
        "shoulder_z_waist_frame_m": shoulder_z_waist,
        "alignment_d_th_m": float(al.d_th),
        "alignment_l_m": float(al.l),

        "step_dt": float(u.step_dt),
        "max_lookahead_step": int(u.rds.env.max_lookahead_step),
        "num_hits": int(u.rds.cfg.num_hits),
        "hit_window_step": int(u.hit_detector.max_window),
        "episode_length_step": int(u.episode_length_step),

        "reset_pose_probe": {
            "note": "env.reset() 직후 1회 관측 (env 프레임: z 에 robot_waist_joint_offset_z 포함). 실기 solve_fk 와 대조용.",
            "joint_pos_resolved_order_rad": jpos,
            "tip_pos_env_frame_m": tip,
            "wrist_pos_env_frame_m": wrist,
            "drum_pos_env_frame_m": drum,
            "robot_waist_joint_offset_z": z_off,
            "robot_waist_joint_to_link_z": float(env_cfg.robot_waist_joint_to_link_z),
        },
    }

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("관절 순서 (articulation = obs/action 순서)")
    print(f"  {'obs':>3}  {'이름':<20} {'모터 id':>8}  {'max_lead':>9}  {'scale':>6}  고정")
    for i, n in enumerate(resolved):
        print(f"  {i:>3}  {n:<20} {motor_by_obs[i]:>8}  {max_lead[i]:>9.4f}  {action_scale_vec[i]:>6.1f}  {'o' if n in frozen_names else ''}")
    print(f"\n  motor_id_by_obs_index = {motor_by_obs}")
    print(f"  specs.py 목록 순서와 동일? {resolved == requested}")
    print(f"  shoulder_z: env {shoulder_z_env:.4f} m -> 허리 기준 {shoulder_z_waist:.4f} m")
    print(f"  lookahead {payload['max_lookahead_step']} 스텝, K {payload['num_hits']}, step_dt {payload['step_dt']}")
    print(f"\n저장: {out}")
    print("=" * 72 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
