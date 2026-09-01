"""
실기 배포에 필요한 "규약"을 Isaac 안에서 찍어 낸다.

Isaac 을 띄우지 않으면 알 수 없는 것이 하나 있다 — obs/action 의 관절 순서다.
robot_interface.py:27 이 robot.find_joints() 를 preserve_order 없이 호출하므로
USD articulation 순서가 이기고, 그 순서는 specs.py 의 ctrl_joint_names 목록 순서도
URDF 선언 순서도 아니다. USD 는 바이너리 crate 라 파일에서 읽을 수도 없다.

export_policy.py 는 이 값을 --joint-order 로 받는다(기본값은 BFS 추정치).
이 스크립트가 실측값을 내보내고, 둘이 다르면 여기서 바로 드러난다.

기존 학습 코드는 건드리지 않는다. 환경을 만들어 읽기만 한다.

사용법
    cd <IsaacLab 루트>
    ./isaaclab.sh -p source/extensions/drum_robot/drum_robot/scripts/deploy/dump_conventions.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --headless \
        --out source/extensions/drum_robot/deploy_export/conventions.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher   # type: ignore[arg-type]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="DrumRobot-drum_robot-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default="deploy_export/conventions.json")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
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
    resolved = list(ri.ctrl_joint_names)          # articulation 순서로 재정렬된 이름
    resolved_ids = [int(i) for i in ri.ctrl_joint_ids]
    requested = list(u.robot_spec.ctrl_joint_names) if hasattr(u, "robot_spec") else []

    # specs.py 의 이름 -> 실기 모터 id
    name_to_motor = dict(ri.robot_spec.joint_name_to_ik_result_idx)
    motor_by_obs = [int(name_to_motor[n]) for n in resolved]

    # ---- 관절 한계 (정규화 상수의 근거) ----
    lo = ri.joint_low.flatten().tolist()
    hi = ri.joint_high.flatten().tolist()
    usd_dir = ri.dir_tensor.flatten().tolist()

    # ---- 팁 좌표 규약 확인용: 기본 자세에서의 tip / drum ----
    # 실기 solve_fk 와 대조하면 팁 정의 불일치(0.373 vs 0.385)가 바로 드러난다.
    env.reset()
    tip = ri.get_body_pos(robot).detach().cpu()[0].tolist()     # (2, 3) 스틱 팁, env 프레임
    wrist = ri.get_wrist_pos(robot).detach().cpu()[0].tolist()  # (2, 3) 손목 링크, env 프레임
    jpos = ri.get_joint_pos(robot).detach().cpu()[0].tolist()   # (9,) resolved 순서
    drum = u.drum_pos.detach().cpu()[0].tolist()                # (8, 3) env 프레임 (+z offset)

    payload = {
        "_comment": "dump_conventions.py 가 Isaac 안에서 생성. export_policy.py 의 --joint-order 검증용.",
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

        "reset_pose_probe": {
            "note": "env.reset() 직후 1회 관측. 실기 solve_fk 와 대조해 팁 정의를 검증한다.",
            "joint_pos_resolved_order_rad": jpos,
            "tip_pos_env_frame_m": tip,
            "wrist_pos_env_frame_m": wrist,
            "drum_pos_env_frame_m": drum,
            "robot_waist_joint_offset_z": float(env_cfg.robot_waist_joint_offset_z),
            "robot_waist_joint_to_link_z": float(env_cfg.robot_waist_joint_to_link_z),
        },
    }

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- 콘솔 요약 ----
    print("\n" + "=" * 72)
    print("관절 순서 (articulation = obs/action 순서)")
    print(f"  {'obs':>3}  {'이름':<20} {'실기 모터 id':>12}")
    for i, n in enumerate(resolved):
        print(f"  {i:>3}  {n:<20} {motor_by_obs[i]:>12}")
    print(f"\n  motor_id_by_obs_index = {motor_by_obs}")
    print(f"  specs.py 목록 순서와 동일? {resolved == requested}")
    if resolved != requested:
        print(f"    요청: {requested}")
        print(f"    실제: {resolved}")
    print(f"\n저장: {out}")
    print("=" * 72 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
