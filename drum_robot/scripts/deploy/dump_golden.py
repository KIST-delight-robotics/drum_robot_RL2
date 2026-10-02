"""
골든 테스트용 덤프 — sim 한 에피소드의 (입력, 기대 obs, act) 를 파일로 뽑는다.

목적: 실기 C++ obs_builder 가 학습과 같은 값을 만드는지 요소별로 대조한다.
obs_builder 는 스케줄러 90여 줄과 재장전 상태머신을 손으로 옮긴 것이라
어딘가 틀렸을 가능성이 실재하고, 그걸 실기 전에 잡는 것이 이 파일의 존재 이유다.

구동 방식: export 한 policy.onnx 로 sim 을 직접 굴린다.
  - 무작위 액션보다 현실적인 궤적이 나와 재장전/스케줄러가 제대로 자극된다
  - export 결과가 sim 루프 안에서도 맞는지 함께 검증된다
  - skrl Runner 기계장치가 필요 없다 (onnxruntime 만)

덤프 내용 (스텝별)
  입력  : joint_pos/vel  — 실기 모터 id 순서. C++ 이 이걸 그대로 먹인다
          t_score        — 스텝 x step_dt
  기대  : 원시 버퍼 9개   — C++ 이 만들어야 하는 값
          obs_148        — 정규화까지 끝난 값 (그래프 검증용)
          act_9          — ONNX 출력

에피소드 상수
  drum_pos  : sim 이 쓴 값. per-episode 노이즈(drum_noise_scale=0.02)가 섞여 있으므로
              C++ 은 자기 드럼 표를 이 값으로 덮어써야 요소별 비교가 성립한다.
  z_offset  : env 프레임 -> 허리 기준 변환용. C++ 은 허리 기준으로 일한다.

사용법
    cd <IsaacLab 루트>
    ./isaaclab.sh -p source/extensions/drum_robot/drum_robot/scripts/deploy/dump_golden.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --headless \
        --onnx  source/extensions/drum_robot/deploy_export/policy.onnx \
        --steps 300 \
        --out   source/extensions/drum_robot/deploy_export/golden.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher   # type: ignore[arg-type]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="DrumRobot-drum_robot-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--onnx", type=str, default="deploy_export/policy.onnx")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="deploy_export/golden.json")
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
import numpy as np
import torch               # type: ignore[arg-type]

import drum_robot          # gym.register  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config   # type: ignore[arg-type]

IN_NAMES = ["joint_pos", "joint_vel", "tip_pos", "drum_pos", "next_hits",
            "hit_armed", "arm_role", "per_arm_pos", "per_arm_time"]


def f(t):
    """텐서를 파이썬 리스트로 (환경 0번만, float32 정밀도 유지)."""
    a = t.detach().cpu().numpy()
    if a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    return np.asarray(a, dtype=np.float32).reshape(-1).tolist()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    import onnxruntime as ort

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    ri = u.robot_interface

    sess = ort.InferenceSession(args_cli.onnx, providers=["CPUExecutionProvider"])
    print(f"[dump_golden] ONNX: {args_cli.onnx}")

    # obs 인덱스 -> 실기 모터 id (dump_conventions.py 와 같은 계산)
    name_to_motor = dict(ri.robot_spec.joint_name_to_ik_result_idx)
    motor_by_obs = [int(name_to_motor[n]) for n in ri.ctrl_joint_names]

    z_off = float(env_cfg.robot_waist_joint_offset_z)
    step_dt = float(u.step_dt)

    env.reset()
    steps = []

    for i in range(args_cli.steps):
        # ---- 이 시점의 원시 버퍼를 env 내부에서 읽는다 ----
        # DirectRLEnv 는 step() 끝에서 _get_observations 를 부르므로,
        # 지금 값들은 방금 반환된 obs 와 짝이 맞는다.
        tip_pos, hit_armed = u.hit_detector.get_value_for_obs()
        joint_pos = ri.get_joint_pos(u.robot)      # (N,9) obs 순서
        joint_vel = ri.get_joint_vel(u.robot)

        raw = {
            "joint_pos":    f(joint_pos),
            "joint_vel":    f(joint_vel),
            "tip_pos":      f(tip_pos),
            "drum_pos":     f(u.drum_pos),
            "next_hits":    f(u.next_hits),
            "hit_armed":    f(hit_armed.float()),
            "arm_role":     f(u.arm_role),
            "per_arm_pos":  f(u.per_arm_pos),
            "per_arm_time": f(u.per_arm_time),
        }

        # ---- ONNX 로 액션을 만든다 (sim 을 이걸로 굴린다) ----
        shapes = {
            "joint_pos": (1, 9), "joint_vel": (1, 9), "tip_pos": (1, 2, 3),
            "drum_pos": (1, 8, 3), "next_hits": (1, 6, 11), "hit_armed": (1, 2, 8),
            "arm_role": (1, 2), "per_arm_pos": (1, 2, 2, 3), "per_arm_time": (1, 2, 2),
        }
        # ★ 프레임 변환. 그래프는 실기와 같은 "허리 기준"을 먹는다.
        #   raw 는 env 프레임이므로 위치 3종의 z 에서 z_offset 을 뺀다.
        #   이걸 빠뜨리면 정규화가 1.0 만큼 어긋나 z 채널이 clamp 에 포화된다.
        feed = {k: np.asarray(raw[k], dtype=np.float32).reshape(shapes[k]) for k in IN_NAMES}
        for k in ("tip_pos", "drum_pos", "per_arm_pos"):
            feed[k] = feed[k].copy()
            feed[k][..., 2] -= z_off
        act = sess.run(["action_mean"], feed)[0].reshape(-1)

        # ---- C++ 이 먹일 입력: 모터 id 순서 ----
        jp_obs = raw["joint_pos"]
        jv_obs = raw["joint_vel"]
        jp_motor = [0.0] * 9
        jv_motor = [0.0] * 9
        for k, m in enumerate(motor_by_obs):
            jp_motor[m] = jp_obs[k]
            jv_motor[m] = jv_obs[k]

        steps.append({
            "step": i,
            "t_score": i * step_dt,
            "joint_pos_motor_order": jp_motor,
            "joint_vel_motor_order": jv_motor,
            "raw": raw,
            "act": np.asarray(act, dtype=np.float32).tolist(),
        })

        obs_dict, _, term, trunc, _ = env.step(torch.tensor(
            act.reshape(1, -1), dtype=torch.float32, device=u.device))
        # 정규화된 148 도 함께 남긴다 (그래프 검증용)
        steps[-1]["obs_148_next"] = f(obs_dict["policy"])

        if bool(term[0]) or bool(trunc[0]):
            print(f"[dump_golden] 에피소드 종료 (step {i})")
            break

    payload = {
        "_comment": "dump_golden.py 생성. 실기 obs_builder 요소별 대조용.",
        "task": args_cli.task,
        "onnx": args_cli.onnx,
        "seed": args_cli.seed,
        "step_dt": step_dt,
        "z_offset": z_off,
        "motor_id_by_obs_index": motor_by_obs,
        "joint_order": list(ri.ctrl_joint_names),
        "note_frames": (
            "tip_pos / drum_pos / per_arm_pos 는 env 프레임이다. "
            "허리 기준으로 쓰려면 z 에서 z_offset 을 뺀다. "
            "drum_pos 에는 per-episode 노이즈가 섞여 있으므로 C++ 은 이 값으로 덮어써야 한다."
        ),
        "num_steps": len(steps),
        "steps": steps,
    }

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\n[dump_golden] {len(steps)} 스텝 저장: {out}  ({size_mb:.1f} MB)")
    print(f"  step_dt {step_dt:.6f} s, z_offset {z_off}")
    print(f"  motor_id_by_obs_index {motor_by_obs}")

    # 간단 요약 — 상태머신이 실제로 자극됐는지
    armed_sum = sum(sum(s["raw"]["hit_armed"]) for s in steps)
    role_on   = sum(1 for s in steps if max(s["raw"]["arm_role"]) > 0.5)
    valid_h0  = sum(1 for s in steps if s["raw"]["next_hits"][9] > 0.5)
    print(f"  hit_armed 합계 {armed_sum:.0f} (0 이면 상태머신이 자극되지 않음)")
    print(f"  arm_role 활성 스텝 {role_on}/{len(steps)}")
    print(f"  next_hits[0].valid 스텝 {valid_h0}/{len(steps)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
