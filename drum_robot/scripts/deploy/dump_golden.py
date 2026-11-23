"""
골든 테스트용 덤프 — sim 을 고정 곡으로 한 에피소드 굴리며 (입력, 기대 obs, 플래너·정렬 상태, act) 를 남긴다.

목적: 실기 C++ obs_builder 가 스티킹 플래너와 정렬 목표 상태기계를 손으로 옮긴 결과가
학습과 같은 값을 내는지 요소별로 대조한다.

구동 방식: export 한 policy.onnx 로 sim 을 굴린다 (skrl 불필요, onnxruntime 만).
조건 고정 (play_song.py 와 같음): 곡 = RDS 텍스트, 드럼 노이즈 0, 시작 자세 = (snare, snare) 쌍, 관절 노이즈 0.
실기는 같은 곡·같은 시작 자세로 obs_builder 를 돌려 비교한다.

덤프 (스텝별, 모두 env 0번)
  입력  : joint_pos/joint_vel (실기 모터 id 순서), t_score
  raw   : 그래프 입력 11버퍼 + next_hits(플래너 입력, golden_check 가 주입). 위치는 허리 기준(z 에서 z_offset 을 뺀 값) = C++ 이 만들어야 하는 값
          last_action 은 env.actions (직전 행동, clamp·frozen 0 적용 후. 리셋 직후 0)
  obs   : env 가 반환한 정규화 obs 90 (그래프 검증용, raw 와 같은 시점)
  act   : ONNX 출력 (sim 을 이걸로 굴렸다)
  state : target_q(누적 목표), 플래너 ev_arm/ev_off, 정렬 target_l/r/waist, prev_drum, arm_drum/t/valid, last_hit_step, virtual_hit

사용법 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/deploy/dump_golden.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --headless \
        --song drum_robot/RDS/DrumSolo.txt \
        --onnx drum_robot/deploy_export/policy.onnx \
        --out  drum_robot/deploy_export/golden.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher   # type: ignore[arg-type]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="DrumRobot-drum_robot-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--song", type=str, default="drum_robot/RDS/DrumSolo.txt")
parser.add_argument("--onnx", type=str, default="drum_robot/deploy_export/policy.onnx")
parser.add_argument("--out", type=str, default="drum_robot/deploy_export/golden.json")
parser.add_argument("--max_steps", type=int, default=0, help="0 = 곡 끝까지")
parser.add_argument("--seed", type=int, default=0)
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
import torch.nn.functional as F   # type: ignore[arg-type]

import drum_robot          # gym.register  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config   # type: ignore[arg-type]

IN_NAMES = ["joint_pos", "joint_vel", "lead", "tip_pos", "drum_pos", "arm_next", "waist_des", "waist_vel_des",
            "yaw_des", "yaw_vel_des", "last_action"]
SHAPES = {"joint_pos": (1, 9), "joint_vel": (1, 9), "lead": (1, 9), "tip_pos": (1, 2, 3),
          "drum_pos": (1, 8, 3), "arm_next": (1, 2, 9), "waist_des": (1, 1), "waist_vel_des": (1, 1),
          "yaw_des": (1, 2), "yaw_vel_des": (1, 2),
          "last_action": (1, 9)}
OFF_WD = 75                # obs 90 에서 waist_des 칸 (export_policy.py 와 같음)
OFF_WV = 76                # obs 90 에서 waist_vel_des 칸
OFF_YD = 77                # obs 90 에서 yaw_des 칸 [오른, 왼]
OFF_YV = 79                # obs 90 에서 yaw_vel_des 칸 [오른, 왼]

# 실기 악기 번호 -> 시뮬 드럼 (snare, floor, mid, high, hihat, ride, crash_r, crash_l). 9 오픈 하이햇 -> 하이햇
NOTE_TO_DRUM = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 4}
START_PAIR = ("snare", "snare")


def read_song(path: Path):
    """RDS 텍스트 -> bpm, [(시각 s, [시뮬 드럼, ...]), ...]  (play_song.py / 실기 make_drum_event 와 같은 계산)"""
    bpm, t, events = 100.0, 0.0, []
    for line in open(path):
        it = line.split()
        if not it:
            continue
        if it[0] == "bpm":
            bpm = float(it[1])
            continue
        if it[0] == "end":
            break
        t += float(it[1]) * 100.0 / bpm
        drums = sorted({NOTE_TO_DRUM[int(n)] for n in (it[2], it[3]) if int(n) != 0})   # 오른손, 왼손 열 (킥 열은 안 씀)
        if drums:
            events.append((t, drums))
    if not events:
        raise SystemExit(f"타격이 없는 곡입니다: {path}")
    return bpm, events


def f(t):
    """텐서 -> 파이썬 리스트 (env 0번만, float32)."""
    a = t.detach().cpu().numpy()
    if a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    return np.asarray(a, dtype=np.float32).reshape(-1).tolist()


def fi(t):
    a = t.detach().cpu().numpy()
    if a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    return np.asarray(a).reshape(-1).astype(int).tolist()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    import onnxruntime as ort

    song_path = Path(args_cli.song).resolve()
    bpm, events = read_song(song_path)
    song_len = events[-1][0] + 1.0                       # 마지막 음 + 1초
    print(f"[dump_golden] 곡 {song_path.name}: {bpm} BPM, 이벤트 {len(events)}개, 길이 {song_len:.2f}s")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = song_len                  # 악보·판정 표·플래너 버퍼가 이 길이로 만들어짐
    # 제어 지연 DR 끄기 (2026-10-26 17:11 에 env 에 추가된 기능. 골든은 결정적이어야 하고, 15-18-20 런은 지연 없이 학습됨)
    if hasattr(env_cfg, "action_delay_steps_max"):
        env_cfg.action_delay_steps_min = 0
        env_cfg.action_delay_steps_max = 0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    ri = u.robot_interface
    al = u.reward_computer.alignment
    T, M, dt = u.episode_length_step, u.drum_pos.shape[1], float(u.step_dt)
    L, K = int(u.rds.env.max_lookahead_step), int(u.rds.cfg.num_hits)
    z_off = float(env_cfg.robot_waist_joint_offset_z)
    drum_names = list(u.instruments.all.keys())

    # ---- 곡 -> 시뮬 악보 (T, M): 리셋 때 무작위 악보 대신 이 곡 ----
    song = torch.zeros((T, M), dtype=torch.int64, device=u.device)
    song_events = []
    for t, drums in events:
        s = round(t / dt)
        if s >= T - u.hit_detector.max_window:
            raise SystemExit(f"곡 시각이 에피소드 끝을 넘음: {t:.3f}s")
        song[s, drums] = 1
        song_events.append({"bin": s, "t_s": s * dt, "drums": drums})
    u.rds._reset_random = lambda env_ids: song[None].expand(len(env_ids), -1, -1).clone()

    # ---- 조건 고정: 드럼 노이즈 0, 시작 자세 = START_PAIR (관절 노이즈 0) ----
    u.instruments.drum_noise_scale = 0.0
    init = u.robot_initializer
    k = init.cfg.drum_pairs.index(START_PAIR)
    init.reset_init_pos = lambda env_ids: (init.pos_angle[k].expand(len(env_ids), -1).clone(),
                                           init.pair_drum_idx[k].expand(len(env_ids), -1).clone())

    # ---- 플래너 출력 가로채기 (env 는 지역변수로만 쓴다) ----
    plan_box = {}
    orig_schedule = u.planner.schedule
    def schedule(*a, **kw):
        out = orig_schedule(*a, **kw)
        plan_box["ev_arm"] = out[5].clone()
        plan_box["ev_off"] = out[6].clone()
        plan_box["total"] = out[7]["total"].clone()
        return out
    u.planner.schedule = schedule

    sess = ort.InferenceSession(args_cli.onnx, providers=["CPUExecutionProvider"])
    print(f"[dump_golden] ONNX: {args_cli.onnx}")
    name_to_motor = dict(ri.robot_spec.joint_name_to_ik_result_idx)
    motor_by_obs = [int(name_to_motor[n]) for n in ri.ctrl_joint_names]

    obs_dict, _ = env.reset()
    start_q = f(ri.get_joint_pos(u.robot))
    drum_used = u.drum_pos[0].clone()                    # (M,3) env 프레임
    steps = []
    n_steps = T - 1 if args_cli.max_steps <= 0 else min(args_cli.max_steps, T - 1)
    wrist_prev = None                                    # 손목 속도 유한차분용 직전 위치 (리셋 뒤 첫 관측은 0)

    for i in range(n_steps):
        # ---- 이 시점의 원시 버퍼 (방금 반환된 obs 와 같은 시점) ----
        joint_pos = ri.get_joint_pos(u.robot)
        joint_vel = ri.get_joint_vel(u.robot)
        # 실기 분해능: env _get_observations 와 같은 식 (위치·속도 칸 버림, 손목 속도 유한차분)
        joint_pos = u._real_feedback_pos(joint_pos)
        if u.cfg.obs_vel_res > 0:
            v = joint_vel * u.real_sign
            v = torch.floor((v + u.cfg.mit_v_limit) / u.cfg.obs_vel_res) * u.cfg.obs_vel_res - u.cfg.mit_v_limit
            joint_vel = torch.where(u.res_mask, v * u.real_sign, joint_vel)
        if u.cfg.wrist_pos_res > 0:
            wq = joint_pos[:, u.wrist_ids]
            joint_vel = joint_vel.clone()
            joint_vel[:, u.wrist_ids] = torch.zeros_like(wq) if wrist_prev is None else (wq - wrist_prev) / u.step_dt
            wrist_prev = wq.clone()
        lead = u.target_q - joint_pos
        tip = ri.get_body_pos(u.robot).clone()           # (1,2,3) env 프레임
        drum = u.drum_pos.clone()                        # (1,M,3) env 프레임
        tip[..., 2] -= z_off                             # -> 허리 기준
        drum[..., 2] -= z_off
        arm_target = torch.stack([al.target_l, al.target_r], dim=1)
        arm_next = torch.cat([F.one_hot(arm_target, M).float(), u.arm_t[..., None]], dim=-1)

        raw = {
            "joint_pos": f(joint_pos), "joint_vel": f(joint_vel), "lead": f(lead),
            "tip_pos": f(tip), "drum_pos": f(drum), "next_hits": f(u.next_hits),
            "arm_next": f(arm_next),
            # env 가 obs 에 넣은 허리 목표 [rad] 를 역정규화 (메인: θ0*, 22: θ0_d). 정규화는 그래프
            "waist_des": f(obs_dict["policy"][:, OFF_WD] * u.joint_half_range[0, al.j_waist] + u.joint_center[0, al.j_waist]),
            "waist_vel_des": f(obs_dict["policy"][:, OFF_WV] * u.joint_vel_scale),   # θ̇0_d [rad/s] 역정규화
            # 어깨 yaw 기준 θk_d [rad], θ̇k_d [rad/s] [오른, 왼] 역정규화 (env yaw_ctrl_ids 순서)
            "yaw_des": f(obs_dict["policy"][:, OFF_YD:OFF_YD + 2] * u.joint_half_range[0, u.yaw_ctrl_ids] + u.joint_center[0, u.yaw_ctrl_ids]),
            "yaw_vel_des": f(obs_dict["policy"][:, OFF_YV:OFF_YV + 2] * u.joint_vel_scale),
            "last_action": f(u.actions),                 # 방금 obs 에 들어간 직전 행동 (env.step 전이라 아직 안 바뀜)
        }
        feed = {k_: np.asarray(raw[k_], dtype=np.float32).reshape(SHAPES[k_]) for k_ in IN_NAMES}
        act = sess.run(["action_mean"], feed)[0].reshape(-1)

        jp_motor, jv_motor = [0.0] * 9, [0.0] * 9
        for k_, m in enumerate(motor_by_obs):
            jp_motor[m] = raw["joint_pos"][k_]
            jv_motor[m] = raw["joint_vel"][k_]

        state = {
            "target_q": f(u.target_q),
            "target_l": int(al.target_l[0]), "target_r": int(al.target_r[0]),
            "target_waist": float(al.target_waist[0]),
            "prev_drum": fi(al.prev_drum),
            "arm_drum": fi(u.arm_drum), "arm_t": f(u.arm_t), "arm_valid": fi(u.arm_valid),
            "last_hit_step": fi(u.last_hit_step),
            "virtual_hit": fi(u._virtual_hits().any(dim=-1)),      # 이번 스텝 가상 타격 [왼, 오른]
            "tip_vel": f(u._get_tip_velocity()),                  # (2,3) 팁 속도 [m/s] 플래너 입력 (바디 속도 기반, 허리 기준과 동일)
            "ev_arm": fi(plan_box["ev_arm"]) if plan_box else [],   # (K*M) 0 왼 / 1 오른 / -1
            "ev_off": fi(plan_box["ev_off"]) if plan_box else [],   # (K) 이벤트까지 남은 스텝
            "planner_total": float(plan_box["total"][0]) if plan_box else None,   # NaN 은 JSON 표준 밖 (C++ 파서가 거부)
        }

        steps.append({
            "step": i, "t_score": i * dt,
            "joint_pos_motor_order": jp_motor, "joint_vel_motor_order": jv_motor,
            "raw": raw,
            "obs": f(obs_dict["policy"]),
            "act": np.asarray(act, dtype=np.float32).tolist(),
            "state": state,
        })

        obs_dict, _, term, trunc, _ = env.step(torch.tensor(act.reshape(1, -1), dtype=torch.float32, device=u.device))
        if bool(term[0]) or bool(trunc[0]):
            print(f"[dump_golden] 에피소드 종료 (step {i})")
            break

    drum_w = drum_used.clone()
    drum_w[:, 2] -= z_off
    payload = {
        "_comment": "dump_golden.py 생성. 실기 obs_builder(플래너·정렬 상태기계) 요소별 대조용.",
        "task": args_cli.task, "onnx": args_cli.onnx, "seed": args_cli.seed,
        "song": str(song_path), "bpm": bpm,
        "song_events": song_events,                      # bin = round(t/step_dt), drums = 시뮬 드럼 인덱스
        "step_dt": dt, "z_offset": z_off,
        "max_lookahead_step": L, "num_hits": K, "episode_length_step": T,
        "motor_id_by_obs_index": motor_by_obs,
        "joint_order": list(ri.ctrl_joint_names),
        "drum_names": drum_names,
        "start_pair": list(START_PAIR),
        "start_joint_pos_obs_order": start_q,
        "drum_pos_waist_frame": drum_w.cpu().tolist(),   # 노이즈 0 이라 specs 와 같아야 함
        "shoulder_z_waist_frame_m": float(al.z_s) - z_off,
        "frame_note": "raw 의 tip_pos/drum_pos 는 허리 기준 (env 값에서 z_offset 을 뺌). C++ 이 만들어야 하는 값 그대로.",
        "num_steps": len(steps),
        "steps": steps,
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    size_mb = out.stat().st_size / (1024 * 1024)
    n_hit = sum(sum(s["state"]["virtual_hit"]) for s in steps)
    n_valid = sum(1 for s in steps if s["raw"]["next_hits"][M + 1] > 0.5)
    n_arm = sum(1 for s in steps if max(s["state"]["arm_valid"]) > 0)
    waist_changes = sum(1 for a, b in zip(steps, steps[1:]) if a["state"]["target_waist"] != b["state"]["target_waist"])
    print(f"\n[dump_golden] {len(steps)} 스텝 저장: {out}  ({size_mb:.1f} MB)")
    print(f"  가상 타격 {n_hit}회 (곡 이벤트 {len(song_events)}개), next_hits[0].valid 스텝 {n_valid}, 손 배정 있는 스텝 {n_arm}")
    print(f"  목표 허리 변경 {waist_changes}회")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
