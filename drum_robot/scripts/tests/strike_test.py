"""
[확인용 임시 스크립트] 드럼 물리 타격 테스트

확인이 끝나면 아래 두 폴더만 지우면 됨 (환경 코드·설정은 건드리지 않음)
    drum_robot/scripts/tests/
    logs/tests/

내용
    - 환경 4개, 오른손 스틱을 스네어 위에서 "놓고 치기"로 내려쳐서 스네어 마찰계수별 튕김 비교
      env별 스네어 재질: 마찰 FRICTIONS (정지·운동 같은 값), 반발 RESTITUTION
      나머지 드럼은 환경이 랜덤으로 넣은 값 그대로
    - 물리·재질 설정(contact_offset, 장면 기본 재질 등)은 환경 설정(drumrobot_cfg.py) 그대로 사용
    - 테스트용으로만 바꾸는 것 (측정 목적)
        * decimation 1: 물리 스텝(7.5ms)마다 기록
        * 관절을 절대 목표각으로 제어 (실기 호스트 PD와 같은 방식)
          환경의 행동 해석(현재각 + 행동 x pi x dt)으로는 손목이 스틱 무게를 못 버텨서,
          이 스크립트 안에서만 env._pre_physics_step 을 바꿔서 사용
        * 드럼 위치 노이즈 0, 시작 자세 고정 (오른손 스틱 끝이 스네어 10cm 위)
    - 놓고 치기: 팁이 RELEASE_HEIGHT 아래로 오면 손목 목표각을 그 자리에 고정해서,
      충돌 순간 손목 PD가 스틱을 헤드로 누르지 않고 관성으로만 부딪히게 함

확인할 것
    1) PhysX에서 읽은 스네어 재질 값이 FRICTIONS / RESTITUTION 과 같은지
    2) 마찰 0 env: 반발비율(충돌 직후 팁 속도 / (e x 충돌 직전 팁 속도)) ~= 1.0  -> 반발 버그 수정 확인
    3) 마찰이 클수록 튕김이 줄어드는지 -> 마찰 랜덤화(스틱 쪽 min 결합)가 실제로 적용되는지
    4) 최저 높이 > -1cm (헤드 관통 없음), 잘못된 접촉 0

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/strike_test.py
    옵션: --slow 0.05 (GUI 슬로모션, 스텝마다 대기 초 / 0이면 실시간), --headless
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="드럼 물리 타격 테스트")
parser.add_argument("--slow", type=float, default=0.05, help="GUI 슬로모션: 스텝마다 대기 시간 [s] (0이면 실시간)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Sim 실행 후에만 import 가능 ----
import csv
import os
import time
from datetime import datetime

import torch                                    # pyright: ignore[reportMissingImports]
from isaaclab.utils import math as math_utils   # pyright: ignore[reportMissingImports]

from drum_robot.tasks.drumrobot.drumrobot_cfg import DrumRobotEnvCfg
from drum_robot.tasks.drumrobot.drumrobot_env import DrumRobotEnv

# ---- 테스트 설정 ----
FRICTIONS = [0.0, 0.3, 0.6, 1.0]    # env별 스네어 마찰계수 (정지·운동 같은 값)
RESTITUTION = 0.85                  # 모든 env 스네어 반발계수

SNARE = 0                           # Instruments 순서상 스네어 인덱스
RIGHT = 1                           # 팔 인덱스 [왼손, 오른손]
INIT_PAIR = ("hihat", "snare")      # 시작 자세 (왼손, 오른손): 스틱 끝이 드럼 10cm 위

LIFT_ANGLE = 0.4        # 스틱 들어올리는 각도 [rad]
LIFT_TIME = 0.3         # 들어올리는 시간 [s]
SETTLE_TIME = 0.3       # 들어올린 뒤 대기 [s]
STRIKE_RATE = 6.0       # 내려칠 때 손목 목표각 이동 속도 [rad/s]
RELEASE_HEIGHT = 0.025  # 팁이 이 높이[m] 아래로 오면 손목 목표각 고정 (놓고 치기)
STRIKE_TIMEOUT = 1.0    # 내려치기 최대 시간 [s]
AFTER_CONTACT = 0.5     # 접촉 후 관찰 시간 [s]
FALSE_CONTACT_H = 0.03  # 이 높이[m]보다 위에서 접촉이 잡히면 잘못된 접촉으로 셈

# 관절 속도 표에 쓸 짧은 이름
JOINT_ABBR = {
    "waist_joint": "waist",
    "left_shoulder_1": "Lsh1", "left_shoulder_2": "Lsh2", "left_elbow": "Lelb",
    "right_shoulder_1": "Rsh1", "right_shoulder_2": "Rsh2", "right_elbow": "Relb",
    "left_wrist": "Lwr", "right_wrist": "Rwr",
}

OUT_DIR = os.path.join("logs", "tests", f"strike_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def make_env() -> DrumRobotEnv:
    cfg = DrumRobotEnvCfg()
    cfg.scene.num_envs = len(FRICTIONS)
    cfg.decimation = 1                      # 물리 스텝(7.5ms)마다 기록 (물리 dt는 그대로)
    cfg.sim.render_interval = 1
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    for c in (cfg.contact_L_cfg, cfg.contact_R_cfg):
        c.history_length = 1                # decimation 1에 맞춤
        c.debug_vis = True                  # GUI 접촉 표시
    cfg.viewer.eye = (3.5, -3.5, 2.5)       # 환경 4개가 보이도록 카메라 위치
    cfg.viewer.lookat = (0.0, 0.0, 0.6)

    m = cfg.sim.physics_material
    print(f"[test] 환경 설정 그대로 사용 - 장면 기본 재질(스틱): "
          f"마찰 {m.static_friction} ({m.friction_combine_mode}), 반발 {m.restitution} ({m.restitution_combine_mode})")
    return DrumRobotEnv(cfg)


def fix_test_conditions(env: DrumRobotEnv):
    # (a) 드럼 위치 노이즈 제거
    env.instruments.drum_noise_scale = 0.0

    # (b) 시작 자세 고정 + 관절 노이즈 제거
    init = env.robot_initializer
    k = init.cfg.drum_pairs.index(INIT_PAIR)
    init.pos_angle = init.pos_angle[k:k + 1]
    init.cfg.drum_pairs = [INIT_PAIR]
    init.cfg.joint_noise_scale = 0.0

    # (c) 스네어 재질 env별 고정 (env._randomize_drum_material 과 같은 방식으로 PhysX 재질 값 덮어씀)
    view = env.drums.root_physx_view
    N, M = env.num_envs, env.drums.num_objects
    mats = view.get_material_properties().reshape(M, N, -1, 3)     # [정지마찰, 동마찰, 반발], 드럼 우선 순서
    mu = torch.tensor(FRICTIONS).unsqueeze(-1)
    mats[SNARE, :, :, 0] = mu
    mats[SNARE, :, :, 1] = mu
    mats[SNARE, :, :, 2] = RESTITUTION
    view.set_material_properties(mats.reshape(M * N, -1, 3), torch.arange(M * N))
    env.drum_friction[:, SNARE] = torch.tensor(FRICTIONS, device=env.device)
    env.drum_restitution[:, SNARE] = RESTITUTION

    readback = view.get_material_properties().reshape(M, N, -1, 3)[SNARE, :, 0]     # (N, 3)
    print("[test] 스네어 재질 (설정값 -> PhysX에서 읽은 값)")
    for i, (s, d, e) in enumerate(readback.tolist()):
        print(f"    env {i}: 마찰 {FRICTIONS[i]:.2f} -> 정지 {s:.2f} / 운동 {d:.2f},  반발 {RESTITUTION:.2f} -> {e:.2f}")


def use_absolute_joint_targets(env: DrumRobotEnv):
    # 테스트에서만: 행동 = 절대 목표각 (로봇 좌표계, ctrl 관절 순서)
    def _pre_physics_step(actions):
        env.actions = actions.clone()
        env.target_joint_pos = env.robot_interface.clip_and_convert(actions)
    env._pre_physics_step = _pre_physics_step


def step(env: DrumRobotEnv, q_des: torch.Tensor):
    env.step(q_des)
    if not args_cli.headless and args_cli.slow > 0:
        time.sleep(args_cli.slow)


def measure(env: DrumRobotEnv):
    tip = env.robot_interface.get_body_pos(env.robot)[:, RIGHT]            # (N, 3) env 좌표
    h = tip[:, 2] - env.drum_pos[:, SNARE, 2]                               # 스네어 윗면 기준 팁 높이
    force = env.contact_R.data.force_matrix_w[:, 0, SNARE].norm(dim=-1)    # 오른손-스네어 접촉력
    contact = env._get_drum_contact()[:, RIGHT, SNARE]                     # 환경 함수 그대로 사용
    return h, force, contact


def tip_vz(env: DrumRobotEnv) -> torch.Tensor:
    # 순간 팁 속도 z (스텝 끝 시점): 손목 링크 속도 + 각속도 x (링크 원점 -> 팁)
    idx = env.robot_interface.body_to_idx["right_wrist"]
    v = env.robot.data.body_link_lin_vel_w[:, idx]
    w = env.robot.data.body_link_ang_vel_w[:, idx]
    r = math_utils.quat_apply(env.robot.data.body_link_quat_w[:, idx], env.robot_interface.tip_offset_R)
    return (v + torch.cross(w, r, dim=-1))[:, 2]


def find_wrist_down_sign(env: DrumRobotEnv, q_hold: torch.Tensor, w: int) -> float:
    # 손목 각도를 + 방향으로 움직여 보고 팁이 내려가면 + 가 "아래"
    h0, _, _ = measure(env)
    q = q_hold.clone()
    q[:, w] += 0.2
    for _ in range(20):
        step(env, q)
    h1, _, _ = measure(env)
    sign = 1.0 if (h1 - h0).mean().item() < 0 else -1.0
    print(f"[test] 손목 방향 확인: 각도 +0.2rad -> 팁 높이 변화 {(h1 - h0).mean().item() * 100:+.1f}cm -> 아래 방향 부호 {sign:+.0f}")
    return sign


def analyze(series: dict, contact_step: torch.Tensor, release_step: torch.Tensor, w: int, dt: float) -> list:
    # series[name]: (steps, N), qd: (steps, N, 9)
    h, vtip, contact, qd = series["h"], series["vtip"], series["contact"], series["qd"]
    after = round(AFTER_CONTACT / dt)
    results = []
    for i, mu in enumerate(FRICTIONS):
        k0 = int(contact_step[i])
        kr = int(release_step[i])
        r = {
            "env": i, "mu": mu, "contact": k0 >= 0,
            "h_min": h[:, i].min().item(),
            "false": int((contact[:, i] & (h[:, i] > FALSE_CONTACT_H)).sum()),
            "t_release": kr * dt if kr >= 0 else float("nan"),
        }
        if k0 >= 1:
            v_in = -vtip[k0 - 1, i].item()      # 충돌 직전 순간 팁 속도 (아래 방향 +)
            v_out = vtip[k0, i].item()          # 접촉 스텝 끝 순간 팁 속도 (위 방향 +)
            w_in = qd[k0 - 1, i, w].item()      # 충돌 직전 손목 각속도
            w_out = qd[k0, i, w].item()         # 접촉 스텝 끝 손목 각속도
            r.update(
                t_contact=k0 * dt,
                v_in=v_in, v_out=v_out,
                ratio=v_out / (RESTITUTION * v_in) if v_in > 1e-6 else float("nan"),
                w_in=w_in, w_out=w_out,
                w_ratio=-w_out / w_in if abs(w_in) > 1e-6 else float("nan"),
                rebound=(h[k0:k0 + after, i].max() - h[k0, i]).item(),     # 접촉 후 올라간 높이
            )
        results.append(r)
    return results


def print_table(results: list):
    print(f"\n[test] 결과 (오른손 -> 스네어, 놓고 치기, 스네어 반발 {RESTITUTION}, 손목 실제 게인)")
    print(" env  마찰  접촉  접촉시각  충돌속도  튕김속도  반발비율  손목(직전->직후)  손목비율  튕김높이  최저높이  잘못된접촉")
    for r in results:
        if "v_in" not in r:
            print(f"  {r['env']}   {r['mu']:.2f}   X   (접촉 없음)  최저높이 {r['h_min'] * 100:.2f}cm  잘못된접촉 {r['false']}")
            continue
        print(f"  {r['env']}   {r['mu']:.2f}   O   {r['t_contact']:6.3f}s  {r['v_in']:5.2f}m/s  {r['v_out']:+5.2f}m/s  "
              f"{r['ratio']:6.2f}   {r['w_in']:+6.2f} -> {r['w_out']:+6.2f}   {r['w_ratio']:6.2f}  "
              f"{r['rebound'] * 100:6.2f}cm  {r['h_min'] * 100:6.2f}cm  {r['false']:4d}")
    print(f"  * 충돌속도·튕김속도: 순간 팁 속도 (충돌 직전 아래 방향 / 접촉 스텝 끝 위 방향)")
    print(f"  * 반발비율 = 튕김속도 / (e x 충돌속도). 마찰 0 env가 ~1.0 이면 반발 그대로 적용 (반발 버그 수정 OK)")
    print(f"  * 손목비율 = 손목 각속도 (직후 / 직전). 마찰 0 이면 ~e({RESTITUTION})")
    print(f"  * 마찰이 클수록 튕김속도·튕김높이가 줄어들면 마찰 랜덤화가 적용되는 것")
    print(f"  * 최저높이가 -1cm 아래면 헤드 관통 의심, 잘못된접촉은 0이어야 정상\n")


def print_joint_table(series: dict, contact_step: torch.Tensor, joint_names: list, dt: float):
    # 접촉 전후 순간 팁 속도와 관절 9개 속도 (로봇 좌표계, rad/s)
    abbr = [JOINT_ABBR.get(n, n) for n in joint_names]
    for i, mu in enumerate(FRICTIONS):
        k0 = int(contact_step[i])
        if k0 < 1:
            continue
        print(f"[test] 접촉 전후 관절 속도 (env {i}, 마찰 {mu:.2f}, 접촉 스텝 = *)")
        print("    t      h[cm]  팁vz[m/s] 접촉 " + "".join(f"{a:>7s}" for a in abbr))
        for k in range(k0 - 1, min(k0 + 4, series["h"].shape[0])):
            mark = "*" if k == k0 else " "
            qd = "".join(f"{v:7.2f}" for v in series["qd"][k, i].tolist())
            print(f"  {mark}{k * dt:.4f} {series['h'][k, i].item() * 100:6.2f} {series['vtip'][k, i].item():8.2f}"
                  f"   {int(series['contact'][k, i].item())}  {qd}")
        print()


def save_outputs(series: dict, joint_names: list, dt: float):
    os.makedirs(OUT_DIR, exist_ok=True)
    steps, N = series["h"].shape

    with open(os.path.join(OUT_DIR, "strike.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "env", "mu", "e", "h", "vz", "vtip", "force", "contact", "q_wrist", "q_wrist_des"]
                        + [f"qd_{JOINT_ABBR.get(n, n)}" for n in joint_names])
        for k in range(steps):
            for i in range(N):
                writer.writerow([
                    f"{k * dt:.4f}", i, FRICTIONS[i], RESTITUTION,
                    f"{series['h'][k, i].item():.5f}", f"{series['vz'][k, i].item():.4f}",
                    f"{series['vtip'][k, i].item():.4f}",
                    f"{series['force'][k, i].item():.4f}", int(series["contact"][k, i].item()),
                    f"{series['q_w'][k, i].item():.4f}", f"{series['q_w_des'][k, i].item():.4f}",
                ] + [f"{v:.4f}" for v in series["qd"][k, i].tolist()])

    import matplotlib                   # pyright: ignore[reportMissingImports]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt     # pyright: ignore[reportMissingImports]

    t = [k * dt for k in range(steps)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, mu in enumerate(FRICTIONS):
        label = f"mu={mu:.2f}"
        axes[0].plot(t, (series["h"][:, i] * 100).tolist(), label=label)
        axes[1].plot(t, series["vtip"][:, i].tolist(), label=label)
        axes[2].plot(t, series["force"][:, i].tolist(), label=label)
    axes[0].axhline(0.0, color="gray", lw=0.8, ls="--")
    axes[0].set_title(f"snare restitution={RESTITUTION}, wrist real gains, release-stroke")
    axes[0].set_ylabel("tip height above snare [cm]")
    axes[1].set_ylabel("tip vz (instant) [m/s]")
    axes[2].set_ylabel("contact force [N]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "strike.png"), dpi=120)
    print(f"[test] 저장: {OUT_DIR}/strike.csv, strike.png")


def main():
    env = make_env()
    fix_test_conditions(env)
    use_absolute_joint_targets(env)

    joint_names = env.robot_interface.get_ctrl_joint_name()
    w = joint_names.index("right_wrist")
    dt = env.step_dt
    N = env.num_envs

    # 손목 "아래" 방향 찾기
    env.reset()
    q_hold = env.robot_interface.get_joint_pos(env.robot).clone()
    sign = find_wrist_down_sign(env, q_hold, w)

    # 본 테스트: 시작 자세로 리셋 후 나머지 관절은 고정, 오른손목만 들었다가 내려치기
    env.reset()
    q_hold = env.robot_interface.get_joint_pos(env.robot).clone()
    q_w0 = q_hold[:, w].clone()
    q_lift = q_w0 - sign * LIFT_ANGLE
    strike_start = LIFT_TIME + SETTLE_TIME
    max_steps = round((strike_start + STRIKE_TIMEOUT + AFTER_CONTACT) / dt)
    after_steps = round(AFTER_CONTACT / dt)

    contact_step = torch.full((N,), -1, dtype=torch.long, device=env.device)
    release_step = torch.full((N,), -1, dtype=torch.long, device=env.device)
    q_w_release = torch.zeros(N, device=env.device)
    q_w_frozen = torch.zeros(N, device=env.device)
    log = {name: [] for name in ("h", "vz", "vtip", "force", "contact", "q_w", "q_w_des", "qd")}
    h_prev, _, _ = measure(env)

    for k in range(max_steps):
        t = k * dt
        if t < LIFT_TIME:
            q_w_des = q_w0 - sign * LIFT_ANGLE * (t / LIFT_TIME)               # 들어올리기
        elif t < strike_start:
            q_w_des = q_lift.clone()                                           # 대기
        else:
            q_w_des = q_lift + sign * STRIKE_RATE * (t - strike_start)         # 내려치기
        q_w_des = torch.where(release_step >= 0, q_w_release, q_w_des)         # 놓은 env는 놓은 순간 각도로 고정
        q_w_des = torch.where(contact_step >= 0, q_w_frozen, q_w_des)          # 닿은 env는 닿은 순간 각도로 고정

        q_des = q_hold.clone()
        q_des[:, w] = q_w_des
        step(env, q_des)

        h, force, contact = measure(env)
        q_w = env.robot_interface.get_joint_pos(env.robot)[:, w]
        qd = env.robot_interface.get_joint_vel(env.robot)          # (N, 9) 로봇 좌표, ctrl 관절 순서
        vtip = tip_vz(env)                                         # (N,) 순간 팁 속도 z
        vz = (h - h_prev) / dt
        h_prev = h

        # 내려치는 중 팁이 RELEASE_HEIGHT 아래로 오면 손목을 놓음 (그 순간 실제 각도로 목표 고정)
        if t >= strike_start:
            new_release = (h < RELEASE_HEIGHT) & (release_step < 0) & (contact_step < 0)
            release_step[new_release] = k
            q_w_release[new_release] = q_w[new_release]

        new_hit = contact & (contact_step < 0)
        contact_step[new_hit] = k
        q_w_frozen[new_hit] = q_w[new_hit]      # 닿은 순간의 실제 손목 각도에서 고정

        for name, val in (("h", h), ("vz", vz), ("vtip", vtip), ("force", force), ("contact", contact),
                          ("q_w", q_w), ("q_w_des", q_w_des), ("qd", qd)):
            log[name].append(val.detach().clone())

        if (contact_step >= 0).all() and k - int(contact_step.max()) >= after_steps:
            break

    series = {name: torch.stack(vals).cpu() for name, vals in log.items()}
    results = analyze(series, contact_step.cpu(), release_step.cpu(), w, dt)
    print_table(results)
    print_joint_table(series, contact_step.cpu(), joint_names, dt)
    save_outputs(series, joint_names, dt)

    # GUI면 창을 닫을 때까지 마지막 장면 유지
    if not args_cli.headless:
        print("[test] 완료. 창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
