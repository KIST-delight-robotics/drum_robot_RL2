"""
[확인용 임시 스크립트] drum_robot.usd 로봇만 띄워서 허리만 움직여 보기 (GUI)
학습과 같은 로봇 설정 사용 (drumrobot_cfg.py 의 robot_cfg: USD, Kp/Kd/armature, 물리 7.5 ms), 환경·드럼·정책 없음
    시작   : PhysX가 실제로 쓰는 질량·izz 출력 (허리, shoulder_2, elbow) + 허리 Kp / Kd / armature
    움직임 : 허리 목표만 바꿈, 나머지 관절 목표는 0° 고정
             step = 0 -> +amp -> -amp 를 period_s 마다 바꾸며 반복, sine = amp·sin(2π t / period_s)
    터미널 : 0.1초마다 허리 목표 / 실제 / 오차 [deg]
    실시간 속도로 재생 (창을 닫으면 끝)
실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/waist_move.py
    옵션: --mode step|sine, --amp_deg 30, --period_s 2.0
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="허리만 움직여 보기")
parser.add_argument("--mode", choices=["step", "sine"], default="step", help="step: 계단, sine: 사인파")
parser.add_argument("--amp_deg", type=float, default=30.0, help="허리를 돌리는 크기 [deg]")
parser.add_argument("--period_s", type=float, default=2.0, help="계단 하나 유지 시간 / 사인파 주기 [s]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Sim 실행 후에만 import 가능 ----
import math
import time

import isaaclab.sim as sim_utils                # pyright: ignore[reportMissingImports]
from isaaclab.assets import Articulation        # pyright: ignore[reportMissingImports]

from drum_robot.tasks.drumrobot.drumrobot_cfg import SIM_DT, DrumRobotEnvCfg

BODIES = ("waist", "left_shoulder_2", "right_shoulder_2", "left_elbow", "right_elbow")   # URDF_MOD 에서 바꾼 링크


def main():
    env_cfg = DrumRobotEnvCfg()
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    sim = sim_utils.SimulationContext(env_cfg.sim)
    sim.set_camera_view(eye=(1.8, -1.8, 2.0), target=(0.0, 0.0, 1.0))

    sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)
    robot = Articulation(env_cfg.robot_cfg.replace(prim_path="/World/Robot"))   # 학습과 같은 USD·액추에이터, 로봇 1대
    sim.reset()

    # PhysX가 실제로 쓰는 값 (USD 를 읽은 결과)
    masses = robot.root_physx_view.get_masses()[0]          # (링크,)
    inertias = robot.root_physx_view.get_inertias()[0]      # (링크, 9) 3x3 관성 행렬
    print(f"\n[waist_move] USD: {env_cfg.robot_cfg.spawn.usd_path}")
    for name in BODIES:
        b = robot.body_names.index(name)
        print(f"[waist_move] {name:17s} 질량 {masses[b].item():.5f} kg, izz {inertias[b, 8].item():.6g} kg·m²")
    jw = robot.joint_names.index("waist_joint")
    print(f"[waist_move] 허리 Kp {robot.data.joint_stiffness[0, jw].item():g}, Kd {robot.data.joint_damping[0, jw].item():g}, "
          f"armature {robot.data.joint_armature[0, jw].item():g}")
    print(f"[waist_move] 움직임: {args_cli.mode}, 크기 {args_cli.amp_deg}°, 주기 {args_cli.period_s} s\n")

    target = robot.data.default_joint_pos.clone()           # (1, 관절) 모두 0
    amp = math.radians(args_cli.amp_deg)
    steps = (0.0, amp, -amp)                                # 계단 순서
    print_every = round(0.1 / SIM_DT)
    deg = math.degrees

    t, k = 0.0, 0
    t0 = time.time()
    while simulation_app.is_running():
        if args_cli.mode == "step":
            w = steps[int(t // args_cli.period_s) % len(steps)]
        else:
            w = amp * math.sin(2 * math.pi * t / args_cli.period_s)
        target[:, jw] = w
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        t += SIM_DT
        k += 1

        if k % print_every == 0:
            q = robot.data.joint_pos[0, jw].item()
            print(f"{t:6.2f}s | 허리 목표 {deg(w):+6.1f}°  실제 {deg(q):+6.1f}°  오차 {deg(w - q):+5.1f}°")

        lag = t - (time.time() - t0)                        # 실시간보다 앞서 있으면 기다림
        if lag > 0:
            time.sleep(lag)


if __name__ == "__main__":
    main()
    simulation_app.close()
