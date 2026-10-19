"""
[확인용 임시 스크립트] 허리·어깨 yaw 정렬 보상 GUI 확인

설계한 정렬 보상이 맞는지 확인: 보상이 계산한 정답 각도(목표 허리 θ0*, 목표 어깨 yaw θ1*, θ2*)를
정책 대신 이 스크립트가 직접 관절 목표로 넣어서 로봇을 움직이고, 그때 보상이 어떻게 나오는지 봄
확인이 끝나면 drum_robot/scripts/tests/ 폴더를 지우면 됨 (환경 코드는 건드리지 않음)

모드
    follow (기본): 허리 -> θ0*, 어깨 yaw -> θ1*, θ2* (현재 허리 기준), 나머지 관절은 시작 자세 유지
                   -> 다음에 칠 드럼(노랑)이 바뀔 때 로봇이 그쪽으로 도는지, 보상이 0 근처인지
    hold         : 모든 관절을 시작 자세로 고정
                   -> 어긋난 만큼 보상이 음수인지, 타격이 가까울수록 커지는지(시간 가중)

화면
    곧 칠 드럼은 노랑, 1초 안에 칠 드럼은 어두운 노랑 (환경의 Visualizer를 이 스크립트에서만 켬)
터미널
    목표가 바뀌거나 0.25초마다: 다음 타격, 손 배정, 목표/실제 허리, 어깨 yaw 오차, 보상

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/align_test.py
    옵션: --mode follow|hold, --episodes 3, --slow 0.02 (스텝마다 대기 초)
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="정렬 보상 GUI 확인")
parser.add_argument("--mode", choices=["follow", "hold"], default="follow")
parser.add_argument("--episodes", type=int, default=3, help="확인할 에피소드 수 (에피소드 5초)")
parser.add_argument("--slow", type=float, default=0.02, help="스텝마다 대기 시간 [s] (0이면 최대 속도)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Sim 실행 후에만 import 가능 ----
import math
import time

import torch                    # pyright: ignore[reportMissingImports]

import drum_robot.tasks.drumrobot.drumrobot_env as drumrobot_env_module
from drum_robot.tasks.drumrobot.drumrobot_cfg import DrumRobotEnvCfg
from drum_robot.tasks.drumrobot.drumrobot_env import DrumRobotEnv
from drum_robot.tasks.drumrobot.components.visualizer import VisualizerCfg
from drum_robot.tasks.drumrobot.components.specs import Instruments

NAMES = list(Instruments().all.keys())


def make_env() -> DrumRobotEnv:
    # 드럼 색 표시 켜기 (환경 생성 때 VisualizerCfg()를 부르므로 이 스크립트에서만 바꿔치기)
    drumrobot_env_module.VisualizerCfg = lambda: VisualizerCfg(enable_visualization=True)

    cfg = DrumRobotEnvCfg()
    cfg.scene.num_envs = 1
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    cfg.viewer.eye = (0.9, -1.1, 1.9)       # 로봇 뒤쪽 위에서 드럼 쪽을 봄
    cfg.viewer.lookat = (0.0, 0.5, 0.6)
    return DrumRobotEnv(cfg)


def use_absolute_joint_targets(env: DrumRobotEnv):
    # 테스트에서만: 행동 = 절대 목표각 (로봇 좌표계, ctrl 관절 순서)
    def _pre_physics_step(actions):
        env.actions = actions.clone()
        env.target_joint_pos = env.robot_interface.clip_and_convert(actions)
        env._substep = 0                                    # 환경의 원래 함수가 하던 물리 스텝별 접촉 지점 기록 초기화
        env.contact_from_tip.fill_(float("nan"))
    env._pre_physics_step = _pre_physics_step


def yaw_targets(env: DrumRobotEnv, q0: torch.Tensor):
    # 정렬 보상과 같은 식: 현재 허리 q0 기준 목표 어깨 yaw
    ar = env.reward_computer.alignment
    idx = torch.arange(env.num_envs, device=env.device)
    pR = env.drum_pos[idx, ar.target_r, :2]
    pL = env.drum_pos[idx, ar.target_l, :2]
    th1 = torch.atan2(pR[:, 1] - ar.l * torch.sin(q0), pR[:, 0] - ar.l * torch.cos(q0)) - q0
    th2 = torch.atan2(pL[:, 1] + ar.l * torch.sin(q0), pL[:, 0] + ar.l * torch.cos(q0)) - q0
    return th1, th2


def main():
    env = make_env()
    use_absolute_joint_targets(env)
    ar = env.reward_computer.alignment
    jw, jr, jl = ar.j_waist, ar.j_r, ar.j_l
    dt = env.step_dt
    M = len(NAMES)

    # 보상 값 가로채기 (환경 코드는 그대로)
    last = {}
    orig_compute = ar.compute
    def compute(*a, **kw):
        reward, log = orig_compute(*a, **kw)
        last["reward"], last["log"] = reward, log
        return reward, log
    ar.compute = compute

    print(f"[align] 모드 = {args_cli.mode}")
    for ep in range(args_cli.episodes):
        env.reset()
        q_hold = env.robot_interface.get_joint_pos(env.robot).clone()
        print(f"\n[align] 에피소드 {ep + 1}: 시작 자세 왼손 {NAMES[ar.prev_drum[0, 0]]}, 오른손 {NAMES[ar.prev_drum[0, 1]]}")
        print("   시각         다음 타격 (남은 시간)  ->  배정 오른손 / 왼손   목표 허리  실제 허리  yaw 오차    보상")

        prev_target = None
        for k in range(env.episode_length_step - 2):
            # 관절 명령: follow면 허리·어깨 yaw를 보상의 정답 각도로, 나머지는 시작 자세
            q = env.robot_interface.get_joint_pos(env.robot)
            q_des = q_hold.clone()
            if args_cli.mode == "follow":
                th1, th2 = yaw_targets(env, q[:, jw])
                q_des[:, jw] = ar.target_waist
                q_des[:, jr] = th1
                q_des[:, jl] = th2
            env.step(q_des)
            if args_cli.slow > 0:
                time.sleep(args_cli.slow)

            # 출력: 목표가 바뀌거나 0.25초마다
            nh = env.rds.get_next_hits(env.steps)
            drums = [NAMES[m] for m in range(M) if nh[0, 0, m] > 0.5 and nh[0, 0, M + 1] > 0.5]
            t_next = nh[0, 0, M].item()
            target = (ar.target_r[0].item(), ar.target_l[0].item())
            if target != prev_target or k % round(0.25 / dt) == 0:
                log = last["log"]
                q0 = env.robot_interface.get_joint_pos(env.robot)[0, jw].item()
                next_str = f"{'+'.join(drums) if drums else '(1초 안에 없음)'} ({t_next:.2f}s)"
                mark = "  <- 목표 바뀜" if target != prev_target and prev_target is not None else ""
                print(f"  {k * dt:5.2f}s  {next_str:>24s}  ->  {NAMES[target[0]]:>7s} / {NAMES[target[1]]:<7s}  "
                      f"{math.degrees(ar.target_waist[0].item()):+7.1f}°  {math.degrees(q0):+7.1f}°  "
                      f"{log['align_yaw_err_deg'][0].item():6.1f}°  {last['reward'][0].item():+.3f}{mark}")
            prev_target = target

    if not args_cli.headless:
        print("\n[align] 완료. 창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            env.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
