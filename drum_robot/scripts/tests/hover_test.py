"""
[확인용 임시 스크립트] 배정 안 된 팔이 직전 드럼 위에 살짝 떠서 가만히 있는지 GUI 확인

학습된 모델이 없으므로 이 스크립트가 행동(-1~1)을 직접 넣음 (환경의 행동 처리는 그대로 사용)
확인이 끝나면 drum_robot/scripts/tests/ 폴더를 지우면 됨 (환경 코드는 건드리지 않음)

시나리오 (양손의 '직전 드럼' = 시작 자세의 드럼)
    1) 자리 잡기 1초    : 양손 스틱 팁을 각자 직전 드럼 위 --hover cm 간격으로 맞춤 (손목만 움직임)
    2) 오른손만 치기 x3 : 오른손 = 배정된 팔 (손목 들기 -> 내려치기 -> 다시 떠 있기), 왼손 = 배정 안 된 팔 (행동 0)
    3) 둘 다 대기 2초   : 양손 행동 0

볼 것
    왼손 스틱이 드럼에 닿지 않고 --hover cm 위에 떠서 움직이지 않는지 (오른손이 치는 동안에도)
    화면: 타격하면 그 손 팁에 빨간 공 (환경의 Visualizer를 이 스크립트에서만 켬)
    터미널: 양손 팁과 직전 드럼 윗면 사이 간격(0 = 닿음), 접촉, 타격 / 단계마다 왼손 결과 요약
    (이 스크립트에서만 에피소드 길이를 10초로 늘림: 시나리오 도중 자동 리셋 방지)

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/hover_test.py
    옵션: --hover 3.0 (떠 있을 높이 cm), --slow 0.03 (스텝마다 대기 초)
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="배정 안 된 팔 대기 자세 GUI 확인")
parser.add_argument("--hover", type=float, default=3.0, help="떠 있을 간격 [cm] (스틱 표면 - 드럼 윗면)")
parser.add_argument("--slow", type=float, default=0.03, help="스텝마다 대기 시간 [s] (0이면 최대 속도)")
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
L, R = 0, 1                         # 손 순서 [왼손, 오른손]
LIFT_DEG = 40.0                     # 오른손이 칠 때 드는 손목 각도 [deg]
STICK_TIP = 0.385                   # 손목 축 -> 팁 거리 [m] (높이 맞추기 계산용)
STICK_RADIUS = 0.0075               # 스틱 캡슐 반지름 [m] (팁 끝점 높이 -> 표면 간격)
HOVER_GAIN = 0.1                    # 높이 맞추기: 스텝마다 오차의 10%씩 (크면 손목이 못 따라가서 지나침)


def make_env() -> DrumRobotEnv:
    # 타격 공 표시 켜기 (환경 생성 때 VisualizerCfg()를 부르므로 이 스크립트에서만 바꿔치기)
    drumrobot_env_module.VisualizerCfg = lambda: VisualizerCfg(enable_visualization=True)

    cfg = DrumRobotEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 10.0             # 이 스크립트에서만: 시나리오 도중 자동 리셋 방지
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    cfg.viewer.eye = (0.9, -1.1, 1.9)       # 로봇 뒤쪽 위에서 드럼 쪽을 봄
    cfg.viewer.lookat = (0.0, 0.5, 0.6)
    return DrumRobotEnv(cfg)


class Runner:
    def __init__(self, env: DrumRobotEnv):
        self.env = env
        names = env.robot_interface.get_ctrl_joint_name()
        self.jw = [names.index("left_wrist"), names.index("right_wrist")]
        self.dt = env.step_dt
        self.wrist_scale = env.action_scale_vec[0, self.jw]
        self.t = 0
        empty = torch.zeros((2, len(NAMES)), dtype=torch.bool, device=env.device)
        self.last = {"hit": empty, "contact": empty}

        # 타격·접촉 결과 가로채기 (환경 코드는 그대로)
        orig = env.hit_detector.step
        def wrapped(contact_force, steps):
            r = orig(contact_force, steps)
            self.last["hit"] = r.hit_per_arm[0]         # (2, M)
            self.last["contact"] = r.contact[0]         # (2, M)
            return r
        env.hit_detector.step = wrapped

    def wrist_step(self, a_left, a_right):
        # 손목만 행동, 나머지 관절은 0 (목표 유지)
        a = torch.zeros((1, 9), device=self.env.device)
        a[0, self.jw[L]] = float(a_left)
        a[0, self.jw[R]] = float(a_right)
        self.env.step(a)
        self.t += 1
        if args_cli.slow > 0:
            time.sleep(args_cli.slow)

    def wrist_q(self):
        return self.env.robot_interface.get_joint_pos(self.env.robot)[0, self.jw]   # (2,)

    def height(self):
        # 손별 스틱 표면과 직전 드럼 윗면 사이 간격 [m] (0이면 닿음)
        env = self.env
        tip = env.robot_interface.get_body_pos(env.robot)[0]                # (2, 3)
        drum = env.reward_computer.alignment.prev_drum[0]                           # (2,) [왼손, 오른손]
        return tip[:, 2] - STICK_RADIUS - env.drum_pos[0, drum, 2]

    def tip(self):
        return self.env.robot_interface.get_body_pos(self.env.robot)[0].clone()

    def hover_action(self, hand: int, goal_m: float) -> float:
        # 팁 간격을 goal로 맞추는 손목 행동 (오차의 HOVER_GAIN씩 보정, 목표 누적이 중력 처짐까지 보정)
        err = goal_m - self.height()[hand].item()
        dq = HOVER_GAIN * err / STICK_TIP
        return max(-1.0, min(1.0, dq / (self.wrist_scale[hand].item() * self.dt)))

    def wrist_to_action(self, hand: int, goal_q: float) -> float:
        # 손목 목표를 goal_q로 옮기는 행동
        dq = goal_q - self.env.target_q[0, self.jw[hand]].item()
        return max(-1.0, min(1.0, dq / (self.wrist_scale[hand].item() * self.dt)))

    @staticmethod
    def drums(mask) -> str:
        names = [NAMES[m] for m in range(len(NAMES)) if mask[m]]
        return "+".join(names) if names else "-"

    def print_row(self, label: str):
        h = self.height() * 100
        c, hit = self.last["contact"], self.last["hit"]
        line = (f"  {self.t * self.dt:4.2f}s {label:<10s} | 드럼과 간격 L {h[L]:+5.1f} R {h[R]:+5.1f} cm "
                f"| 접촉 L {self.drums(c[L]):<6s} R {self.drums(c[R]):<6s}")
        if hit.any():
            line += f" | 타격 L {self.drums(hit[L])} R {self.drums(hit[R])}"
        print(line)


class LeftWatch:
    # 왼손(배정 안 된 팔) 기록: 높이, 움직임, 접촉
    def __init__(self, r: Runner):
        self.r = r
        self.tip0 = r.tip()[L]
        self.h, self.move, self.touch = [], [], 0

    def update(self):
        self.h.append(self.r.height()[L].item() * 100)
        self.move.append((self.r.tip()[L] - self.tip0).norm().item() * 100)
        self.touch += int(self.r.last["contact"][L].any())

    def report(self, label: str):
        print(f"[{label}] 왼손 결과: 드럼과 간격 {min(self.h):+.1f} ~ {max(self.h):+.1f} cm, "
              f"처음 위치에서 최대 {max(self.move):.2f} cm 움직임, 드럼에 닿은 스텝 {self.touch}번")


def main():
    env = make_env()
    r = Runner(env)
    env.reset()
    goal = args_cli.hover / 100.0
    prev = env.reward_computer.alignment.prev_drum[0].tolist()
    print(f"[hover] 직전 드럼: 왼손 {NAMES[prev[L]]}, 오른손 {NAMES[prev[R]]} / 떠 있을 간격 {args_cli.hover:.1f} cm")

    # 1) 자리 잡기: 양손 팁을 직전 드럼 위 goal 간격으로
    print(f"\n[1 자리 잡기] 양손 팁을 드럼 위 {args_cli.hover:.1f} cm 간격으로")
    for k in range(round(1.0 / r.dt)):
        r.wrist_step(r.hover_action(L, goal), r.hover_action(R, goal))
        if k % round(0.2 / r.dt) == 0:
            r.print_row("자리잡기")
    r.print_row("자리잡기")

    # 2) 오른손만 치기 (왼손 행동 0)
    print("\n[2 오른손만 치기] 오른손 = 배정된 팔, 왼손 = 배정 안 된 팔 (행동 0)")
    watch = LeftWatch(r)
    q_hover = r.wrist_q()[R].item()
    for s in range(3):
        for _ in range(round(0.25 / r.dt)):                                         # 들기
            r.wrist_step(0.0, r.wrist_to_action(R, q_hover + math.radians(LIFT_DEG))); watch.update()
        r.print_row("오른손 듦")
        for _ in range(round(0.3 / r.dt)):                                          # 내려치기
            r.wrist_step(0.0, -1.0); watch.update()
            if r.last["hit"][R].any():
                r.print_row("오른손 침")
                break
        for _ in range(round(0.15 / r.dt)):                                         # 힘 빼기 (목표 = 지금 각도)
            r.wrist_step(0.0, r.wrist_to_action(R, r.wrist_q()[R].item())); watch.update()
    for _ in range(round(0.6 / r.dt)):                                              # 오른손도 다시 떠 있기
        r.wrist_step(0.0, r.hover_action(R, goal)); watch.update()
    r.print_row("오른손 복귀")
    watch.report("2 오른손만 치기")

    # 3) 둘 다 대기 (행동 0)
    print("\n[3 둘 다 대기] 양손 행동 0")
    watch = LeftWatch(r)
    for k in range(round(2.0 / r.dt)):
        r.wrist_step(0.0, 0.0); watch.update()
        if k % round(0.25 / r.dt) == 0:
            r.print_row("대기")
    watch.report("3 둘 다 대기")

    if not args_cli.headless:
        print("\n[hover] 완료. 창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            env.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
