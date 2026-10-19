"""
[확인용 임시 스크립트] 드럼 뚫림(터널링) 대책 GUI 확인: contact_offset / 북 두께

레포 설정은 그대로 두고, 이 스크립트 안에서만 아래 값을 덮어써서 비교
    --offset      드럼 8개 contact_offset [m]                         (레포 값 0.01)
    --drum_thick  북 4개(snare, floor, mid, high) 두께 [m]             (레포 값 0.015)
                  윗면 높이는 그대로 두고 아래로만 두꺼워짐 (심벌 4개는 0.015 그대로)
덮어쓰는 곳 (레포 값을 바꾸면 전달되는 곳과 같음)
    cfg.drums_cfg          : contact_offset, 원판 두께(spawn.height), 처음 위치(init_state.pos)
    env.drum_half_thickness: 리셋 때 원판 위치 계산
확인이 끝나면 drum_robot/scripts/tests/ 폴더를 지우면 됨 (환경 코드는 건드리지 않음)

순서 (0은 처음 한 번, 1~2는 창을 닫을 때까지 반복)
    0) 설정 확인 : 시뮬레이션에 실제로 들어간 원판 두께·contact_offset, 리셋 뒤 윗면 높이가 바꾸기 전과 같은지
    1) 뚫림      : 손목을 들었다가 최대로 내려치고 0.3초 누르기 x4번 -> 원판 밑으로 들어간 손을 둘로 나눠 집계 (북 / 심벌 따로)
                   관통 = 원판 자리를 위에서 아래로 통과 (진짜 뚫림)
                   바깥으로 돌아 들어감 = 원판을 빗나가 허공으로 내려갔다가 원호를 따라 밑으로 들어옴 (뚫림 아님)
    2) 가짜 접촉 : 스틱을 직전 드럼 위 2cm에 띄워 1초 대기 -> 허리를 좌우로 돌려 드럼 위를 스쳐 지나가기
                   -> 닿지 않았는데 접촉력이 생기는지, 팁 궤적이 막히거나 튀는지
    반복할 때마다 다른 시드(다른 시작 자세)를 쓰고, 매 반복 끝에 누적 결과를 출력
    시드 순서가 같으므로 설정만 바꿔 실행한 결과를 같은 반복 횟수에서 비교하면 됨

화면
    타격하면 그 손 팁에 빨간 공 (환경의 Visualizer를 이 스크립트에서만 켬), 원판 밑으로 빠지는 스틱을 눈으로 확인
실행 (레포 루트에서)
    기준 (지금 레포)  : ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/tunnel_test.py --offset 0.01 --drum_thick 0.015
    offset만 5cm     : ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/tunnel_test.py --offset 0.05 --drum_thick 0.015
    offset + 북 7cm  : ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/tunnel_test.py --offset 0.05 --drum_thick 0.07
    옵션: --num_envs 4, --slow 0.0 (스텝마다 대기 초), --rounds 0 (반복 횟수, 0이면 창을 닫을 때까지)
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="드럼 뚫림 대책 GUI 확인")
parser.add_argument("--offset", type=float, default=0.05, help="드럼 contact_offset [m] (레포 값 0.01)")
parser.add_argument("--drum_thick", type=float, default=0.015, help="북 4개 두께 [m] (레포 값 0.015)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--slow", type=float, default=0.0, help="스텝마다 대기 시간 [s]")
parser.add_argument("--rounds", type=int, default=0, help="반복 횟수 (0이면 창을 닫을 때까지 반복)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Sim 실행 후에만 import 가능 ----
import math
import time

import torch                    # pyright: ignore[reportMissingImports]
import omni.usd                 # pyright: ignore[reportMissingImports]
from pxr import Usd, UsdGeom    # pyright: ignore[reportMissingImports]

import drum_robot.tasks.drumrobot.drumrobot_env as drumrobot_env_module
from drum_robot.tasks.drumrobot.drumrobot_cfg import DrumRobotEnvCfg
from drum_robot.tasks.drumrobot.drumrobot_env import DrumRobotEnv
from drum_robot.tasks.drumrobot.components.visualizer import VisualizerCfg
from drum_robot.tasks.drumrobot.components.specs import Instruments

INSTS = Instruments().all
NAMES = list(INSTS.keys())
DRUMS = ("snare", "floor", "mid", "high")      # 두께를 바꾸는 북 (심벌은 그대로)
STICK_TIP = 0.385                               # 손목 축 -> 팁 거리 [m]
STICK_RADIUS = 0.0075                           # 스틱 캡슐 반지름 [m]
HOVER = 0.02                                    # 가짜 접촉 시험에서 떠 있을 간격 [m] (offset 5cm 안)
SEED = 0


def make_env() -> DrumRobotEnv:
    drumrobot_env_module.VisualizerCfg = lambda: VisualizerCfg(enable_visualization=True)
    cfg = DrumRobotEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 10.0                 # 이 스크립트에서만: 시나리오 도중 자동 리셋 방지 (길면 악보 데이터가 커져 GPU 메모리 부족)
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    cfg.viewer.eye = (0.9, -1.1, 1.9)
    cfg.viewer.lookat = (0.0, 0.5, 0.6)

    # ---- 덮어쓰기 (레포의 drumrobot_cfg.py:36, 39, 49와 같은 값) ----
    waist_z = cfg.robot_waist_joint_offset_z
    for name, obj in cfg.drums_cfg.rigid_objects.items():
        obj.spawn.collision_props.contact_offset = args_cli.offset
        if name in DRUMS:
            x, y, z = INSTS[name].position
            obj.spawn.height = args_cli.drum_thick
            obj.init_state.pos = (x, y, z + waist_z - args_cli.drum_thick / 2)

    env = DrumRobotEnv(cfg)
    # ---- 덮어쓰기 (레포의 drumrobot_env.py:59-60과 같은 값, 리셋 전에) ----
    for name in DRUMS:
        env.drum_half_thickness[NAMES.index(name)] = args_cli.drum_thick / 2
    return env


def thickness_of(m: int) -> float:
    return args_cli.drum_thick if NAMES[m] in DRUMS else INSTS[NAMES[m]].thickness


def check_settings(env: DrumRobotEnv):
    print(f"\n[0 설정 확인] 요청: contact_offset {args_cli.offset} m, 북 두께 {args_cli.drum_thick} m")
    stage = omni.usd.get_context().get_stage()
    for name in NAMES:
        root = stage.GetPrimAtPath(f"/World/envs/env_0/Drum_{name}")
        height, offset = None, None
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Cylinder):
                height = UsdGeom.Cylinder(p).GetHeightAttr().Get()
            a = p.GetAttribute("physxCollision:contactOffset")
            if a and a.Get() is not None:
                offset = a.Get()
        print(f"  {name:>7s}: 시뮬레이션 원판 두께 {height:.3f} m, contact_offset {offset}")
    # 스틱 충돌체 contact_offset (실제 적용 거리 = 드럼 + 스틱)
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        a = p.GetAttribute("physxCollision:contactOffset")
        if "stick" in p.GetPath().pathString.lower() and a and a.HasAuthoredValue():
            print(f"  스틱 충돌체 {p.GetPath().pathString.split('/')[-1]}: contact_offset {a.Get()}")
    # 리셋 뒤 윗면 높이 = env.drum_pos (바꾸기 전과 같은 위치여야 함)
    env.step(torch.zeros((env.num_envs, 9), device=env.device))
    center_z = env.drums.data.object_pos_w[:, :, 2] - env.scene.env_origins[:, None, 2]       # (N, M) 원판 중심
    top_z = center_z + env.drum_half_thickness[None]
    err = (top_z - env.drum_pos[:, :, 2]).abs().max().item()
    print(f"  리셋 뒤 원판 윗면 높이 - 정해진 윗면 높이: 최대 {err * 1000:.2f} mm (0이면 윗면 위치 그대로)")


class Runner:
    def __init__(self, env: DrumRobotEnv):
        self.env = env
        names = env.robot_interface.get_ctrl_joint_name()
        self.jw = [names.index("left_wrist"), names.index("right_wrist")]
        self.j_waist = names.index("waist_joint")
        self.dt = env.step_dt
        self.wrist_scale = env.action_scale_vec[0, self.jw]
        self.waist_scale = env.action_scale_vec[0, self.j_waist]
        self.ar = torch.arange(env.num_envs, device=env.device)

    def reset(self, seed: int = SEED):
        # 설정끼리 비교할 수 있게 시드 고정 (시행마다 다른 시드)
        torch.manual_seed(seed)
        self.env.reset()

    def step(self, a: torch.Tensor):
        if not simulation_app.is_running():     # 창을 닫으면 반복을 멈춤
            raise WindowClosed
        self.env.step(a)
        if args_cli.slow > 0:
            time.sleep(args_cli.slow)

    def zeros(self):
        return torch.zeros((self.env.num_envs, 9), device=self.env.device)

    def start_drum(self):
        return self.env.reward_computer.alignment.prev_drum      # (N, 2) 손별 직전 드럼 [왼손, 오른손]

    def gap(self):
        # 손별 스틱 표면 - 직전 드럼 윗면 [m]
        tip = self.env.robot_interface.get_body_pos(self.env.robot)
        top = self.env.drum_pos[self.ar[:, None], self.start_drum(), 2]
        return tip[..., 2] - STICK_RADIUS - top

    def own_contact_force(self):
        # 손별 직전 드럼(떠 있는 그 드럼)과의 접촉력 크기 (마지막 물리 스텝) [N]
        fL = self.env.contact_L.data.force_matrix_w[:, 0].norm(dim=-1)    # (N, M)
        fR = self.env.contact_R.data.force_matrix_w[:, 0].norm(dim=-1)
        f = torch.stack([fL, fR], dim=1)                                   # (N, 2, M)
        return f.gather(2, self.start_drum()[..., None]).squeeze(-1)       # (N, 2)

    def over_own_drum(self):
        # 손별 팁이 직전 드럼 원판 위(수평으로 반지름 안)에 있는지
        tip = self.env.robot_interface.get_body_pos(self.env.robot)
        dp = self.env.drum_pos[self.ar[:, None], self.start_drum()]
        return (tip[..., :2] - dp[..., :2]).norm(dim=-1) < self.env.reward_computer.radius[self.start_drum()]


class WindowClosed(Exception):
    pass


def scenario_tunnel(r: Runner, round_idx: int, tot: dict, n_trials: int = 4, lift_deg: float = 60.0):
    # tot: 반복 전체 누적 {"북": [관통, 바깥으로 돌아 들어감, 치기 전부터 밑, 전체 손], "심벌": [...]}
    # 관통 = 이전 스텝에 원판 자리(반지름 안, 밑면 위) -> 이번 스텝에 원판 자리 밑 (위에서 아래로 원판을 통과)
    # 바깥으로 돌아 들어감 = 이전 스텝에 반지름 바깥 -> 이번 스텝에 원판 자리 밑 (허공으로 빗나갔다가 원호를 따라 밑으로)
    print(f"\n[1 뚫림] 손목 {lift_deg:.0f}° 들기 -> 최대로 내려치고 0.3초 누르기 x{n_trials}번 (env {r.env.num_envs}개)")
    env = r.env
    for t in range(n_trials):
        r.reset(seed=round_idx * n_trials + t)
        start = r.start_drum()
        q0 = env.robot_interface.get_joint_pos(env.robot)[:, r.jw].clone()
        a = r.zeros()
        for _ in range(20):                                                        # 들기
            a[:, r.jw] = ((q0 + math.radians(lift_deg) - env.target_q[:, r.jw]) / (r.wrist_scale * r.dt)).clamp(-1, 1)
            r.step(a)
        thick = torch.tensor([thickness_of(m) for m in range(len(NAMES))], device=env.device)[start]   # (N, 2)
        R = env.reward_computer.radius[start]
        dp = env.drum_pos[r.ar[:, None], start]                                    # (N, 2, 3) 손별 직전 드럼 윗면 중심

        def tip_state():
            tip = env.robot_interface.get_body_pos(env.robot)
            inside = (tip[..., :2] - dp[..., :2]).norm(dim=-1) < R                 # 수평으로 원판 반지름 안
            below = tip[..., 2] - dp[..., 2] < -thick                              # 원판 밑면보다 아래
            return inside, below

        through = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)   # 관통 (진짜 뚫림)
        around = torch.zeros_like(through)                                              # 바깥(허공)으로 빗나갔다가 밑으로 들어옴
        inside_p, below_p = tip_state()
        already = inside_p & below_p                                                    # 내려치기 전부터 원판 자리 밑에 있음
        a[:, r.jw] = -1.0
        for k in range(30):                                                        # 내려치고 0.3초 누르기 -> 0.15초 놓기
            if k == 20:
                a[:, r.jw] = 0.0
            r.step(a)
            inside, below = tip_state()
            new_under = below & ~below_p & inside                                   # 이번 스텝에 원판 자리 밑으로 들어감
            through |= new_under & inside_p                                         # 이전에도 원판 자리 안 (위에서 아래로 통과)
            around |= new_under & ~inside_p                                         # 이전엔 원판 바깥 (가장자리로 돌아 들어옴)
            inside_p, below_p = inside, below
        around &= ~through
        is_drum = torch.tensor([n in DRUMS for n in NAMES], device=env.device)[start]
        for key, mask in (("북", is_drum), ("심벌", ~is_drum)):
            tot[key][0] += int((through & mask).sum()); tot[key][1] += int((around & mask).sum())
            tot[key][2] += int((already & mask).sum()); tot[key][3] += int(mask.sum())
        print(f"  {t + 1}번째: 관통 {int(through.sum())}, 바깥으로 돌아 들어감 {int(around.sum())}, "
              f"치기 전부터 밑 {int(already.sum())} / 손 {through.numel()}")
    for key, (u, o, al, n) in tot.items():
        print(f"[1 뚫림] 누적 {key}: 관통(진짜 뚫림) {u} / {n} 손 ({(u / max(n, 1)) * 100:.1f}%), "
              f"바깥(허공)으로 빗나갔다가 밑으로 들어감 {o} ({(o / max(n, 1)) * 100:.1f}%), 치기 전부터 원판 밑 {al} ({(al / max(n, 1)) * 100:.1f}%)")


def scenario_ghost(r: Runner, round_idx: int, tot: dict):
    # tot: 반복 전체 누적 {"hold": 대기 중 접촉력 최대, "sweep": 스윙 중 접촉력 최대, "dgap": 스윙 중 간격 변화 최대, "hands": 집계한 손 수}
    print(f"\n[2 가짜 접촉] 스틱을 직전 드럼 위 {HOVER * 100:.0f}cm에 띄워 대기 -> 허리를 좌우로 돌려 드럼 위를 스쳐 지나가기")
    env = r.env
    r.reset(seed=1000 + round_idx)
    a = r.zeros()
    for _ in range(round(1.0 / r.dt)):                                             # 자리 잡기 (오차의 10%씩)
        dq = 0.1 * (HOVER - r.gap()) / STICK_TIP
        a[:, r.jw] = (dq / (r.wrist_scale * r.dt)).clamp(-1, 1)
        r.step(a)
    g0 = r.gap().clone()
    a[:] = 0.0
    f_hold = torch.zeros((env.num_envs, 2), device=env.device)
    for _ in range(round(1.0 / r.dt)):                                             # 대기 (행동 0)
        r.step(a); f_hold = torch.maximum(f_hold, r.own_contact_force())
    hovering = (g0 > 0.01) & (g0 < 0.03)                                           # 실제로 1~3cm에 떠 있던 손만 집계
    if not hovering.any():
        print("  이번 반복에는 1~3cm에 떠 있는 손이 없어 건너뜀")
        return
    print(f"  떠 있는 손 {int(hovering.sum())} / {hovering.numel()} (간격 1~3cm), 그 손들의 간격 평균 {g0[hovering].mean() * 100:.1f} cm")
    print(f"  이번: 대기 1초 동안 떠 있는 드럼과의 접촉력 최대 {f_hold[hovering].max():.3f} N (0이어야 함)")

    # 허리 좌우 스윙: 목표를 ±15° 범위로 움직임 (팁이 드럼 위를 수평으로 약 1~2 m/s로 지나감)
    # 떠 있던 드럼 위에 있는 동안만 집계 (다른 드럼에 실제로 부딪히는 것은 제외)
    f_sweep = torch.zeros((env.num_envs, 2), device=env.device)
    dgap_max = torch.zeros((env.num_envs, 2), device=env.device)
    swing = [+1] * 7 + [-1] * 14 + [+1] * 7
    for s in swing:
        a[:] = 0.0
        a[:, r.j_waist] = s * math.radians(15) / 7 / (r.waist_scale * r.dt)
        r.step(a)
        over = r.over_own_drum() & hovering
        f_sweep = torch.where(over, torch.maximum(f_sweep, r.own_contact_force()), f_sweep)
        dgap_max = torch.where(over, torch.maximum(dgap_max, (r.gap() - g0).abs()), dgap_max)
    a[:] = 0.0
    for _ in range(20):
        r.step(a)
    print(f"  이번: 스윙 중 떠 있던 드럼 위를 지나는 동안 그 드럼과의 접촉력 최대 {f_sweep[hovering].max():.3f} N, "
          f"간격 변화 최대 {dgap_max[hovering].max() * 100:.2f} cm")
    tot["hold"] = max(tot["hold"], f_hold[hovering].max().item())
    tot["sweep"] = max(tot["sweep"], f_sweep[hovering].max().item())
    tot["dgap"] = max(tot["dgap"], dgap_max[hovering].max().item())
    tot["hands"] += int(hovering.sum())
    print(f"[2 가짜 접촉] 누적 (손 {tot['hands']}개): 대기 중 접촉력 최대 {tot['hold']:.3f} N, 스윙 중 접촉력 최대 {tot['sweep']:.3f} N, "
          f"간격 변화 최대 {tot['dgap'] * 100:.2f} cm (기준 설정과 비교: 접촉력이 새로 생기거나 간격 변화가 커지면 가짜 접촉)")


def main():
    env = make_env()
    r = Runner(env)
    r.reset()
    check_settings(env)

    tunnel_tot = {"북": [0, 0, 0, 0], "심벌": [0, 0, 0, 0]}     # [관통, 바깥으로 돌아 들어감, 치기 전부터 밑, 전체 손]
    ghost_tot = {"hold": 0.0, "sweep": 0.0, "dgap": 0.0, "hands": 0}
    k = 0
    try:
        while args_cli.rounds == 0 or k < args_cli.rounds:
            print(f"\n========== {k + 1}번째 반복 (offset {args_cli.offset} m, 북 두께 {args_cli.drum_thick} m) ==========")
            scenario_tunnel(r, k, tunnel_tot)
            scenario_ghost(r, k, ghost_tot)
            k += 1
    except WindowClosed:
        print(f"\n[tunnel] 창을 닫아서 {k}번 반복 후 종료")

    if not args_cli.headless and simulation_app.is_running():
        print("\n[tunnel] 완료. 창을 닫으면 종료됩니다.")
        while simulation_app.is_running():
            env.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
