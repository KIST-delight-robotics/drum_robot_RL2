"""
[확인용 임시 스크립트] 시작 자세 보기 (GUI)

시작 자세 = 실기 타격점 IK 타격 자세 + 손목 20° 든 자세 (robot_initializer.py)
스틱·손목·팔이 드럼/심벌에 닿는지 눈으로 확인

내용
    - env 마다 시작 드럼 쌍을 하나씩 고정: env i = drum_pairs[i % 36] (36개 env 면 모든 쌍)
    - 행동 0 (관절 목표 = 시작 자세 유지) 으로 창을 닫을 때까지 계속 시뮬레이션
    - settle 스텝 뒤 (그리고 300 스텝마다) 손목 링크 접촉을 env 별로 출력
      손목 링크 = 스틱 캡슐 + 손목 캡슐. 팔(팔꿈치·어깨) 링크 접촉은 센서가 없어서 화면으로만 확인

옵션
    --num_envs 36        env 개수 (기본 36 = 모든 시작 쌍)
    --pair hihat,snare   이 쌍만 (왼손,오른손 드럼 이름, robot_initializer.drum_pairs 에 있는 조합)
    --env 0              카메라를 둘 env 번호
    --clean              허리·어깨 yaw 노이즈와 드럼 위치 노이즈 끔 (설계 자세 그대로)
    --settle 30          접촉을 처음 출력할 스텝

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/start_pose_view.py
"""

import argparse

from isaaclab.app import AppLauncher   # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser(description="시작 자세 보기 (GUI)")
parser.add_argument("--num_envs", type=int, default=36, help="env 개수 (36 = 모든 시작 쌍)")
parser.add_argument("--pair", type=str, default=None, help="이 쌍만: 왼손,오른손 (예: hihat,snare)")
parser.add_argument("--env", type=int, default=0, help="카메라를 둘 env 번호")
parser.add_argument("--clean", action="store_true", help="허리·yaw 노이즈와 드럼 위치 노이즈 끔")
parser.add_argument("--settle", type=int, default=30, help="접촉을 처음 출력할 스텝")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Sim 실행 후에만 import 가능 ----
import torch   # pyright: ignore[reportMissingImports]

from drum_robot.tasks.drumrobot.drumrobot_cfg import DrumRobotEnvCfg
from drum_robot.tasks.drumrobot.drumrobot_env import DrumRobotEnv
from drum_robot.tasks.drumrobot.components.specs import Instruments

cfg = DrumRobotEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.viewer.origin_type = "env"          # 카메라를 env 원점 기준으로
cfg.viewer.env_index = args_cli.env
cfg.viewer.eye = (0.0, -1.3, 2.0)       # 로봇 뒤 위 (드럼은 +y 쪽)
cfg.viewer.lookat = (0.1, 0.5, 0.8)
env = DrumRobotEnv(cfg=cfg)

names = list(Instruments().all.keys())
ri = env.robot_initializer
pairs = ri.cfg.drum_pairs
P = len(pairs)

fixed = None
if args_cli.pair:
    l_name, r_name = [s.strip() for s in args_cli.pair.split(",")]
    if (l_name, r_name) not in pairs:
        raise SystemExit(f"[start_pose_view] 쌍 ({l_name}, {r_name}) 이 drum_pairs 에 없음: {pairs}")
    fixed = pairs.index((l_name, r_name))

if args_cli.clean:
    env.instruments.drum_noise_scale = 0.0      # 드럼 위치 노이즈 끔


def reset_init_pos(env_ids):
    # env 번호로 시작 쌍 고정 (원래는 무작위), 노이즈는 robot_initializer 와 같은 방식 (허리·yaw 만)
    ids = torch.as_tensor(env_ids, device=ri.device).long()
    idx = torch.full_like(ids, fixed) if fixed is not None else ids % P
    pos = ri.pos_angle[idx]
    if not args_cli.clean:
        pos = pos + ri.cfg.joint_noise_scale * torch.randn_like(pos) * ri.noise_mask
    return pos, ri.pair_drum_idx[idx]


ri.reset_init_pos = reset_init_pos

env.reset()
N, A = env.num_envs, env.actions.shape[1]
pair_of = [pairs[fixed if fixed is not None else i % P] for i in range(N)]
print(f"[start_pose_view] env {N}개, 노이즈 {'끔' if args_cli.clean else '허리·yaw 5° + 드럼 위치 2 cm'}, 카메라 env {args_cli.env}")
for i in range(N):
    print(f"[start_pose_view]   env {i:2d}: 왼손 {pair_of[i][0]:8s} 오른손 {pair_of[i][1]}")


def report(step: int):
    # 손목 링크(스틱 캡슐 + 손목 캡슐)가 드럼에서 받는 힘 > 1 mN 이면 접촉
    f = torch.stack([env.contact_L.data.force_matrix_w[:, 0].norm(dim=-1),
                     env.contact_R.data.force_matrix_w[:, 0].norm(dim=-1)], dim=1)     # (N, 2, M) [왼손, 오른손]
    d = env._get_contact_from_tip()                                                    # (N, 2, M) 접촉 지점-스틱 끝 거리 [m]
    c = f > 1e-3
    n_env = int(c.any(dim=(1, 2)).sum())
    print(f"[start_pose_view] step {step}: 손목 링크 접촉 env {n_env}/{N}")
    for i in torch.where(c.any(dim=(1, 2)))[0].tolist():
        items = []
        for a, hand in enumerate(("왼손", "오른손")):
            for m in torch.where(c[i, a])[0].tolist():
                dist = d[i, a, m].item()
                part = "?" if dist != dist else ("팁" if dist <= 0.01 else ("스틱" if dist <= 0.31 else "손목 캡슐"))
                items.append(f"{hand}-{names[m]} {f[i, a, m].item():.1f} N ({part}, 끝에서 {dist * 100:.0f} cm)")
        print(f"[start_pose_view]   env {i:2d} ({pair_of[i][0]}, {pair_of[i][1]}): " + ", ".join(items))


zero = torch.zeros((N, A), device=env.device)
step = 0
with torch.inference_mode():
    while simulation_app.is_running():
        env.step(zero)
        step += 1
        if step == args_cli.settle or step % 300 == 0:
            report(step)

env.close()
simulation_app.close()
