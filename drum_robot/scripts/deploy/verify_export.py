"""export 한 그래프가 학습된 에이전트와 같은 액션을 내는지 대조한다.

왜 필요한가 — 2026-09-02 사고: 정규화 z 부호를 잘못 넣어 정책이 높이를 못 봤는데
골든 대조(정규화 이전 원시값)와 PyTorch-vs-ORT(같은 상수)는 둘 다 통과했다.
빠져 있던 것은 "학습된 에이전트가 실제로 내는 액션"과의 대조다. 재export 후 설치 전에 반드시 통과시킬 것.

비교
    A. 배포 경로   raw 11버퍼(허리 기준) -> ONNX -> action
    B. 학습 경로   env 가 반환한 obs (90) -> skrl 스케일러 -> MLP -> action
golden.json 의 각 스텝은 raw 와 obs 를 같은 시점으로 갖고 있다.

사용법
    python drum_robot/scripts/deploy/verify_export.py \
        --checkpoint logs/skrl/drum_robot/<run>/checkpoints/best_agent.pt \
        --onnx   drum_robot/deploy_export/policy.onnx \
        --golden drum_robot/deploy_export/golden.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

IN_NAMES = ["joint_pos", "joint_vel", "lead", "tip_pos", "drum_pos", "arm_next", "waist_des", "waist_vel_des",
            "yaw_des", "yaw_vel_des", "last_action"]
SHAPES = {"joint_pos": (1, 9), "joint_vel": (1, 9), "lead": (1, 9), "tip_pos": (1, 2, 3),
          "drum_pos": (1, 8, 3), "arm_next": (1, 2, 9), "waist_des": (1, 1), "waist_vel_des": (1, 1),
          "yaw_des": (1, 2), "yaw_vel_des": (1, 2),
          "last_action": (1, 9)}


def build_mlp(sd):
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net_container.")})
    layers, prev = [], None
    for slot in idx:
        w, b = sd[f"net_container.{slot}.weight"], sd[f"net_container.{slot}.bias"]
        if prev is not None:
            if slot - prev != 2:
                raise RuntimeError(f"Linear 슬롯 간격이 2가 아닙니다 ({prev} -> {slot}).")
            layers.append(nn.ELU())
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w); lin.bias.copy_(b)
        layers.append(lin); prev = slot
    return nn.Sequential(*layers).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd, sp = ck["policy"], ck["state_preprocessor"]
    mean, var = sp["running_mean"].float().reshape(-1), sp["running_variance"].float().reshape(-1)
    net = build_mlp(sd)
    print(f"[1/3] 체크포인트: {a.checkpoint.name}   MLP {[l.out_features for l in net if isinstance(l, nn.Linear)]}   스케일러 {tuple(mean.shape)}")

    g = json.loads(a.golden.read_text())
    steps = g["steps"]
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    print(f"[2/3] 골든 {len(steps)} 스텝 ({Path(g.get('song', '?')).name})")

    errs, errs_rec = [], []
    for st in steps:
        feed = {k: np.asarray(st["raw"][k], dtype=np.float32).reshape(SHAPES[k]) for k in IN_NAMES}
        got = sess.run(["action_mean"], feed)[0].reshape(-1)

        x = torch.tensor(st["obs"], dtype=torch.float32)
        z = ((x - mean) / (torch.sqrt(var) + 1e-8)).clamp(-5.0, 5.0)
        with torch.no_grad():
            exp = net(z.unsqueeze(0)).squeeze(0).numpy()

        errs.append(np.abs(got - exp).max())
        errs_rec.append(np.abs(got - np.asarray(st["act"], dtype=np.float32)).max())   # 덤프 당시 ONNX 출력과 재현 일치

    e = np.asarray(errs)
    worst = int(e.argmax())
    print(f"[3/3] 배포 경로 vs 학습 경로  ({len(e)} 스텝)")
    print(f"      최대 {e.max():.4e}   p99 {np.percentile(e, 99):.4e}   중앙 {np.median(e):.4e}   최악 스텝 {worst}")
    print(f"      (덤프 당시 ONNX 출력 재현 오차 최대 {max(errs_rec):.2e})")

    if e.max() > a.tol:
        print(f"\n      !! 불일치 (허용 {a.tol:.1e}). 그래프가 학습과 다른 액션을 냅니다.")
        print(f"         의심 순서: 1) z 프레임(z_offset 접기) 2) max_lead / joint_center 3) 버퍼 순서 4) 스케일러 clip 5) last_action 시점")
        print(f"         이 상태로 배포하면 안 됩니다.")
        return 1
    print("\n      통과 — 배포 그래프가 학습된 에이전트와 같은 액션을 냅니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
