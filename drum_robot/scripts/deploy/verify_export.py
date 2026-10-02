"""export 한 그래프가 학습된 에이전트와 같은 액션을 내는지 대조한다.

왜 필요한가 — 2026-09-02 사고:
    drum_center 의 z 오프셋 부호를 잘못 넣어 z 채널 14개가 clamp(±1.5)에
    포화됐다. 정책이 높이를 관측하지 못해 타격이 0회가 됐는데,
    기존 검증은 전부 통과했다:

      golden_check          원시버퍼끼리 비교 — 정규화 이전이라 멀쩡
      PyTorch vs ORT        양쪽이 같은 잘못된 상수를 씀
      ort_check             이름·형상만 확인

    빠져 있던 것은 "학습된 에이전트가 실제로 내는 액션"과의 대조다.
    이 스크립트가 그것이다. 재export 후 설치 전에 반드시 통과시킬 것.

무엇을 비교하나
    A. 배포 경로   raw 버퍼(허리 기준) -> ONNX -> action
    B. 학습 경로   env 정규화된 obs_148 -> skrl 스케일러 -> MLP -> action

    B 는 체크포인트에서 직접 만든다 (state_preprocessor + net_container).
    golden.json 이 두 입력을 짝지어 갖고 있으므로 같은 관측에 대한
    두 경로의 출력을 비교할 수 있다.

사용법
    python verify_export.py \
        --checkpoint logs/.../checkpoints/best_agent.pt \
        --onnx       <policy.onnx> \
        --golden     <golden.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

IN_NAMES = ["joint_pos", "joint_vel", "tip_pos", "drum_pos", "next_hits",
            "hit_armed", "arm_role", "per_arm_pos", "per_arm_time"]
SHAPES = {"joint_pos": (1, 9), "joint_vel": (1, 9), "tip_pos": (1, 2, 3),
          "drum_pos": (1, 8, 3), "next_hits": (1, 6, 11), "hit_armed": (1, 2, 8),
          "arm_role": (1, 2), "per_arm_pos": (1, 2, 2, 3), "per_arm_time": (1, 2, 2)}
POS_KEYS = ("tip_pos", "drum_pos", "per_arm_pos")   # z 에 z_offset 이 실려 있는 버퍼


def build_mlp(sd):
    """net_container.{0,2,4,6} 을 nn.Sequential 로 되살린다 (export 와 동일 규약)."""
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="최대 허용 오차. 부동소수 정밀도는 보통 1e-6 수준")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd, sp = ck["policy"], ck["state_preprocessor"]
    mean, var = sp["running_mean"].float(), sp["running_variance"].float()
    net = build_mlp(sd)
    print(f"[1/3] 체크포인트: {a.checkpoint.name}")
    print(f"      MLP {[l.out_features for l in net if isinstance(l, nn.Linear)]}"
          f"   스케일러 {tuple(mean.shape)}")

    g = json.loads(a.golden.read_text())
    steps, z_off = g["steps"], float(g.get("z_offset", 1.0))
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    print(f"[2/3] 골든 {len(steps)} 스텝, z_offset {z_off}")

    errs = []
    for t in range(len(steps) - 1):
        # 배포 경로 — 실기와 같은 허리 기준 입력
        feed = {k: np.asarray(steps[t + 1]["raw"][k], dtype=np.float32).reshape(SHAPES[k])
                for k in IN_NAMES}
        for k in POS_KEYS:
            feed[k] = feed[k].copy(); feed[k][..., 2] -= z_off
        got = sess.run(["action_mean"], feed)[0].reshape(-1)

        # 학습 경로 — obs_148_next[t] 는 step t 이후 관측 = raw[t+1] 과 같은 시점
        x = torch.tensor(steps[t]["obs_148_next"], dtype=torch.float32)
        z = ((x - mean) / torch.sqrt(var + 1e-8)).clamp(-5.0, 5.0)
        with torch.no_grad():
            exp = net(z.unsqueeze(0)).squeeze(0).numpy()

        errs.append(np.abs(got - exp).max())

    e = np.asarray(errs)
    worst = int(e.argmax())
    print(f"[3/3] 배포 경로 vs 학습 경로  ({len(e)} 스텝)")
    print(f"      최대 {e.max():.4e}   p99 {np.percentile(e, 99):.4e}   중앙 {np.median(e):.4e}")
    print(f"      최악 스텝 {worst}")

    if e.max() > a.tol:
        print(f"\n      !! 불일치 (허용 {a.tol:.1e}). 그래프가 학습과 다른 액션을 냅니다.")
        print(f"         의심 순서:")
        print(f"           1. 정규화 상수 프레임 — drum_center 가 허리 기준인가")
        print(f"           2. joint_center / joint_half_range / joint_vel_scale")
        print(f"           3. 버퍼 이어붙이는 순서")
        print(f"           4. 스케일러 clip 값 (기본 ±5)")
        print(f"         이 상태로 배포하면 안 됩니다. 중단합니다.")
        return 1

    print(f"\n      통과 — 배포 그래프가 학습된 에이전트와 같은 액션을 냅니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
