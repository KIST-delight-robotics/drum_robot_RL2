"""
[확인용 임시 스크립트] 같은 곡의 실기 로그와 sim play CSV 를 같은 시간축에 띄워 비교 (matplotlib 창, 확대·이동 가능)

시간축 = 곡 시각 [s]
    실기: policy.csv 의 t_score (정책이 곡을 시작한 뒤. 그 전 서버 자세 잡기 구간은 뺌)
    sim : play CSV 의 t (에피소드 시각 = 곡 시각)
그래프: θ0 허리, θ1 오른 어깨 yaw, θ2 왼 어깨 yaw 마다 [실기] [sim ...] 칸 (같은 관절은 y 범위 같음, 시간축 공유)
    RL_desired (주황) = 정책 명령, output (초록) = 실제 각도 [deg]
    세로선 = 음표 시각 (첫 sim CSV 의 타격, 파랑 왼손 / 빨강 오른손)
    칸 제목: lag = desired 변화량과 output 변화량의 상관이 최대인 시간차, RMS = RMS(desired − output)

실행 (레포 루트에서, Isaac Sim 필요 없음)
    python drum_robot/scripts/tests/plot_real_vs_sim.py \
        --real ~/RL_Real/Phil-drum-robot/drumrobot_server/log/log_1028_1605_policy.csv \
        --sim logs/play_song/izz_search/<파일>.csv [--sim <다른 sim CSV> ...]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (제목, 실기 q_cmd 번호 = 모터 id 순서, 실기 joint_pos 번호 = obs_constants joint_order 순서, sim 열 이름)
JOINTS = [("θ0  waist", 0, 0, "waist"),
          ("θ1  right shoulder yaw", 1, 2, "yaw_r"),
          ("θ2  left shoulder yaw", 2, 1, "yaw_l")]


def load_real(path: Path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("joint_pos_8") not in (None, "") and float(r["t_score"]) > 0]
    t = np.array([float(r["t_score"]) for r in rows])
    q = {key: (np.degrees([float(r[f"q_cmd_{jc}"]) for r in rows]), np.degrees([float(r[f"joint_pos_{jp}"]) for r in rows]))
         for _, jc, jp, key in JOINTS}
    return t, q


def load_sim(path: Path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    q = {key: (np.array([float(r[f"{key}_cmd"]) for r in rows]), np.array([float(r[f"{key}_act"]) for r in rows]))
         for *_, key in JOINTS}
    hits = {k: t[np.array([float(r[k]) for r in rows]) > 0] for k in ("hit_l", "hit_r")}
    return t, q, hits


def sim_label(path: Path) -> str:
    # play_song 이 남긴 txt 의 허리 게인·izz (없으면 파일 이름만)
    txt = path.with_suffix(".txt")
    info = next((ln[len("waist: "):] for ln in open(txt) if ln.startswith("waist:")), "") if txt.exists() else ""
    return f"sim {path.stem[-15:]}  {info}"


def lag_rms(t, c, a) -> tuple[float, float]:
    dt = np.median(np.diff(t))
    dc, da = np.diff(c), np.diff(a)
    r = np.array([np.corrcoef(dc[:len(dc) - k], da[k:])[0, 1] for k in range(30)])
    k = int(np.argmax(r))
    frac = 0.5 * (r[k - 1] - r[k + 1]) / (r[k - 1] - 2 * r[k] + r[k + 1]) if 0 < k < len(r) - 1 else 0.0
    return (k + frac) * dt * 1000, float(np.sqrt(np.mean((c - a) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="실기 vs sim 같은 곡 비교")
    parser.add_argument("--real", type=Path, required=True, help="실기 log_..._policy.csv")
    parser.add_argument("--sim", type=Path, action="append", required=True, help="sim play CSV (여러 번 가능)")
    args = parser.parse_args()

    t_real, q_real = load_real(args.real.expanduser())
    sims = [(sim_label(p), *load_sim(p)) for p in args.sim]
    hits = sims[0][3]
    sources = [(f"real {args.real.stem.replace('_policy', '')}", t_real, q_real)] + [(lab, t, q) for lab, t, q, _ in sims]

    n = len(sources)
    fig, axes = plt.subplots(3 * n, 1, figsize=(14, min(14, 1.6 * 3 * n + 1)), sharex=True)
    for g, (title, *_, key) in enumerate(JOINTS):
        lo = min(min(q[key][0].min(), q[key][1].min()) for _, _, q in sources)
        hi = max(max(q[key][0].max(), q[key][1].max()) for _, _, q in sources)
        for s, (label, t, q) in enumerate(sources):
            ax = axes[g * n + s]
            c, a = q[key]
            ax.plot(t, c, "-", color="tab:orange", lw=1.1, label="RL_desired",
                    drawstyle="steps-post")                             # 정책 스텝 동안 목표 유지 = 계단
            ax.plot(t, a, "-", color="tab:green", lw=1.4, label="output")
            for k, color in (("hit_l", "tab:blue"), ("hit_r", "tab:red")):
                for x in hits[k]:
                    ax.axvline(x, color=color, alpha=0.25, lw=0.8)
            lag, rms = lag_rms(t, c, a)
            ax.set_title(f"{title}  |  {label}  |  lag {lag:.0f} ms,  RMS {rms:.1f} deg", fontsize=9, loc="left")
            ax.set_ylim(lo - 3, hi + 3)
            ax.set_ylabel("deg")
            ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("song time [s]   (real: t_score, sim: t)   vertical lines: note hit time, blue = left, red = right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
