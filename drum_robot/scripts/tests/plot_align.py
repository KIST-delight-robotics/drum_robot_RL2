"""
[확인용 임시 스크립트] play_align.py / play_song.py가 저장한 CSV를 matplotlib 창으로 보기 (확대·이동 가능)

실행 (레포 루트에서, env_isaaclab의 python으로. Isaac Sim 필요 없음)
    python drum_robot/scripts/tests/plot_align.py                      # 가장 최근 기록 폴더 (play_align 또는 play_song)
    python drum_robot/scripts/tests/plot_align.py <폴더>               # 그 폴더의 *.csv 전부 (play_song 곡 폴더 = 모델별 CSV)
    python drum_robot/scripts/tests/plot_align.py <episode_001.csv>    # 파일 하나
조작
    ← / →        이전 / 다음 에피소드 (play_song 곡 폴더면 이전 / 다음 모델)
    v            큰 칸: 각도 보기 <-> 각속도 보기 (확대한 시간 구간은 그대로 유지)
    축 범위      모든 모델·에피소드 같은 x, y 범위 (칸·보기별 전체 최소~최대). 모델을 넘겨도 보던 범위(확대 포함) 유지
    창 아래 도구 막대: 돋보기 = 확대, 십자 = 이동, 집 = 공통 범위 (여섯 칸이 시간축을 같이 씀)
그래프: 6칸 = θ0, θ1, θ2 각각 [큰 칸: 각도 또는 각속도] + [바로 아래 작은 칸: 위치 오차]
    θ0 = 허리, θ1 = 오른 어깨 yaw, θ2 = 왼 어깨 yaw
    큰 칸 (각도) [deg]
        desired    (검정 점선) : 정렬 보상이 계산한 목표 θ0*, θ1*, θ2*
        RL_desired (주황 실선) : 정책이 명령한 관절 목표 (env.target_q, 예전 CSV에는 없음)
        output     (초록 실선) : 실제 관절 각도 (피드백)
    큰 칸 (각속도) [rad/s] = 각도 차이 / 스텝 시간
        RL_desired (주황) : 정책 명령의 속도 = 행동 × 13 rad/s (행동 1 = 13 rad/s),  output (초록) : 실제 각속도
        desired는 계단이라 속도가 의미 없어서 그리지 않음. 제목: |각속도| 최대값
    작은 칸 (위치 오차) [deg]
        desired − output    (검정) : 과제 오차 (정렬 보상이 벌주는 값, 부호 포함)
        RL_desired − output (주황) : 추종 오차 (관절이 정책 명령을 따라가지 못한 만큼)
        제목: 평균 |오차|, 치기 직전(가장 가까운 음표까지 0.1초 이하) 평균 |오차|
    세로선: 음표 시각 (파랑 = 왼손, 빨강 = 오른손)
"""

import argparse
import csv
import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt

# matplotlib 기본 단축키와 겹치는 키 해제 (← / → / v 는 이 스크립트에서 씀)
for _k, _keys in (("keymap.back", ("left", "c")), ("keymap.forward", ("right", "v"))):
    plt.rcParams[_k] = [x for x in plt.rcParams[_k] if x not in _keys]

REPO = Path(__file__).resolve().parents[3]
PANELS = [("θ0  waist", "waist_tgt", "waist_cmd", "waist_act"),
          ("θ1  right shoulder yaw", "yaw_r_tgt", "yaw_r_cmd", "yaw_r_act"),
          ("θ2  left shoulder yaw", "yaw_l_tgt", "yaw_l_cmd", "yaw_l_act")]
NOTE_TEXT = "time [s]   (vertical lines: note hit time, blue = left hand, red = right hand)"


def find_files(arg: str | None) -> list[Path]:
    if arg is None:                             # 가장 최근 기록 폴더 (play_align 에피소드 폴더, play_song 곡 폴더)
        dirs = glob.glob(str(REPO / "logs" / "skrl" / "drum_robot" / "*" / "play_align" / "*")) \
             + glob.glob(str(REPO / "logs" / "play_song" / "*"))
        dirs = [d for d in dirs if glob.glob(str(Path(d) / "*.csv"))]
        if not dirs:
            raise SystemExit("기록 폴더가 없습니다. 먼저 play_align.py 또는 play_drumsolo.py / play_snare0.py를 실행하세요.")
        arg = max(dirs, key=lambda d: max(Path(f).stat().st_mtime for f in glob.glob(str(Path(d) / "*.csv"))))
    p = Path(arg)
    files = [p] if p.is_file() else sorted(p.glob("*.csv"), key=lambda f: f.stat().st_mtime)   # 곡 폴더: 모델별 CSV (오래된 것부터)
    if not files:
        raise SystemExit(f"CSV가 없습니다: {p}")
    return files


def load(path: Path) -> dict:
    cols: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    v = float(v)
                except ValueError:
                    pass
                cols.setdefault(k, []).append(v)
    return cols


_CACHE: dict = {}


def load_cached(path: Path) -> dict:
    if path not in _CACHE:
        _CACHE[path] = load(path)
    return _CACHE[path]


def velocity(t: list, y: list) -> tuple[list, list]:
    # 각속도 [rad/s] = 각도 차이 (CSV는 deg) / 시간 차이, 시각은 두 스텝의 가운데
    tv = [(t[k] + t[k - 1]) / 2 for k in range(1, len(t))]
    v = [math.radians(y[k] - y[k - 1]) / (t[k] - t[k - 1]) for k in range(1, len(t))]
    return tv, v


def errors(d: dict, tgt: str, cmd: str, act: str) -> tuple[list, list | None]:
    # 위치 오차 [deg]: desired − output, RL_desired − output (명령 열이 없으면 None)
    e_task = [a - b for a, b in zip(d[tgt], d[act])]
    e_track = [a - b for a, b in zip(d[cmd], d[act])] if cmd in d else None
    return e_task, e_track


def mean(v: list) -> float:
    return sum(v) / len(v) if v else float("nan")


def near_hit(d: dict) -> list:
    # 가장 가까운 음표까지 0.1초 이하인 스텝
    return [min([x for x in (a, b) if not math.isnan(x)], default=99.0) <= 0.1 for a, b in zip(d["note_l_s"], d["note_r_s"])]


def common_limits(files: list[Path]) -> dict:
    # 모든 파일(모델·에피소드)에 같은 축 범위: x = 0 ~ 가장 긴 기록, y = 칸·보기별 전체 최소 ~ 최대 (+5% 여유)
    lim = {"x": (0.0, max(load_cached(f)["t"][-1] for f in files))}
    for i, (_, tgt, cmd, act) in enumerate(PANELS):
        for mode in ("angle", "velocity", "error"):
            vals = []
            for f in files:
                d = load_cached(f)
                if mode == "angle":
                    vals += [x for c in (tgt, cmd, act) if c in d for x in d[c]]
                elif mode == "velocity":
                    vals += [x for c in (cmd, act) if c in d for x in velocity(d["t"], d[c])[1]]
                else:
                    e_task, e_track = errors(d, tgt, cmd, act)
                    vals += e_task + (e_track or [])
            lo, hi = min(vals), max(vals)
            pad = max(0.05 * (hi - lo), 0.1 if mode == "velocity" else 1.0)
            lim[(mode, i)] = (lo - pad, hi + pad)
    return lim


def note_lines(ax, d: dict):
    for k, color in (("hit_l", "tab:blue"), ("hit_r", "tab:red")):
        for x, h in zip(d["t"], d[k]):
            if h:
                ax.axvline(x, color=color, alpha=0.35, lw=1)


def draw(fig, ax_main, ax_err, path: Path, i: int, n: int, mode: str, lim: dict):
    d = load_cached(path)
    t = d["t"]
    near = near_hit(d)
    for k, (title, tgt, cmd, act) in enumerate(PANELS):
        # 큰 칸: 각도 또는 각속도
        ax = ax_main[k]
        ax.clear()
        if mode == "angle":
            ax.plot(t, d[tgt], "--", color="black", lw=1.2, label="desired")
            if cmd in d:                                                # 예전 CSV에는 명령 열이 없음
                ax.plot(t, d[cmd], "-", color="tab:orange", lw=1.3, label="RL_desired")
            ax.plot(t, d[act], "-", color="tab:green", lw=1.5, label="output")
            ax.set_ylabel("deg")
            ax.set_title(f"{title}   angle", fontsize=10)
        else:
            peaks = []
            if cmd in d:
                tv, v = velocity(t, d[cmd])
                ax.plot(tv, v, "-", color="tab:orange", lw=1.3, label="RL_desired")
                peaks.append(f"RL_desired {max(abs(x) for x in v):.1f}")
            tv, v = velocity(t, d[act])
            ax.plot(tv, v, "-", color="tab:green", lw=1.5, label="output")
            peaks.append(f"output {max(abs(x) for x in v):.1f}")
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_ylabel("rad/s")
            ax.set_title(f"{title}   angular velocity   max |ω|  " + " / ".join(peaks) + " rad/s", fontsize=10)
        note_lines(ax, d)
        ax.grid(alpha=0.3)
        ax.set_ylim(*lim[(mode, k)])

        # 작은 칸: 위치 오차 (항상)
        ax = ax_err[k]
        ax.clear()
        e_task, e_track = errors(d, tgt, cmd, act)
        ax.plot(t, e_task, "-", color="black", lw=1.2, label="desired − output")
        text = f"desired−output  mean |e| {mean([abs(x) for x in e_task]):.1f},  near hit {mean([abs(x) for x, m in zip(e_task, near) if m]):.1f}"
        if e_track is not None:
            ax.plot(t, e_track, "-", color="tab:orange", lw=1.1, label="RL_desired − output")
            text += f"   |   RL_desired−output  mean |e| {mean([abs(x) for x in e_track]):.1f},  near hit {mean([abs(x) for x, m in zip(e_track, near) if m]):.1f}"
        ax.axhline(0, color="gray", lw=0.8)
        note_lines(ax, d)
        ax.set_ylabel("err deg")
        ax.set_title(f"{title.split()[0]}  position error   {text}  deg", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(*lim[("error", k)])

    ax_main[0].set_xlim(*lim["x"])
    ax_main[0].legend(loc="upper right", fontsize=8)
    ax_err[0].legend(loc="upper right", fontsize=8)
    ax_err[-1].set_xlabel(NOTE_TEXT)
    other = "angle" if mode == "velocity" else "angular velocity"
    fig.suptitle(f"{path.parent.name}/{path.name}   ({i + 1}/{n})   [{'angle' if mode == 'angle' else 'angular velocity'} + position error]"
                 f"   ← / → : model,  v : {other}", fontsize=11)
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="play_align / play_song CSV 보기")
    parser.add_argument("path", nargs="?", default=None, help="기록 폴더 또는 CSV (기본: 가장 최근)")
    args = parser.parse_args()
    files = find_files(args.path)

    lim = common_limits(files)                                          # 모든 모델·에피소드 같은 x, y 범위
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True, gridspec_kw={"height_ratios": [2, 1, 2, 1, 2, 1]})
    ax_main, ax_err = list(axes[0::2]), list(axes[1::2])                # θ0 각도, θ0 오차, θ1 각도, θ1 오차, θ2 각도, θ2 오차
    state = {"i": 0, "mode": "angle"}

    def redraw():
        draw(fig, ax_main, ax_err, files[state["i"]], state["i"], len(files), state["mode"], lim)
        tb = fig.canvas.toolbar                                         # 도구 막대 '집' = 공통 범위
        if tb is not None:
            tb.update()
            tb.push_current()

    def on_key(event):
        if event.key in ("right", "left"):                              # 모델 전환: 보던 범위(확대 포함) 그대로
            view = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
            state["i"] = (state["i"] + (1 if event.key == "right" else -1)) % len(files)
            redraw()
            for ax, (xl, yl) in zip(axes, view):
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            fig.canvas.draw_idle()
        elif event.key == "v":                                          # 각도 <-> 각속도: 시간 구간·오차 칸 범위 유지, 큰 칸 y는 그 보기의 공통 범위
            view = [(ax.get_xlim(), ax.get_ylim()) for ax in ax_err]
            state["mode"] = "velocity" if state["mode"] == "angle" else "angle"
            redraw()
            for ax, (xl, yl) in zip(ax_err, view):
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    # 터미널에서 띄운 창은 키보드 포커스를 못 받는 경우가 있어서, 마우스가 그래프 위에 오면 포커스를 가져옴 (Tk)
    if hasattr(fig.canvas, "get_tk_widget"):
        fig.canvas.mpl_connect("figure_enter_event", lambda e: fig.canvas.get_tk_widget().focus_force())
    draw(fig, ax_main, ax_err, files[0], 0, len(files), "angle", lim)
    fig.tight_layout()
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
