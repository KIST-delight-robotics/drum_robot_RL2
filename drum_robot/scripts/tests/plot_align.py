"""
[확인용 임시 스크립트] play_align.py / play_song.py가 저장한 CSV를 matplotlib 창으로 보기 (확대·이동 가능)

실행 (레포 루트에서, env_isaaclab의 python으로. Isaac Sim 필요 없음)
    python drum_robot/scripts/tests/plot_align.py                      # 가장 최근 기록 폴더 (play_align 또는 play_song)
    python drum_robot/scripts/tests/plot_align.py <폴더>               # 그 폴더의 *.csv 전부 (play_song 곡 폴더 = 모델별 CSV)
    python drum_robot/scripts/tests/plot_align.py <episode_001.csv>    # 파일 하나
    python drum_robot/scripts/tests/plot_align.py <폴더> --share_y <다른 폴더> ...   # y 범위를 다른 폴더 CSV까지 합쳐서 정함 (보여주는 건 <폴더>만)
    python drum_robot/scripts/tests/plot_align.py <폴더> --joints 0     # θ0(허리) 칸만 (0 = 허리, 1 = 오른 어깨 yaw, 2 = 왼 어깨 yaw, 예: 0,2)
조작
    ← / →        이전 / 다음 에피소드 (play_song 곡 폴더면 이전 / 다음 모델)
    v            큰 칸: 각도 보기 <-> 각속도 보기 (확대한 시간 구간은 그대로 유지)
    축 범위      y = 모든 모델·에피소드 같은 범위 (칸·보기별 전체 최소~최대), 모델을 넘겨도 보던 y 범위(확대 포함) 유지
                 x = 파일마다 자기 기록 전체 (0 ~ 끝), 모델을 넘기면 새 파일 전체로 (1배·10배처럼 길이가 크게 달라도 보기 좋게)
    창 아래 도구 막대: 돋보기 = 확대, 십자 = 이동, 집 = 이 파일 전체 x + 공통 y (여섯 칸이 시간축을 같이 씀)
그래프: 9칸 = θ0, θ1, θ2 각각 [큰 칸: 각도 또는 각속도] + [작은 칸: 위치 오차] + [작은 칸: 토크]
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
    작은 칸 (토크) [N·m] = PD 추정값 (CSV에 토크가 없어서 계산)
        Kp·(RL_desired − output) − Kd·각속도 (각속도 = 직전 스텝과의 차이), 토크 한계로 자름
        Kp·Kd·한계 = 학습 폴더 params/env.yaml 의 액추에이터 값 (θ0 = waist, θ1·θ2 = arm). play 때 cfg 가 학습 때와 같다고 가정
        제목: |토크| 최대값
    맨 아래 칸 (보상 항) [스텝당 보상] = play 때 env가 계산한 값 그대로 (예전 CSV에는 없음)
        정렬 (파랑), 행동 변화 (주황), 허리 속도 (초록), 회색 바탕 = 속도 벌 켜짐 (s ≥ vel_free_ratio·D)
        점선: 허리 오차 1°, 10°일 때의 정렬 벌 (가중치는 기록에서 역산: Σ(−정렬 벌) / Σ|오차 rad|)
        범례: 에피소드 합 Σ 와 세 항 중 비중. 값은 play 때 reward.py 가중치 기준 (학습 때 가중치와 다를 수 있음)
    세로선: 음표 시각 (파랑 = 왼손, 빨강 = 오른손)
    점 (각도 칸, 위치 오차 칸): 타격 스텝의 기록값. 점을 클릭하면 그 스텝의 desired / RL_desired / output / 오차를 점 옆에 표시하고 터미널에도 출력
    y 눈금 (각도 칸, 위치 오차 칸): 1° 간격 옅은 격자 + 5° 간격 격자. 숫자는 겹치지 않게 칸 높이에 맞춰 1° / 2° / 5° / 10° / 20° / 50°마다 (확대하면 1°마다)
"""

import argparse
import csv
import glob
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# matplotlib 기본 단축키와 겹치는 키 해제 (← / → / v 는 이 스크립트에서 씀)
for _k, _keys in (("keymap.back", ("left", "c")), ("keymap.forward", ("right", "v"))):
    plt.rcParams[_k] = [x for x in plt.rcParams[_k] if x not in _keys]

REPO = Path(__file__).resolve().parents[3]
PANELS = [("θ0  waist", "waist_tgt", "waist_cmd", "waist_act"),
          ("θ1  right shoulder yaw", "yaw_r_tgt", "yaw_r_cmd", "yaw_r_act"),
          ("θ2  left shoulder yaw", "yaw_l_tgt", "yaw_l_cmd", "yaw_l_act")]
ACTUATOR = ["waist", "arm", "arm"]          # 칸별 액추에이터 그룹 (env.yaml 이름)
REW = [("rew_align", "align", "tab:blue"),  # 보상 항 칸 (CSV 열, 이름, 색)
       ("rew_act_rate", "action rate", "tab:orange"),
       ("rew_vel", "waist vel", "tab:green")]
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


_GAINS: dict = {}


def gains(path: Path) -> dict:
    # 학습 폴더 params/env.yaml 의 액추에이터별 Kp·Kd·토크 한계 {"waist": {...}, "arm": {...}} (없으면 {})
    if path in _GAINS:
        return _GAINS[path]
    run = path.name.split("__")[0] if "__" in path.name \
        else next((p.name for p in path.parents if p.parent.name == "drum_robot"), "")   # play_song / play_align
    y = REPO / "logs" / "skrl" / "drum_robot" / run / "params" / "env.yaml"
    txt = path.with_suffix(".txt")                                          # play_song txt 의 checkpoint (실험 복사본 등 다른 레포 모델)
    ckpt = next((ln[len("checkpoint: "):].strip() for ln in open(txt) if ln.startswith("checkpoint:")), None) if txt.exists() else None
    if ckpt:
        run = Path(ckpt).parents[1].name
        y_ckpt = Path(ckpt).parents[1] / "params" / "env.yaml"
        y = y_ckpt if y_ckpt.exists() else REPO / "logs" / "skrl" / "drum_robot" / run / "params" / "env.yaml"   # 학습 폴더를 옮겼으면 메인 레포 logs
    out, name, inside = {}, None, False
    for line in (y.read_text().splitlines() if run and y.exists() else []):
        if line.strip() == "actuators:":
            inside = True
        elif inside:
            if line and not line.startswith("    "):                       # actuators 블록 끝
                break
            m = re.match(r"^    (\w+):\s*$", line)
            if m:
                name = m.group(1)
                out[name] = {}
            m = re.match(r"^      (stiffness|damping|effort_limit_sim):\s*([-+\d.eE]+)\s*$", line)
            if m and name:
                out[name][m.group(1)] = float(m.group(2))
    _GAINS[path] = {k: v for k, v in out.items() if len(v) == 3}
    return _GAINS[path]


def torque(d: dict, cmd: str, act: str, g: dict) -> list | None:
    # PD 추정 토크 [N·m] = Kp·(명령 − 실제) − Kd·각속도 (CSV는 deg, 각속도 = 직전 스텝과의 차이), 토크 한계로 자름
    if cmd not in d or not g:
        return None
    t, q, lim = d["t"], d[act], g["effort_limit_sim"]
    tau = []
    for k in range(len(t)):
        w = math.radians(q[k] - q[k - 1]) / (t[k] - t[k - 1]) if k else 0.0
        x = g["stiffness"] * math.radians(d[cmd][k] - q[k]) - g["damping"] * w
        tau.append(max(-lim, min(lim, x)))
    return tau


def velocity(t: list, y: list) -> tuple[list, list]:
    # 각속도 [rad/s] = 각도 차이 (CSV는 deg) / 시간 차이, 시각은 두 스텝의 가운데
    tv = [(t[k] + t[k - 1]) / 2 for k in range(1, len(t))]
    v = [math.radians(y[k] - y[k - 1]) / (t[k] - t[k - 1]) for k in range(1, len(t))]
    return tv, v


def errors(d: dict, tgt: str, cmd: str, act: str) -> tuple[list, list | None]:
    # 위치 오차 [deg]: desired − output, RL_desired − output (명령 열이 없으면 None)
    des = d["waist_ref"] if tgt == "waist_tgt" and not all(math.isnan(x) for x in d.get("waist_ref", [math.nan])) else d[tgt]   # θ0_d 재생이면 θ0_d
    e_task = [a - b for a, b in zip(des, d[act])]
    e_track = [a - b for a, b in zip(d[cmd], d[act])] if cmd in d else None
    return e_task, e_track


def mean(v: list) -> float:
    return sum(v) / len(v) if v else float("nan")


def near_hit(d: dict) -> list:
    # 가장 가까운 음표까지 0.1초 이하인 스텝
    return [min([x for x in (a, b) if not math.isnan(x)], default=99.0) <= 0.1 for a, b in zip(d["note_l_s"], d["note_r_s"])]


def common_limits(files: list[Path]) -> dict:
    # 모든 파일(모델·에피소드)에 같은 y 범위: 칸·보기별 전체 최소 ~ 최대 (+5% 여유). x 는 파일마다 따로 (draw)
    lim = {}
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
        vals = [x for f in files for x in (torque(load_cached(f), cmd, act, gains(f).get(ACTUATOR[i])) or [])]
        lo, hi = (min(vals), max(vals)) if vals else (-1.0, 1.0)
        pad = max(0.05 * (hi - lo), 0.5)
        lim[("torque", i)] = (lo - pad, hi + pad)
    vals = [x for f in files for k, _, _ in REW if k in load_cached(f) for x in load_cached(f)[k]]
    lo = min(vals + [-0.2])                                             # 보상 항 칸: 0 위로는 안 감 (전부 벌)
    lim["reward"] = (lo * 1.05, -0.02 * lo)
    return lim


def note_lines(ax, d: dict):
    for k, color in (("hit_l", "tab:blue"), ("hit_r", "tab:red")):
        for x, h in zip(d["t"], d[k]):
            if h:
                ax.axvline(x, color=color, alpha=0.35, lw=1)


def hit_points(ax, t: list, series: list, idx: list, info):
    # 타격 스텝(세로선 위치)마다 그 스텝의 기록값에 점. 클릭하면 info(스텝) 글을 표시 (main 의 on_pick). series = [(값 목록, 색), ...]
    for ys, color in series:
        pts, = ax.plot([t[j] for j in idx], [ys[j] for j in idx], "o", color=color, ms=4, zorder=5, picker=5, label="_nolegend_")
        pts.hit_idx, pts.hit_info = idx, info


def deg_ticks(ax):
    # 각도 칸 y 눈금: 1° 간격 옅은 격자 + 5° 간격 격자, 숫자는 보이는 범위와 칸 높이에 맞춰 겹치지 않는 간격으로 (그릴 때마다 다시 정함)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    def every():
        lo, hi = ax.get_ylim()
        room = max(2.0, ax.get_window_extent().height / 14)            # 숫자 하나에 약 14 px
        return next((s for s in (1, 2, 5, 10, 20, 50) if (hi - lo) / s <= room), 100)

    def fmt_major(y, _):                                                # 5°의 배수
        return f"{y:.0f}" if round(y) % every() == 0 else ""

    def fmt_minor(y, _):                                                # 나머지 1° 눈금 (5°의 배수는 위에서 씀)
        e = every()
        return f"{y:.0f}" if e < 5 and round(y) % e == 0 and round(y) % 5 != 0 else ""

    ax.yaxis.set_major_formatter(FuncFormatter(fmt_major))
    ax.yaxis.set_minor_formatter(FuncFormatter(fmt_minor))
    ax.fmt_ydata = lambda y: f"{y:.4f}"                                 # 커서 위치 y값은 소수 2자리 (눈금 형식과 따로)
    ax.grid(which="minor", axis="y", alpha=0.12, lw=0.5)
    ax.tick_params(axis="y", which="both", labelsize=8)


def draw_reward(ax, d: dict, lim: dict):
    # 보상 항 칸: 스텝당 정렬·행동 변화·허리 속도, 회색 = 속도 벌 켜짐, 점선 = 허리 오차 1°·10°의 정렬 벌
    ax.clear()
    t = d["t"]
    if "rew_align" in d:
        ax.fill_between(t, 0, 1, where=[g > 0.5 for g in d["vel_gate"]], step="post", color="gray", alpha=0.15,
                        transform=ax.get_xaxis_transform(), label="vel penalty on")
        tot = {k: sum(d[k]) for k, _, _ in REW}
        all_ = sum(abs(v) for v in tot.values()) or 1.0
        for k, name, color in REW:
            ax.plot(t, d[k], "-", color=color, lw=1.2, drawstyle="steps-post",
                    label=f"{name}  Σ {tot[k]:.2f} ({abs(tot[k]) / all_:.0%})")
        e = sum(abs(math.radians(a - b)) for a, b in zip(d["waist_tgt"], d["waist_act"]))
        w = -tot["rew_align"] / e if e > 0 else 1.0                       # 정렬 가중치 (오차 rad 당 벌, 기록에서 역산)
        for deg in (1, 10):
            y = -w * math.radians(deg)
            ax.axhline(y, color="black", lw=0.8, ls=":")
            ax.text(t[-1], y, f"{deg}° err  ", fontsize=8, ha="right", va="bottom")
        ax.set_title(f"reward terms per step (env values at play time, align weight {w:.2f})   gray = vel penalty on   "
                     f"Σ total {sum(tot.values()):.2f}", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.set_title("reward terms: n/a (old CSV, re-run play_song)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.8)
    note_lines(ax, d)
    ax.set_ylabel("reward")
    ax.grid(alpha=0.3)
    ax.set_ylim(*lim["reward"])
    ax.set_xlabel(NOTE_TEXT)


def draw(fig, ax_main, ax_err, ax_tau, ax_rew, path: Path, i: int, n: int, mode: str, lim: dict):
    d = load_cached(path)
    t = d["t"]
    near = near_hit(d)
    hit_idx = [j for j in range(len(t)) if d["hit_l"][j] or d["hit_r"][j]]   # 타격 스텝 (세로선 위치)
    for k, (title, tgt, cmd, act) in enumerate(PANELS):
        des = d["waist_ref"] if tgt == "waist_tgt" and not all(math.isnan(x) for x in d.get("waist_ref", [math.nan])) else d[tgt]   # errors() 와 같은 desired

        def info(j, title=title, des=des, cmd=cmd, act=act):          # 점 클릭 때 보일 그 스텝 값 [deg]
            hands = "+".join(h for h, c in (("L", "hit_l"), ("R", "hit_r")) if d[c][j])
            s = f"{title.split()[0]}  t {t[j]:.3f}s  hit {hands}\ndesired {des[j]:.2f}"
            s += f"\nRL_desired {d[cmd][j]:.2f}" if cmd in d else ""
            s += f"\noutput {d[act][j]:.2f}\ndesired−output {des[j] - d[act][j]:.2f}"
            s += f"\nRL_desired−output {d[cmd][j] - d[act][j]:.2f}" if cmd in d else ""
            return s + "  deg"

        # 큰 칸: 각도 또는 각속도
        ax = ax_main[k]
        ax.clear()
        if mode == "angle":
            if tgt == "waist_tgt" and not all(math.isnan(x) for x in d.get("waist_ref", [math.nan])):
                ax.plot(t, d["waist_ref"], "-", color="black", lw=1.0, label="desired θ0_d")   # θ0_d 재생이면 θ0* 대신 (24 = 직선, 25 = 최소 저크)
            else:
                ax.plot(t, d[tgt], "--", color="black", lw=1.2, label="desired")
            if cmd in d:                                                # 예전 CSV에는 명령 열이 없음
                ax.plot(t, d[cmd], "-", color="tab:orange", lw=1.3, label="RL_desired",
                        drawstyle="steps-post")                         # 정책 스텝(15 ms) 동안 목표 유지 = 계단 (실기 motors.csv 와 같은 모양)
            ax.plot(t, d[act], "-", color="tab:green", lw=1.5, label="output")
            hit_points(ax, t, [(des, "black")] + ([(d[cmd], "tab:orange")] if cmd in d else []) + [(d[act], "tab:green")], hit_idx, info)
            ax.set_ylabel("deg")
            ax.set_title(f"{title}   angle", fontsize=10)
        else:
            peaks = []
            if tgt == "waist_tgt" and not all(math.isnan(x) for x in d.get("waist_ref", [math.nan])):
                tv, v = velocity(t, d["waist_ref"])                    # θ̇0_d: 기준 궤적 θ0_d 의 스텝 차이 (속도 추종 보상의 목표)
                ax.plot(tv, v, "-", color="black", lw=1.0, label="desired θ̇0_d")
                peaks.append(f"desired {max(abs(x) for x in v):.1f}")
            if cmd in d:
                tv, v = velocity(t, d[cmd])
                ax.plot(t[:-1], v, "-", color="tab:orange", lw=1.3, label="RL_desired",
                        drawstyle="steps-post")                         # 정책 스텝 [t_k-1, t_k] 동안 같은 명령 속도 = 계단 (구간 시작에 놓음)
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
        if mode == "angle":
            deg_ticks(ax)

        # 작은 칸: 위치 오차 (항상)
        ax = ax_err[k]
        ax.clear()
        e_task, e_track = errors(d, tgt, cmd, act)
        ax.plot(t, e_task, "-", color="black", lw=1.2, label="desired − output")
        text = f"desired−output  mean |e| {mean([abs(x) for x in e_task]):.1f},  near hit {mean([abs(x) for x, m in zip(e_task, near) if m]):.1f}"
        if e_track is not None:
            ax.plot(t, e_track, "-", color="tab:orange", lw=1.1, label="RL_desired − output")
            text += f"   |   RL_desired−output  mean |e| {mean([abs(x) for x in e_track]):.1f},  near hit {mean([abs(x) for x, m in zip(e_track, near) if m]):.1f}"
        hit_points(ax, t, [(e_task, "black")] + ([(e_track, "tab:orange")] if e_track is not None else []), hit_idx, info)
        ax.axhline(0, color="gray", lw=0.8)
        note_lines(ax, d)
        ax.set_ylabel("err deg")
        ax.set_title(f"{title.split()[0]}  position error   {text}  deg", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(*lim[("error", k)])
        deg_ticks(ax)

        # 작은 칸: 토크 (PD 추정, 항상)
        ax = ax_tau[k]
        ax.clear()
        g = gains(path).get(ACTUATOR[k])
        tau = torque(d, cmd, act, g)
        if tau is not None:
            ax.plot(t, tau, "-", color="tab:purple", lw=1.2, label="torque (est.)")
            ax.set_title(f"{title.split()[0]}  torque (est.) = Kp {g['stiffness']:g}·(RL_desired − output) − Kd {g['damping']:g}·ω,"
                         f" clip ±{g['effort_limit_sim']:g}   max |τ| {max(abs(x) for x in tau):.1f} N·m", fontsize=9)
        else:
            ax.set_title(f"{title.split()[0]}  torque: n/a (no RL_desired column or no params/env.yaml)", fontsize=9)
        ax.axhline(0, color="gray", lw=0.8)
        note_lines(ax, d)
        ax.set_ylabel("N·m")
        ax.grid(alpha=0.3)
        ax.set_ylim(*lim[("torque", k)])

    draw_reward(ax_rew, d, lim)                                         # 맨 아래 칸: 보상 항
    ax_main[0].set_xlim(0.0, t[-1])                                     # x = 이 파일 기록 전체
    ax_main[0].legend(loc="upper right", fontsize=8)
    ax_err[0].legend(loc="upper right", fontsize=8)
    other = "angle" if mode == "velocity" else "angular velocity"
    fig.suptitle(f"{path.parent.name}/{path.name}   ({i + 1}/{n})   [{'angle' if mode == 'angle' else 'angular velocity'} + position error + torque + reward]"
                 f"   ← / → : model,  v : {other}", fontsize=11)
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="play_align / play_song CSV 보기")
    parser.add_argument("path", nargs="?", default=None, help="기록 폴더 또는 CSV (기본: 가장 최근)")
    parser.add_argument("--share_y", nargs="+", default=[], help="y 범위만 같이 맞출 다른 기록 폴더 또는 CSV (창에는 안 나옴)")
    parser.add_argument("--joints", type=str, default="0,1,2", help="보여줄 관절 칸 (0 = 허리, 1 = 오른 어깨 yaw, 2 = 왼 어깨 yaw), 예: 0")
    args = parser.parse_args()
    files = find_files(args.path)
    others = [f for p in args.share_y for f in find_files(p)]

    global PANELS, ACTUATOR                                             # 고른 관절 칸만 남김 (draw·common_limits 가 이 목록을 씀)
    sel = [int(x) for x in args.joints.split(",")]
    PANELS, ACTUATOR = [PANELS[j] for j in sel], [ACTUATOR[j] for j in sel]

    lim = common_limits(files + others)                                 # 모든 모델·에피소드 같은 y 범위 (--share_y 파일 포함)
    fig, axes = plt.subplots(3 * len(PANELS) + 1, 1, figsize=(13, 3.5 + 4 * len(PANELS)), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1] * len(PANELS) + [1.5]})
    joint_axes, ax_rew = axes[:-1], axes[-1]                            # 맨 아래 칸 = 보상 항
    ax_main, ax_err, ax_tau = list(joint_axes[0::3]), list(joint_axes[1::3]), list(joint_axes[2::3])   # θ0 각도, θ0 오차, θ0 토크, θ1 ..., θ2 ...
    state = {"i": 0, "mode": "angle"}

    def redraw():
        draw(fig, ax_main, ax_err, ax_tau, ax_rew, files[state["i"]], state["i"], len(files), state["mode"], lim)
        tb = fig.canvas.toolbar                                         # 도구 막대 '집' = 공통 범위
        if tb is not None:
            tb.update()
            tb.push_current()

    def on_key(event):
        if event.key in ("right", "left"):                              # 모델 전환: 보던 y 범위(확대 포함) 그대로, x 는 새 파일 전체
            view = [ax.get_ylim() for ax in axes]
            state["i"] = (state["i"] + (1 if event.key == "right" else -1)) % len(files)
            redraw()
            for ax, yl in zip(axes, view):
                ax.set_ylim(yl)
            fig.canvas.draw_idle()
        elif event.key == "v":                                          # 각도 <-> 각속도: 시간 구간·오차·토크·보상 칸 범위 유지, 큰 칸 y는 그 보기의 공통 범위
            view = [(ax.get_xlim(), ax.get_ylim()) for ax in ax_err + ax_tau + [ax_rew]]
            state["mode"] = "velocity" if state["mode"] == "angle" else "angle"
            redraw()
            for ax, (xl, yl) in zip(ax_err + ax_tau + [ax_rew], view):
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            fig.canvas.draw_idle()

    note = {"ann": None, "mouse": None}

    def on_pick(event):                                                 # 타격 점 클릭: 그 스텝 값을 점 옆에 표시 + 터미널 출력
        a = event.artist
        if not hasattr(a, "hit_idx") or event.mouseevent is note["mouse"]:   # 한 번 클릭에 점이 여러 개 걸리면 첫 점만
            return
        note["mouse"] = event.mouseevent
        n = min(event.ind, key=lambda m: abs(a.get_xdata()[m] - event.mouseevent.xdata))   # 클릭한 x 에 가장 가까운 점
        text = a.hit_info(a.hit_idx[n])
        if note["ann"] is not None:
            try:
                note["ann"].remove()
            except (ValueError, NotImplementedError):                   # 다시 그리면서 (ax.clear) 이미 지워짐
                pass
        note["ann"] = a.axes.annotate(text, (a.get_xdata()[n], a.get_ydata()[n]), xytext=(10, 10), textcoords="offset points",
                                      fontsize=8, zorder=10, bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
        print(text.replace("\n", "   "))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    # 터미널에서 띄운 창은 키보드 포커스를 못 받는 경우가 있어서, 마우스가 그래프 위에 오면 포커스를 가져옴 (Tk)
    if hasattr(fig.canvas, "get_tk_widget"):
        fig.canvas.mpl_connect("figure_enter_event", lambda e: fig.canvas.get_tk_widget().focus_force())
    draw(fig, ax_main, ax_err, ax_tau, ax_rew, files[0], 0, len(files), "angle", lim)
    fig.tight_layout()
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
