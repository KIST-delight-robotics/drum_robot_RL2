"""정책 명령 실시간 스코프 — play1.py --telemetry 가 쏘는 UDP 를 받아 그린다.

sim 프로세스와 완전히 독립이다. 아무 때나 켜고 끄고 재시작해도 되고,
sim 이 --headless 로 원격에서 돌아도 UDP 주소만 맞으면 된다.

    # sim
    ./isaaclab.sh -p drum_robot/scripts/reinforcement_learning/skrl/play1.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 \
        --checkpoint <ckpt> --midi_file <곡> --telemetry 9999

    # 뷰어 (다른 터미널)
    python drum_robot/scripts/analysis/scope.py --port 9999

화면: 관절 9개를 3×3 블록으로, 각 블록이 3단.
    1단 action   정책 원출력 [-1,1]  — ±1 에 붙으면 포화
    2단 angle    target(주황)=명령 / q(파랑)=실측 / 관절한계(빨강 점선)
                 주황과 파랑이 벌어지면 명령을 못 따라가는 것.
    3단 torque   액추에이터 토크 [N·m] / effort_limit(빨강 점선)
                 한계에 눕으면 포화 = PD 감쇠가 소실되는 구간.
"""
import argparse
import socket
import struct
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# play1.py 와 공유하는 패킷 포맷
HDR = struct.Struct("<4sI9f9f9f144s")   # magic, njoint, lo(9), hi(9), effort(9), names(9×16B)
DAT = struct.Struct("<4sIf36f")         # magic, seq, t, action(9), target(9), q(9), tau(9)
R2D = 180.0 / np.pi

# 눈금 간격 후보 — 5 계열 우선. 축 범위를 target 개로 나누기에 가장 가까운 값을 고른다.
#   그냥 5 로 고정하면 손목 토크(±2.5)는 눈금 1개, action(±1.2)은 0개가 되어 못 쓴다.
_STEPS = (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200)


def tick_step(span, target):
    return min(_STEPS, key=lambda s: abs(span / s - target))


def style(p, span, target):
    """주 눈금 + 보조 눈금(절반) + 양쪽 grid."""
    maj = tick_step(span, target)
    p.yaxis.set_major_locator(MultipleLocator(maj))
    p.yaxis.set_minor_locator(MultipleLocator(maj / 2))
    p.xaxis.set_minor_locator(MultipleLocator(0.1))          # 0.1초
    p.grid(which="major", alpha=.35, lw=.6)
    p.grid(which="minor", alpha=.15, lw=.4)
    p.tick_params(axis="both", which="major", labelsize=7)
    p.tick_params(axis="both", which="minor", length=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--host", default="0.0.0.0", help="바인드 주소 (원격 sim 이면 0.0.0.0)")
    ap.add_argument("--window", type=float, default=3.0, help="화면에 보이는 시간 [초]")
    ap.add_argument("--every", type=int, default=3, help="패킷 N개마다 재드로우 (3 ≈ 22Hz)")
    ap.add_argument("--dt", type=float, default=0.015, help="정책 주기 [초] — 창 길이 계산용")
    ap.add_argument("--gridlines", type=int, default=10,
                    help="y축 주 눈금 개수 목표. 크게 하면 촘촘해진다 (기본 10)")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((a.host, a.port))
    sock.setblocking(False)
    print(f"[SCOPE] {a.host}:{a.port} 수신 대기 — sim 의 헤더 패킷을 기다립니다 (Ctrl+C 종료)")

    # ── 헤더 대기 (관절이름·한계를 sim 이 알려준다) ────────────────────
    names, lo, hi, eff, nj = None, None, None, None, 0
    while names is None:
        try:
            buf, _ = sock.recvfrom(4096)
        except BlockingIOError:
            time.sleep(0.05)
            continue
        if len(buf) == HDR.size and buf[:4] == b"DRM0":
            f = HDR.unpack(buf)
            nj = f[1]
            lo = np.array(f[2:11], dtype=np.float64)
            hi = np.array(f[11:20], dtype=np.float64)
            eff = np.array(f[20:29], dtype=np.float64)
            names = [f[29][i * 16:(i + 1) * 16].rstrip(b"\0").decode() for i in range(nj)]
            print(f"[SCOPE] 연결됨 — {nj}관절 {names}")
            print(f"[SCOPE] effort_limit [N·m]: {[round(v, 2) for v in eff]}")

    W = max(20, int(round(a.window / a.dt)))
    t_ax = np.arange(W) * a.dt
    act = np.zeros((W, nj)); tgt = np.zeros((W, nj)); q = np.zeros((W, nj)); tau = np.zeros((W, nj))
    for j in range(nj):                                    # 각도는 한계 중앙에서 시작 (그림 튐 방지)
        tgt[:, j] = q[:, j] = 0.5 * (lo[j] + hi[j])

    # ── 그림: 관절 3×3 블록, 각 블록 2단 = 6행 3열 ────────────────────
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(9, 3, figsize=(15.5, 13.5), sharex=True)
    fig.canvas.manager.set_window_title(f"policy scope  udp:{a.port}")
    L = []
    for j in range(nj):
        br, c = j // 3, j % 3
        pa, pq, pt = ax[3 * br][c], ax[3 * br + 1][c], ax[3 * br + 2][c]

        pa.set_title(names[j], fontsize=9, pad=3)
        pa.axhline(1, color="crimson", ls="--", lw=.8)
        pa.axhline(-1, color="crimson", ls="--", lw=.8)
        pa.axhline(0, color="0.85", lw=.6)
        (l_a,) = pa.plot(t_ax, act[:, j], color="tab:green", lw=1.0, animated=True)
        pa.set_ylim(-1.2, 1.2); style(pa, 2.4, a.gridlines)
        if c == 0:
            pa.set_ylabel("action", fontsize=8)

        pad = 0.06 * (hi[j] - lo[j])
        pq.axhline(lo[j] * R2D, color="crimson", ls="--", lw=.8)
        pq.axhline(hi[j] * R2D, color="crimson", ls="--", lw=.8)
        (l_t,) = pq.plot(t_ax, tgt[:, j] * R2D, color="tab:orange", lw=1.1, animated=True)
        (l_q,) = pq.plot(t_ax, q[:, j] * R2D, color="tab:blue", lw=1.0, animated=True)
        pq.set_ylim((lo[j] - pad) * R2D, (hi[j] + pad) * R2D)
        style(pq, (hi[j] - lo[j] + 2 * pad) * R2D, a.gridlines)
        if c == 0:
            pq.set_ylabel("deg", fontsize=8)

        # 3단 — 액추에이터 토크 + effort_limit
        e = max(float(eff[j]), 1e-3)
        pt.axhline(e, color="crimson", ls="--", lw=.8)
        pt.axhline(-e, color="crimson", ls="--", lw=.8)
        pt.axhline(0, color="0.85", lw=.6)
        (l_u,) = pt.plot(t_ax, tau[:, j], color="tab:red", lw=1.0, animated=True)
        pt.set_ylim(-1.15 * e, 1.15 * e); style(pt, 2.3 * e, a.gridlines)
        if c == 0:
            pt.set_ylabel("N·m", fontsize=8)
        L.append((l_a, l_t, l_q, l_u))

    for c in range(3):
        ax[8][c].set_xlabel(f"last {a.window:.0f}s   [s]", fontsize=8)
    # 라벨은 ASCII (matplotlib 기본 폰트에 CJK 없음). 터미널 출력은 한글 유지.
    fig.suptitle("policy scope    green = action (+-1 saturated)    orange = target (commanded)    "
                 "blue = q (measured)    red = torque / limits", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .97])
    plt.show(block=False)
    fig.canvas.draw()

    bg = [fig.canvas.copy_from_bbox(fig.bbox)]

    def on_draw(_evt):                    # 창 크기 변경 등으로 재드로우되면 배경 다시 캡처
        bg[0] = fig.canvas.copy_from_bbox(fig.bbox)
    fig.canvas.mpl_connect("draw_event", on_draw)

    # ── 수신 루프 ──────────────────────────────────────────────────────
    n_rx = n_lost = n_draw = 0
    last_seq = None
    t_stat = time.time()
    try:
        while plt.fignum_exists(fig.number):
            got = 0
            while True:                    # 대기 중인 패킷을 모두 소비 (지연 누적 방지)
                try:
                    buf, _ = sock.recvfrom(4096)
                except BlockingIOError:
                    break
                if len(buf) != DAT.size or buf[:4] != b"DRM1":
                    continue               # 헤더 패킷 등은 무시
                f = DAT.unpack(buf)
                seq = f[1]
                if last_seq is not None and seq > last_seq + 1:
                    n_lost += seq - last_seq - 1
                last_seq = seq
                v = np.asarray(f[3:], dtype=np.float64)
                act[:-1] = act[1:]; act[-1] = v[0:9]
                tgt[:-1] = tgt[1:]; tgt[-1] = v[9:18]
                q[:-1] = q[1:];     q[-1] = v[18:27]
                tau[:-1] = tau[1:]; tau[-1] = v[27:36]
                n_rx += 1; got += 1

            if got and n_rx % a.every < got:
                fig.canvas.restore_region(bg[0])
                for j in range(nj):
                    l_a, l_t, l_q, l_u = L[j]
                    l_a.set_ydata(act[:, j])
                    l_t.set_ydata(tgt[:, j] * R2D)
                    l_q.set_ydata(q[:, j] * R2D)
                    l_u.set_ydata(tau[:, j])
                    br, c = j // 3, j % 3
                    ax[3 * br][c].draw_artist(l_a)
                    ax[3 * br + 1][c].draw_artist(l_t)
                    ax[3 * br + 1][c].draw_artist(l_q)
                    ax[3 * br + 2][c].draw_artist(l_u)
                fig.canvas.blit(fig.bbox)
                n_draw += 1

            fig.canvas.flush_events()
            if not got:
                time.sleep(0.002)

            now = time.time()
            if now - t_stat >= 2.0:
                print(f"\r[SCOPE] {n_rx/(now-t_stat):6.1f} pkt/s   draw {n_draw/(now-t_stat):5.1f}/s"
                      f"   유실 {n_lost}", end="", flush=True)
                n_rx = n_draw = 0
                t_stat = now
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("\n[SCOPE] 종료")


if __name__ == "__main__":
    main()
