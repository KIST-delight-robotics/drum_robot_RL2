"""MIDI 연주가능성 분류 — 로봇이 구조상 못 치는 곡을 학습 코퍼스에서 제외한다.

시뮬/GPU 없이 MIDI 파싱 + CPU 산술만 쓴다. 추정 파라미터는 하나도 안 들어간다
(이동속도 상한 v_max 같은 미측정 값은 의도적으로 제외 — 그걸로 데이터를 지우면 안 되니까).

판정에 쓰는 물리 제약 3개. 전부 코드에 있는 값에서 유도된다:
  ① 같은 드럼 재타  min_gap = max(2W+1, min_gap_s/step_dt) = 8스텝 = 120ms
       판정창(±W) 겹침 + 재장전 리프트 시간. RDS 로더가 이미 '조용히 드롭'하는 제약.
  ② 스틱 2개        한 시각에 팔 타격 3개 이상 불가. 킥·페달하이햇은 발이라 제외.
  ③ 한 팔 재타      t_rearm = 2W+1 = 7스텝 = 105ms. 드럼이 달라도 한 팔은 이보다 빨리 못 친다.

①과 ③은 결과가 다르므로 지표를 둘로 나눈다:
  fidel  = ①/노트   로더가 지워준다 → 학습엔 무해. "이 곡의 몇 %를 아예 안 치는가" (곡 변형도)
  poison = (②+③)/노트  로더가 안 지운다 → 악보에 남아 miss 페널티가 영구히 걸린다. "학습을 망치는 비율"

③ 배정은 drumrobot_env._compute_arm_schedule 과 같은 greedy (CROSS 페널티 포함),
학습이 실제로 보는 단위인 '마디 세그먼트'(팔 상태가 매 에피소드 리셋)로 끊어서 판정한다.

사용:
    /home/shy/anaconda3/envs/env_isaaclab/bin/python scripts/analysis/midi_playability.py --csv out.csv
    python scripts/analysis/midi_playability.py --file MIDIs/xxx.mid -v
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))

from tasks.drumrobot.components.robotic_drum_score import (   # noqa: E402
    GENERAL_MIDI_PERCUSSION_KEY_MAP,
    RDSCfg,
    read_rds_txt,
)

from mido import MidiFile   # noqa: E402


# ===== 환경 상수 (drumrobot_cfg / hit_detector 와 같은 값) =====
SIM_DT = 0.0075
DECIMATION = 2
STEP_DT = SIM_DT * DECIMATION          # 0.015 s = 66.667 Hz (정책 주기)
HIT_WINDOW_STEP = 3                    # HitDetectorCfg.hit_window_step
T_REARM = 2 * HIT_WINDOW_STEP + 1      # 7 스텝 = 105 ms — 한 팔 최소 재타 간격
EPISODE_LENGTH_S = 5.0
EPISODE_T = int(EPISODE_LENGTH_S / STEP_DT)
SAFE_T = EPISODE_T - HIT_WINDOW_STEP   # 에피소드 끝 W 스텝은 채점 안 함
CROSS_PENALTY = 1.5                    # DrumRobotEnvCfg.sched_cross_penalty
BIG = 1.0e3


def read_midi(path: Path) -> tuple[float, list[tuple[float, str, int]]]:
    """robotic_drum_score._read_midi_file 과 같은 규칙 (채널 9, note_on vel>0)."""
    mid = MidiFile(path)
    tpb = mid.ticks_per_beat
    bpm = 120.0
    for track in mid.tracks:        # 첫 set_tempo 를 전체 bpm 으로 (로더와 동일)
        found = False
        for msg in track:
            if msg.type == "set_tempo":
                bpm, found = 60_000_000 / msg.tempo, True
                break
        if found:
            break

    events: list[tuple[float, str, int]] = []
    for track in mid.tracks:
        cur, tempo = 0, 500000
        for msg in track:
            cur += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0 and getattr(msg, "channel", None) == 9:
                inst = GENERAL_MIDI_PERCUSSION_KEY_MAP.get(msg.note, f"Unknown({msg.note})")
                events.append(((cur * tempo) / (tpb * 1_000_000), inst, msg.velocity))
    events.sort()
    return bpm, events


def split_measures(events, bpm, slow_factor, cfg):
    """RDS._generate_rds_from_midi 의 마디 분할을 그대로 재현.
    각 세그먼트 = 그 구간 첫 이벤트부터 1마디(4/4) 안의 이벤트들. 학습이 보는 에피소드 1개에 대응.
    반환: [[(time_idx, drum_idx), ...], ...] (① 적용 전), 세그먼트별 드롭 수"""
    measure_duration = 4 * (60.0 / bpm)
    segments, drops = [], 0
    i, n = 0, len(events)
    while i < n:
        first_t = events[i][0]
        placed: list[tuple[int, int]] = []
        last_idx: dict[int, int] = {}
        j = i
        while j < n and events[j][0] <= first_t + measure_duration:
            t, inst, _vel = events[j]
            j += 1
            if inst in cfg.pedal_instruments or inst not in cfg.note_to_drum_idx:
                continue                                   # 페달/미지원 악기는 팔 작업이 아님
            drum = cfg.note_to_drum_idx[inst]
            time_idx = int(round(slow_factor * (t - first_t) / STEP_DT)) + cfg.start_offset_steps
            if time_idx >= SAFE_T:
                continue
            if time_idx - last_idx.get(drum, -10 ** 9) < MIN_GAP:
                drops += 1                                 # ① 로더가 드롭하는 노트
                continue
            placed.append((time_idx, drum))
            last_idx[drum] = time_idx
        if placed:
            segments.append(placed)
        i = j if j > i else i + 1
    return segments, drops


def schedule_segment(placed, drum_pos):
    """세그먼트 하나를 양팔에 배정. drumrobot_env._compute_arm_schedule 과 같은 greedy.
    반환: (②초과분, ③불가 노트 수)"""
    by_step: dict[int, list[int]] = {}
    for s, d in placed:
        by_step.setdefault(s, []).append(d)

    excess = 0
    arm_pos: list[tuple[float, float, float] | None] = [None, None]   # 팔 투영위치 (0=왼, 1=오른)
    arm_free = [0.0, 0.0]        # 그 팔이 다시 칠 수 있는 최소 스텝
    busy = 0

    for s in sorted(by_step):
        drums = by_step[s]
        if len(drums) > 2:
            # env 스케줄러는 드럼 인덱스가 큰 상위 2개만 본다. 나머지는 물리적으로 불가.
            excess += len(drums) - 2
            drums = sorted(drums, reverse=True)[:2]

        def score(a: int, d: int) -> float:
            p = drum_pos[d]
            w = s - arm_free[a]
            cross = max(p[0], 0.0) if a == 0 else max(-p[0], 0.0)   # 반대편 침범량 [m]
            travel = (0.0 if arm_pos[a] is None else math.dist(arm_pos[a], p)) + CROSS_PENALTY * cross
            return travel / max(w, 1.0) + BIG * max(-w, 0.0)

        if len(drums) == 1:
            d = drums[0]
            a = 0 if score(0, d) <= score(1, d) else 1
            chosen = [(d, a)]
        else:
            d0, d1 = drums
            chosen = [(d0, 0), (d1, 1)] if score(0, d0) + score(1, d1) <= score(1, d0) + score(0, d1) \
                else [(d0, 1), (d1, 0)]

        for d, a in chosen:
            if s - arm_free[a] < 0:
                busy += 1                     # ③ 그 팔이 아직 재장전 중 — 어느 팔로도 불가
            arm_pos[a] = drum_pos[d]
            arm_free[a] = s + T_REARM
    return excess, busy


def analyze(path: Path, cfg: RDSCfg, drum_pos, slow_factor: float) -> dict:
    bpm, events = read_rds_txt(path) if path.suffix.lower() == ".txt" else read_midi(path)
    segments, drops = split_measures(events, bpm, slow_factor, cfg)
    n_placed = sum(len(s) for s in segments)
    n_raw = n_placed + drops
    if n_raw == 0:
        return {"file": path.name, "n_raw": 0, "skip": True}

    excess = busy = 0
    for seg in segments:
        e, b = schedule_segment(seg, drum_pos)
        excess += e
        busy += b

    span = max((s[-1][0] - s[0][0]) for s in segments) if segments else 0
    dur = sum((s[-1][0] - s[0][0] + 1) for s in segments) * STEP_DT
    return {
        "file": path.name,
        "skip": False,
        "bpm": round(bpm, 1),
        "n_seg": len(segments),
        "n_raw": n_raw,
        "density": round(n_raw / dur, 2) if dur > 0 else 0.0,
        "drop_same": drops,
        "chord_excess": excess,
        "busy": busy,
        "fidel": round(drops / n_raw, 4),                      # ① 사라지는 비율 (곡 변형도)
        "poison": round((excess + busy) / n_raw, 4),           # ②+③ 학습 오염 비율
        "_span": span,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None, help="MIDI 폴더 (기본: RDSCfg.midi_folder_path)")
    ap.add_argument("--file", default=None, help="한 파일만 상세 분석")
    ap.add_argument("--slow_factor", type=float, default=1.0, help="악보 시간 배율 (학습과 맞출 것)")
    ap.add_argument("--csv", default=None, help="곡별 결과 CSV 경로")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    cfg = RDSCfg()
    from tasks.drumrobot.components.specs import Instruments
    drum_pos = [i.position for i in Instruments().all.values()]   # idx 순서 = note_to_drum_idx 값

    files = [Path(a.file)] if a.file else sorted(Path(a.folder or cfg.midi_folder_path).glob("*.mid"))
    print(f"[분류] {len(files)}개 | step_dt={STEP_DT*1000:.1f}ms  ①min_gap={MIN_GAP}스텝({MIN_GAP*STEP_DT*1000:.0f}ms)"
          f"  ③t_rearm={T_REARM}스텝({T_REARM*STEP_DT*1000:.0f}ms)  slow={a.slow_factor}  (추정 파라미터 없음)")

    rows = []
    for f in files:
        try:
            r = analyze(f, cfg, drum_pos, a.slow_factor)
        except Exception as e:
            r = {"file": f.name, "skip": True, "n_raw": 0, "err": str(e)[:80]}
        rows.append(r)
        if a.verbose or a.file:
            print(r)

    ok = [r for r in rows if not r["skip"]]
    n_err = len(rows) - len(ok)
    tot = sum(r["n_raw"] for r in ok)
    print(f"\n=== 전체 {len(ok)}곡 / 팔 노트 {tot}개 (파싱 실패·빈 곡 {n_err}개) ===")
    print(f"  ① 드롭 {sum(r['drop_same'] for r in ok)/tot*100:.1f}%"
          f"  | ② 동시3타 {sum(r['chord_excess'] for r in ok)}개"
          f"  | ③ 한팔 재장전 위반 {sum(r['busy'] for r in ok)/tot*100:.1f}%"
          f"  → poison {sum(r['chord_excess']+r['busy'] for r in ok)/tot*100:.1f}%")

    print("\n=== 임계값별 생존 ===")
    print(f"{'poison<':>8} {'fidel<':>7} {'곡':>6} {'노트':>8} {'잔여poison':>11} {'잔여fidel':>10}")
    for p, f_ in ((1.01, 1.01), (0.15, 0.40), (0.10, 0.30), (0.05, 0.20), (0.02, 0.10)):
        s = [r for r in ok if r["poison"] < p and r["fidel"] < f_]
        if not s:
            continue
        n = sum(r["n_raw"] for r in s)
        pp = sum((r["chord_excess"] + r["busy"]) for r in s) / n
        ff = sum(r["drop_same"] for r in s) / n
        print(f"{p:8.2f} {f_:7.2f} {len(s):6d} {n:8d} {pp*100:10.1f}% {ff*100:9.1f}%")

    print("\n=== poison 최악 10곡 (팔이 못 따라가는 곡) ===")
    for r in sorted(ok, key=lambda r: -r["poison"])[:10]:
        print(f"  poison {r['poison']*100:5.1f}%  fidel {r['fidel']*100:5.1f}%  {r['density']:5.1f}노트/s  {r['file']}")
    print("\n=== fidel 최악 10곡 (더블스트로크로 통째로 사라지는 곡) ===")
    for r in sorted(ok, key=lambda r: -r["fidel"])[:10]:
        print(f"  poison {r['poison']*100:5.1f}%  fidel {r['fidel']*100:5.1f}%  {r['density']:5.1f}노트/s  {r['file']}")

    if a.csv:
        keys = ["file", "bpm", "n_seg", "n_raw", "density", "drop_same", "chord_excess", "busy", "fidel", "poison"]
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in ok:
                w.writerow(r)
        print(f"\n[CSV] {a.csv}  ({len(ok)}곡)")


MIN_GAP = max(2 * HIT_WINDOW_STEP + 1, int(round(RDSCfg.min_gap_s / STEP_DT)))

if __name__ == "__main__":
    main()
