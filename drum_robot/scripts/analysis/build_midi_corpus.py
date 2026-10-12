"""학습용 MIDI 코퍼스 생성 — midi_playability 판정으로 통과한 곡만 새 폴더에 모은다.

원본 MIDIs/ 는 읽기만 하고 절대 수정하지 않는다 (원본이 곧 백업).
되돌리려면 생성된 폴더를 지우면 끝.

생성물:
    <out>/*.mid           통과한 곡 (복사본)
    <out>/manifest.json   생성 조건 + 곡별 지표 + 제외 목록 (제외 사유 포함)

manifest 에 제약값(step_dt, W, min_gap, t_rearm)을 박아두는 이유:
제어주기나 hit_window 를 바꾸면 이 코퍼스는 낡은 것이 된다. 나중에 "무슨 기준으로 뽑았지"가 되지 않도록.

사용:
    /home/shy/anaconda3/envs/env_isaaclab/bin/python scripts/analysis/build_midi_corpus.py \
        --out MIDIs_train --max_fidel 0.20 --max_poison 0.05
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("midi_playability", _HERE / "midi_playability.py")
mp = importlib.util.module_from_spec(_spec)
sys.modules["midi_playability"] = mp
_spec.loader.exec_module(mp)

from tasks.drumrobot.components.robotic_drum_score import RDSCfg      # noqa: E402
from tasks.drumrobot.components.specs import Instruments              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="원본 MIDI 폴더 (기본: RDSCfg.midi_folder_path)")
    ap.add_argument("--out", default="MIDIs_train", help="생성할 폴더 (원본 폴더 옆에 만든다)")
    ap.add_argument("--max_fidel", type=float, default=0.20,
                    help="① 로 사라지는 노트 비율 상한. 이보다 많이 지워지는 곡은 제외 (곡 변형 과다)")
    ap.add_argument("--max_poison", type=float, default=0.05,
                    help="②+③ 불가 노트 비율 상한. 학습을 오염시키는 노트가 많은 곡은 제외")
    ap.add_argument("--slow_factor", type=float, default=1.0)
    ap.add_argument("--symlink", action="store_true", help="복사 대신 심볼릭 링크")
    ap.add_argument("--dry_run", action="store_true", help="판정만 하고 파일은 안 만든다")
    a = ap.parse_args()

    cfg = RDSCfg()
    src = Path(a.src or cfg.midi_folder_path).resolve()
    out = (Path(a.out) if Path(a.out).is_absolute() else src.parent / a.out).resolve()
    if out == src:
        raise SystemExit("출력 폴더가 원본과 같습니다 — 원본은 보존해야 합니다")

    drum_pos = [i.position for i in Instruments().all.values()]
    files = sorted(src.glob("*.mid"))
    print(f"[코퍼스] 원본 {src} ({len(files)}곡) → {out}")
    print(f"[기준] fidel < {a.max_fidel}  poison < {a.max_poison}  slow_factor={a.slow_factor}")

    keep, drop = [], []
    for f in files:
        try:
            r = mp.analyze(f, cfg, drum_pos, a.slow_factor)
        except Exception as e:
            drop.append({"file": f.name, "reason": f"parse_error: {str(e)[:60]}"})
            continue
        if r["skip"]:
            drop.append({"file": f.name, "reason": "no_arm_notes"})
            continue
        rec = {k: r[k] for k in ("file", "bpm", "n_seg", "n_raw", "density",
                                 "drop_same", "chord_excess", "busy", "fidel", "poison")}
        if r["fidel"] >= a.max_fidel:
            drop.append({**rec, "reason": f"fidel {r['fidel']:.3f} >= {a.max_fidel}"})
        elif r["poison"] >= a.max_poison:
            drop.append({**rec, "reason": f"poison {r['poison']:.3f} >= {a.max_poison}"})
        else:
            keep.append(rec)

    n_keep_notes = sum(r["n_raw"] for r in keep)
    n_all_notes = n_keep_notes + sum(r.get("n_raw", 0) for r in drop)
    res_fidel = sum(r["drop_same"] for r in keep) / max(n_keep_notes, 1)
    res_poison = sum(r["chord_excess"] + r["busy"] for r in keep) / max(n_keep_notes, 1)

    print(f"\n[결과] 유지 {len(keep)}곡 / 제외 {len(drop)}곡")
    print(f"       노트 {n_keep_notes} / {n_all_notes} ({n_keep_notes/max(n_all_notes,1)*100:.1f}%)")
    print(f"       잔여 fidel {res_fidel*100:.1f}%  잔여 poison {res_poison*100:.2f}%")
    reasons = {}
    for d in drop:
        reasons[d["reason"].split()[0]] = reasons.get(d["reason"].split()[0], 0) + 1
    print(f"       제외 사유: {reasons}")

    if a.dry_run:
        print("\n[dry-run] 파일은 만들지 않았습니다")
        return

    if out.exists():
        raise SystemExit(f"{out} 이미 존재합니다 — 지우고 다시 실행하세요 (원본 {src} 은 영향 없음)")
    out.mkdir(parents=True)
    for r in keep:
        s, d = src / r["file"], out / r["file"]
        if a.symlink:
            d.symlink_to(s)
        else:
            shutil.copy2(s, d)

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_folder": str(src),
        "note": "원본은 수정하지 않음. 이 폴더는 생성물이므로 지우고 다시 만들어도 된다.",
        "constraints": {
            "step_dt_s": mp.STEP_DT,
            "hit_window_step_W": mp.HIT_WINDOW_STEP,
            "min_gap_steps": mp.MIN_GAP,
            "min_gap_ms": round(mp.MIN_GAP * mp.STEP_DT * 1000, 1),
            "t_rearm_steps": mp.T_REARM,
            "t_rearm_ms": round(mp.T_REARM * mp.STEP_DT * 1000, 1),
            "episode_length_s": mp.EPISODE_LENGTH_S,
            "slow_factor": a.slow_factor,
            "comment": "추정 파라미터 없음. 이동속도(v_max) 제약은 미측정이라 판정에 쓰지 않았다.",
        },
        "thresholds": {"max_fidel": a.max_fidel, "max_poison": a.max_poison},
        "summary": {
            "kept": len(keep), "dropped": len(drop),
            "kept_notes": n_keep_notes, "total_notes": n_all_notes,
            "residual_fidel": round(res_fidel, 4), "residual_poison": round(res_poison, 5),
        },
        "kept": keep,
        "dropped": drop,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    made = len(list(out.glob("*.mid")))
    print(f"\n[완료] {out}  — mid {made}개 + manifest.json")
    print(f"[사용] RDSCfg.midi_folder_path 를 이 경로로 바꾸면 학습이 이 코퍼스를 쓴다")


if __name__ == "__main__":
    main()
