"""
[확인용 임시 스크립트] 정해진 곡(RDS 텍스트)으로 play: 모델끼리 같은 곡, 같은 조건으로 비교

play.py를 그대로 실행하고, 환경 생성(gym.make)만 감싸서 아래를 적용 (레포 코드는 그대로)
    곡       : RDS 텍스트 (bpm, 행마다 t += 박 × 100 / bpm, 드럼 번호 - 1 = 시뮬 드럼, 9 오픈 하이햇 -> 하이햇, 킥은 버림)
    길이     : 에피소드 = 마지막 음 + 1초 (정책 관측에 에피소드 시간이 없어서 5초보다 길어도 됨)
    고정     : 드럼 위치 노이즈 0, 시작 자세 = 스네어·스네어 쌍 (관절 노이즈 0), 손 배정 = 플래너 (악보의 R/L은 안 씀)
    기록     : align_record.py (θ0·θ1·θ2 desired / RL_desired / output, 다음 음표, 가상 타격), 1초마다 CSV 덮어쓰기
    종료     : 곡이 끝나면 (에피소드 마지막 스텝) CSV·정보 txt 저장, 요약 출력 후 시뮬레이션 종료
    저장     : logs/play_song/<곡>/<학습 폴더>__<체크포인트>__<실행 시각>.csv / .txt
    호환     : 관측 105, 85로 학습한 예전 모델은 재생할 때만 예전 관측으로 바꿔 끼움 (레포 파일은 그대로)
               --compat auto (기본): 학습 폴더 params/env.yaml의 observation_space가 105면 old1, 85면 old2, 아니면 none
               old1 = 다음 타격 3개 + 정렬 오차 칸 없음 (10-21 13:48 ~ 10-22 오전 모델)
               old0 = old1 + 쉬는 손 드럼 칸 0 (10-21 11:41 이전 모델)
               old2 = θ0* 진행률 칸 없음 (관측 85, 11-09 시간 기준 속도 벌 전 모델. 보상은 지금 reward.py로 기록)
               --freeze auto (기본): 환경이 허리·어깨 yaw 외 관절을 고정. 단 2026-10-22 14:33 ~ 16:46에 시작한 학습은
                                     관절을 풀고 학습했으므로 재생 때도 풀어 줌 (yes = 항상 고정, no = 항상 풂)

실행 (레포 루트에서): 곡 전용 실행 파일을 쓰면 됨
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/play_drumsolo.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --checkpoint <.../checkpoints/best_agent.pt>
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/play_snare0.py ... (같은 옵션)
    직접: play_song.py --song=<RDS 텍스트> ... / 옵션: --print_s 0.25, --save_s 1.0, 그 밖은 play.py와 같음
    --waist_gain 180,5,0.08 : 재생할 때만 허리 Kp,Kd,armature 를 바꿈 (없으면 drumrobot_cfg.py 값. 레포 cfg는 그대로)
    --slow 2 : 곡을 2배 느리게 (음표 시각 × 2 = bpm / 2). 저장은 logs/play_song/<곡>_slow/ (배속끼리 ← / → 로 비교)
그래프
    python drum_robot/scripts/tests/plot_align.py logs/play_song/<곡>      (← / → : 모델, v : 각도 <-> 각속도)
"""

import argparse
import os
import re
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PLAY = HERE.parent / "reinforcement_learning" / "skrl" / "play.py"

# 이 스크립트 옵션만 떼어 내고 나머지는 play.py로 넘김
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--song", type=str, required=True)
_parser.add_argument("--print_s", type=float, default=0.25)
_parser.add_argument("--save_s", type=float, default=1.0)
_parser.add_argument("--checkpoint", type=str, required=True)     # 저장 이름에도 씀 (play.py에 그대로 넘김)
# _parser.add_argument("--compat", type=str, default="auto", choices=["auto", "none", "old0", "old1", "old2"])
_parser.add_argument("--compat", type=str, default="auto", choices=["auto", "none", "old0", "old1"])
_parser.add_argument("--freeze", type=str, default="auto", choices=["auto", "yes", "no"])
_parser.add_argument("--waist_gain", type=str, default=None, help="허리 Kp,Kd,armature (예: 180,5,0.08)")
_parser.add_argument("--slow", type=float, default=1.0, help="곡을 K배 느리게 (음표 시각 × K)")
_parser.add_argument("--waist_ref", type=str, default="step", choices=["step", "jerk"], help="jerk = 관측 θ0* 칸에 최소 저크 θ0_d (재생만)")
_args, _rest = _parser.parse_known_args()
sys.argv = [str(PLAY)] + _rest + [f"--checkpoint={_args.checkpoint}"]

# 실기 번호 -> 시뮬 드럼 (snare, floor, mid, high, hihat, ride, crash_r, crash_l)
NOTE_TO_DRUM = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 4}
START_PAIR = ("snare", "snare")


GRADES = []                                                          # 이벤트별 {시뮬 드럼: 세기 1~7} (실기 make_drum_event: 5·6열 = velocity_R, velocity_L)


def read_song(path: Path):
    # RDS 텍스트 -> bpm, [(시각 s, [시뮬 드럼, ...]), ...]  (실기 서버 make_drum_event와 같은 시간 계산)
    bpm, t, events = 100.0, 0.0, []
    for line in open(path):
        it = line.split()
        if not it:
            continue
        if it[0] == "bpm":
            bpm = float(it[1])
            continue
        if it[0] == "end":
            break
        t += float(it[1]) * 100.0 / bpm
        drums = sorted({NOTE_TO_DRUM[int(n)] for n in (it[2], it[3]) if int(n) != 0})   # 오른손, 왼손 열 (킥 열은 안 씀)
        if drums:
            events.append((t, drums))
            g = {}
            for n, v in ((it[2], it[4]), (it[3], it[5])):            # (오른손 음, 오른손 세기), (왼손 음, 왼손 세기)
                if int(n) != 0:
                    g[NOTE_TO_DRUM[int(n)]] = max(g.get(NOTE_TO_DRUM[int(n)], 0), int(v))   # 같은 드럼 두 손이면 센 쪽 (학습 악보와 같음)
            GRADES.append(g)
    if not events:
        raise SystemExit(f"타격이 없는 곡입니다: {path}")
    return bpm, events


SONG = Path(_args.song).resolve()
BPM, EVENTS = read_song(SONG)
SLOW = _args.slow
if SLOW != 1.0:                                                     # K배 느리게: 음표 시각 × K (= bpm / K)
    BPM, EVENTS = BPM / SLOW, [(t * SLOW, d) for t, d in EVENTS]
SONG_LEN = EVENTS[-1][0] + 1.0                                      # 마지막 음 + 1초
CKPT = Path(_args.checkpoint).resolve()
STEM = f"{CKPT.parents[1].name}__{CKPT.stem}__" + (f"slow{SLOW:g}x__" if SLOW != 1.0 else "") \
       + ("jerkref__" if _args.waist_ref == "jerk" else "") + time.strftime('%Y%m%d_%H%M%S')
OUT_DIR = REPO / "logs" / "play_song" / (f"{SONG.stem}_slow" if SLOW != 1.0 else SONG.stem)
OUT_CSV = OUT_DIR / f"{STEM}.csv"
OUT_TXT = OUT_DIR / f"{STEM}.txt"
print(f"[play_song] 곡 {SONG.name} ({SLOW:g}배 느리게): {BPM:g} BPM, 타격 이벤트 {len(EVENTS)}개 "
      f"(두 드럼 동시 {sum(len(d) > 1 for _, d in EVENTS)}개), 길이 {SONG_LEN:.2f}s")


def _train_obs_dim(ckpt: Path):
    # 학습 폴더 params/env.yaml에 남은 학습 당시 관측 크기 (없으면 None)
    p = ckpt.parents[1] / "params" / "env.yaml"
    m = re.search(r"^observation_space:\s*(\d+)", p.read_text(), re.M) if p.exists() else None
    return int(m.group(1)) if m else None


TRAIN_OBS = _train_obs_dim(CKPT)
# COMPAT = _args.compat if _args.compat != "auto" else {105: "old1", 85: "old2"}.get(TRAIN_OBS, "none")
COMPAT = _args.compat if _args.compat != "auto" else {105: "old1"}.get(TRAIN_OBS, "none")
UNFROZEN = ("2026-10-22_14-33-18", "2026-10-22_16-46-43")          # 이 사이에 시작한 학습만 관절을 풀고 학습 (나머지는 고정)
_run = CKPT.parents[1].name[:19]
FREEZE = not (UNFROZEN[0] <= _run < UNFROZEN[1]) if _args.freeze == "auto" else (_args.freeze == "yes")
print(f"[play_song] 학습 관측 {TRAIN_OBS}, 호환 모드 {COMPAT}, 관절 고정 {FREEZE}")

import gymnasium as gym                     # play.py가 쓰는 같은 모듈 (make만 감쌈)

sys.path.insert(0, str(HERE))
from align_record import AlignRecorder      # 기록 공통 부분 (play_align.py와 같이 씀)


def _apply_compat(cfg):
    # 관측 105로 학습한 예전 모델: 다음 타격 3개, 정렬 오차 칸 없음 (old0은 쉬는 손 드럼 칸 0도). 이 프로세스 안에서만 바꿔 끼움
    from dataclasses import dataclass
    import drum_robot.tasks.drumrobot.drumrobot_env as env_mod

    @dataclass
    class _RDSCfgOld(env_mod.RDSCfg):
        num_hits: int = 3                                           # 관측과 플래너 모두 다음 타격 3개

    env_mod.RDSCfg = _RDSCfgOld
    cfg.observation_space = 105                                     # 정책 입력 크기 = 체크포인트

    cls = env_mod.DrumRobotEnv
    orig_init_obs, orig_pack = cls._init_obs, cls._normalize_and_pack_obs

    def _init_obs_old(self):
        self.cfg.observation_space += 3                             # 원래 크기 검사가 정렬 오차 3칸까지 세므로 잠깐 맞춰 줌
        try:
            orig_init_obs(self)
        finally:
            self.cfg.observation_space -= 3
        self.obs_dim_align = 0

    def _pack_old(self, **kw):
        kw["align_err"] = kw["align_err"][:, :0]                    # 정렬 오차 칸 빼기
        if COMPAT == "old0":                                        # 관측 수정 전: 쉬는 손의 드럼 칸 = 0
            arm_next = kw["arm_next"].clone()
            arm_next[..., :-1] *= self.arm_valid[..., None]
            kw["arm_next"] = arm_next
        return orig_pack(self, **kw)

    cls._init_obs, cls._normalize_and_pack_obs = _init_obs_old, _pack_old


# def _apply_compat_old2(cfg):
#     # 관측 85로 학습한 모델: θ0* 진행률 칸 없음 (나머지 순서는 같음). 이 프로세스 안에서만 바꿔 끼움
#     import drum_robot.tasks.drumrobot.drumrobot_env as env_mod

#     cfg.observation_space = 85                                      # 정책 입력 크기 = 체크포인트

#     cls = env_mod.DrumRobotEnv
#     orig_init_obs, orig_pack = cls._init_obs, cls._normalize_and_pack_obs

#     def _init_obs_old(self):
#         self.cfg.observation_space += 1                             # 원래 크기 검사가 진행률 1칸까지 세므로 잠깐 맞춰 줌
#         try:
#             orig_init_obs(self)
#         finally:
#             self.cfg.observation_space -= 1
#         self.obs_dim_waist_phase = 0

#     def _pack_old(self, **kw):
#         kw["waist_phase"] = kw["waist_phase"][:, :0]                # 진행률 칸 빼기
#         return orig_pack(self, **kw)

#     cls._init_obs, cls._normalize_and_pack_obs = _init_obs_old, _pack_old


def _apply_jerk_ref(u):
    # 재생할 때만: 관측 θ0* 칸에 최소 저크 θ0_d. τ = clip(s / D, 0, 1), θ0_d = θ_from + (θ0* - θ_from)·(10τ³ - 15τ⁴ + 6τ⁵)
    import torch
    al = u.reward_computer.alignment
    al.ref_waist = al.target_waist.clone()                          # θ0_d
    al.ref_from = al.target_waist.clone()                           # θ_from: θ0*가 바뀐 순간의 θ0_d
    orig_compute, orig_reset, orig_pack = al.compute, al.reset, u._normalize_and_pack_obs
    j0 = al.j_waist

    def compute(*args, **kwargs):
        out = orig_compute(*args, **kwargs)
        al.ref_from = torch.where(al.waist_s == 0, al.ref_waist, al.ref_from)   # θ0*가 바뀐 스텝 (s = 0)
        tau = (al.waist_s / al.waist_D).clamp(0.0, 1.0)
        al.ref_waist = al.ref_from + (al.target_waist - al.ref_from) * tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
        return out

    def reset(env_ids, *args, **kwargs):
        orig_reset(env_ids, *args, **kwargs)
        al.ref_waist[env_ids] = al.target_waist[env_ids]             # 시작: θ0_d = θ0*
        al.ref_from[env_ids] = al.target_waist[env_ids]

    def pack(**kw):
        kw["waist_des"] = ((al.ref_waist - u.joint_center[0, j0]) / u.joint_half_range[0, j0])[:, None]   # θ0* 대신 θ0_d (같은 정규화)
        return orig_pack(**kw)

    al.compute, al.reset, u._normalize_and_pack_obs = compute, reset, pack


def _shrink_midi_dataset():
    # 재생은 MIDI 마디를 안 씀 (env 1개라 int(1 × score_ratio) = 0, 악보 = 곡). 그런데 env 생성 때 MIDI 마디 전체 × 에피소드 길이
    # 텐서를 GPU에 만들어서 긴 곡(--slow 3 이상)은 메모리 부족 -> 이 프로세스 안에서만 MIDI 파일 하나로 만듦 (레포 파일은 그대로)
    from drum_robot.tasks.drumrobot.components.robotic_drum_score import RDS

    orig_build = RDS._build_rds_dataset

    def _build_one(self, midi_files):
        for f in midi_files:                                         # 쓸 수 있는 마디가 나오는 첫 파일
            try:
                return orig_build(self, [f])
            except RuntimeError:
                continue
        return orig_build(self, midi_files)

    RDS._build_rds_dataset = _build_one


def _patched_make(*args, **kwargs):
    import torch

    kwargs["cfg"].episode_length_s = SONG_LEN                        # 악보·판정 표·플래너 버퍼가 이 길이로 만들어짐
    _shrink_midi_dataset()
    # if COMPAT == "old2":
    #     _apply_compat_old2(kwargs["cfg"])
    # elif COMPAT != "none":
    #     _apply_compat(kwargs["cfg"])
    if COMPAT != "none":
        _apply_compat(kwargs["cfg"])
    if _args.waist_gain:                                             # 재생할 때만 허리 게인 바꿈 (레포 cfg는 그대로)
        kp, kd, arm = (float(x) for x in _args.waist_gain.split(","))
        acts = kwargs["cfg"].robot_cfg.actuators
        acts["waist"] = acts["waist"].replace(stiffness=kp, damping=kd, armature=arm)
    env = _make(*args, **kwargs)
    u = env.unwrapped
    jw = u.robot.find_joints("waist_joint")[0][0]
    bw = u.robot.find_bodies("waist")[0][0]
    waist_info = (f"Kp {u.robot.data.joint_stiffness[0, jw].item():g}, Kd {u.robot.data.joint_damping[0, jw].item():g}, "
                  f"armature {u.robot.data.joint_armature[0, jw].item():g}, waist link izz {u.robot.root_physx_view.get_inertias()[0, bw, 8].item():g}")
    print(f"[play_song] 허리: {waist_info}")
    if not FREEZE:
        u.frozen_ids = []                                            # 관절을 풀고 학습한 모델: 환경의 관절 고정 해제
    T, M, dt = u.episode_length_step, u.drum_pos.shape[1], u.step_dt

    # 곡 -> 시뮬 악보 (T, M): 리셋 때 무작위 악보 대신 이 곡
    song = torch.zeros((T, M), dtype=torch.int64, device=u.device)
    # for t, drums in EVENTS:
    for i, (t, drums) in enumerate(EVENTS):
        s = round(t / dt)
        if s >= T - u.hit_detector.max_window:
            raise SystemExit(f"곡 시각이 에피소드 끝을 넘음: {t:.3f}s")
        # song[s, drums] = 1
        for d in drums:
            song[s, d] = max(int(song[s, d]), GRADES[i][d])         # 악보 값 = 세기 1~7 (54: 관측·판정이 이 값을 세기로 씀)
    u.rds._reset_random = lambda env_ids: song[None].expand(len(env_ids), -1, -1).clone()
    u.rds._reset_midi = lambda env_ids, selection_strength=0.0: song[None].expand(len(env_ids), -1, -1).clone()   # 54: score_ratio 1.0 이라 env 1개도 MIDI 경로로 리셋됨

    # 조건 고정: 드럼 위치 노이즈 0, 시작 자세 = 스네어·스네어 쌍 (관절 노이즈 0)
    u.instruments.drum_noise_scale = 0.0
    ri = u.robot_initializer
    k = ri.cfg.drum_pairs.index(START_PAIR)
    ri.reset_init_pos = lambda env_ids: (ri.pos_angle[k].expand(len(env_ids), -1).clone(),
                                         ri.pair_drum_idx[k].expand(len(env_ids), -1).clone())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[play_song] 저장: {OUT_CSV}")
    save_every = max(1, round(_args.save_s / dt))

    def finish(rows):
        rec.save_csv(OUT_CSV, rows)
        with open(OUT_TXT, "w") as f:
            f.write(f"checkpoint: {CKPT}\n")
            f.write(f"song: {SONG}\nslow: {SLOW:g}\nbpm: {BPM:g}\nlength_s: {SONG_LEN:.3f}\nevents: {len(EVENTS)}\n")
            f.write(f"drum_noise: 0\nstart_pair: {START_PAIR}\njoint_noise: 0\nhand_assignment: planner (악보 R/L 안 씀)\n")
            f.write(f"waist: {waist_info}\n")
            f.write(f"recorded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for line in rec.reward_summary():
                f.write(line + "\n")
        # rec.summary(rows, SONG.stem)
        print(f"\n[play_song] {SONG.stem} 결과")
        for line in rec.reward_summary():
            print(f"[play_song] {line}")
        print(f"[play_song] 곡 끝: {OUT_CSV}")
        print(f"[play_song] 그래프: python drum_robot/scripts/tests/plot_align.py {OUT_DIR}\n")
        sys.stdout.flush()
        os._exit(0)                                                  # 시뮬레이션 바로 종료 (CSV는 이미 저장됨)

    def on_step(step, rows):
        if step >= T - 1:                                            # 에피소드 마지막 스텝 = 곡 끝
            finish(rows)
        elif step % save_every == 0:                                 # 1초마다 덮어쓰기 (중간에 창을 닫아도 남음)
            rec.save_csv(OUT_CSV, rows)

    if _args.waist_ref == "jerk":
        _apply_jerk_ref(u)
    # rec = AlignRecorder(u, print_s=_args.print_s, on_step=on_step, tag="play_song")
    rec = AlignRecorder(u, print_s=None, on_step=on_step, tag="play_song")     # 중간 각도 출력 없음 (곡 끝에 보상·성공만)
    return env


_make = gym.make
gym.make = _patched_make

runpy.run_path(str(PLAY), run_name="__main__")
