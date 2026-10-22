"""
[확인용 임시 스크립트] 정렬 학습 결과 play + 허리·어깨 yaw 목표(desired) / 정책 명령(RL) / 실제(output) 기록

play.py를 그대로 실행하고, 환경 생성(gym.make)만 감싸서 매 스텝 env 0번을 기록함 (레포 코드는 그대로)
    터미널   : print_s 초마다 다음 음표, 목표/명령/실제 각도, 오차(목표 - 실제)
    에피소드 : CSV + 평균 오차 요약 (그래프는 plot_align.py로 matplotlib 창에서 봄)
    저장     : <체크포인트 폴더>/../play_align/<실행 시각>/episode_001.csv ...  (에피소드 끝날 때마다 바로 저장)
desired: 정렬 보상의 목표 (허리 = 쌍 표, 어깨 yaw = 같은 식으로 다시 계산: θ1* = atan2(y_R - l·sin q0, x_R - l·cos q0) - q0, 왼쪽은 부호 반대)
RL     : 정책이 명령한 관절 목표 env.target_q (행동 = 목표 속도를 누적, 실제 ± max_lead 안으로 자름)
output : 실제 관절 각도

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/play_align.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --checkpoint <.../checkpoints/best_agent.pt>
    옵션: --print_s 0.25 (터미널 출력 간격 [s]), 그 밖의 옵션은 play.py와 같음 (--real-time 등)
"""

import argparse
import runpy
import sys
import time
from pathlib import Path

# 이 스크립트 옵션만 떼어 내고 나머지는 play.py로 넘김
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--print_s", type=float, default=0.25)
_parser.add_argument("--checkpoint", type=str, default=None)      # 저장 위치 정하는 데만 씀 (play.py에도 그대로 넘김)
_args, _rest = _parser.parse_known_args()
PLAY = Path(__file__).resolve().parents[1] / "reinforcement_learning" / "skrl" / "play.py"
sys.argv = [str(PLAY)] + _rest + ([f"--checkpoint={_args.checkpoint}"] if _args.checkpoint else [])

if _args.checkpoint:
    OUT = Path(_args.checkpoint).resolve().parents[1] / "play_align" / time.strftime("%Y%m%d_%H%M%S")
else:
    OUT = Path("logs") / "play_align" / time.strftime("%Y%m%d_%H%M%S")

import gymnasium as gym                     # play.py가 쓰는 같은 모듈 (make만 감쌈)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_record import AlignRecorder      # 기록 공통 부분 (play_song.py와 같이 씀)


def _patched_make(*args, **kwargs):
    env = _make(*args, **kwargs)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[play_align] 저장 위치: {OUT}")
    count = {"ep": 0}

    def on_episode_end(rows):                   # 에피소드 끝날 때마다 CSV + 요약
        count["ep"] += 1
        name = f"episode_{count['ep']:03d}"
        rec.save_csv(OUT / f"{name}.csv", rows)
        rec.summary(rows, f"에피소드 {count['ep']}")
        print(f"[play_align] 저장: {OUT / name}.csv  (그래프: python drum_robot/scripts/tests/plot_align.py {OUT})\n")

    rec = AlignRecorder(env.unwrapped, print_s=_args.print_s, on_episode_end=on_episode_end, tag="play_align")
    return env


_make = gym.make
gym.make = _patched_make

runpy.run_path(str(PLAY), run_name="__main__")
