"""
[확인용 임시 스크립트] Snare_0.txt 전용 play (play_song.py를 이 곡으로 실행)

실행 (레포 루트에서)
    ~/RL_workspace/IsaacLab/isaaclab.sh -p drum_robot/scripts/tests/play_snare0.py \
        --task=DrumRobot-drum_robot-Direct-v0 --num_envs=1 --checkpoint <.../checkpoints/best_agent.pt>
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [str(HERE / "play_song.py"), f"--song={HERE.parents[1] / 'RDS' / 'Snare_0.txt'}"] + sys.argv[1:]
runpy.run_path(str(HERE / "play_song.py"), run_name="__main__")
