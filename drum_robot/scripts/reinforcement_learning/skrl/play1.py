"""
Script to evaluate a checkpoint on REAL MIDI scores (play1).

play.py와 동일하되, 실제 MIDI 악보로 평가 (세기 포함). --eval_mode 로 3가지:
  1: MIDI 강제        — 전체 MIDI 폴더에서 마디 단위 랜덤 샘플 (랜덤 RDS 0%)
  2: 특정 곡          — --midi_file 하나의 마디들만
  3: 곡 전체 연속연주  — --midi_file 을 1개 긴 악보로, 처음부터 끝까지 (에피소드 길이 자동)

세기·타격 성적은 env 로거가 출력: success_rate, dyn_impact_p/mf/f(n), dyn_wrist_*, intensity_match
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher                                                                    # type: ignore[arg-type]

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a checkpoint on real MIDI scores (play1).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ===== play1 전용 인자 =====
parser.add_argument(
    "--eval_mode", type=int, default=1, choices=[1, 2, 3],
    help="1: MIDI강제(전체,마디랜덤)  2: 특정곡(마디)  3: 곡전체 연속연주",
)
parser.add_argument("--midi_file", type=str, default=None, help="레벨 2/3 에서 평가할 MIDI 파일 경로")
parser.add_argument("--slow_factor", type=float, default=None, help="RDS slow_factor override (기본 1.5=학습값, 1.0=원곡 템포)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym                                                                                 # type: ignore[arg-type]
import skrl
import torch                                                                                            # type: ignore[arg-type]
from packaging import version                                                                           # type: ignore[arg-type]
import drum_robot   # 사용자 패키지를 강제로 로드하여 gym.register 실행

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner                                                          # type: ignore[arg-type]
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner                                                            # type: ignore[arg-type]

from isaaclab.envs import (                                                                             # type: ignore[arg-type]
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict                                                              # type: ignore[arg-type]

from isaaclab_rl.skrl import SkrlVecEnvWrapper                                                          # type: ignore[arg-type]
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint                 # type: ignore[arg-type]

import isaaclab_tasks  # noqa: F401                                                                     # type: ignore[arg-type]
from isaaclab_tasks.utils import get_checkpoint_path                                                    # type: ignore[arg-type]
from isaaclab_tasks.utils.hydra import hydra_task_config                                                # type: ignore[arg-type]

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _midi_span_seconds(path: str):
    """MIDI 파일의 (첫 노트 시각, 마지막 노트 시각)[초] — 레벨3 에피소드 길이 계산용 (채널 9만)."""
    from mido import MidiFile   # pyright: ignore[reportMissingImports]
    mid = MidiFile(path)
    tpb = mid.ticks_per_beat
    first_t = None
    last_t = 0.0
    for track in mid.tracks:
        ct = 0
        tempo = 500000
        for msg in track:
            ct += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0 and getattr(msg, "channel", None) == 9:
                t = (ct * tempo) / (tpb * 1_000_000)
                if first_t is None:
                    first_t = t
                last_t = max(last_t, t)
    return (first_t or 0.0), last_t


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):

    print("================================ PLAY1 (MIDI EVAL) ================================")

    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ===== play1 평가 설정: 실제 MIDI 강제 =====
    env_cfg.eval_midi_ratio = 1.0   # 세 모드 모두 순수 MIDI (랜덤 0%)
    if args_cli.slow_factor is not None:
        env_cfg.eval_slow_factor = args_cli.slow_factor   # 원곡 템포 테스트용 override (1.0 등)
    if args_cli.eval_mode >= 2:
        if not args_cli.midi_file:
            raise ValueError("--eval_mode 2/3 은 --midi_file 이 필요합니다.")
        env_cfg.eval_midi_file = os.path.abspath(args_cli.midi_file)
    if args_cli.eval_mode == 3:
        env_cfg.eval_full_song = True
        # 곡 전체가 에피소드에 담기도록 길이 자동 계산 (override된 slow_factor 반영)
        from drum_robot.tasks.drumrobot.components.robotic_drum_score import RDSCfg
        rc = RDSCfg()
        first_t, last_t = _midi_span_seconds(env_cfg.eval_midi_file)
        step_dt = env_cfg.sim.dt * env_cfg.decimation
        needed_steps = round(env_cfg.eval_slow_factor * (last_t - first_t) / step_dt) + rc.start_offset_steps + 60
        env_cfg.episode_length_s = needed_steps * step_dt
        print(f"[PLAY1] full-song 에피소드 길이 = {env_cfg.episode_length_s:.1f}s ({needed_steps} steps)")

    print(
        f"[PLAY1] eval_mode={args_cli.eval_mode} | midi_ratio={env_cfg.eval_midi_ratio} | "
        f"file={env_cfg.eval_midi_file or '(all)'} | full_song={env_cfg.eval_full_song} | slow_factor={env_cfg.eval_slow_factor}"
    )

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # [mode 3(곡 전체)에서만] 곡 1회 끝날 때마다 그 곡의 지표가 출력되도록 로거 flush 주기를 에피소드 길이에 맞춤.
    # mode 1/2·학습은 영향 없음 (여기서만, 런타임에 play1 env 인스턴스에만 적용).
    if args_cli.eval_mode == 3:
        try:
            base_env = env.unwrapped
            base_env.logger.cfg.interval = base_env.episode_length_step
            print(f"[PLAY1] (mode3) 로그 flush = 곡 1회마다 (에피소드 {base_env.episode_length_step} 스텝)")
        except Exception as e:
            print(f"[PLAY1] 로거 주기 설정 실패(무시): {e}")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
