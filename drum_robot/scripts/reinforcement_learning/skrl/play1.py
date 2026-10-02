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
parser.add_argument("--export_traj", type=str, default=None, help="env0 관절 궤적을 CSV로 저장(한 에피소드). MATLAB 확인용. --num_envs=1 권장")
parser.add_argument("--telemetry", type=str, default=None,
                    help="정책 명령을 UDP 송신 (HOST:PORT 또는 PORT). scripts/analysis/scope.py 로 실시간 관찰. "
                         "sim 비용 ~50us/스텝 (15ms 예산의 0.3%)")
parser.add_argument("--wrist_fd", action="store_true", default=False,
                    help="[sim2real 점검] 손목 속도 obs 를 실기와 같은 유한차분으로 대체. "
                         "같은 체크포인트로 켜고/끄고 두 번 돌린 성공률 차이 = 이 채널의 배포 갭")
parser.add_argument("--random_drums", action="store_true", default=False,
                    help="드럼 위치 랜덤화(DR) 유지. 기본(play1)은 실기 위치로 고정(drum_noise_scale=0)")
parser.add_argument("--log_every", type=int, default=300,
                    help="play1 로그 flush 주기(스텝). 짧게 하면 곡 안 끝나도 로그가 자주 뜸. 0=곡 1회마다(mode3)")

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
    """악보 파일의 (첫 노트 시각, 마지막 노트 시각)[초] — 레벨3 에피소드 길이 계산용.
    .txt(실기 서버 형식)는 RDS 의 텍스트 로더로, .mid 는 MIDI 채널 9 노트로 계산."""
    if path.lower().endswith(".txt"):
        from drum_robot.tasks.drumrobot.components.robotic_drum_score import read_rds_txt
        _, events = read_rds_txt(path)
        arm_t = [t for t, name, _ in events if name != "Bass Drum 1"]
        if not arm_t:
            raise ValueError(f"텍스트 악보에 팔 타격이 없습니다: {path}")
        # .txt 는 절대 시각 유지(RDS _build_full_song 과 동일) → 앞 침묵까지 에피소드에 포함
        return 0.0, max(arm_t)
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


# ===== 실기(phill-drum-robot) trajectory_log CSV 양식으로 export =====
# 서버 trajectory_log 헤더 = "t,joint 0,...,joint 12" (13관절, 실기 모터 id 순서, 관절각 rad).
# 그 양식과 동일하게 뽑아야 실기 재생기(REPLAY)가 변환 없이 그대로 스트리밍한다.
# sim ctrl 관절이름 -> 실기 모터 id (motors.json id 순서)
SIM_NAME_TO_REAL_ID = {
    "waist_joint":      0,
    "right_shoulder_1": 1,
    "left_shoulder_1":  2,
    "right_shoulder_2": 3,
    "right_elbow":      4,
    "left_shoulder_2":  5,
    "left_elbow":       6,
    "right_wrist":      7,
    "left_wrist":       8,
}
NUM_REAL_JOINT = 13   # 실기 NUM_JOINT (0~8 팔, 9~10 페달, 11~12 머리)
REAL_CTRL_DT = 0.005  # 실기 제어주기 DT_SECOND (200Hz). export CSV를 이 고정 그리드로 리샘플.


def sim_to_real_joint_rad(real_id: int, q_sim_rad: float) -> float:
    """sim 관절각(robot-frame rad) -> 실기 관절각(rad) 규약 변환. (규약 변환은 여기 한 곳에서만)
    ⚠ TODO(손목): sim 손목(-10~90°) ↔ 실기 손목(-90~100°) 물리 정합 미확정 → 현재 identity.
       (real_id 7=right_wrist, 8=left_wrist)  HW 타격 전 반드시 확정.
    그 외 관절: sim↔real 관절각 규약 동일 가정(미검증) → identity."""
    return q_sim_rad


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
    env_cfg.wrist_vel_finite_diff = args_cli.wrist_fd   # [sim2real 점검] 손목 속도 유한차분 A/B
    if args_cli.slow_factor is not None:
        env_cfg.eval_slow_factor = args_cli.slow_factor   # 원곡 템포 테스트용 override (1.0 등)

    # ===== export: '지정한 곡 전체 연주'를 뽑는 용도 → full-song(mode 3)으로 맞춘다 =====
    #   midi_file 이 있으면 그 곡 전체를 처음부터 끝까지 재생한 궤적이 CSV로 저장된다.
    #   (mode 1/2 는 마디 랜덤/부분이라 곡이 이어지지 않으므로 export 에 부적합)
    if args_cli.export_traj is not None:
        if args_cli.midi_file:
            if args_cli.eval_mode != 3:
                print(f"[EXPORT] --export_traj + --midi_file → 곡 전체 연속연주(mode 3)로 자동 설정")
                args_cli.eval_mode = 3
        else:
            print("[EXPORT] 경고: --midi_file 미지정 → 랜덤 마디가 저장됩니다. "
                  "특정 곡 전체를 뽑으려면 --midi_file <경로> 를 지정하세요.")

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
        f"file={env_cfg.eval_midi_file or '(all)'} | full_song={env_cfg.eval_full_song} | slow_factor={env_cfg.eval_slow_factor}\n"
        f"[PLAY1] wrist_vel = {'유한차분(실기 모사)' if env_cfg.wrist_vel_finite_diff else '참값(학습 그대로)'}"
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

    # 로그 flush 주기 설정 (play1 env 인스턴스에만, 런타임). 학습/다른 스크립트 영향 없음.
    #   --log_every N (기본 300): N 스텝마다 로그 출력 → 곡이 안 끝나도 자주 뜸. (n이 작아 rate는 다소 요동)
    #   --log_every 0: mode3에서 곡 1회(에피소드 전체)마다 출력 (per-song 깔끔한 집계).
    try:
        base_env = env.unwrapped
        if args_cli.export_traj is not None:
            base_env.logger.cfg.interval = 10**9   # export: 중간 flush 끔 → 곡 끝에 딱 1회만 (아래 break에서 flush)
            print("[PLAY1] export 모드 → 로그는 곡 끝에 1회만 출력")
        elif args_cli.log_every > 0:
            base_env.logger.cfg.interval = args_cli.log_every
            print(f"[PLAY1] 로그 flush = {args_cli.log_every} 스텝마다 (--log_every 0 이면 곡 1회마다)")
        elif args_cli.eval_mode == 3:
            base_env.logger.cfg.interval = base_env.episode_length_step
            print(f"[PLAY1] (mode3) 로그 flush = 곡 1회마다 (에피소드 {base_env.episode_length_step} 스텝)")
    except Exception as e:
        print(f"[PLAY1] 로거 주기 설정 실패(무시): {e}")

    # ===== 궤적 export 준비 (env0, MATLAB 확인용) =====
    export_base = None
    export_rows = []
    export_sim2real = None
    if args_cli.export_traj is not None:
        export_base = env.unwrapped
        export_joint_names = export_base.robot_interface.get_ctrl_joint_name()   # sim ctrl 순서 (9)
        # sim ctrl index -> 실기 모터 id (양식을 실기 순서로 맞추기 위함)
        try:
            export_sim2real = [SIM_NAME_TO_REAL_ID[n] for n in export_joint_names]
        except KeyError as e:
            raise ValueError(f"[EXPORT] sim 관절이름 {e} 을 실기 id로 매핑 못함 — SIM_NAME_TO_REAL_ID 확인")
        if export_base.num_envs != 1:
            print(f"[EXPORT] 경고: num_envs={export_base.num_envs} → env0만 기록 (--num_envs=1 권장)")
        print(f"[EXPORT] env0 궤적 기록 → {args_cli.export_traj} "
              f"(phill trajectory_log 양식: t,joint 0..{NUM_REAL_JOINT-1})")

    # ===== 정책 명령 UDP 텔레메트리 (scripts/analysis/scope.py 가 수신) =====
    #   sim 은 sendto 만 하고 그림은 별 프로세스가 그린다 → sim 루프 영향 사실상 0.
    #   패킷 포맷은 scope.py 와 공유 (아래 두 Struct). 헤더는 1초마다 재전송 →
    #   뷰어를 언제 켜도/재시작해도 관절이름·한계를 자동으로 받는다.
    tel = None
    if args_cli.telemetry is not None:
        import socket, struct
        _h, _, _p = args_cli.telemetry.rpartition(":")
        _addr = (_h or "127.0.0.1", int(_p))
        _b = env.unwrapped
        _names = _b.robot_interface.get_ctrl_joint_name()          # articulation 해석 순서 (specs 순서와 다름!)
        _lo, _hi = _b.robot_interface.get_limit()                  # (1,9) robot 프레임 [rad]
        _eff = _b.robot.data.joint_effort_limits[0, _b.robot_interface.get_ctrl_joint_ids()]   # (9,) N·m
        tel = dict(
            sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
            addr=_addr,
            hdr=struct.Struct("<4sI9f9f9f144s"),                   # magic, njoint, lo, hi, effort, names(9×16B)
            dat=struct.Struct("<4sIf36f"),                         # magic, seq, t, action9, target9, q9, tau9
            seq=0,
            scale=_b.action_scale_vec[0],
            dt=float(_b.step_dt),
            lo=_lo[0], hi=_hi[0],
            hdr_bytes=None,
        )
        tel["hdr_bytes"] = tel["hdr"].pack(
            b"DRM0", len(_names),
            *_lo[0].cpu().tolist(), *_hi[0].cpu().tolist(), *_eff.cpu().tolist(),
            b"".join(n.encode()[:16].ljust(16, b"\0") for n in _names),
        )
        print(f"[TEL] 정책 명령 UDP → {_addr[0]}:{_addr[1]}")
        print(f"[TEL] 뷰어:  python drum_robot/scripts/analysis/scope.py --port {_addr[1]}")
        print(f"[TEL] 관절 순서: {_names}")
        print(f"[TEL] effort_limit [N·m]: {[round(v, 2) for v in _eff.cpu().tolist()]}")

    # ===== play1은 드럼 위치를 실기(drum_coordinate.json) 위치로 고정 =====
    #   학습은 드럼 DR(랜덤화)하지만, 평가/CSV export는 실기 layout으로 고정해야 재생과 일치.
    #   nominal(specs.py Instruments) = 실기 R/L 중점(z-0.045)이라, drum_noise_scale=0 이면 실기 위치.
    #   drum_noise_scale은 매 리셋에서 읽히므로(아래 reset 포함) 여기서 0으로 두면 전 에피소드 고정.
    if not args_cli.random_drums:
        env.unwrapped.instruments.drum_noise_scale = 0.0
        print("[PLAY1] 드럼 위치 = 실기 고정 (drum_noise_scale=0, DR 끔). --random_drums 로 랜덤화 유지 가능")
    else:
        print(f"[PLAY1] 드럼 위치 = 랜덤화 유지 (drum_noise_scale={env.unwrapped.instruments.drum_noise_scale})")

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
            # 텔레메트리: 정책이 실제로 본 관절각 (_pre_physics_step 과 같은 시점)
            _q_pre = None
            if tel is not None:
                _rb = env.unwrapped
                _q_pre = _rb.robot_interface.get_joint_pos(_rb.robot)[0].clone()

            # env stepping
            obs, _, terminated, truncated, _ = env.step(actions)

            # 텔레메트리 송신 — env._pre_physics_step 과 똑같은 식으로 명령을 재현
            if tel is not None:
                _rb = env.unwrapped
                _a = torch.clamp(actions[0], -1.0, 1.0)
                _tgt = torch.clamp(_q_pre + _a * tel["scale"] * tel["dt"], tel["lo"], tel["hi"])
                _q = _rb.robot_interface.get_joint_pos(_rb.robot)[0]
                _tau = _rb.robot_interface.get_joint_torque(_rb.robot)[0]
                _v = torch.cat([_a, _tgt, _q, _tau]).cpu().tolist()   # D2H 1회 (36 float)
                if tel["seq"] % 66 == 0:                            # 약 1초마다 헤더 재전송
                    tel["sock"].sendto(tel["hdr_bytes"], tel["addr"])
                tel["sock"].sendto(
                    tel["dat"].pack(b"DRM1", tel["seq"], tel["seq"] * tel["dt"], *_v), tel["addr"])
                tel["seq"] += 1

        # ===== env0 관절각 기록 (실기 trajectory_log 양식: 13관절, 실기 id 순서, 관절각 rad) =====
        if export_base is not None:
            q_sim = export_base.robot_interface.get_joint_pos(export_base.robot)[0]   # (9,) sim rad, sim ctrl 순서
            t = len(export_rows) * export_base.step_dt
            q13 = [0.0] * NUM_REAL_JOINT
            for i, rid in enumerate(export_sim2real):
                q13[rid] = sim_to_real_joint_rad(rid, float(q_sim[i]))
            # 페달(9,10)·머리(11,12)는 sim에 없음 → 0.0 placeholder (재생기가 현재값으로 홀드)
            export_rows.append([t] + q13)
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                print(f"[EXPORT] 에피소드 종료 — {len(export_rows)} 스텝")
                try:
                    with torch.inference_mode():   # 로거 텐서가 inference tensor라 리셋(zero_)도 inference 모드 안에서
                        export_base.logger.flush(prefix="(곡 전체 집계)")   # 곡 끝에 로그 1회 출력
                except Exception as e:
                    print(f"[PLAY1] 최종 로그 flush 실패(무시): {e}")
                break

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # ===== 궤적 CSV 저장 (실기 trajectory_log 양식: "t,joint 0,...,joint 12", 실기 200Hz 고정 그리드) =====
    #   sim 정책주기(step_dt=1/60)로 기록 → 실기 제어주기(0.005s=200Hz)로 선형보간 리샘플.
    #   행당 dt가 실기와 동일(0.005)해져 MATLAB 0.005 가정이 그대로 맞고, 서버 로그 rate와도 일치.
    #   (로봇이 어차피 하던 60→200Hz 보간을 export 단에서 미리 수행 = "보는 것 = 실행되는 것")
    if export_base is not None and len(export_rows) > 0:
        import csv
        import numpy as np
        arr = np.asarray(export_rows, dtype=np.float64)     # (N, 1+13): col0=t(초), col1..13=관절 rad
        t_src = arr[:, 0]
        n_out = int(round(t_src[-1] / REAL_CTRL_DT)) + 1
        t_dst = np.arange(n_out) * REAL_CTRL_DT
        res = np.empty((n_out, arr.shape[1]), dtype=np.float64)
        res[:, 0] = t_dst
        for c in range(1, arr.shape[1]):
            res[:, c] = np.interp(t_dst, t_src, arr[:, c])  # 각 관절을 200Hz 그리드로 선형보간
        with open(args_cli.export_traj, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"joint {k}" for k in range(NUM_REAL_JOINT)])   # 서버 Logger 헤더와 동일
            w.writerows(res.tolist())
        print(f"[EXPORT] 저장 완료: {len(export_rows)} 스텝(60Hz) → {n_out} 행(200Hz, dt={REAL_CTRL_DT}s) "
              f"× {NUM_REAL_JOINT}관절 (phill 양식) → {args_cli.export_traj}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
