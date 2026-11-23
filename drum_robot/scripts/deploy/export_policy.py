"""
정책을 실기 배포용으로 export 한다 (관측 90, 허리 + 어깨 yaw 정렬 모델, yaw 기준 궤적).

    policy.onnx         env 정규화 + 스케일러 + MLP + 결정적 평균 (한 그래프)
    obs_constants.json  실기 C++ (policy_runner / obs_builder) 가 필요로 하는 상수

그래프 입력 11개 (원시 물리량, 위치는 "허리 조인트 기준 미터"):
    joint_pos (1,9)  joint_vel (1,9)  lead (1,9)  tip_pos (1,2,3)  drum_pos (1,8,3)
    arm_next (1,2,9)  waist_des (1,1)  waist_vel_des (1,1)  yaw_des (1,2)  yaw_vel_des (1,2)  last_action (1,9)
출력: action_mean (1,9)

정규화 (drumrobot_env.py::_normalize_and_pack_obs 와 동일)
    joint_pos : (q - center)/half_range, clamp ±1.5
    joint_vel : /joint_vel_scale, clamp ±10
    lead      : /max_lead                      (max_lead = effort_limit / stiffness, 관절별)
    tip, drum : z 에 robot_waist_joint_offset_z 를 더한 뒤 (p - drum_center)/half_range, clamp ±1.5
                drum_center 는 오프셋 없는 Instruments.position 에서 만든다 (env _init_obs 와 같게).
                -> z 채널 10개가 sim 과 같이 +1.5 에 포화된다. 체크포인트가 그 상태로 학습됐으므로
                   고치지 않고 재현한다. (스케일러 std 로 교차 확인한다 — 아래 2c)
    arm_next  : 그대로
    waist_des : 허리 기준 θ0_d [rad] -> (θ0_d - center)/half_range (허리 관절 값), clamp 없음
    waist_vel_des : 허리 기준 속도 θ̇0_d [rad/s] -> /joint_vel_scale, clamp ±10
    yaw_des   : 어깨 yaw 기준 θk_d [오른, 왼] [rad] -> (θk_d - center)/half_range (어깨 yaw 관절 값), clamp 없음
    yaw_vel_des : 어깨 yaw 기준 속도 θ̇k_d [오른, 왼] [rad/s] -> /joint_vel_scale, clamp ±10
    last_action : 그대로 (직전 행동, clamp ±1·frozen 0 적용 후. 리셋 직후 0)

Isaac 을 띄우지 않는다. 체크포인트, 런 params/env.yaml, 소스 파일, conventions.json 만 읽는다.
conventions.json (dump_conventions.py) 은 필수다: 관절 순서, max_lead, 고정 관절, shoulder_z 가 거기서 온다.

사용법 (레포 루트에서)
    python drum_robot/scripts/deploy/export_policy.py \
        --checkpoint  logs/skrl/drum_robot/<run>/checkpoints/best_agent.pt \
        --conventions drum_robot/deploy_export/conventions.json \
        --out         drum_robot/deploy_export \
        [--motors-json /home/shy/RL_Real/Phil-drum-robot/drumrobot_server/config/motors.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ==========================================================================
# 경로 · 상수
# ==========================================================================

HERE = Path(__file__).resolve()
TASK_DIR = HERE.parents[2] / "tasks" / "drumrobot"          # drum_robot/tasks/drumrobot
COMP_DIR = TASK_DIR / "components"
SPECS_PY = COMP_DIR / "specs.py"
REWARD_PY = COMP_DIR / "reward.py"
PLANNER_PY = COMP_DIR / "sticking_planner.py"
RDS_PY = COMP_DIR / "robotic_drum_score.py"
HITDET_PY = COMP_DIR / "hit_detector.py"
INIT_PY = COMP_DIR / "robot_initializer.py"
ENV_PY = TASK_DIR / "drumrobot_env.py"
ENV_CFG_PY = TASK_DIR / "drumrobot_cfg.py"

OBS_DIM = 90
ACT_DIM = 9
NUM_DRUM = 8            # M
NUM_HITS = 6            # K
NH_CH = NUM_DRUM + 2    # next_hits 채널: multi-hot M + time + valid
ARM_CH = NUM_DRUM + 1   # arm_next 채널: one-hot M + 남은 시간
OPSET = 13              # Gemm + Elu + Clip + Add 뿐이라 낮게 고정

INPUT_NAMES = ["joint_pos", "joint_vel", "lead", "tip_pos", "drum_pos",
               "arm_next", "waist_des", "waist_vel_des", "yaw_des", "yaw_vel_des", "last_action"]
INPUT_SHAPES = {
    "joint_pos": (1, ACT_DIM), "joint_vel": (1, ACT_DIM), "lead": (1, ACT_DIM),
    "tip_pos": (1, 2, 3), "drum_pos": (1, NUM_DRUM, 3),
    "arm_next": (1, 2, ARM_CH), "waist_des": (1, 1), "waist_vel_des": (1, 1),
    "yaw_des": (1, 2), "yaw_vel_des": (1, 2),
    "last_action": (1, ACT_DIM),
}
OUTPUT_NAMES = ["action_mean"]

# obs_90 안의 구간 시작 인덱스 (env _normalize_and_pack_obs 의 cat 순서)
OFF_JP, OFF_JV, OFF_LEAD = 0, 9, 18
OFF_TIP, OFF_DRUM = 27, 33
OFF_ARM, OFF_WD = 57, 75
OFF_WVD = 76
OFF_YD = 77
OFF_YVD = 79
OFF_LAST = 81
assert OFF_WD + 1 == OFF_WVD
assert OFF_WVD + 1 == OFF_YD
assert OFF_YD + 2 == OFF_YVD
assert OFF_YVD + 2 == OFF_LAST
assert OFF_LAST + ACT_DIM == OBS_DIM


# ==========================================================================
# 설정 파일 읽기 (Isaac import 회피)
# ==========================================================================

def load_module(path: Path, name: str):
    """패키지 __init__ 을 우회해 파일에서 직접 모듈을 읽는다 (isaaclab 을 import 하지 않는 파일만)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 로드할 수 없습니다: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass 데코레이터가 sys.modules 에서 자기 모듈을 찾는다
    spec.loader.exec_module(mod)
    return mod


def scalar_from_source(path: Path, field: str) -> float:
    """`field: type = <숫자>` 또는 `field: type = <숫자>*math.pi/180` 형태의 스칼라를 소스에서 뽑는다.

    해당 파일들은 isaaclab 또는 상대 import 를 쓰므로 모듈로 불러올 수 없어 소스를 읽는다.
    하드코딩하지 않는 이유: 학습 쪽에서 바꿨을 때 조용히 어긋나는 것을 막는다.
    """
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^\s*{re.escape(field)}\s*:\s*[\w\[\], ]+\s*=\s*([-+0-9.eE]+)(\s*\*\s*math\.pi\s*/\s*180)?",
                  text, re.M)
    if not m:
        raise RuntimeError(f"{path.name} 에서 '{field}' 를 찾을 수 없습니다")
    v = float(m.group(1))
    return math.radians(v) if m.group(2) else v


def load_run_params(run_dir: Path) -> dict:
    """런이 시작할 때 덤프한 params/env.yaml 을 읽는다 (라이브 소스가 아니라 학습 당시 값)."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def ignore_tag(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("", ignore_tag)
    path = run_dir / "params" / "env.yaml"
    if not path.exists():
        raise RuntimeError(f"런 파라미터가 없습니다: {path}")
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)


def run_start_time(run_dir: Path) -> float:
    import datetime as dt
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", run_dir.name)
    if not m:
        return 0.0
    d, hh, mm, ss = m.groups()
    return dt.datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S").timestamp()


def warn_source_drift(run_dir: Path, files: list[Path]) -> list[str]:
    """런 시작 이후에 수정된 소스 파일. 여기서 읽는 상수가 체크포인트와 어긋날 수 있다."""
    t0 = run_start_time(run_dir)
    if t0 == 0.0:
        return []
    return [f.name for f in files if f.exists() and f.stat().st_mtime > t0]


# ==========================================================================
# 상수
# ==========================================================================

def build_constants(conv: dict, run: dict) -> dict:
    specs = load_module(SPECS_PY, "drumrobot_specs")
    robot_spec = specs.RobotSpec()
    instruments = specs.Instruments()
    pl = specs.PartLength()

    # ---- 관절 순서: conventions.json 이 Isaac 에서 찍은 실제 articulation 순서 ----
    joint_order = list(conv["joint_order_resolved"])
    if sorted(joint_order) != sorted(robot_spec.ctrl_joint_names) or len(joint_order) != ACT_DIM:
        raise RuntimeError(f"conventions 관절 목록이 specs.ctrl_joint_names 와 다릅니다: {joint_order}")

    lo = torch.tensor([robot_spec.joint_limit[n][0] for n in joint_order], dtype=torch.float32)
    hi = torch.tensor([robot_spec.joint_limit[n][1] for n in joint_order], dtype=torch.float32)
    conv_lo = torch.tensor(conv["joint_limit_low_resolved_order"], dtype=torch.float32)
    conv_hi = torch.tensor(conv["joint_limit_high_resolved_order"], dtype=torch.float32)
    if (lo - conv_lo).abs().max() > 1e-6 or (hi - conv_hi).abs().max() > 1e-6:
        raise RuntimeError("specs.joint_limit 과 conventions 의 관절 한계가 다릅니다 (소스가 학습 뒤 바뀌었나)")
    joint_center = 0.5 * (lo + hi)
    joint_half_range = 0.5 * (hi - lo) + 1e-6           # env _init_obs 와 동일

    # ---- 드럼: Instruments.all 의 dict 순서 = 드럼 인덱스 (env 와 같은 규약) ----
    drum_names = list(instruments.all.keys())
    drum_ids = [inst.id for inst in instruments.all.values()]
    if drum_ids != list(range(1, NUM_DRUM + 1)):
        raise RuntimeError(f"악기 id 가 1..{NUM_DRUM} 순서가 아닙니다: {drum_ids}")
    drum_pos = torch.tensor([inst.position for inst in instruments.all.values()], dtype=torch.float32)  # (8,3) 허리 기준
    drum_min = drum_pos.min(dim=0).values
    drum_max = drum_pos.max(dim=0).values
    drum_center = 0.5 * (drum_min + drum_max)
    drum_half_range = 0.5 * (drum_max - drum_min) + 1e-6   # env _init_obs 와 동일 (오프셋 없음)

    # ---- z 프레임: env 는 drum_pos/tip_pos 의 z 에 이 값을 더한 채로 정규화한다 ----
    z_off = float(run["robot_waist_joint_offset_z"])
    if abs(z_off - float(conv["reset_pose_probe"]["robot_waist_joint_offset_z"])) > 1e-9:
        raise RuntimeError("robot_waist_joint_offset_z 가 런과 conventions 에서 다릅니다")

    # ---- 런이 기록한 값 ----
    joint_vel_scale = float(run["joint_vel_scale"])
    action_scale = float(run["action_scale"])
    wrist_action_scale = float(run["wrist_action_scale"])
    decimation = int(run["decimation"])
    sim_dt = float(run["sim"]["dt"])
    train_step_dt = sim_dt * decimation
    acts = run["robot_cfg"]["actuators"]
    gains = {k: {"stiffness": float(v["stiffness"]), "damping": float(v["damping"]),
                 "effort_limit_sim": float(v["effort_limit_sim"])} for k, v in acts.items()}

    # ---- max_lead: Isaac 실측값 (conventions) 과 런 값(effort/stiffness) 교차 확인 ----
    max_lead = torch.tensor(conv["max_lead_obs_order"], dtype=torch.float32)
    def group_of(name: str) -> str:
        if name == "waist_joint":
            return "waist"
        return "wrist" if "wrist" in name else "arm"
    max_lead_run = torch.tensor([gains[group_of(n)]["effort_limit_sim"] / gains[group_of(n)]["stiffness"]
                                 for n in joint_order], dtype=torch.float32)
    if (max_lead - max_lead_run).abs().max() > 1e-4:
        raise RuntimeError(f"max_lead 불일치: conventions {max_lead.tolist()} vs 런 {max_lead_run.tolist()}")

    # ---- 고정 관절 (정렬 전용 학습: 행동 0) ----
    frozen_names = list(conv["frozen_joint_names"])
    frozen_idx = [joint_order.index(n) for n in frozen_names]
    active_names = [n for n in joint_order if n not in frozen_names]

    # ---- 어깨 높이 (정렬 reach 판정용), 허리 기준 ----
    shoulder_z = float(conv["shoulder_z_waist_frame_m"])

    # ---- 스티킹 플래너 상수 (sticking_planner.py 는 torch 만 import 하므로 모듈로 읽는다) ----
    planner_mod = load_module(PLANNER_PY, "drumrobot_sticking_planner")
    planner_cfg = dataclasses.asdict(planner_mod.StickingPlannerCfg(step_dt=train_step_dt))
    missing = [n for n in drum_names if n not in planner_cfg["role_cost"]]
    if missing:
        raise RuntimeError(f"role_cost 에 없는 악기: {missing}")
    planner_cfg["role_cost"] = {n: list(planner_cfg["role_cost"][n]) for n in drum_names}   # 드럼 인덱스 순서로

    # ---- 정렬 상수 (reward.py 는 상대 import 라 소스에서 뽑는다) ----
    align = {f: scalar_from_source(REWARD_PY, f)
             for f in ("w_waist", "w_yaw", "w1", "w2", "k_time", "waist_step_deg", "reach_ratio")}
    align["l_waist_to_shoulder_m"] = pl.waist / 2
    align["arm_max_length_m"] = pl.upper_arm + pl.lower_arm + pl.stick
    align["d_th_m"] = align["reach_ratio"] * align["arm_max_length_m"]
    align["shoulder_z_waist_frame_m"] = shoulder_z
    align["joint_limit_rad"] = {n: list(robot_spec.joint_limit[n]) for n in ("waist_joint", "right_shoulder_1", "left_shoulder_1")}
    align["part_length_m"] = dataclasses.asdict(pl)

    # ---- 악보 (RDS) ----
    max_lookahead_s = scalar_from_source(RDS_PY, "max_lookahead_s")
    num_hits = int(scalar_from_source(RDS_PY, "num_hits"))
    if num_hits != NUM_HITS:
        raise RuntimeError(f"num_hits 가 {NUM_HITS} 이 아닙니다: {num_hits}")
    max_lookahead_step = int(max_lookahead_s / train_step_dt)          # env __init__ 와 같은 식 (int 절삭)
    max_match_window_s = scalar_from_source(HITDET_PY, "max_match_window_s")
    hit_window_step = round(max_match_window_s / train_step_dt)        # env: round(...)

    # ---- 학습 시작 자세 (실기 START 자세와 비교할 근거) ----
    start_pose = {
        "height_above_drum_m": scalar_from_source(INIT_PY, "height_above_drum"),
        "joint_noise_scale_rad": scalar_from_source(INIT_PY, "joint_noise_scale"),
        "wrist_deg": 25.0,     # robot_initializer.py theta7/theta8 (소스 고정값)
        "note": "드럼 쌍 (왼손, 오른손) 위 height_above_drum 에 팁을 두는 IK 자세 + 관절 노이즈. 쌍 목록은 robot_initializer.py drum_pairs",
    }

    return dict(
        joint_order=joint_order,
        joint_name_to_motor_id={n: int(robot_spec.joint_name_to_ik_result_idx[n]) for n in joint_order},
        motor_id_by_obs_index=[int(robot_spec.joint_name_to_ik_result_idx[n]) for n in joint_order],
        joint_usd_dir=[int(robot_spec.joint_usd_dir[n]) for n in joint_order],
        joint_low=lo, joint_high=hi,
        joint_center=joint_center, joint_half_range=joint_half_range,
        joint_vel_scale=joint_vel_scale,
        max_lead=max_lead,
        frozen_names=frozen_names, frozen_idx=frozen_idx, active_names=active_names,
        drum_names=drum_names, drum_ids=drum_ids, drum_pos=drum_pos,
        drum_center=drum_center, drum_half_range=drum_half_range,
        z_offset=z_off,
        action_scale=action_scale, wrist_action_scale=wrist_action_scale,
        decimation=decimation, sim_dt=sim_dt, train_step_dt=train_step_dt,
        gains=gains,
        action_delay_steps_min=int(run.get("action_delay_steps_min", 0)),
        action_delay_steps_max=int(run.get("action_delay_steps_max", 0)),
        planner=planner_cfg,
        align=align,
        max_lookahead_s=max_lookahead_s, max_lookahead_step=max_lookahead_step,
        max_match_window_s=max_match_window_s, hit_window_step=hit_window_step,
        start_pose=start_pose,
    )


# ==========================================================================
# 네트워크 재구성
# ==========================================================================

def build_net(policy_sd: dict) -> nn.Sequential:
    """skrl 체크포인트의 net_container 를 nn.Sequential 로 되살린다 (Linear/ELU 교대)."""
    idx = sorted({int(k.split(".")[1]) for k in policy_sd if k.startswith("net_container.")})
    layers: list[nn.Module] = []
    prev_slot = None
    for slot in idx:
        w = policy_sd[f"net_container.{slot}.weight"]
        b = policy_sd[f"net_container.{slot}.bias"]
        if prev_slot is not None:
            if slot - prev_slot != 2:
                raise RuntimeError(f"Linear 슬롯 간격이 2가 아닙니다 ({prev_slot} -> {slot}). 활성함수 구조를 확인하세요.")
            layers.append(nn.ELU())
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        prev_slot = slot
    net = nn.Sequential(*layers)
    if layers[0].in_features != OBS_DIM:
        raise RuntimeError(f"입력 차원이 {OBS_DIM} 이 아닙니다: {layers[0].in_features}")
    if layers[-1].out_features != ACT_DIM:
        raise RuntimeError(f"출력 차원이 {ACT_DIM} 이 아닙니다: {layers[-1].out_features}")
    return net


# ==========================================================================
# 배포 그래프
# ==========================================================================

class DeployPolicy(nn.Module):
    """원시 물리량(허리 기준) -> env 정규화 -> 스케일러 -> MLP -> 결정적 평균."""

    CLIP_OBS = 1.5          # env 정규화 clamp (joint_pos, tip, drum)
    CLIP_VEL = 10.0
    CLIP_SCALER = 5.0       # skrl RunningStandardScaler.clip_threshold
    EPS_SCALER = 1e-8       # sqrt() 바깥에 더한다 — skrl 구현과 동일

    def __init__(self, net: nn.Module, c: dict, scaler_mean, scaler_var):
        super().__init__()
        self.net = net

        def buf(name, t):
            self.register_buffer(name, torch.as_tensor(t).reshape(-1).clone().float())

        buf("joint_center", c["joint_center"])
        buf("joint_half", c["joint_half_range"])
        self.j_waist = c["joint_order"].index("waist_joint")   # waist_des 정규화용 허리 인덱스 (obs 순서)
        self.j_yaw = [c["joint_order"].index("right_shoulder_1"), c["joint_order"].index("left_shoulder_1")]   # yaw_des 정규화용 [오른, 왼] (env yaw_ctrl_ids)
        buf("max_lead", c["max_lead"])
        buf("drum_center", c["drum_center"])
        buf("drum_half", c["drum_half_range"])
        buf("z_shift", torch.tensor([0.0, 0.0, c["z_offset"]]))     # 허리 기준 -> env 프레임
        buf("scaler_mean", scaler_mean)
        buf("scaler_std", torch.as_tensor(scaler_var).reshape(-1).clone().float().sqrt())
        self.joint_vel_scale = float(c["joint_vel_scale"])

    def _norm_pos(self, p: torch.Tensor) -> torch.Tensor:
        """허리 기준 (..., 3) 위치 -> env 프레임(z + offset) -> 드럼 상자 정규화 (env 와 동일)."""
        return torch.clamp((p + self.z_shift - self.drum_center) / self.drum_half,
                           -self.CLIP_OBS, self.CLIP_OBS)

    def forward(self,
                joint_pos: torch.Tensor,    # (1, 9)      rad
                joint_vel: torch.Tensor,    # (1, 9)      rad/s
                lead: torch.Tensor,         # (1, 9)      rad  (누적 목표 - 측정)
                tip_pos: torch.Tensor,      # (1, 2, 3)   m, 허리 기준 [왼, 오른]
                drum_pos: torch.Tensor,     # (1, 8, 3)   m, 허리 기준
                arm_next: torch.Tensor,     # (1, 2, 9)   [왼, 오른] one-hot 8 + 남은 시간
                waist_des: torch.Tensor,    # (1, 1)      허리 기준 θ0_d [rad]
                waist_vel_des: torch.Tensor,  # (1, 1)    허리 기준 속도 θ̇0_d [rad/s]
                yaw_des: torch.Tensor,      # (1, 2)      어깨 yaw 기준 θk_d [오른, 왼] [rad]
                yaw_vel_des: torch.Tensor,  # (1, 2)      어깨 yaw 기준 속도 θ̇k_d [오른, 왼] [rad/s]
                last_action: torch.Tensor,  # (1, 9)      직전 행동 (clamp·frozen 0 적용 후), 정규화 불필요
                ) -> torch.Tensor:
        n = joint_pos.shape[0]
        jp = torch.clamp((joint_pos - self.joint_center) / self.joint_half, -self.CLIP_OBS, self.CLIP_OBS)
        jv = torch.clamp(joint_vel / self.joint_vel_scale, -self.CLIP_VEL, self.CLIP_VEL)
        ld = lead / self.max_lead
        tp = self._norm_pos(tip_pos)
        dp = self._norm_pos(drum_pos)
        wd = (waist_des - self.joint_center[self.j_waist]) / self.joint_half[self.j_waist]   # env waist_des 와 동일, clamp 없음
        wv = torch.clamp(waist_vel_des / self.joint_vel_scale, -self.CLIP_VEL, self.CLIP_VEL)   # env waist_vel_des_n 과 동일
        yd = (yaw_des - self.joint_center[self.j_yaw]) / self.joint_half[self.j_yaw]   # env yaw_des 와 동일, clamp 없음
        yv = torch.clamp(yaw_vel_des / self.joint_vel_scale, -self.CLIP_VEL, self.CLIP_VEL)   # env yaw_vel_des_n 과 동일

        obs = torch.cat([
            jp, jv, ld,
            tp.reshape(n, -1),
            dp.reshape(n, -1),
            arm_next.reshape(n, -1),
            wd.reshape(n, -1),
            wv.reshape(n, -1),
            yd.reshape(n, -1),
            yv.reshape(n, -1),
            last_action.reshape(n, -1),
        ], dim=-1)

        obs = torch.clamp((obs - self.scaler_mean) / (self.scaler_std + self.EPS_SCALER),
                          -self.CLIP_SCALER, self.CLIP_SCALER)
        return self.net(obs)


def dummy_inputs() -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros(*INPUT_SHAPES[k]) for k in INPUT_NAMES)


def random_inputs(gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """검증용. 실제 물리량 규모에 가깝게 (허리 기준 미터)."""
    def r(shape, lo=-1.0, hi=1.0):
        return torch.rand(*shape, generator=gen) * (hi - lo) + lo
    return (
        r(INPUT_SHAPES["joint_pos"], -1.5, 1.5),
        r(INPUT_SHAPES["joint_vel"], -8.0, 8.0),
        r(INPUT_SHAPES["lead"], -0.6, 0.6),
        r(INPUT_SHAPES["tip_pos"], -0.6, 0.6),
        r(INPUT_SHAPES["drum_pos"], -0.6, 0.6),
        r(INPUT_SHAPES["arm_next"], 0.0, 1.0),
        r(INPUT_SHAPES["waist_des"], -1.5, 1.5),
        r(INPUT_SHAPES["waist_vel_des"], -3.0, 3.0),
        r(INPUT_SHAPES["yaw_des"], 0.0, 3.2),
        r(INPUT_SHAPES["yaw_vel_des"], -3.0, 3.0),
        r(INPUT_SHAPES["last_action"], -1.0, 1.0),
    )


# ==========================================================================
# main
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--conventions", required=True, type=Path, help="dump_conventions.py 산출물")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--motors-json", type=Path, default=None,
                    help="실기 motors.json. 주면 손목 CST 게인이 학습과 맞는지 대조한다.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 0. 런 파라미터 · 소스 드리프트 ----
    run_dir = args.checkpoint.resolve().parents[1]
    run = load_run_params(run_dir)
    conv = json.loads(args.conventions.read_text(encoding="utf-8"))
    print(f"[0/6] 런: {run_dir.name}")
    if int(run.get("observation_space", 0) if isinstance(run.get("observation_space"), int) else 0) not in (0, OBS_DIM):
        print(f"      !! 런의 observation_space 가 {OBS_DIM} 이 아닙니다: {run.get('observation_space')}")

    drift = warn_source_drift(run_dir, [SPECS_PY, REWARD_PY, PLANNER_PY, RDS_PY, HITDET_PY, INIT_PY, ENV_PY, ENV_CFG_PY])
    if drift:
        print(f"      !! 경고 — 런 시작 이후 수정된 소스: {', '.join(drift)}")
        print(f"         이 파일에서 읽는 상수가 체크포인트와 어긋날 수 있습니다.")

    # ---- 1. 체크포인트 ----
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for key in ("policy", "state_preprocessor"):
        if key not in ck:
            raise RuntimeError(f"체크포인트에 '{key}' 가 없습니다")
    sp = ck["state_preprocessor"]
    scaler_mean = sp["running_mean"].reshape(-1).float()
    scaler_var = sp["running_variance"].reshape(-1).float()
    if scaler_mean.numel() != OBS_DIM:
        raise RuntimeError(f"스케일러 차원이 {OBS_DIM} 이 아닙니다: {scaler_mean.numel()}")
    net = build_net(ck["policy"])
    print(f"[1/6] 네트워크: {[l.out_features for l in net if isinstance(l, nn.Linear)]}")

    # ---- 2. 상수 ----
    c = build_constants(conv, run)
    print(f"[2/6] 상수: train_step_dt={c['train_step_dt']:.6f}s, z_offset={c['z_offset']}, "
          f"lookahead {c['max_lookahead_s']}s = {c['max_lookahead_step']}스텝")
    print(f"      action_scale {c['action_scale']} / 손목 {c['wrist_action_scale']}, "
          f"max_lead {[round(v, 4) for v in c['max_lead'].tolist()]}")
    print(f"      움직이는 관절 {c['active_names']}, 고정 {c['frozen_names']}")
    print(f"      shoulder_z(허리 기준) {c['align']['shoulder_z_waist_frame_m']:.4f} m")

    # ---- 2b. 실기 손목 게인 교차검증 ----
    if args.motors_json:
        real = json.loads(args.motors_json.read_text(encoding="utf-8"))
        wrist = [m for m in real["motors"] if m.get("name", "").endswith("_wrist")]
        sim_w = c["gains"]["wrist"]
        bad = []
        for m in wrist:
            gr = float(m["gear_ratio"])
            k_eq = float(m["control_kp"]) * gr / 1000.0
            d_eq = float(m["control_kd"]) * gr / 1000.0
            ok = abs(k_eq - sim_w["stiffness"]) < 1e-3 and abs(d_eq - sim_w["damping"]) < 1e-4
            print(f"      대조 {m['name']:<12} 실기 {k_eq:.3f}/{d_eq:.3f} vs 학습 {sim_w['stiffness']}/{sim_w['damping']}  {'OK' if ok else '불일치'}")
            if not ok:
                bad.append(m["name"])
        if bad:
            print(f"\n      !! 손목 게인 불일치: {bad}. 중단합니다.")
            return 2

    # ---- 2c. z 포화 교차 확인 ----
    # 그래프가 실제 드럼 좌표(허리 기준)를 넣었을 때 포화시키는 obs 채널 = sim 에서도 상수였던 채널이어야 한다.
    # 스케일러 std 가 그 증거다. 포화 채널인데 std 가 크면 프레임이 학습과 다른 것이다.
    model = DeployPolicy(net, c, scaler_mean, scaler_var).eval()
    dn = model._norm_pos(c["drum_pos"])                         # (8,3) 실제 드럼 좌표의 정규화값
    sat_ch = [OFF_DRUM + 3 * d + k for d in range(NUM_DRUM) for k in range(3) if abs(dn[d, k].item()) >= 1.5]
    std = scaler_var.sqrt()
    degenerate = (std < 1e-3).nonzero().flatten().tolist()
    bad_sat = [i for i in sat_ch if i not in degenerate]
    print(f"      z 포화 채널(드럼) {len(sat_ch)}개 {sat_ch}")
    print(f"      스케일러 std<1e-3 채널 {len(degenerate)}개 {degenerate}")
    if bad_sat:
        print(f"\n      !! 그래프가 포화시키는 채널 {bad_sat} 이 학습 중에는 상수가 아니었습니다.")
        print(f"         z 프레임이 학습과 다릅니다 (drum_center / z_offset 확인). 중단합니다.")
        return 3
    xy_sat = [i for i in sat_ch if (i - OFF_DRUM) % 3 != 2]
    if xy_sat:
        print(f"\n      !! x/y 채널이 포화됩니다: {xy_sat}. 중단합니다.")
        return 3

    # ---- 3. 그래프 ----
    onnx_path = args.out / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(model, dummy_inputs(), str(onnx_path),
                          input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)
    print(f"[3/6] export: {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB, opset {OPSET})")

    # ---- 4. PyTorch vs onnxruntime ----
    gen = torch.Generator().manual_seed(0)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        worst = 0.0
        for _ in range(32):
            xs = random_inputs(gen)
            with torch.no_grad():
                ref = model(*xs).numpy()
            got = sess.run(OUTPUT_NAMES, {n: x.numpy() for n, x in zip(INPUT_NAMES, xs)})[0]
            worst = max(worst, float(abs(ref - got).max()))
        print(f"[4/6] 검증 {'OK' if worst < 1e-5 else '불일치'}: PyTorch vs ORT 최대 오차 {worst:.3e} (32회)")
        if worst >= 1e-5:
            return 1
    except ImportError:
        print("[4/6] 검증 건너뜀 — onnxruntime 이 없습니다.")

    # ---- 5. 상수 파일 ----
    def lst(t):
        return [round(float(v), 9) for v in torch.as_tensor(t).reshape(-1).tolist()]

    payload = {
        "_comment": "export_policy.py 가 생성. 손으로 고치지 말 것 — 재export 하면 덮어씀.",
        "checkpoint": str(args.checkpoint),
        "run": run_dir.name,
        "conventions": str(args.conventions),
        "source_drift_after_run": drift,
        "task": "alignment_only",
        "task_note": "허리 + 어깨 yaw 가 움직이는 정렬 전용 정책. frozen 관절은 행동 0 (누적 목표 유지). 타격은 가상 타격.",
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "opset": OPSET,

        "graph_input_names": INPUT_NAMES,
        "graph_input_shapes": {k: list(INPUT_SHAPES[k]) for k in INPUT_NAMES},
        "graph_output_names": OUTPUT_NAMES,
        "graph_does_normalization": True,
        "position_frame": "waist_relative_meters",
        "obs_layout": {
            "joint_pos": [OFF_JP, 9], "joint_vel": [OFF_JV, 9], "lead": [OFF_LEAD, 9],
            "tip_pos": [OFF_TIP, 6], "drum_pos": [OFF_DRUM, 24],
            "arm_next": [OFF_ARM, 2 * ARM_CH], "waist_des": [OFF_WD, 1],
            "waist_vel_des": [OFF_WVD, 1],
            "yaw_des": [OFF_YD, 2],
            "yaw_vel_des": [OFF_YVD, 2],
            "last_action": [OFF_LAST, ACT_DIM],
        },
        "last_action_note": "직전 스텝에 적용한 행동 (obs 순서, clamp ±1 후 frozen 관절 0). 리셋 직후 0. drumrobot_env.py self.actions",

        "joint_order": c["joint_order"],
        "joint_order_source": "conventions.json (Isaac articulation 순서 실측)",
        "motor_id_by_obs_index": c["motor_id_by_obs_index"],
        "joint_name_to_motor_id": c["joint_name_to_motor_id"],
        "joint_limit_low_rad": lst(c["joint_low"]),
        "joint_limit_high_rad": lst(c["joint_high"]),

        # ===== 행동 -> 관절 목표 (drumrobot_env.py _pre_physics_step) =====
        #   target += clamp(a,-1,1) * scale * policy_dt     (frozen 관절은 a = 0)
        #   target  = clamp(target, q_now - max_lead, q_now + max_lead)
        #   target  = clamp(target, joint_low, joint_high)
        #   lead    = target - q_now  (다음 obs)
        #   초기값: 인계 시점 자세 (sim 은 리셋 자세)
        "action_scale": c["action_scale"],
        "wrist_action_scale": c["wrist_action_scale"],
        "action_scale_by_obs_index": [c["wrist_action_scale"] if "wrist" in n else c["action_scale"] for n in c["joint_order"]],
        "max_lead_by_obs_index": lst(c["max_lead"]),
        "frozen_joint_names": c["frozen_names"],
        "frozen_obs_index": c["frozen_idx"],
        "active_joint_names": c["active_names"],

        "drum_names": c["drum_names"],
        "drum_ids": c["drum_ids"],
        "train_drum_pos_waist_frame_m": [lst(c["drum_pos"][d]) for d in range(NUM_DRUM)],
        "real_to_sim_drum": {
            "method": "midpoint(real.right.position, real.left.position) - (0, 0, z_shift)",
            "z_shift": 0.045,
            "verified_max_xy_error_m": 0.0005,
            "note": "2026-10-26 재확인: 8개 전부 xy 0.5mm 이내, z +45mm.",
        },

        "policy_tick_stride": 3,
        "policy_dt": round(3 * 0.005, 6),
        "train_step_dt": round(c["train_step_dt"], 12),
        "sim_dt": round(c["sim_dt"], 12),
        "decimation": c["decimation"],
        "actuator_gains": c["gains"],
        "action_delay_steps_min": c["action_delay_steps_min"],
        "action_delay_steps_max": c["action_delay_steps_max"],

        # ===== 악보 (robotic_drum_score.py get_next_hits) =====
        "max_lookahead_s": c["max_lookahead_s"],
        "max_lookahead_step": c["max_lookahead_step"],
        "time_norm_denom_s": round((c["max_lookahead_step"] - 1) * c["train_step_dt"], 12),
        "num_hits": NUM_HITS,
        "next_hits_channels": NH_CH,
        "next_hits_note": "채널 = multi-hot M + time(offset/(L-1), 0~1) + valid. 세기 채널 없음.",
        "hit_window_step": c["hit_window_step"],
        "max_match_window_s": c["max_match_window_s"],

        # ===== 스티킹 플래너 (sticking_planner.py) =====
        "sticking_planner": c["planner"],

        # ===== 정렬 목표 (reward.py AlignmentReward) =====
        "alignment": c["align"],
        "arm_next_channels": ARM_CH,
        "arm_next_note": "손별 [왼, 오른]: 목표 드럼 one-hot(플래너 배정 다음 음표, 없으면 그 손이 직전에 친 드럼) + 남은 시간(없으면 1)",
        "waist_des_note": "허리 기준 θ0_d [rad] (reward.py AlignmentReward.ref_waist, θ0* 로 가는 최소 저크). 그래프가 joint_pos 와 같은 center/half_range 로 정규화, clamp 없음",
        "waist_vel_des_note": "허리 기준 속도 θ̇0_d [rad/s] (reward.py AlignmentReward.ref_vel). 그래프가 /joint_vel_scale, clamp ±10",
        "yaw_des_note": "어깨 yaw 기준 θk_d [오른, 왼] [rad] (reward.py AlignmentReward.ref_yaw: 그 손 목표 드럼이 바뀌면 직전 θk_d 에서 θk*(실제 q0 기준) 로 최소 저크). 그래프가 어깨 yaw 관절 center/half_range 로 정규화, clamp 없음",
        "yaw_vel_des_note": "어깨 yaw 기준 속도 θ̇k_d [오른, 왼] [rad/s] (reward.py AlignmentReward.ref_yaw_vel = wrap(θk_d − 직전 θk_d) / step_dt). 그래프가 /joint_vel_scale, clamp ±10",

        "train_start_pose": c["start_pose"],

        "_normalization_in_graph": {
            "note": "C++ 은 정규화하지 않는다. 아래는 참고·검증용.",
            "joint_center": lst(c["joint_center"]),
            "joint_half_range": lst(c["joint_half_range"]),
            "joint_vel_scale": c["joint_vel_scale"],
            "max_lead": lst(c["max_lead"]),
            "drum_center_waist_frame": lst(c["drum_center"]),
            "drum_half_range": lst(c["drum_half_range"]),
            "z_offset_added_in_graph": c["z_offset"],
            "saturated_z_channels_by_design": sat_ch,
            "degenerate_scaler_channels": degenerate,
        },
    }
    json_path = args.out / "obs_constants.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[5/6] 상수: {json_path}")

    print("\n관절 매핑 (obs 인덱스 -> 실기 모터 id)")
    for i, name in enumerate(c["joint_order"]):
        tag = "고정" if i in c["frozen_idx"] else "정책"
        print(f"  obs[{i}]  {name:<18} -> motor {c['motor_id_by_obs_index'][i]}  ({tag})")
    print("[6/6] 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
