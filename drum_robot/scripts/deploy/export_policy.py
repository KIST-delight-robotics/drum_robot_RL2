"""
정책을 실기 배포용으로 export 한다.

    policy.onnx         env 정규화 + 스케일러 + MLP + 결정적 평균 (한 그래프)
    obs_constants.json  obs_builder 가 조립 로직을 짜는 데 필요한 상수
핵심 두 가지
  1) 정규화를 C++ 이 아니라 그래프 안에서 한다.
     그래프 입력은 원시 물리량 9개 버퍼이고, 위치는 전부 "허리 기준 미터".
     -> eps 관례, clamp 값, 148개 인덱스 순서를 C++ 이 재현할 필요가 없어진다.

  2) drum_center 는 허리 기준 그대로 쓴다 (z 오프셋은 정규화에서 상쇄된다).
     sim 은 런타임 drum_pos/tip_pos 를 월드 프레임(+robot_waist_joint_offset_z)에서
     정규화하는데, 정규화 상수는 오프셋 없는 배열에서 만들어졌다(env.py:817 vs 748).
     그 결과 z 14채널이 clamp(+-1.5)에 상시 포화해 스케일러가 상수로 학습했다
     (체크포인트 측정: mean 1.5000, std 8.9e-05).
     C++ 이 허리 기준 z 를 그대로 넣어도 sim 과 같은 값이 나오도록 상수에 접어 둔다.

이 스크립트는 Isaac 을 띄우지 않는다. 체크포인트와 설정 파일만 읽는다.
따라서 관절 순서를 알 수 없어 --joint-order 로 받는다. 기본값은 articulation
BFS 순서 추정치이고, dump_golden.py 가 실기 대조용으로 "실제" 순서를 함께 내보내므로
골든 비교 도구가 둘의 불일치를 기계적으로 잡는다. 사람이 기억할 필요가 없다.

사용법
    python export_policy.py \
        --checkpoint <...>/checkpoints/best_agent.pt \
        --out        <...>/deploy_export
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
SPECS_PY = TASK_DIR / "components" / "specs.py"
HITDET_PY = TASK_DIR / "components" / "hit_detector.py"
ENV_CFG_PY = TASK_DIR / "drumrobot_cfg.py"

# articulation(USD) 순서 추정치.
#   근거: robot_interface.py:35 의 body_names 출력이 링크 트리의 너비우선 열거와
#   정확히 일치했다 (base_link, waist, head, left_shoulder_1, right_shoulder_1,
#   head_2, left_shoulder_2, right_shoulder_2, left_elbow, right_elbow,
#   left_wrist, right_wrist). 관절은 루트를 제외한 링크와 1:1 이고 자식 링크
#   이름을 갖는다. head/head_2 는 ctrl_joint_names 에 없어 제외.
#
#   주의: specs.py 의 ctrl_joint_names 목록 순서도, URDF 선언 순서도 아니다.
#   robot.find_joints() 가 preserve_order 없이 호출되므로 articulation 순서가 이긴다.
DEFAULT_JOINT_ORDER = [
    "waist_joint",
    "left_shoulder_1",
    "right_shoulder_1",
    "left_shoulder_2",
    "right_shoulder_2",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# 실기 모터 id (motors.json). specs.py 의 joint_name_to_ik_result_idx 와 같은 값이지만,
# 그 파일을 읽어 검증하므로 여기서는 하드코딩하지 않는다.

OBS_DIM = 148
ACT_DIM = 9
NUM_DRUM = 8
NUM_HITS = 6            # K
SCHED_NOTES = 2         # 팔별 담당 노트 개수
OPSET = 13              # Gemm + Elu + Clip 뿐이므로 낮게 고정해 ORT 버전 요구를 느슨하게


# ==========================================================================
# 설정 파일 읽기 (Isaac import 회피)
# ==========================================================================

def load_module(path: Path, name: str):
    """패키지 __init__ 을 우회해 파일에서 직접 모듈을 읽는다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 로드할 수 없습니다: {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass 데코레이터가 sys.modules 에서 자기 모듈을 찾으므로 먼저 등록한다.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def scalar_from_source(path: Path, field: str) -> float:
    """`field: type = <숫자>` 형태의 스칼라를 소스에서 뽑는다.

    이 값을 하드코딩하지 않는 이유: 학습 쪽에서 바꿨을 때 조용히 어긋나는 것을 막는다.
    (해당 파일들은 isaaclab 을 import 하므로 모듈로 불러올 수 없어 소스를 읽는다.)
    """
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^\s*{re.escape(field)}\s*:\s*\w+\s*=\s*([-+0-9.eE]+)", text, re.M)
    if not m:
        raise RuntimeError(f"{path.name} 에서 '{field}' 를 찾을 수 없습니다")
    return float(m.group(1))


def load_run_params(run_dir: Path) -> dict:
    """런이 시작할 때 덤프한 params/env.yaml 을 읽는다.

    라이브 소스 파일이 아니라 이걸 봐야 하는 이유: 학습 후 소스가 바뀌면
    조용히 어긋난다. 실제로 그런 일이 있었다 — 손목 게인이 학습 뒤에 수정됐다.
    """
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
    """디렉터리 이름(2026-08-31_09-55-09_...)에서 런 시작 시각을 뽑는다."""
    import datetime as dt
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", run_dir.name)
    if not m:
        return 0.0
    d, hh, mm, ss = m.groups()
    return dt.datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S").timestamp()


def warn_source_drift(run_dir: Path, files: list[Path]) -> list[str]:
    """런 시작 이후에 수정된 소스 파일을 찾는다.

    이 검사가 있으면 '학습 뒤에 상수를 바꿨는데 export 는 새 값을 읽어서
    체크포인트와 어긋나는' 사고가 조용히 지나가지 않는다.
    """
    t0 = run_start_time(run_dir)
    if t0 == 0.0:
        return []
    return [f.name for f in files if f.exists() and f.stat().st_mtime > t0]


# ==========================================================================
# 정규화 상수
# ==========================================================================

def build_constants(joint_order: list[str], run: dict) -> dict:
    specs = load_module(SPECS_PY, "drumrobot_specs")

    robot_spec = specs.RobotSpec()
    instruments = specs.Instruments()

    # ---- 관절 한계 → center / half_range (obs 순서대로) ----
    unknown = [n for n in joint_order if n not in robot_spec.joint_limit]
    if unknown:
        raise RuntimeError(f"joint_limit 에 없는 관절: {unknown}")
    if len(joint_order) != ACT_DIM:
        raise RuntimeError(f"관절 {ACT_DIM}개가 필요합니다 (받음: {len(joint_order)})")

    lo = torch.tensor([robot_spec.joint_limit[n][0] for n in joint_order], dtype=torch.float32)
    hi = torch.tensor([robot_spec.joint_limit[n][1] for n in joint_order], dtype=torch.float32)
    joint_center = 0.5 * (lo + hi)
    joint_half_range = 0.5 * (hi - lo) + 1e-6          # env.py:742-743 과 동일

    # ---- 드럼 위치 → center / half_range ----
    # 악기 id 순서(1..8)로 정렬해야 obs 의 drum_pos 배치와 맞는다.
    by_id = sorted(instruments.all.items(), key=lambda kv: kv[1].id)
    drum_ids = [inst.id for _, inst in by_id]
    drum_names = [name for name, _ in by_id]
    if drum_ids != list(range(1, NUM_DRUM + 1)):
        raise RuntimeError(f"악기 id 가 1..{NUM_DRUM} 연속이 아닙니다: {drum_ids}")

    drum_pos = torch.tensor([inst.position for _, inst in by_id], dtype=torch.float32)   # (8,3)
    drum_min = drum_pos.min(dim=0).values
    drum_max = drum_pos.max(dim=0).values
    drum_center = 0.5 * (drum_min + drum_max)
    drum_half_range = 0.5 * (drum_max - drum_min) + 1e-6                                 # env.py:751-752

    # ---- z 오프셋: 접지 않는다 ★2026-09-02 수정 ----
    #
    # 이전에 drum_center_eff[2] -= z_off 를 했고, 그것이 z 채널 14개를 전부
    # clamp(+1.5) 에 포화시켜 정책이 높이를 관측하지 못하게 만들었다.
    #
    # 올바른 관계:
    #   env  : (env값 − env중심),  env중심 = 허리중심 + z_off   (drumrobot_env.py:798-801)
    #   실기 : (허리값 − 허리중심)
    #   env값 = 허리값 + z_off 이므로 두 식은 항등이다. z_off 가 상쇄된다.
    #
    # 따라서 실기(허리 기준 입력)용 상수는 drum_center 그대로다.
    # inst.position 이 이미 허리 기준이므로 아무것도 더하거나 빼지 않는다.
    z_off = float(run["robot_waist_joint_offset_z"])
    drum_center_eff = drum_center.clone()   # = 허리 기준. 이름은 하위호환으로 유지

    # ---- 나머지 상수는 런이 기록한 값에서 (라이브 소스가 아니라) ----
    joint_vel_scale = float(run["joint_vel_scale"])
    action_scale = float(run["action_scale"])
    wrist_action_scale = float(run["wrist_action_scale"])
    decimation = int(run["decimation"])
    sim_dt = float(run["sim"]["dt"])
    train_step_dt = sim_dt * decimation          # 학습 정책 주기 (= 1/60 s)

    # 액추에이터 게인 — 실기와 대조할 근거로 남긴다
    acts = run["robot_cfg"]["actuators"]
    gains = {k: {"stiffness": float(v["stiffness"]), "damping": float(v["damping"])}
             for k, v in acts.items()}

    hit_cfg = {
        f: scalar_from_source(HITDET_PY, f)
        for f in ("alpha", "drum_xy_radius", "drum_z_range",
                  "min_impact_velocity", "rearm_height", "hit_window_step")
    }
    max_lookahead_s = scalar_from_source(TASK_DIR / "components" / "robotic_drum_score.py",
                                         "max_lookahead_s")
    # 학습(drumrobot_env.py:76)과 같은 식. C++ 이 다시 나누면 부동소수 반올림으로 119가 될 수 있다.
    max_lookahead_step = int(max_lookahead_s / train_step_dt)

    return dict(
        # 학습 지연은 이제 범위 DR — 배포 메타에는 [min, max] 를 둘 다 싣는다.
        #   구 런(env.yaml 에 action_delay_steps 단일 키)도 읽히도록 폴백.
        action_delay_steps_min=int(run.get("action_delay_steps_min", run.get("action_delay_steps", 0))),
        action_delay_steps_max=int(run.get("action_delay_steps_max", run.get("action_delay_steps", 0))),
        joint_order=joint_order,
        joint_name_to_motor_id={n: int(robot_spec.joint_name_to_ik_result_idx[n]) for n in joint_order},
        motor_id_by_obs_index=[int(robot_spec.joint_name_to_ik_result_idx[n]) for n in joint_order],
        joint_center=joint_center,
        joint_half_range=joint_half_range,
        joint_vel_scale=joint_vel_scale,
        drum_names=drum_names,
        drum_ids=drum_ids,
        drum_center=drum_center,
        drum_center_eff=drum_center_eff,
        drum_half_range=drum_half_range,
        z_offset=z_off,
        action_scale=action_scale,
        wrist_action_scale=wrist_action_scale,
        decimation=decimation,
        sim_dt=sim_dt,
        train_step_dt=train_step_dt,
        gains=gains,
        hit=hit_cfg,
        max_lookahead_s=max_lookahead_s,
        max_lookahead_step=max_lookahead_step,
    )


# ==========================================================================
# 네트워크 재구성
# ==========================================================================

def build_net(policy_sd: dict) -> nn.Sequential:
    """skrl 체크포인트의 net_container 를 nn.Sequential 로 되살린다.

    체크포인트가 자기 서술적이다 — net_container.{0,2,4,6} 이 Linear,
    홀수 인덱스가 활성함수(elu). 그래서 skrl 도 IsaacLab 도 필요 없다.
    """
    idx = sorted({int(k.split(".")[1]) for k in policy_sd if k.startswith("net_container.")})
    layers: list[nn.Module] = []
    prev_slot = None

    for slot in idx:
        w = policy_sd[f"net_container.{slot}.weight"]
        b = policy_sd[f"net_container.{slot}.bias"]
        if prev_slot is not None:
            gap = slot - prev_slot
            if gap != 2:
                raise RuntimeError(
                    f"Linear 슬롯 간격이 2가 아닙니다 ({prev_slot} -> {slot}). "
                    "활성함수 구조가 바뀐 것으로 보입니다 — 스크립트를 확인하세요."
                )
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
    """원시 물리량 -> env 정규화 -> 스케일러 -> MLP -> 결정적 평균.

    위치 입력(tip_pos, drum_pos, per_arm_pos)의 단위·프레임은 '허리 기준 미터'.
    z 오프셋은 DRUM_CENTER_EFF 에 접혀 있으므로 C++ 은 변환하지 않는다.
    """

    CLIP_OBS = 1.5          # env 정규화 clamp
    CLIP_VEL = 10.0
    CLIP_SCALER = 5.0       # skrl RunningStandardScaler.clip_threshold
    EPS_SCALER = 1e-8       # sqrt() 바깥에 더한다 — skrl 구현과 동일

    def __init__(self, net: nn.Module, c: dict, scaler_mean, scaler_var):
        super().__init__()
        self.net = net

        def buf(name, t):
            self.register_buffer(name, t.reshape(-1).clone().float())

        buf("joint_center", c["joint_center"])
        buf("joint_half", c["joint_half_range"])
        buf("drum_center_eff", c["drum_center_eff"])
        buf("drum_half", c["drum_half_range"])
        buf("scaler_mean", scaler_mean)
        buf("scaler_std", scaler_var.reshape(-1).clone().float().sqrt())
        self.joint_vel_scale = float(c["joint_vel_scale"])

    def _norm_pos(self, p: torch.Tensor) -> torch.Tensor:
        """(..., 3) 위치를 드럼 상자로 정규화. 마지막 축이 xyz."""
        return torch.clamp((p - self.drum_center_eff) / self.drum_half,
                           -self.CLIP_OBS, self.CLIP_OBS)

    def forward(self,
                joint_pos: torch.Tensor,      # (1, 9)      rad
                joint_vel: torch.Tensor,      # (1, 9)      rad/s
                tip_pos: torch.Tensor,        # (1, 2, 3)   m, 허리 기준
                drum_pos: torch.Tensor,       # (1, 8, 3)   m, 허리 기준
                next_hits: torch.Tensor,      # (1, 6, 11)  정규화 불필요
                hit_armed: torch.Tensor,      # (1, 2, 8)   0/1
                arm_role: torch.Tensor,       # (1, 2)      0/1
                per_arm_pos: torch.Tensor,    # (1, 2, 2, 3) m, 허리 기준
                per_arm_time: torch.Tensor,   # (1, 2, 2)   이미 [0,1]
                ) -> torch.Tensor:
        n = joint_pos.shape[0]

        # ---- A-2 : env 정규화 (env.py:_normalize_and_pack_obs 와 동일) ----
        jp = torch.clamp((joint_pos - self.joint_center) / self.joint_half,
                         -self.CLIP_OBS, self.CLIP_OBS)
        jv = torch.clamp(joint_vel / self.joint_vel_scale, -self.CLIP_VEL, self.CLIP_VEL)
        tp = self._norm_pos(tip_pos)
        dp = self._norm_pos(drum_pos)
        pa = torch.cat([self._norm_pos(per_arm_pos), per_arm_time.unsqueeze(-1)], dim=-1)

        obs = torch.cat([
            jp,
            jv,
            tp.reshape(n, -1),
            dp.reshape(n, -1),
            next_hits.reshape(n, -1),
            hit_armed.reshape(n, -1),
            arm_role.reshape(n, -1),
            pa.reshape(n, -1),
        ], dim=-1)

        # ---- B : 스케일러 (skrl RunningStandardScaler._compute 와 동일) ----
        obs = torch.clamp((obs - self.scaler_mean) / (self.scaler_std + self.EPS_SCALER),
                          -self.CLIP_SCALER, self.CLIP_SCALER)

        # ---- C : 신경망. GaussianMixin 의 평균만 (샘플링 없음) ----
        return self.net(obs)


INPUT_NAMES = ["joint_pos", "joint_vel", "tip_pos", "drum_pos", "next_hits",
               "hit_armed", "arm_role", "per_arm_pos", "per_arm_time"]
OUTPUT_NAMES = ["action_mean"]


def dummy_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, ACT_DIM),
        torch.zeros(1, ACT_DIM),
        torch.zeros(1, 2, 3),
        torch.zeros(1, NUM_DRUM, 3),
        torch.zeros(1, NUM_HITS, NUM_DRUM + 3),
        torch.zeros(1, 2, NUM_DRUM),
        torch.zeros(1, 2),
        torch.zeros(1, 2, SCHED_NOTES, 3),
        torch.zeros(1, 2, SCHED_NOTES),
    )


def random_inputs(gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """검증용. 실제 물리량 규모에 가깝게 만든다 (허리 기준 미터)."""
    def r(*shape, lo=-1.0, hi=1.0):
        return torch.rand(*shape, generator=gen) * (hi - lo) + lo
    return (
        r(1, ACT_DIM, lo=-1.5, hi=1.5),
        r(1, ACT_DIM, lo=-8.0, hi=8.0),
        r(1, 2, 3, lo=-0.5, hi=0.5),
        r(1, NUM_DRUM, 3, lo=-0.5, hi=0.5),
        r(1, NUM_HITS, NUM_DRUM + 3, lo=0.0, hi=1.0),
        (r(1, 2, NUM_DRUM, lo=0.0, hi=1.0) > 0.5).float(),
        (r(1, 2, lo=0.0, hi=1.0) > 0.5).float(),
        r(1, 2, SCHED_NOTES, 3, lo=-0.5, hi=0.5),
        r(1, 2, SCHED_NOTES, lo=0.0, hi=1.0),
    )


# ==========================================================================
# main
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--motors-json", type=Path, default=None,
                    help="실기 motors.json. 주면 손목 CST 게인이 학습과 맞는지 대조한다.")
    ap.add_argument("--joint-order", nargs=ACT_DIM, default=DEFAULT_JOINT_ORDER,
                    help="obs/action 의 관절 순서 (articulation 순서). "
                         "기본값은 BFS 추정치이며 dump_golden.py 가 실제 값을 검증한다.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 0. 런 파라미터 · 소스 드리프트 검사 ----
    run_dir = args.checkpoint.resolve().parents[1]
    run = load_run_params(run_dir)
    print(f"[0/6] 런: {run_dir.name}")

    drift = warn_source_drift(run_dir, [SPECS_PY, HITDET_PY, ENV_CFG_PY,
                                        TASK_DIR / "components" / "robotic_drum_score.py"])
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
    print(f"[1/6] 네트워크 재구성: {[l.out_features for l in net if isinstance(l, nn.Linear)]}")

    # ---- 2. 상수 ----
    c = build_constants(list(args.joint_order), run)
    print(f"[2/6] 상수: z_offset={c['z_offset']}, "
          f"drum_center_eff={[round(v, 4) for v in c['drum_center_eff'].tolist()]}, "
          f"train_step_dt={c['train_step_dt']:.6f}s")
    for k, g in c["gains"].items():
        print(f"      게인 {k:6s} stiffness={g['stiffness']:<7} damping={g['damping']}")

    # ---- 2b. 실기 게인 교차검증 ----
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
            print(f"      대조 {m['name']:<12} 실기 {k_eq:.3f}/{d_eq:.3f}  "
                  f"vs 학습 {sim_w['stiffness']}/{sim_w['damping']}  {'OK' if ok else '불일치'}")
            if not ok:
                bad.append(m["name"])
        if bad:
            print(f"\n      !! 손목 게인 불일치: {bad}")
            print(f"         학습이 모델링한 법칙과 실기가 실행하는 법칙이 다릅니다. 중단합니다.")
            return 2

    # ---- 2c. 정규화 포화 가드 ★2026-09-02 추가 ----
    #
    # 왜 필요한가: 2026-09-02 에 drum_center 의 z 오프셋 부호를 잘못 넣어
    # z 채널 14개(tip 2 + drum 8 + per_arm 4)가 전부 clamp(±1.5)에 붙었다.
    # 정책이 높이를 관측하지 못해 타격이 0회가 됐는데, 원시버퍼 대조와
    # PyTorch-vs-ORT 검증은 둘 다 통과했다 — 같은 상수를 쓰기 때문이다.
    #
    # 이 가드는 "이 상수로 실제 드럼 좌표를 정규화하면 범위 안에 들어오는가"를
    # 직접 본다. 상수가 프레임을 잘못 잡으면 즉시 드러난다.
    # 드럼 좌표의 극값을 center/half_range 에서 복원한다 (build_constants 가 그렇게 만들었다).
    dc  = c["drum_center"]           # 허리 기준
    dhr = c["drum_half_range"]
    dce = c["drum_center_eff"]       # 그래프가 실제로 쓰는 상수
    drum_wr = torch.stack([dc - (dhr - 1e-6), dc + (dhr - 1e-6)])   # (2,3) min/max
    dn = (drum_wr - dce) / dhr
    sat = (dn.abs() > 1.5)
    n_sat = int(sat.sum())
    print(f"      정규화 포화 가드: 드럼 극값 2점 x 3축 중 포화 {n_sat}개")
    if n_sat > 0:
        worst = dn.abs().max().item()
        ax = ["x", "y", "z"]
        per_ax = {ax[i]: int(sat[:, i].sum()) for i in range(3)}
        print(f"\n      !! 정규화 상수가 드럼 좌표를 clamp(±1.5) 밖으로 보냅니다.")
        print(f"         축별 포화: {per_ax}   최대 |정규화값| = {worst:.2f}")
        print(f"         drum_center_eff = {[round(v, 4) for v in dce.tolist()]}")
        print(f"         drum_half_range = {[round(v, 4) for v in dhr.tolist()]}")
        print(f"         드럼 좌표 범위   = {[round(v, 4) for v in drum_wr[0].tolist()]}"
              f" ~ {[round(v, 4) for v in drum_wr[1].tolist()]}")
        print(f"\n         이 상태로 배포하면 해당 축이 상수가 되어 정책이 그 축을")
        print(f"         관측하지 못합니다. 프레임(허리 기준 vs env)을 확인하세요.")
        print(f"         drumrobot_env.py 의 drum_center 계산과 같은 프레임이어야 합니다.")
        print(f"         중단합니다.")
        return 3

    # 스케일러 std 진단 (학습 중 상수였던 채널)
    std = scaler_var.sqrt()
    degenerate = (std < 1e-3).nonzero().flatten().tolist()
    print(f"      스케일러 std < 1e-3 채널 {len(degenerate)}개: {degenerate}")
    print(f"      -> 이 채널들은 학습 중 상수였다.")

    # ---- 3. 그래프 ----
    model = DeployPolicy(net, c, scaler_mean, scaler_var).eval()

    onnx_path = args.out / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_inputs(), str(onnx_path),
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
    print(f"[3/6] export: {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB, opset {OPSET})")

    # ---- 4. 검증 — PyTorch vs onnxruntime ----
    gen = torch.Generator().manual_seed(0)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        worst = 0.0
        for _ in range(32):
            xs = random_inputs(gen)
            with torch.no_grad():
                ref = model(*xs).numpy()
            got = sess.run(OUTPUT_NAMES,
                           {n: x.numpy() for n, x in zip(INPUT_NAMES, xs)})[0]
            worst = max(worst, float(abs(ref - got).max()))
        status = "OK" if worst < 1e-5 else "불일치"
        print(f"[4/6] 검증 {status}: PyTorch vs ORT 최대 오차 {worst:.3e} (32회)")
        if worst >= 1e-5:
            return 1
    except ImportError:
        print("[4/6] 검증 건너뜀 — onnxruntime 이 없습니다. "
              "`pip install onnxruntime` 후 다시 실행하면 대조합니다.")

    # ---- 5. 상수 파일 ----
    def lst(t):
        return [round(float(v), 9) for v in t.reshape(-1).tolist()]

    payload = {
        "_comment": "export_policy.py 가 생성. 손으로 고치지 말 것 — 재export 하면 덮어씀.",
        "checkpoint": str(args.checkpoint),
        "run": run_dir.name,
        "source_drift_after_run": drift,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "opset": OPSET,

        "graph_input_names": INPUT_NAMES,
        "graph_output_names": OUTPUT_NAMES,
        "graph_does_normalization": True,
        "position_frame": "waist_relative_meters",

        "joint_order": c["joint_order"],
        "joint_order_source": "articulation BFS 추정 (dump_golden.py 가 검증)",
        "motor_id_by_obs_index": c["motor_id_by_obs_index"],
        "joint_name_to_motor_id": c["joint_name_to_motor_id"],

        "drum_names": c["drum_names"],
        "drum_ids": c["drum_ids"],

        # 실기 drum_coordinate.json -> 그래프 입력 변환.
        #   실기는 악기마다 right/left 좌표를 따로 갖는데 학습은 하나만 쓴다.
        #   전수 대조 결과 "실기 중점 - (0,0,0.045)" 가 학습 좌표와 일치한다
        #   (XY 오차 0.5mm 이내 = 반올림, Z 는 8개 전부 정확히 +45mm).
        #   z 는 clamp 에 흡수되므로 결과에 영향이 없지만, XY 중점 변환은 필수다.
        "real_to_sim_drum": {
            "method": "midpoint(real.right.position, real.left.position) - (0, 0, z_shift)",
            "z_shift": 0.045,
            "verified_max_xy_error_m": 0.0005,
            "note": "z 오차는 정규화 clamp(+-1.5)에 흡수된다. XY 는 흡수되지 않는다.",
        },

        # 실기 정책 주기. 학습 60Hz 를 실기 200Hz 에서 3틱(66.7Hz)으로 받는다.
        #   200/60 = 3.33 이 정수가 아니라 3틱으로 근사한 것. 적분은 q = q_now + a·scale·policy_dt 이고
        #   실제 dt(0.015)를 써야 지령 각속도가 보존된다. 1.0/60.0 하드코딩 금지.
        #   ⚠️ 실기 C++ 의 POLICY_TICK_STRIDE 와 반드시 같아야 한다 (불일치 시 실기가 정책을 거절).
        #   2026-09-01 100Hz(stride 2) 로 바꿨다가 학습 실패로 되돌림 — SIM_DT 주석 참조.
        #     100Hz 재시도 시 여기도 2 / round(2*0.005,6) 로 함께 갱신할 것.
        "policy_tick_stride": 3,
        "policy_dt": round(3 * 0.005, 6),
        "train_step_dt": round(c["train_step_dt"], 12),
        "sim_dt": round(c["sim_dt"], 12),
        "decimation": c["decimation"],
        "actuator_gains": c["gains"],
        "action_delay_steps_min": c["action_delay_steps_min"],
        "action_delay_steps_max": c["action_delay_steps_max"],
        "action_scale": c["action_scale"],
        "wrist_action_scale": c["wrist_action_scale"],

        "hit_detector": {
            "drum_xy_radius": c["hit"]["drum_xy_radius"],
            "drum_z_range": c["hit"]["drum_z_range"],
            "min_impact_velocity": c["hit"]["min_impact_velocity"],
            "rearm_height": c["hit"]["rearm_height"],
            "tip_vel_filter_alpha": c["hit"]["alpha"],
            # 스텝 단위 상수는 학습 스텝 주기(1/60s)로 초 변환한다 (명세 04절)
            "hit_window_step": int(c["hit"]["hit_window_step"]),
            "hit_window_s": round(c["hit"]["hit_window_step"] * c["train_step_dt"], 6),
            "rearm_s": round((2 * c["hit"]["hit_window_step"] + 1) * c["train_step_dt"], 6),
        },
        "max_lookahead_s": c["max_lookahead_s"],
        # 스텝 수를 직접 싣는다. C++ 이 max_lookahead_s / train_step_dt 로 재계산하면
        # 반올림 때문에 119가 나올 수 있고, time_norm 분모가 (L-1) 이라 오차가 그대로 들어간다.
        "max_lookahead_step": c["max_lookahead_step"],
        "time_norm_denom_s": round((c["max_lookahead_step"] - 1) * c["train_step_dt"], 12),
        "sched_notes": SCHED_NOTES,
        "num_hits": NUM_HITS,

        "_normalization_in_graph": {
            "note": "C++ 은 정규화하지 않는다. 아래는 참고·검증용.",
            "joint_center": lst(c["joint_center"]),
            "joint_half_range": lst(c["joint_half_range"]),
            "joint_vel_scale": c["joint_vel_scale"],
            "drum_center_raw": lst(c["drum_center"]),
            "drum_center_eff": lst(c["drum_center_eff"]),
            "drum_half_range": lst(c["drum_half_range"]),
            "z_offset_folded": c["z_offset"],
            "degenerate_channels": degenerate,
        },
    }

    json_path = args.out / "obs_constants.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"[5/6] 상수: {json_path}")

    print("\n관절 매핑 (obs 인덱스 -> 실기 모터 id)")
    for i, name in enumerate(c["joint_order"]):
        print(f"  obs[{i}]  {name:<18} -> motor {c['motor_id_by_obs_index'][i]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
