"""
로봇 초기 위치 초기화 클래스 
"""

from __future__ import annotations

from dataclasses import dataclass, field
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]
import math

from .specs import RobotSpec, Instruments
from .ik_solver import IKSolver

@dataclass
class RobotInitializerCfg:
    # 제어하는 관절 개수
    num_ctrl_joint: int = 9

    # 양 팔이 위치 가능한 드럼 조합 [L, R]
    drum_pairs: list = field(default_factory=lambda: [
        ("snare", "snare"),   ("hihat", "snare"),
        ("snare", "floor"),   ("floor", "floor"),   ("mid", "floor"),     ("high", "floor"),    ("hihat", "floor"),  ("ride", "floor"),    ("crash_l", "floor"),
        ("snare", "mid"),     ("mid", "mid"),       ("high", "mid"),      ("hihat", "mid"),     ("crash_l", "mid"),
        ("snare", "high"),    ("high", "high"),     ("hihat", "high"),    ("crash_l", "high"),
        ("snare", "hihat"),   ("hihat", "hihat"),
        ("snare", "ride"),    ("floor", "ride"),    ("mid", "ride"),      ("high", "ride"),     ("hihat", "ride"),   ("ride", "ride"),     ("crash_l", "ride"),
        ("floor", "crash_r"), ("mid", "crash_r"),   ("high", "crash_r"),  ("ride", "crash_r"), ("crash_r", "crash_r"),
        ("snare", "crash_l"), ("high", "crash_l"),  ("hihat", "crash_l"), ("crash_l", "crash_l"),
    ])

    height_above_drum: float = 0.1

    # 실기 서버 drum_coordinate.json 의 손별 타격점 [m] (허리 조인트 기준) 과 손목 각도 [deg]
    #   이름 -> (오른손 위치, 오른손 손목, 왼손 위치, 왼손 손목). 시작 자세 = 이 타격 자세 + 손목 대기 각도 (실기 리셋과 같음)
    hit_points: dict = field(default_factory=lambda: {
        "snare":   ((-0.061, 0.363, -0.443), 10.0, (-0.139, 0.358, -0.427), 10.0),
        "floor":   (( 0.284, 0.358, -0.423), 10.0, ( 0.180, 0.359, -0.457), 10.0),
        "mid":     (( 0.244, 0.585, -0.327),  5.0, ( 0.188, 0.608, -0.339),  5.0),
        "high":    ((-0.031, 0.614, -0.263),  5.0, (-0.106, 0.600, -0.288),  5.0),   # json "top"
        "hihat":   ((-0.266, 0.514, -0.174), 10.0, (-0.318, 0.472, -0.183), 10.0),   # json "closed hihat"
        "ride":    (( 0.357, 0.625, -0.106), 15.0, ( 0.295, 0.663, -0.095), 15.0),
        "crash_r": (( 0.496, 0.404, -0.208), 10.0, ( 0.474, 0.444, -0.200), 10.0),   # json "right crash"
        "crash_l": ((-0.162, 0.676, -0.083), 10.0, (-0.205, 0.661, -0.121), 10.0),   # json "left crash"
    })
    wrist_stay_deg: float = 20.0            # 실기 스트로크 규칙 손목 대기 각도: 타격 자세에서 손목을 이만큼 들어 올린 자세로 시작 (+ = 드는 방향)

    joint_noise_scale: float = 5*math.pi/180
    noise_joints: tuple = ("waist_joint", "right_shoulder_1", "left_shoulder_1")    # 시작 자세 노이즈를 넣는 관절 (스틱 높이를 안 바꾸는 허리·어깨 yaw만)

class RobotInitializer:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RobotInitializerCfg,
            ctrl_joint_names: list,
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.ctrl_joint_names = ctrl_joint_names
        self.instruments = Instruments()
        self.robot = RobotSpec()

        self._init_pos_angle()
        self.noise_mask = torch.tensor([float(n in cfg.noise_joints) for n in ctrl_joint_names], device=self.device)   # (9,) 노이즈 넣는 관절 1

    def _init_pos_angle(self):
        N = len(self.cfg.drum_pairs)
        ik_solver = IKSolver(device=self.device)

        # 이름 → Instrument lookup
        inst_by_name = self.instruments.all

        # 각 페어의 좌/우 팁 목표 위치를 바로 텐서로
        # pl = torch.tensor(
        #     [inst_by_name[l_name].position for l_name, _ in self.cfg.drum_pairs],
        #     device=self.device, dtype=torch.float32,
        # )  # (N, 3)
        # pr = torch.tensor(
        #     [inst_by_name[r_name].position for _, r_name in self.cfg.drum_pairs],
        #     device=self.device, dtype=torch.float32,
        # )  # (N, 3)
        pl = torch.tensor(
            [self.cfg.hit_points[l_name][2] for l_name, _ in self.cfg.drum_pairs],
            device=self.device, dtype=torch.float32,
        )  # (N, 3) 왼손 타격점 (실기 json)
        pr = torch.tensor(
            [self.cfg.hit_points[r_name][0] for _, r_name in self.cfg.drum_pairs],
            device=self.device, dtype=torch.float32,
        )  # (N, 3) 오른손 타격점 (실기 json)

        # pl[:, 2] = pl[:, 2] + self.cfg.height_above_drum
        # pr[:, 2] = pr[:, 2] + self.cfg.height_above_drum

        pm_xy = (pl[:, 0:2] + pr[:, 0:2]) / 2
        theta0 = torch.atan2(pm_xy[:, 1], pm_xy[:, 0]) - 90*math.pi/180

        # theta7 = torch.full((N,), 25*math.pi/180, device=self.device)
        # theta8 = torch.full((N,), 25*math.pi/180, device=self.device)
        theta7 = torch.tensor([math.radians(self.cfg.hit_points[r_name][1]) for _, r_name in self.cfg.drum_pairs], device=self.device)   # 오른손 타격 손목 각도 (실기 json)
        theta8 = torch.tensor([math.radians(self.cfg.hit_points[l_name][3]) for l_name, _ in self.cfg.drum_pairs], device=self.device)   # 왼손 타격 손목 각도 (실기 json)

        out, err = ik_solver.solve_geometric_ik(pr, pl, theta0, theta7, theta8)    # (N, 9)

        if err.any():
            failed = [self.cfg.drum_pairs[i] for i in torch.where(err > 0.5)[0].tolist()]
            raise RuntimeError(f"[RobotInitializer] IK failed for {len(failed)}/{N} pairs: {failed}")

        out[:, 7] = out[:, 7] + math.radians(self.cfg.wrist_stay_deg)    # 오른손 손목 대기 각도만큼 들기 (IK 결과 7 = right_wrist)
        out[:, 8] = out[:, 8] + math.radians(self.cfg.wrist_stay_deg)    # 왼손 손목 (IK 결과 8 = left_wrist)

        self.pos_angle = torch.zeros((N, self.cfg.num_ctrl_joint), device=self.device)

        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """
        for i in range(self.cfg.num_ctrl_joint):
            name = self.ctrl_joint_names[i]
            idx = self.robot.joint_name_to_ik_result_idx[name]
            self.pos_angle[:, i] = out[:, idx]

        # 드럼 쌍 이름 -> 드럼 번호 (Instruments 순서), 시작 자세 드럼을 알려주기 위해
        names = list(inst_by_name.keys())
        self.pair_drum_idx = torch.tensor(
            [[names.index(l_name), names.index(r_name)] for l_name, r_name in self.cfg.drum_pairs],
            device=self.device,
            dtype=torch.long,
        )   # (N, 2) [왼손, 오른손]

    def reset_init_pos(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = len(self.cfg.drum_pairs)

        rand = torch.randint(0, N, (len(env_ids),), device=self.device)

        init_pos = self.pos_angle[rand, :]
        # init_pos = init_pos + self.cfg.joint_noise_scale * torch.randn_like(init_pos)
        init_pos = init_pos + self.cfg.joint_noise_scale * torch.randn_like(init_pos) * self.noise_mask   # 허리·어깨 yaw만 노이즈

        return init_pos, self.pair_drum_idx[rand]   # (n, 9) 시작 관절각, (n, 2) 시작 자세 드럼 [왼손, 오른손]