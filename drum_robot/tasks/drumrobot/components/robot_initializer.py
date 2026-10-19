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

    joint_noise_scale: float = 5*math.pi/180

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

    def _init_pos_angle(self):
        N = len(self.cfg.drum_pairs)
        ik_solver = IKSolver(device=self.device)

        # 이름 → Instrument lookup
        inst_by_name = self.instruments.all

        # 각 페어의 좌/우 팁 목표 위치를 바로 텐서로
        pl = torch.tensor(
            [inst_by_name[l_name].position for l_name, _ in self.cfg.drum_pairs],
            device=self.device, dtype=torch.float32,
        )  # (N, 3)
        pr = torch.tensor(
            [inst_by_name[r_name].position for _, r_name in self.cfg.drum_pairs],
            device=self.device, dtype=torch.float32,
        )  # (N, 3)

        pl[:, 2] = pl[:, 2] + self.cfg.height_above_drum
        pr[:, 2] = pr[:, 2] + self.cfg.height_above_drum

        pm_xy = (pl[:, 0:2] + pr[:, 0:2]) / 2
        theta0 = torch.atan2(pm_xy[:, 1], pm_xy[:, 0]) - 90*math.pi/180

        theta7 = torch.full((N,), 25*math.pi/180, device=self.device)
        theta8 = torch.full((N,), 25*math.pi/180, device=self.device)

        out, err = ik_solver.solve_geometric_ik(pr, pl, theta0, theta7, theta8)    # (N, 9)

        if err.any():
            failed = [self.cfg.drum_pairs[i] for i in torch.where(err > 0.5)[0].tolist()]
            raise RuntimeError(f"[RobotInitializer] IK failed for {len(failed)}/{N} pairs: {failed}")

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
        init_pos = init_pos + self.cfg.joint_noise_scale * torch.randn_like(init_pos)

        return init_pos, self.pair_drum_idx[rand]   # (n, 9) 시작 관절각, (n, 2) 시작 자세 드럼 [왼손, 오른손]