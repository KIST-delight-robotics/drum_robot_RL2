"""
로봇 인터페이스 클래스
"""

from __future__ import annotations
                                                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch                                    # pyright: ignore[reportMissingImports]
from isaaclab.assets import Articulation        # pyright: ignore[reportMissingImports]
from isaaclab.utils import math as math_utils   # pyright: ignore[reportMissingImports]

from .specs import EnvRuntimeSpec, RobotSpec

class RobotInterface:
    def __init__(
            self,
            device: torch.device | str,
            env: EnvRuntimeSpec,
            robot: Articulation,
            env_origins: torch.Tensor,
    ):
        self.device = device
        self.env = env
        self.robot_spec = RobotSpec()
        self.env_origins = env_origins

        # body/joint id resolve
        ids, names = robot.find_joints(self.robot_spec.ctrl_joint_names)
        
        self.ctrl_joint_ids = ids       # 항상 articulation 순서
        self.ctrl_joint_names = names
        if len(self.ctrl_joint_ids) != len(self.robot_spec.ctrl_joint_names):
            raise RuntimeError(f"ctrl_joint_ids mismatch: {len(self.ctrl_joint_ids)} (expected 9). names={self.ctrl_joint_names}")
        
        self.body_to_idx = {name: self._get_body_idx(robot, name) for name in robot.data.body_names}
        print("body_names:", list(self.body_to_idx.keys()))

        # offset wrist link to tip
        L_off = torch.tensor(self.robot_spec.tip_offset_left, device=self.device, dtype=torch.float32)  # (3,)
        R_off = torch.tensor(self.robot_spec.tip_offset_right, device=self.device, dtype=torch.float32)

        N = self.env.num_envs
        self.tip_offset_L = L_off.unsqueeze(0).expand(N, 3)
        self.tip_offset_R = R_off.unsqueeze(0).expand(N, 3)

        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """

        # 관절 제한값
        self.joint_low = torch.tensor(
            [self.robot_spec.joint_limit[name][0] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)  # unsqueeze(n) n번째 차원에 1인 차원을 생성

        self.joint_high = torch.tensor(
            [self.robot_spec.joint_limit[name][1] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        # 로봇 관절-USD 관절 방향
        self.dir_tensor = torch.tensor(
            [self.robot_spec.joint_usd_dir[name] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)
    
    def get_joint_pos(self, robot: Articulation) -> torch.Tensor:
        # 로봇의 관절 위치
        usd_pos = robot.data.joint_pos[:, self.ctrl_joint_ids]        # (num_envs, 9)
        joint_pos = self._convert_usd_to_robot(usd_pos)
        return joint_pos
    
    def get_joint_vel(self, robot: Articulation) -> torch.Tensor:
        # 로봇의 관절 속도
        usd_vel = robot.data.joint_vel[:, self.ctrl_joint_ids]        # (num_envs, 9)
        joint_vel = self._convert_usd_to_robot(usd_vel)
        return joint_vel

    def get_joint_torque(self, robot: Articulation) -> torch.Tensor:
        # 액추에이터가 실제로 가한 관절 토크 (N·m). 일률 페널티(|τ·q̇|)용, 제곱/절댓값이라 방향 변환 불필요
        return robot.data.applied_torque[:, self.ctrl_joint_ids]      # (num_envs, 9)

    def clip_and_convert(self, q: torch.Tensor) -> torch.Tensor:
        # clip
        robot_q = torch.max(torch.min(q, self.joint_high), self.joint_low)

        # convert
        usd_q = self._convert_robot_to_usd(robot_q)

        return usd_q

    def get_ctrl_joint_ids(self):
        return self.ctrl_joint_ids
    
    def get_ctrl_joint_name(self):
        return self.ctrl_joint_names
    
    def get_body_pos(self, robot: Articulation) -> torch.Tensor:
        # 스틱 링크의 월드 좌표계 위치 가져오기
        all_body_pos = robot.data.body_pos_w       # (N, num_bodies, 3)
        all_body_quat = robot.data.body_quat_w     # (N, num_bodies, 4)  (w,x,y,z)인 경우가 많음

        # tip
        name = "left_wrist"

        idx = self.body_to_idx[name]
        pos = all_body_pos[:, idx]                  # (N, 3)
        quat = all_body_quat[:, idx]

        # 팁 위치 구하기
        tip_w = pos + math_utils.quat_apply(quat, self.tip_offset_L)

        # 월드 기준 -> env 기준
        L_tip = tip_w - self.env_origins

        name = "right_wrist"

        idx = self.body_to_idx[name]
        pos = all_body_pos[:, idx]                  # (N, 3)
        quat = all_body_quat[:, idx]

        # 팁 위치 구하기
        tip_w = pos + math_utils.quat_apply(quat, self.tip_offset_R)

        # 월드 기준 -> env 기준
        R_tip = tip_w - self.env_origins

        tip_pos = torch.stack([L_tip, R_tip], dim=1)   # (N, 2, 3)

        return tip_pos
    
    def get_wrist_pos(self, robot: Articulation) -> torch.Tensor:
        # 왼쪽/오른쪽 wrist 링크의 env 기준 위치 반환
        all_body_pos = robot.data.body_pos_w  # (N, num_bodies, 3)

        l_idx = self.body_to_idx["left_wrist"]
        r_idx = self.body_to_idx["right_wrist"]

        L_wrist = all_body_pos[:, l_idx] - self.env_origins  # (N, 3)
        R_wrist = all_body_pos[:, r_idx] - self.env_origins  # (N, 3)

        return torch.stack([L_wrist, R_wrist], dim=1)  # (N, 2, 3)

    def get_elbow_pos(self, robot: Articulation) -> torch.Tensor:
        # 왼쪽/오른쪽 elbow 링크의 env 기준 위치 반환
        all_body_pos = robot.data.body_pos_w  # (N, num_bodies, 3)

        l_idx = self.body_to_idx["left_elbow"]
        r_idx = self.body_to_idx["right_elbow"]

        L_elbow = all_body_pos[:, l_idx] - self.env_origins  # (N, 3)
        R_elbow = all_body_pos[:, r_idx] - self.env_origins  # (N, 3)

        return torch.stack([L_elbow, R_elbow], dim=1)  # (N, 2, 3)

    def get_limit(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.joint_low, self.joint_high
    
    def reset(self, robot: Articulation, env_ids: torch.Tensor, init_robot_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_joint_pos = robot.data.default_joint_pos[env_ids]  # 기본 자세 가져오기
        default_joint_vel = robot.data.default_joint_vel[env_ids]

        joint_pos = default_joint_pos.clone()
        init_usd_pos = self._convert_robot_to_usd(init_robot_pos)
        joint_pos[:,self.ctrl_joint_ids] = init_usd_pos

        joint_vel = torch.zeros_like(default_joint_vel)

        return joint_pos, joint_vel

    def _get_body_idx(self, robot: Articulation, body_name: str) -> int:
        ids, _ = robot.find_bodies(body_name) # ([id], ['body name'])
        
        if len(ids) == 0:
            raise RuntimeError("body not found. Check USD body names.")
        
        idx = ids[0] # int 값만 저장

        return idx
    
    def _convert_usd_to_robot(self, usd_data: torch.Tensor) -> torch.Tensor:
        return self.dir_tensor * usd_data

    def _convert_robot_to_usd(self, robot_data: torch.Tensor) -> torch.Tensor:
        return self.dir_tensor * robot_data