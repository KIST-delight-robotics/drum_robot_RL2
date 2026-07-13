from __future__ import annotations
                                                                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch                                                    # pyright: ignore[reportMissingImports]
from typing import Sequence

import numpy as np                                              # pyright: ignore[reportMissingImports]
import gymnasium as gym                                         # pyright: ignore[reportMissingImports]

import isaaclab.sim as sim_utils                                # pyright: ignore[reportMissingImports]
from isaaclab.assets import Articulation, ArticulationCfg       # pyright: ignore[reportMissingImports]
from isaaclab.envs import DirectRLEnv                           # pyright: ignore[reportMissingImports]
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane     # pyright: ignore[reportMissingImports]

from .drumrobot_cfg import DrumRobotEnvCfg
from .components.specs import EnvRuntimeSpec, Instruments
from .components.robot_interface import RobotInterface
from .components.robotic_drum_score import RDSCfg, RDS
from .components.hit_detector import HitDetector, HitDetectorCfg
from .components.reward import RewardComputerCfg, RewardComputer
from .components.robot_initializer import RobotInitializerCfg, RobotInitializer
from .components.visualizer import VisualizerCfg, Visualizer

from drum_robot.utils.logger import EnvLogger, LoggerCfg


class DrumRobotEnv(DirectRLEnv):
    cfg: DrumRobotEnvCfg

    # ============================================================
    # [Override Functions from DirectRLEnv]
    # These functions are automatically called by the RL loop.
    # DO NOT change signature.
    # 실행 순서 
    # 초기화: _setup_scene -> __init__
    # 루프: _get_observations -> _pre_physics_step -> _apply_action (*decimation) -> _get_dones -> _get_rewards
    # ============================================================

    def __init__(self, cfg: DrumRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Space 정의 (반드시 single-env shape로 정의)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32
        )

        # episode length (step)
        self.episode_length_step = int(self.cfg.episode_length_s / self.step_dt)

        # step
        N = self.num_envs
        self.steps = torch.zeros((N,), device=self.device, dtype=torch.int64)

        # drum
        self.instruments = Instruments()
        self.default_drum_pos = torch.tensor(
            [inst.position for inst in self.instruments.all.values()],
            device=self.device,
            dtype=torch.float32,
        )
        M = len(self.instruments.all)
        self.drum_pos = torch.zeros((N, M, 3), device=self.device, dtype=torch.float32)

        # spec
        hit_detector_cfg = HitDetectorCfg()
        rds_cfg = RDSCfg()
        max_lookahead_step = int(rds_cfg.max_lookahead_s / self.step_dt)
        env_specs = EnvRuntimeSpec(
            num_envs=self.num_envs,
            episode_length_step=self.episode_length_step,
            max_lookahead_step=max_lookahead_step,
            hit_window_step=hit_detector_cfg.hit_window_step,
            step_dt=self.step_dt,
        )

        # robot
        self.robot_interface = RobotInterface(
            device=self.device,
            env=env_specs,
            robot=self.robot,
            env_origins=self.scene.env_origins,
        )

        # 관절별 action_scale (손목만 상한 크게 → 손목 스냅 가능). ctrl_joint_names는 resolve된 순서
        ctrl_joint_names = self.robot_interface.get_ctrl_joint_name()
        scale_list = [
            self.cfg.wrist_action_scale if "wrist" in name else self.cfg.action_scale
            for name in ctrl_joint_names
        ]
        self.action_scale_vec = torch.tensor(
            scale_list, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # 손목 관절 인덱스 [왼, 오] (손목 각속도 계산용). hit_per_arm 팔 축(0=왼,1=오)과 맞춤
        name_to_idx = {name: i for i, name in enumerate(ctrl_joint_names)}
        self.wrist_joint_ids = torch.tensor(
            [name_to_idx["left_wrist"], name_to_idx["right_wrist"]],
            device=self.device, dtype=torch.long,
        )

        # RDS
        self.rds = RDS(
            device=self.device,
            cfg=rds_cfg,
            env=env_specs,
        )

        # 타격 감지
        self.hit_detector = HitDetector(
            device=self.device,
            cfg=hit_detector_cfg,
            env=env_specs,
        )

        # 보상 계산
        self.reward_computer = RewardComputer(
            device=self.device,
            cfg=RewardComputerCfg(),
            env=env_specs,
            ctrl_joint_names=self.robot_interface.get_ctrl_joint_name(),
        )

        # 로봇 초기 위치 initializer
        ctrl_joint_names = self.robot_interface.get_ctrl_joint_name()
        self.robot_initializer = RobotInitializer(
            device=self.device,
            cfg=RobotInitializerCfg(),
            ctrl_joint_names=ctrl_joint_names,
        )

        # 관측 벡터 및 정규화를 위한 변수 초기화
        self._init_obs()

        # 로그
        self.logger = EnvLogger(
            num_envs=self.num_envs,
            device=self.device,
            cfg=LoggerCfg(interval=2000, sample_env_id=0),
        )

        # 시각화
        self.visualizer = Visualizer(
            device=self.device,
            cfg=VisualizerCfg(),
            env=env_specs,
        )
        
        drum_names = list(self.instruments.all.keys())
        self.visualizer.init_visualization(drum_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 광원 추가
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # 로봇의 관절 위치
        joint_pos = self.robot_interface.get_joint_pos(self.robot)

        # 로봇의 관절 속도
        joint_vel = self.robot_interface.get_joint_vel(self.robot)

        # 위치, 로봇 상태
        tip_pos, hit_armed = self.hit_detector.get_value_for_obs()
        drum_pos = self.drum_pos

        # 다음 타격
        self.next_hits = self.rds.get_next_hits(self.steps)

        obs = self._normalize_and_pack_obs(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            tip_pos=tip_pos,
            drum_pos=drum_pos,
            next_hits=self.next_hits,
            hit_armed=hit_armed.float(),
            )
        
        if torch.isnan(obs).any():
            raise RuntimeError("NaN obs")
        
        return {"policy": obs}

    def _pre_physics_step(self, actions: torch.Tensor):
        # RL 에이전트로부터 받은 actions 저장
        actions = actions.to(torch.float32)

        # action 범위를 [-1, 1]로 강제
        actions = torch.clamp(actions, -1.0, 1.0)

        # 로봇 위치 가져오기
        robot_q = self.robot_interface.get_joint_pos(self.robot)

        # 목표 위치 = 현재 위치 + actions (관절별 상한: 손목만 큼)
        target_q_d = actions * self.action_scale_vec
        robot_q_next = robot_q + target_q_d * self.step_dt
        usd_q_next = self.robot_interface.clip_and_convert(robot_q_next)

        self.actions = actions.clone()
        self.target_joint_pos = usd_q_next

        if torch.isnan(actions).any():
            raise RuntimeError("NaN actions")

    def _apply_action(self):
        ctrl_joint_ids = self.robot_interface.get_ctrl_joint_ids()
        self.robot.set_joint_position_target(self.target_joint_pos, joint_ids=ctrl_joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # episode time out
        self.steps = self.steps + 1
        time_out = self.steps >= self.episode_length_step - 1
        died = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        tip_pos = self.robot_interface.get_body_pos(self.robot) # (N, 2, 3)

        (
            hit_mask,
            tip_vel, prev_tip_pos,
            prev_hit_armed, hit_per_arm,
        ) = self.hit_detector.detect_hit(tip_pos=tip_pos, drum_pos=self.drum_pos)

        # 관절각, 관절 속도 (손목 각속도 계산 위해 먼저 가져옴)
        robot_pos = self.robot_interface.get_joint_pos(self.robot)
        robot_vel = self.robot_interface.get_joint_vel(self.robot)
        robot_torque = self.robot_interface.get_joint_torque(self.robot)

        # 타격 순간 드럼별 하강 속도 (양팔 중 max)
        down_vel = torch.clamp(-tip_vel[:, :, 2], 0.0)                         # (N, 2)
        hit_vel_per_drum = (hit_per_arm.float() * down_vel[:, :, None]).max(dim=1).values  # (N, M)

        # 타격 순간 드럼별 손목 각속도 (양팔 중, 그 드럼 친 팔의 손목)
        wrist_vel_abs = robot_vel[:, self.wrist_joint_ids].abs()                # (N, 2)
        hit_wrist_per_drum = (hit_per_arm.float() * wrist_vel_abs[:, :, None]).max(dim=1).values  # (N, M)

        # 타격 시간/속도/손목각속도 기록
        self.rds.set_rds_visit(steps=self.steps, hit_mask=hit_mask, hit_vel=hit_vel_per_drum, hit_wrist=hit_wrist_per_drum)

        # 결과
        rds = self.rds.get_rds()
        rds_visit = self.rds.get_rds_visit()
        rds_visit_vel = self.rds.get_rds_visit_vel()
        rds_visit_wrist = self.rds.get_rds_visit_wrist()
        success, wrong_hit, missed_target, time_error, success_impact_vel, success_wrist_ang_vel, target_done = self.hit_detector.get_result(
            hit_mask=hit_mask,
            steps=self.steps,
            rds=rds,
            rds_visit=rds_visit,
            rds_visit_vel=rds_visit_vel,
            rds_visit_wrist=rds_visit_wrist,
        )

        joint_low, joint_high = self.robot_interface.get_limit()

        wrist_pos = self.robot_interface.get_wrist_pos(self.robot)  # (N, 2, 3)
        elbow_pos = self.robot_interface.get_elbow_pos(self.robot)  # (N, 2, 3)

        reward, log_terms, rate_log_terms = self.reward_computer.compute(
            success=success,
            wrong_hit=wrong_hit,
            missed_target=missed_target,
            time_error=time_error,
            tip_pos=tip_pos,
            prev_tip_pos=prev_tip_pos,
            drum_pos=self.drum_pos,
            next_hits=self.next_hits,
            tip_vel=tip_vel,
            prev_hit_armed=prev_hit_armed,
            robot_vel=robot_vel,
            robot_torque=robot_torque,
            actions=self.actions,
            robot_pos=robot_pos,
            joint_low=joint_low,
            joint_high=joint_high,
            wrist_pos=wrist_pos,
            elbow_pos=elbow_pos,
            hit_per_arm=hit_per_arm,
            success_impact_vel=success_impact_vel,
            success_wrist_ang_vel=success_wrist_ang_vel,
            target_done=target_done,
        )

        # 가중치 업데이트
        self.reward_computer.update_difficulty_weights(success=success, missed_target=missed_target)

        # 추종오차 진단 (|목표 − 현재|, rad). 운동학 가정 검증용. USD 프레임 = 부호뒤집기라 |오차| 프레임 무관
        ctrl_ids = self.robot_interface.get_ctrl_joint_ids()
        current_usd = self.robot.data.joint_pos[:, ctrl_ids]                 # (N, 9)
        track_err = (self.target_joint_pos - current_usd).abs()             # (N, 9)
        log_terms["track_err_rad"] = track_err.mean(dim=-1)                 # 전 관절 평균
        log_terms["track_err_wrist_rad"] = track_err[:, self.wrist_joint_ids].mean(dim=-1)  # 손목만

        # 로그 출력
        self.logger.add(log_terms)
        self.logger.add_probability(rate_log_terms)
        self.logger.maybe_flush()

        # 시각화 (타격 표시, 드럼 색상 변경)
        self.visualizer.step(tip_pos=tip_pos, next_hits=self.next_hits, hit_per_arm=hit_per_arm)

        if torch.isnan(reward).any():
            raise RuntimeError("NaN reward")

        return reward
    
    def _reset_idx(self, env_ids: Sequence[int]| None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES   # type: ignore[arg-type]
        # 부모 클래스의 리셋(버퍼 초기화 등) 호출
        super()._reset_idx(env_ids)             # type: ignore[arg-type]

        # 드럼 위치 리셋 (perturbation)
        self._reset_drum(env_ids)

        # RDS 리셋
        self.rds.reset(env_ids=env_ids, score_ratio=0.0, selection_strength=0.0)  # 랜덤으로 RDS 생성해서 사용

        # 로봇 자세 리셋
        init_robot_pos = self.robot_initializer.reset_init_pos(env_ids)
        joint_pos, joint_vel = self.robot_interface.reset(robot=self.robot, env_ids=env_ids, init_robot_pos=init_robot_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # tip 리셋
        tip_pos = self.robot_interface.get_body_pos(self.robot)
        self.hit_detector.reset(env_ids=env_ids, tip_pos=tip_pos)
 
        # 텐서 변수들 리셋
        self._reset_tensors(env_ids)

        # 시각화 리셋
        self.visualizer.reset(self.drum_pos)
        
    # ============================================================
    # [Custom Functions]
    # Internal utility methods (NOT called by RL engine directly)
    # ============================================================

    """ init """
    def _init_obs(self):
        # obs 차원 계산
        M = len(self.instruments.all)
        K = self.rds.cfg.num_hits
        
        self.obs_dim_joint_pos = 9                     # ctrl 관절 수
        self.obs_dim_joint_vel = 9
        self.obs_dim_tip       = 2 * 3                 # 양손 * xyz
        self.obs_dim_drum      = M * 3                 # 드럼 * xyz
        self.obs_dim_next      = K * (M + 2)           # lookahead K개 * (one-hot M + time + valid)
        self.obs_dim_armed     = 2 * M                 # 양손 * 드럼
        
        obs_dim_total = (
            self.obs_dim_joint_pos + self.obs_dim_joint_vel
            + self.obs_dim_tip + self.obs_dim_drum
            + self.obs_dim_next + self.obs_dim_armed
        )

        if obs_dim_total != self.cfg.observation_space:
            raise RuntimeError("observation space dim mismatch.")
        
        # joint
        joint_low, joint_high = self.robot_interface.get_limit()
        self.joint_center = 0.5 * (joint_low + joint_high)
        self.joint_half_range = 0.5 * (joint_high - joint_low) + 1e-6

        self.joint_vel_scale = torch.tensor(self.cfg.joint_vel_scale, device=self.device, dtype=torch.float32)

        # drum
        drum_min = self.default_drum_pos.min(dim=0).values
        drum_max = self.default_drum_pos.max(dim=0).values

        center = 0.5 * (drum_min + drum_max)
        half_range = 0.5 * (drum_max - drum_min) + 1e-6
        
        self.drum_center = torch.cat([center], dim=0).unsqueeze(0).unsqueeze(1)          # (1, 1, 3)
        self.drum_half_range = torch.cat([half_range], dim=0).unsqueeze(0).unsqueeze(1)  # (1, 1, 3)       

    """ func (_get_observations) """
    def _normalize_and_pack_obs(self,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
            tip_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            next_hits: torch.Tensor,
            hit_armed: torch.Tensor,
        ) -> torch.Tensor:
        # joint_pos, joint_vel: (N, 9)
        # tip_pos: (N, 2, 3)
        # drum_pos: (N, M, 3)
        # next_hits: (N, K, M+2)
        # hit_armed: (N, 2, M)

        # normalize
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -10.0, 10.0)

        tip_pos_n = (tip_pos - self.drum_center) / self.drum_half_range
        tip_pos_n = torch.clamp(tip_pos_n, -1.5, 1.5)

        drum_pos_n = (drum_pos - self.drum_center) / self.drum_half_range
        drum_pos_n = torch.clamp(drum_pos_n, -1.5, 1.5)
        
        obs = torch.cat(
            [
                joint_pos_n,
                joint_vel_n,
                tip_pos_n.reshape(self.num_envs, self.obs_dim_tip),
                drum_pos_n.reshape(self.num_envs, self.obs_dim_drum),
                next_hits.reshape(self.num_envs, self.obs_dim_next),
                hit_armed.reshape(self.num_envs, self.obs_dim_armed),
            ],
            dim=-1
        )
        
        return obs

    """ func (_reset_idx) """
    def _reset_drum(self, env_ids: torch.Tensor):
        # 각 env/각 에피소드에서 악기 위치
        drum_pos = self.default_drum_pos.unsqueeze(0).repeat(len(env_ids), 1, 1)  # (M, 3) -차원 추가-> (1, M, 3) -복제-> (N, M, 3)
        drum_pos = drum_pos + self.instruments.drum_noise_scale * torch.randn_like(drum_pos)
        drum_pos[:, :, 2] = drum_pos[:, :, 2] + self.cfg.robot_waist_joint_offset_z

        self.drum_pos[env_ids] = drum_pos

    def _reset_tensors(self, env_ids: torch.Tensor):
        # 스텝 리셋
        self.steps[env_ids] = 0
