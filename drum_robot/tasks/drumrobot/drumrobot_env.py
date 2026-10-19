from __future__ import annotations
                                                                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch                                                    # pyright: ignore[reportMissingImports]
import torch.nn.functional as F                                 # pyright: ignore[reportMissingImports]
from typing import Sequence

import numpy as np                                              # pyright: ignore[reportMissingImports]
import gymnasium as gym                                         # pyright: ignore[reportMissingImports]

import isaaclab.sim as sim_utils                                # pyright: ignore[reportMissingImports]
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollection   # pyright: ignore[reportMissingImports]
from isaaclab.envs import DirectRLEnv                           # pyright: ignore[reportMissingImports]
from isaaclab.sensors import ContactSensor                      # pyright: ignore[reportMissingImports]
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane     # pyright: ignore[reportMissingImports]
from isaaclab.utils import math as math_utils                   # pyright: ignore[reportMissingImports]

from .drumrobot_cfg import DrumRobotEnvCfg
from .components.specs import EnvRuntimeSpec, Instruments, RobotSpec
from .components.robot_interface import RobotInterface
from .components.robotic_drum_score import RDSCfg, RDS
from .components.hit_detector import HitDetector, HitDetectorCfg, HitResult
from .components.reward import RewardComputer, RewardCfg
from .components.robot_initializer import RobotInitializerCfg, RobotInitializer
from .components.sticking_planner import StickingPlanner, StickingPlannerCfg
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
        self.drum_half_thickness = torch.tensor(
            [inst.thickness / 2 for inst in self.instruments.all.values()],
            device=self.device,
            dtype=torch.float32,
        )   # (M,)
        self._randomize_drum_material()        # 반발계수·마찰 랜덤화: 시작할 때 한 번
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
            hit_window_step=round(hit_detector_cfg.max_match_window_s / self.step_dt),     # 최대 매칭 범위 [스텝]
            step_dt=self.step_dt,
        )

        # robot
        self.robot_interface = RobotInterface(
            device=self.device,
            env=env_specs,
            robot=self.robot,
            env_origins=self.scene.env_origins,
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

        # 행동 -> 관절 목표 (직전 목표에 누적)
        ctrl_joint_names = self.robot_interface.get_ctrl_joint_name()
        ctrl_joint_ids = self.robot_interface.get_ctrl_joint_ids()
        self.action_scale_vec = torch.tensor(
            [self.cfg.wrist_action_scale if "wrist" in name else self.cfg.action_scale for name in ctrl_joint_names],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, 9)
        # 목표가 측정값보다 앞설 수 있는 최대 폭 = 토크 제한 / 강성 [rad] (더 앞서도 토크는 안 늘고 되돌리는 시간만 늘어남)
        self.max_lead = (
            self.robot.data.joint_effort_limits[0, ctrl_joint_ids] / self.robot.data.joint_stiffness[0, ctrl_joint_ids]
        ).unsqueeze(0)  # (1, 9)
        self.target_q = torch.zeros((N, len(ctrl_joint_names)), device=self.device)   # 누적 목표 (로봇 좌표계)

        # 실제 접촉 지점이 스틱 끝에서 떨어진 거리 (팁/몸통 구분용), 물리 스텝마다 기록
        # 센서는 마지막 물리 스텝의 접촉 지점만 주므로 _apply_action에서 앞 물리 스텝 것을 받아 둠
        robot_spec = RobotSpec()
        self.wrist_body_ids = [self.robot_interface.body_to_idx["left_wrist"], self.robot_interface.body_to_idx["right_wrist"]]
        self.stick_tip_x = torch.tensor(
            [robot_spec.tip_offset_left[0], robot_spec.tip_offset_right[0]], device=self.device
        ).view(1, 2, 1)                                                         # 손목 링크 x축(스틱 축) 위 스틱 끝 [m]
        self.contact_from_tip = torch.full((N, self.cfg.decimation, 2, M), float("nan"), device=self.device)   # (N, D, 2, M)
        self._substep = 0

        # 로봇 초기 위치 initializer
        self.robot_initializer = RobotInitializer(
            device=self.device,
            cfg=RobotInitializerCfg(),
            ctrl_joint_names=ctrl_joint_names,
        )

        # 보상 (모든 보상 항은 reward.py에서 관리)
        self.reward_computer = RewardComputer(
            device=self.device,
            cfg=RewardCfg(),
            env=env_specs,
            joint_names=ctrl_joint_names,
            num_actions=self.cfg.action_space,
        )

        # 손 배정 (스티킹 플래너): 손별 다음 음표 -> 판정(담당 손), 정렬·접근 보상, 관측
        self.planner = StickingPlanner(
            cfg=StickingPlannerCfg(step_dt=self.step_dt),
            drum_names=list(self.instruments.all.keys()),
            num_events=rds_cfg.num_hits,
            max_lookahead_step=max_lookahead_step,
            episode_length_step=self.episode_length_step,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.last_hit_step = torch.full((N, 2), -10000, dtype=torch.long, device=self.device)   # 손별 마지막 타격 스텝 (플래너 입력)
        self.stick_tip_off = torch.tensor(
            [robot_spec.tip_offset_left, robot_spec.tip_offset_right], device=self.device
        )                                                                                       # (2, 3) 손목 링크 -> 스틱 끝
        self.arm_drum = torch.zeros((N, 2), dtype=torch.long, device=self.device)   # 손별 다음 음표 드럼 [왼손, 오른손]
        self.arm_t = torch.ones((N, 2), device=self.device)                          # 손별 다음 음표까지 남은 시간 (0~1, 없으면 1)
        self.arm_valid = torch.zeros((N, 2), dtype=torch.bool, device=self.device)   # 손별 다음 음표 있음

        # 관측 벡터 및 정규화를 위한 변수 초기화
        self._init_obs()

        # 로그
        self.logger = EnvLogger(
            num_envs=self.num_envs,
            device=self.device,
            cfg=LoggerCfg(
                interval=2000,
                sample_env_id=0,
                groups=(
                    ("rew_total",),                                                                 # 총 보상
                    ("match_rate", "matched", "rew_hit", "rew_pos", "rew_height", "wrist_lift_deg",  # 성공
                     "within_20ms", "mean_abs_err_ms", "tip_ratio"),
                    ("rew_align", "rew_prox", "align_waist_err_deg", "align_yaw_err_deg", "prox_dist_cm"),  # 유도
                    ("wrong", "rew_wrong", "wrong_per_hit", "wrong_arm", "bad", "rew_bad"),        # 잘못 침
                    ("missed", "rew_miss"),                                                         # 놓침
                ),                                                                                  # 나머지(hits, 정규화 등)는 맨 뒤
            ),
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
        self.drums = RigidObjectCollection(self.cfg.drums_cfg)     # 드럼 헤드 강체 (고정)
        self.contact_L = ContactSensor(self.cfg.contact_L_cfg)     # 왼쪽 스틱 접촉 센서
        self.contact_R = ContactSensor(self.cfg.contact_R_cfg)     # 오른쪽 스틱 접촉 센서
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.rigid_object_collections["drums"] = self.drums
        self.scene.sensors["contact_L"] = self.contact_L    # 등록하면 물리 스텝마다 갱신, 리셋도 자동
        self.scene.sensors["contact_R"] = self.contact_R
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 광원 추가
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # 로봇의 관절 위치
        joint_pos = self.robot_interface.get_joint_pos(self.robot)

        # 로봇의 관절 속도
        joint_vel = self.robot_interface.get_joint_vel(self.robot)

        # 목표 앞섬 (누적 목표 - 측정 각도)
        lead = self.target_q - joint_pos

        # 위치
        tip_pos = self.robot_interface.get_body_pos(self.robot)     # (N, 2, 3)
        drum_pos = self.drum_pos

        # 다음 타격
        self.next_hits = self.rds.get_next_hits(self.steps)

        # 손별 다음 음표 (플래너 배정): 드럼 one-hot (없으면 0) + 남은 시간 (없으면 1)
        M = drum_pos.shape[1]
        arm_next = torch.cat(
            [F.one_hot(self.arm_drum, M).float() * self.arm_valid[..., None], self.arm_t[..., None]], dim=-1
        )   # (N, 2, M+1)

        obs = self._normalize_and_pack_obs(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            lead=lead,
            tip_pos=tip_pos,
            drum_pos=drum_pos,
            next_hits=self.next_hits,
            arm_next=arm_next,
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

        # 목표 위치 = 직전 목표 + actions (행동 0이면 목표 유지 -> 중력을 버팀)
        target_q = self.target_q + actions * self.action_scale_vec * self.step_dt
        target_q = torch.max(torch.min(target_q, robot_q + self.max_lead), robot_q - self.max_lead)  # 측정값 ± 최대 앞섬
        low, high = self.robot_interface.get_limit()
        self.target_q = torch.max(torch.min(target_q, high), low)                                      # 관절 제한
        usd_q_next = self.robot_interface.clip_and_convert(self.target_q)

        self.actions = actions.clone()
        self.target_joint_pos = usd_q_next

        # 이번 정책 스텝의 물리 스텝별 접촉 지점 기록 초기화
        self._substep = 0
        self.contact_from_tip.fill_(float("nan"))

        if torch.isnan(actions).any():
            raise RuntimeError("NaN actions")

    def _apply_action(self):
        # 물리 스텝마다 호출됨: 두 번째 물리 스텝부터는 직전 물리 스텝의 접촉 지점을 받아 둠
        if self._substep > 0:
            self.contact_from_tip[:, self._substep - 1] = self._get_contact_from_tip()
        self._substep += 1

        ctrl_joint_ids = self.robot_interface.get_ctrl_joint_ids()
        self.robot.set_joint_position_target(self.target_joint_pos, joint_ids=ctrl_joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # episode time out
        self.steps = self.steps + 1
        time_out = self.steps >= self.episode_length_step - 1
        died = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        # 타격 판정 (정상 타격 / 잘못된 접촉, 팁/몸통 기록) + 악보 매칭
        contact_force = self._get_drum_contact_force()              # (N, D, 2, M, 3)
        self.contact_from_tip[:, -1] = self._get_contact_from_tip() # 마지막 물리 스텝 접촉 지점
        tip_pos = self.robot_interface.get_body_pos(self.robot)     # (N, 2, 3)
        result = self.hit_detector.step(
            contact_force=contact_force,
            contact_from_tip=self.contact_from_tip,
            steps=self.steps,
        )

        # 손 배정: 이번 스텝 타격까지 반영한 뒤 플래너 실행 -> 손별 다음 음표, 판정의 목표별 담당 손
        next_hits = self.rds.get_next_hits(self.steps)
        hit_arm = result.hit_per_arm.any(dim=-1)                                                # (N, 2)
        self.last_hit_step = torch.where(hit_arm, self.steps[:, None].expand(-1, 2), self.last_hit_step)
        plan = self.planner.schedule(self.steps, next_hits, tip_pos, self._get_tip_velocity(), self.drum_pos, self.last_hit_step)
        ev_arm, ev_off = plan[5], plan[6]
        self.hit_detector.set_target_arm(self.steps, ev_arm, ev_off)                           # 다음 스텝부터 판정에 사용
        self.arm_drum, self.arm_t, self.arm_valid = self._arm_next_notes(ev_arm, next_hits)

        # 보상
        joint_pos = self.robot_interface.get_joint_pos(self.robot)
        reward, reward_log = self.reward_computer.compute(
            result=result,
            joint_pos=joint_pos,
            tip_pos=tip_pos,
            drum_pos=self.drum_pos,
            arm_drum=self.arm_drum,
            arm_t=self.arm_t,
            arm_valid=self.arm_valid,
            actions=self.actions,
            steps=self.steps,
            global_step=self.common_step_counter,
        )

        # 로그 출력
        self._log_hit_result(result, extra_terms=reward_log)
        self.logger.maybe_flush()

        # 시각화 (타격 표시, 드럼 색상 변경)
        self.visualizer.step(tip_pos=tip_pos, next_hits=self.next_hits, hit_per_arm=result.hit_per_arm)

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
        init_robot_pos, init_drum = self.robot_initializer.reset_init_pos(env_ids)
        joint_pos, joint_vel = self.robot_interface.reset(robot=self.robot, env_ids=env_ids, init_robot_pos=init_robot_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 누적 목표 = 시작 자세 (관절 제한 안)
        low, high = self.robot_interface.get_limit()
        self.target_q[env_ids] = torch.max(torch.min(init_robot_pos, high), low)

        # 타격 감지 리셋 (rds.reset 이후: 새 악보로 매칭 범위 계산)
        self.hit_detector.reset(env_ids=env_ids, rds=self.rds.get_rds())

        # 보상 리셋 (드럼 위치 리셋 이후: 드럼 쌍별 목표 허리 표 계산)
        self.reward_computer.reset(env_ids=env_ids, drum_pos=self.drum_pos, init_drum=init_drum)

        # 손 배정 리셋
        self.planner.reset(env_ids)                     # 확정된 배정 지움
        self.last_hit_step[env_ids] = -10000
        self.arm_valid[env_ids] = False                 # 리셋 직후 첫 관측은 "배정 없음"
        self.arm_t[env_ids] = 1.0

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
        self.obs_dim_lead      = 9                     # 목표 앞섬 (누적 목표 - 측정 각도)
        self.obs_dim_tip       = 2 * 3                 # 양손 * xyz
        self.obs_dim_drum      = M * 3                 # 드럼 * xyz
        self.obs_dim_next      = K * (M + 2)           # lookahead K개 * (one-hot M + time + valid)
        self.obs_dim_arm       = 2 * (M + 1)           # 손별 다음 음표 (플래너 배정): 드럼 one-hot M + 남은 시간

        obs_dim_total = (
            self.obs_dim_joint_pos + self.obs_dim_joint_vel + self.obs_dim_lead
            + self.obs_dim_tip + self.obs_dim_drum
            + self.obs_dim_next + self.obs_dim_arm
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

    def _randomize_drum_material(self):
        # 환경마다, 드럼마다 반발계수와 마찰계수를 범위 안에서 랜덤으로 설정 (시작할 때 한 번)
        # Isaac Lab 기본 randomize_rigid_body_material은 RigidObjectCollection 미지원이라 직접 구현
        view = self.drums.root_physx_view
        N, M = self.num_envs, self.drums.num_objects
        insts = list(self.instruments.all.values())

        def sample(ranges: list) -> torch.Tensor:
            # 드럼별 (하한, 상한) -> (N, M) 랜덤 값, 0.01 단위 (서로 다른 재질 수 제한, PhysX 약 64000개 한도)
            lo = torch.tensor([r[0] for r in ranges])   # (M,) CPU
            hi = torch.tensor([r[1] for r in ranges])
            return torch.round((lo + (hi - lo) * torch.rand(N, M)) * 100) / 100

        e = sample([inst.restitution_range for inst in insts])     # (N, M) 반발계수
        mu = sample([inst.friction_range for inst in insts])       # (N, M) 마찰계수 (정지·운동 같은 값)

        # 재질 값 = [정지마찰, 동마찰, 반발], view 순서는 드럼 우선 (드럼번호 * N + env번호)
        mats = view.get_material_properties().reshape(M, N, -1, 3)    # CPU 텐서
        mats[..., 0] = mu.T.unsqueeze(-1)
        mats[..., 1] = mu.T.unsqueeze(-1)
        mats[..., 2] = e.T.unsqueeze(-1)
        view.set_material_properties(mats.reshape(M * N, -1, 3), torch.arange(M * N))

        # (N, M) 로그/critic 용 (actor 관측에는 넣지 않음)
        self.drum_restitution = e.to(self.device)
        self.drum_friction = mu.to(self.device)

    """ func (_get_observations) """
    def _normalize_and_pack_obs(self,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
            lead: torch.Tensor,
            tip_pos: torch.Tensor,
            drum_pos: torch.Tensor,
            next_hits: torch.Tensor,
            arm_next: torch.Tensor,
        ) -> torch.Tensor:
        # joint_pos, joint_vel, lead: (N, 9)
        # tip_pos: (N, 2, 3)
        # drum_pos: (N, M, 3)
        # next_hits: (N, K, M+2)
        # arm_next: (N, 2, M+1) 손별 다음 음표 (이미 0~1)

        # normalize
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -10.0, 10.0)

        lead_n = lead / self.max_lead                   # -1 ~ 1 (최대 앞섬으로 나눔)

        tip_pos_n = (tip_pos - self.drum_center) / self.drum_half_range
        tip_pos_n = torch.clamp(tip_pos_n, -1.5, 1.5)

        drum_pos_n = (drum_pos - self.drum_center) / self.drum_half_range
        drum_pos_n = torch.clamp(drum_pos_n, -1.5, 1.5)
        
        obs = torch.cat(
            [
                joint_pos_n,
                joint_vel_n,
                lead_n,
                tip_pos_n.reshape(self.num_envs, self.obs_dim_tip),
                drum_pos_n.reshape(self.num_envs, self.obs_dim_drum),
                next_hits.reshape(self.num_envs, self.obs_dim_next),
                arm_next.reshape(self.num_envs, self.obs_dim_arm),
            ],
            dim=-1
        )
        
        return obs

    """ func (_get_rewards) """
    def _get_drum_contact_force(self) -> torch.Tensor:
        # 스틱이 드럼에서 받은 힘 벡터 (N, D, 2, M, 3) [N], 2 = [왼손, 오른손]
        # D = 정책 스텝 안의 물리 스텝 수 (센서 history_length), 시간 순 (0이 가장 이른 물리 스텝)
        # contact_offset 안에 들어와도 실제로 닿기 전에는 힘이 0이므로 힘으로 판정
        # 윗면을 위에서 치면 z가 + (드럼이 스틱을 위로 밀어냄), 밑면이면 -
        fL = self.contact_L.data.force_matrix_w_history[:, :, 0]   # (N, D, M, 3), 0이 가장 최근
        fR = self.contact_R.data.force_matrix_w_history[:, :, 0]
        return torch.stack([fL, fR], dim=2).flip(1)

    def _get_contact_from_tip(self) -> torch.Tensor:
        # 마지막 물리 스텝의 실제 접촉 지점이 스틱 끝에서 떨어진 거리 (N, 2, M) [m], 접촉 없으면 NaN
        # 접촉 지점을 그 순간의 손목 링크 좌표로 바꾸면 x = 스틱 축 위치 -> 스틱 끝(tip_offset x)까지 거리
        cp = torch.stack([self.contact_L.data.contact_pos_w[:, 0], self.contact_R.data.contact_pos_w[:, 0]], dim=1)   # (N, 2, M, 3) 월드
        pos = self.robot.data.body_pos_w[:, self.wrist_body_ids]                 # (N, 2, 3)
        quat = self.robot.data.body_quat_w[:, self.wrist_body_ids]               # (N, 2, 4)
        M = cp.shape[2]
        local = math_utils.quat_apply_inverse(quat[:, :, None].expand(-1, -1, M, -1), cp - pos[:, :, None])   # (N, 2, M, 3)
        return self.stick_tip_x - local[..., 0]

    def _get_tip_velocity(self) -> torch.Tensor:
        # 팁 속도 (N, 2, 3) = 손목 링크 속도 + 각속도 × (손목 -> 팁)  (플래너 입력, planner_test와 같은 계산)
        d = self.robot.data
        v = d.body_lin_vel_w[:, self.wrist_body_ids]
        w = d.body_ang_vel_w[:, self.wrist_body_ids]
        r = math_utils.quat_apply(d.body_quat_w[:, self.wrist_body_ids], self.stick_tip_off.expand_as(v))
        return v + torch.cross(w, r, dim=-1)

    def _arm_next_notes(self, ev_arm: torch.Tensor, next_hits: torch.Tensor):
        # 플래너 배정 (이벤트별 드럼 담당 손) -> 손별 가장 이른 음표: 드럼 (N, 2), 남은 시간 (N, 2), 있음 (N, 2)
        N, K, M = ev_arm.shape
        drum = torch.zeros((N, 2), dtype=torch.long, device=self.device)
        t = torch.ones((N, 2), device=self.device)
        valid = torch.zeros((N, 2), dtype=torch.bool, device=self.device)
        for a in (0, 1):
            for k in reversed(range(K)):                    # 뒤에서부터 덮어써서 가장 이른 음표가 남음
                mask = ev_arm[:, k] == a                    # (N, M) 이벤트 k에서 손 a가 맡은 드럼
                has = mask.any(dim=-1)
                drum[:, a] = torch.where(has, mask.int().argmax(dim=-1), drum[:, a])
                t[:, a] = torch.where(has, next_hits[:, k, M], t[:, a])
                valid[:, a] |= has
        return drum, t, valid

    def _log_hit_result(self, r: HitResult, extra_terms: dict | None = None):
        # 타격·매칭 결과 로그 (학습 진행 확인용) + 보상 항 로그(extra_terms)
        # logger.add는 스텝마다 한 번만 호출해야 평균이 맞아서 한 번에 넘김
        n_hit = r.hit_per_arm.any(dim=1).float().sum(-1)                   # (N,) 이번 스텝 타격 수
        n_matched = r.matched.float().sum(-1)
        n_wrong = r.wrong_hit.float().sum(-1)
        n_missed = r.missed.float().sum(-1)
        n_bad = r.bad_contact.float().sum(-1)                               # 잘못된 접촉 (밑면·옆면)
        n_wrong_arm = r.wrong_arm.float().sum(-1)                           # 잘못 침 중 담당이 아닌 손으로 친 것
        n_arm_hit = r.hit_per_arm.float().sum(dim=(1, 2))                   # 손별로 센 정상 타격 수
        n_tip = r.tip_per_arm.float().sum(dim=(1, 2))                       # 그중 팁으로 친 수
        abs_err_ms = (r.time_error.abs() * 1000.0 * r.matched).sum(-1)
        n_within = ((r.time_error.abs() <= 0.02) & r.matched).float().sum(-1)

        terms = {"hits": n_hit, "matched": n_matched, "wrong": n_wrong, "wrong_arm": n_wrong_arm, "missed": n_missed, "bad": n_bad}
        extra_terms = extra_terms or {}
        terms.update({k: v for k, v in extra_terms.items() if v.ndim == 1})       # 값 항 (N,)
        ratios = {k: v for k, v in extra_terms.items() if v.ndim == 2}            # 비율 항 (N, 2) [분자, 분모]
        self.logger.add(terms)
        self.logger.add_probability({
            "match_rate": torch.stack([n_matched, n_matched + n_missed], dim=-1),   # 판정 끝난 목표 중 맞힌 비율
            "wrong_per_hit": torch.stack([n_wrong, n_hit], dim=-1),               # 타격 중 잘못 친 비율
            "within_20ms": torch.stack([n_within, n_matched], dim=-1),            # 맞힌 것 중 |오차| <= 20ms 비율
            "mean_abs_err_ms": torch.stack([abs_err_ms, n_matched], dim=-1),      # 평균 |시간 오차| [ms]
            "tip_ratio": torch.stack([n_tip, n_arm_hit], dim=-1),                 # 정상 타격 중 팁(끝 반구)으로 친 비율 (몸통 타격 확인용)
            **ratios,
        })

    """ func (_reset_idx) """
    def _reset_drum(self, env_ids: torch.Tensor):
        # 각 env/각 에피소드에서 악기 위치
        drum_pos = self.default_drum_pos.unsqueeze(0).repeat(len(env_ids), 1, 1)  # (M, 3) -차원 추가-> (1, M, 3) -복제-> (N, M, 3)
        drum_pos = drum_pos + self.instruments.drum_noise_scale * torch.randn_like(drum_pos)
        drum_pos[:, :, 2] = drum_pos[:, :, 2] + self.cfg.robot_waist_joint_offset_z

        self.drum_pos[env_ids] = drum_pos

        # 실제 드럼 강체 위치 쓰기
        pose = torch.zeros((len(env_ids), drum_pos.shape[1], 7), device=self.device, dtype=torch.float32)
        pose[..., :3] = drum_pos + self.scene.env_origins[env_ids].unsqueeze(1)    # env 좌표 -> 월드 좌표
        pose[..., 2] -= self.drum_half_thickness                                    # 윗면 -> 원판 중심
        pose[..., 3] = 1.0                                                          # 회전 없음 (w, x, y, z)
        self.drums.write_object_pose_to_sim(pose, env_ids=env_ids)

    def _reset_tensors(self, env_ids: torch.Tensor):
        # 스텝 리셋
        self.steps[env_ids] = 0
