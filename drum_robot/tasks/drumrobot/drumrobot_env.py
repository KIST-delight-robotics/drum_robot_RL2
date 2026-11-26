from __future__ import annotations
                                                                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch                                                    # pyright: ignore[reportMissingImports]
import torch.nn.functional as F                                 # pyright: ignore[reportMissingImports]
from typing import Sequence
import math

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
from .components.hit_detector import HitDetector, HitDetectorCfg
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
        # self.pos_res = torch.tensor([self.cfg.wrist_pos_res if "wrist" in n else self.cfg.obs_pos_res for n in ctrl_joint_names],
        #                             device=self.device).unsqueeze(0)   # (1, 9)
        motor_ids = [self.robot_interface.robot_spec.joint_name_to_ik_result_idx[n] for n in ctrl_joint_names]
        self.real_sign = torch.tensor([self.cfg.real_dir_sign[m] for m in motor_ids], device=self.device).unsqueeze(0)                 # (1, 9) 모터각 = (관절각 - init) × sign
        self.real_init = torch.tensor([math.radians(self.cfg.real_init_deg[m]) for m in motor_ids], device=self.device).unsqueeze(0)   # (1, 9) [rad]
        self.res_mask = torch.tensor(["wrist" not in n for n in ctrl_joint_names], device=self.device).unsqueeze(0)                  # (1, 9) TMotor만 True
        self.wrist_ids = [i for i, n in enumerate(ctrl_joint_names) if "wrist" in n]
        self.wrist_prev_q = torch.zeros((N, len(self.wrist_ids)), device=self.device)      # 손목 속도 유한차분용 직전 위치
        self.wrist_seeded = torch.zeros((N,), dtype=torch.bool, device=self.device)        # 리셋 뒤 첫 관측은 손목 속도 0 (실기 obs_builder)
        # 목표가 측정값보다 앞설 수 있는 최대 폭 = 토크 제한 / 강성 [rad] (더 앞서도 토크는 안 늘고 되돌리는 시간만 늘어남)
        self.max_lead = (
            self.robot.data.joint_effort_limits[0, ctrl_joint_ids] / self.robot.data.joint_stiffness[0, ctrl_joint_ids]
        ).unsqueeze(0)  # (1, 9)
        self.target_q = torch.zeros((N, len(ctrl_joint_names)), device=self.device)   # 누적 목표 (로봇 좌표계)
        self.actions = torch.zeros((N, len(ctrl_joint_names)), device=self.device)    # 직전 행동 (관측용, 첫 관측이 행동보다 먼저라 미리 만듦)
        # 허리만 학습: 허리 외 관절(어깨 yaw 포함)은 행동 0으로 고정 (= 목표 유지, 시작 자세 그대로)
        # self.frozen_ids = [i for i, name in enumerate(ctrl_joint_names)
        #                    if name not in ("waist_joint",)]
        # self.frozen_ids = [i for i, name in enumerate(ctrl_joint_names)
        #                    if name not in ("waist_joint", "right_shoulder_1", "left_shoulder_1")]
        self.frozen_ids = []                                                # 고정 관절 없음: 9개 관절 모두 학습 (타격 관절 포함)
        self.waist_ctrl_id = ctrl_joint_names.index("waist_joint")
        self.yaw_ctrl_ids = [ctrl_joint_names.index("right_shoulder_1"), ctrl_joint_names.index("left_shoulder_1")]   # 어깨 yaw [오른, 왼]
        self.waist_usd_id = ctrl_joint_ids[self.waist_ctrl_id]            # 외란 넣을 관절 (articulation 번호)
        self.waist_dist = torch.zeros((N, 1), device=self.device)           # 지금 걸린 허리 외란 [N m]
        # self.yaw_usd_ids = [ctrl_joint_ids[i] for i in self.yaw_ctrl_ids]  # 외란 넣을 어깨 yaw 관절 (articulation 번호) [오른, 왼]
        # self.yaw_dist = torch.zeros((N, 2), device=self.device)             # 지금 걸린 어깨 yaw 외란 [N m] [오른, 왼]
        self.arm_usd_ids = [ctrl_joint_ids[i] for i, n in enumerate(ctrl_joint_names) if "shoulder" in n or "elbow" in n]   # 외란 넣을 팔 관절 6개 (어깨 yaw·pitch·팔꿈치, articulation 번호)
        self.arm_dist = torch.zeros((N, 6), device=self.device)             # 지금 걸린 팔 관절 외란 [N m]
        self.wrist_usd_ids = [ctrl_joint_ids[i] for i in self.wrist_ids]   # 외란 넣을 손목 관절 (articulation 번호) [왼, 오른]
        self.wrist_dist = torch.zeros((N, 2), device=self.device)           # 지금 걸린 손목 외란 [N m] [왼, 오른]

        # 제어 지연 버퍼 (물리 스텝 단위, 순환 인덱스). max = 0이면 미사용
        #   _apply_action이 물리 스텝마다 호출되므로 SIM_DT(7.5 ms) 해상도로 지연. env마다 [min, max]에서 뽑고 리셋마다 다시 뽑음
        self.delay_min = int(self.cfg.action_delay_steps_min)
        self.delay_max = int(self.cfg.action_delay_steps_max)
        self._env_arange = torch.arange(N, device=self.device)
        self.target_buf = torch.zeros((self.delay_max + 1, N, len(ctrl_joint_names)), device=self.device)   # 지난 목표들
        self._delay_ptr = 0
        self.env_delay = torch.randint(self.delay_min, self.delay_max + 1, (N,), device=self.device)       # env별 지연 D

        # 실제 접촉 지점이 스틱 끝에서 떨어진 거리 (팁/몸통 구분용), 물리 스텝마다 기록
        # 센서는 마지막 물리 스텝의 접촉 지점만 주므로 _apply_action에서 앞 물리 스텝 것을 받아 둠
        robot_spec = RobotSpec()
        self.wrist_body_ids = [self.robot_interface.body_to_idx["left_wrist"], self.robot_interface.body_to_idx["right_wrist"]]
        self.shoulder_body_ids = [self.robot_interface.body_to_idx["left_shoulder_2"], self.robot_interface.body_to_idx["right_shoulder_2"]]   # 어깨 pitch 관절 = 위팔 시작점 (허리 목표의 거리 조건)
        self.stick_tip_x = torch.tensor(
            [robot_spec.tip_offset_left[0], robot_spec.tip_offset_right[0]], device=self.device
        ).view(1, 2, 1)                                                         # 손목 링크 x축(스틱 축) 위 스틱 끝 [m]
        self.contact_from_tip = torch.full((N, self.cfg.decimation, 2, M), float("nan"), device=self.device)   # (N, D, 2, M)
        self.tip_vz_pre = torch.zeros((N, self.cfg.decimation, 2), device=self.device)   # (N, D, 2) 물리 스텝별 그 물리 스텝 직전 스틱 끝 수직 속도 [m/s] (내려치는 속도)
        self.wrist_vel_pre = torch.zeros((N, self.cfg.decimation, 2), device=self.device)   # (N, D, 2) 물리 스텝별 그 물리 스텝 직전 손목 각속도 [rad/s] [왼손, 오른손], 내려치는 방향 +
        self._substep = 0

        # 로봇 초기 위치 initializer
        self.robot_initializer = RobotInitializer(
            device=self.device,
            cfg=RobotInitializerCfg(),
            ctrl_joint_names=ctrl_joint_names,
        )

        # 보상 (허리·어깨 정렬, reward.py): 허리 목표의 거리 조건에 어깨 높이가 필요 (허리·yaw 회전과 무관한 상수)
        shoulder_z = (self.robot.data.body_pos_w[0, self.shoulder_body_ids, 2] - self.scene.env_origins[0, 2]).mean().item()
        print(f"[DrumRobotEnv] 어깨 높이 z_S = {shoulder_z:.3f} m (env 좌표)")
        self.reward_computer = RewardComputer(
            device=self.device,
            cfg=RewardCfg(),
            env=env_specs,
            joint_names=ctrl_joint_names,
            shoulder_z=shoulder_z,
        )

        # 손 배정 (스티킹 플래너): 손별 다음 음표 -> 가상 타격의 담당 손, 정렬 보상, 관측
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
        self.arm_grade = torch.zeros((N, 2), device=self.device)                     # 손별 다음 음표 악보 세기 1~7 (없으면 0)
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
                    ("rew_align", "align_waist_err_deg", "align_yaw_err_deg"),                     # 정렬
                    # ("rew_act_rate", "rew_waist_act"),                                            # 행동 변화 (모든 관절), 허리 행동 크기  (허리 행동 크기 비용 뺌)
                    ("rew_act_rate",),                                                              # 행동 변화 (모든 관절)
                    # ("rew_vel", "vel_gate"),                                                        # 허리 속도 (목표 근처), 문 평균
                    ("rew_vel",),
                ),
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
        self.self_contacts = [ContactSensor(c) for c in (self.cfg.self_contact_upper_cfg, self.cfg.self_contact_fore_cfg, self.cfg.self_contact_wrist_cfg)]   # 자기 충돌 센서 [왼 상완, 왼 전완, 왼 손목] (오른팔 링크 3개만 받음)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.rigid_object_collections["drums"] = self.drums
        self.scene.sensors["contact_L"] = self.contact_L    # 등록하면 물리 스텝마다 갱신, 리셋도 자동
        self.scene.sensors["contact_R"] = self.contact_R
        for i, s in enumerate(self.self_contacts):
            self.scene.sensors[f"self_contact_{i}"] = s
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 광원 추가
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # 로봇의 관절 위치
        joint_pos = self.robot_interface.get_joint_pos(self.robot)
        if self.cfg.obs_pos_noise_std > 0:
            # noise = torch.zeros_like(joint_pos)
            # noise[:, self.waist_ctrl_id] = self.cfg.obs_pos_noise_std * torch.randn(self.num_envs, device=self.device)
            # noise[:, self.yaw_ctrl_ids] = self.cfg.obs_pos_noise_std * torch.randn(self.num_envs, 2, device=self.device)
            noise = self.cfg.obs_pos_noise_std * torch.randn_like(joint_pos) * self.res_mask                     # TMotor 관절 전부 (허리·어깨 yaw·pitch·팔꿈치)
            noise[:, self.wrist_ids] = self.cfg.obs_wrist_pos_noise_std * torch.randn(self.num_envs, len(self.wrist_ids), device=self.device)   # 손목 (Maxon)
            joint_pos = joint_pos + noise

        # 로봇의 관절 속도
        joint_vel = self.robot_interface.get_joint_vel(self.robot)
        if self.cfg.obs_vel_noise_std > 0:
            # noise = torch.zeros_like(joint_vel)
            # noise[:, self.waist_ctrl_id] = self.cfg.obs_vel_noise_std * torch.randn(self.num_envs, device=self.device)
            # noise[:, self.yaw_ctrl_ids] = self.cfg.obs_vel_noise_std * torch.randn(self.num_envs, 2, device=self.device)
            noise = self.cfg.obs_vel_noise_std * torch.randn_like(joint_vel) * self.res_mask                     # TMotor 관절 전부 (손목 속도는 아래에서 위치 차분으로 덮어씀)
            joint_vel = joint_vel + noise

        # if self.cfg.obs_pos_res > 0:
        #     joint_pos = torch.round(joint_pos / self.pos_res) * self.pos_res          # 실기 위치 엔코더 칸
        # if self.cfg.obs_vel_res > 0:
        #     joint_vel = torch.round(joint_vel / self.cfg.obs_vel_res) * self.cfg.obs_vel_res   # 실기 속도 12비트 칸
        joint_pos = self._real_feedback_pos(joint_pos)
        if self.cfg.obs_vel_res > 0:
            v = joint_vel * self.real_sign                                                  # 모터 좌표 속도
            v = torch.floor((v + self.cfg.mit_v_limit) / self.cfg.obs_vel_res) * self.cfg.obs_vel_res - self.cfg.mit_v_limit   # MIT 12비트 칸 (−50 + k·칸)
            joint_vel = torch.where(self.res_mask, v * self.real_sign, joint_vel)
        if self.cfg.wrist_pos_res > 0:
            wq = joint_pos[:, self.wrist_ids]
            wv = torch.where(self.wrist_seeded[:, None], (wq - self.wrist_prev_q) / self.step_dt, torch.zeros_like(wq))   # 실기: 손목 위치 유한차분 (정책 주기)
            joint_vel = joint_vel.clone()
            joint_vel[:, self.wrist_ids] = wv
            self.wrist_prev_q = wq.clone()
            self.wrist_seeded[:] = True

        # 목표 앞섬 (누적 목표 - 측정 각도)
        lead = self.target_q - joint_pos

        # 위치
        tip_pos = self.robot_interface.get_body_pos(self.robot)     # (N, 2, 3)
        drum_pos = self.drum_pos

        # 다음 타격 (시각화용, 관측에는 안 넣음)
        self.next_hits = self.rds.get_next_hits(self.steps)

        # 손별 목표 드럼 (정렬 항과 같음: 다음 음표 드럼, 없으면 그 손이 직전에 친 드럼) one-hot + 남은 시간 (없으면 1)
        M = drum_pos.shape[1]
        al = self.reward_computer.alignment
        arm_target = torch.stack([al.target_l, al.target_r], dim=1)                 # (N, 2) [왼손, 오른손]
        arm_next = torch.cat([F.one_hot(arm_target, M).float(), self.arm_t[..., None]], dim=-1)   # (N, 2, M+1)
        # arm_next = torch.cat([F.one_hot(arm_target, M).float(), self.arm_grade[..., None] / 7.0, self.arm_t[..., None]], dim=-1)   # (N, 2, M+2) 남은 시간은 마지막 칸 그대로
        # align_err = al.signed_errors(joint_pos, drum_pos) / math.pi                 # (N, 3) 정렬 오차 [허리, 오른 yaw, 왼 yaw], -1 ~ 1  (θ0*로 바꿈)
        j0 = al.j_waist
        # waist_des = ((al.target_waist - self.joint_center[0, j0]) / self.joint_half_range[0, j0])[:, None]   # (N, 1) 목표 허리 θ0*, 관절 각도와 같은 정규화, -1 ~ 1
        waist_des = ((al.ref_waist - self.joint_center[0, j0]) / self.joint_half_range[0, j0])[:, None]   # θ0_d
        waist_vel_des = al.ref_vel[:, None]                                         # (N, 1) θ̇0_d [rad/s] (정규화는 pack 에서 관절 속도와 같게)
        # yaw_err = al.signed_errors(joint_pos, drum_pos)[:, 1:]                      # (N, 2) θ* − q [오른 yaw, 왼 yaw] (현재 허리 기준, 보상과 같은 목표)
        # yaw_des = (joint_pos[:, self.yaw_ctrl_ids] + yaw_err - self.joint_center[0, self.yaw_ctrl_ids]) / self.joint_half_range[0, self.yaw_ctrl_ids]   # (N, 2) 목표 yaw θ1*, θ2*, 관절 각도와 같은 정규화
        yaw_des = (al.ref_yaw - self.joint_center[0, self.yaw_ctrl_ids]) / self.joint_half_range[0, self.yaw_ctrl_ids]   # (N, 2) θ1_d, θ2_d, 관절 각도와 같은 정규화
        yaw_vel_des = al.ref_yaw_vel                                                # (N, 2) θ̇1_d, θ̇2_d [rad/s] (정규화는 pack 에서 관절 속도와 같게)
        # waist_phase = (al.waist_s / al.waist_D).clamp(max=1.0)[:, None]          # (N, 1) θ0* 바뀐 뒤 진행률 s/D, 0 ~ 1 (속도 벌은 vel_free_ratio부터)
        # waist_phase = joint_pos[:, :0]                                              # (N, 0) 진행률 칸 뺌

        obs = self._normalize_and_pack_obs(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            lead=lead,
            tip_pos=tip_pos,
            drum_pos=drum_pos,
            # next_hits=self.next_hits,     # 다음 타격 관측 뺌
            arm_next=arm_next,
            # align_err=align_err,          # θ0*로 바꿈
            waist_des=waist_des,
            waist_vel_des=waist_vel_des,
            yaw_des=yaw_des,
            yaw_vel_des=yaw_vel_des,
            # waist_phase=waist_phase,
            last_action=self.actions,
            )
        
        if torch.isnan(obs).any():
            raise RuntimeError("NaN obs")
        
        return {"policy": obs}

    def _real_feedback_pos(self, q: torch.Tensor) -> torch.Tensor:
        # 실기 피드백 위치: 관절각 -> 모터각 -> 엔코더 칸 버림 -> 관절각 (TMotor: MIT 16비트 −12.5 + k·칸, 손목: Maxon 카운트 k·칸)
        m = (q - self.real_init) * self.real_sign
        if self.cfg.obs_pos_res > 0:
            mt = torch.floor((m + self.cfg.mit_p_limit) / self.cfg.obs_pos_res) * self.cfg.obs_pos_res - self.cfg.mit_p_limit
            m = torch.where(self.res_mask, mt, m)
        if self.cfg.wrist_pos_res > 0:
            mw = torch.floor(m / self.cfg.wrist_pos_res) * self.cfg.wrist_pos_res
            m = torch.where(self.res_mask, m, mw)
        return m * self.real_sign + self.real_init

    def _pre_physics_step(self, actions: torch.Tensor):
        # RL 에이전트로부터 받은 actions 저장
        actions = actions.to(torch.float32)

        # action 범위를 [-1, 1]로 강제
        actions = torch.clamp(actions, -1.0, 1.0)

        # 허리만 학습: 허리 외 관절은 행동 0 (= 목표 유지, 시작 자세 그대로)
        actions[:, self.frozen_ids] = 0.0

        # 로봇 위치 가져오기
        # robot_q = self.robot_interface.get_joint_pos(self.robot)
        robot_q = self._real_feedback_pos(self.robot_interface.get_joint_pos(self.robot))   # 실기 max_lead clamp 기준 = 피드백 값

        # 목표 위치 = 직전 목표 + actions (행동 0이면 목표 유지 -> 중력을 버팀)
        target_q = self.target_q + actions * self.action_scale_vec * self.step_dt
        target_q = torch.max(torch.min(target_q, robot_q + self.max_lead), robot_q - self.max_lead)  # 측정값 ± 최대 앞섬
        low, high = self.robot_interface.get_limit()
        self.target_q = torch.max(torch.min(target_q, high), low)                                      # 관절 제한
        # usd_q_next = self.robot_interface.clip_and_convert(self.target_q)
        q_cmd = self.target_q
        if self.cfg.cmd_pos_res > 0:
            m = (self.target_q - self.real_init) * self.real_sign                                       # 모터각
            m = torch.floor((m + self.cfg.mit_p_limit) / self.cfg.cmd_pos_res) * self.cfg.cmd_pos_res - self.cfg.mit_p_limit   # MIT 16비트 버림 (floatToUint)
            q_cmd = torch.where(self.res_mask, m * self.real_sign + self.real_init, self.target_q)     # 손목(CST)은 그대로
        usd_q_next = self.robot_interface.clip_and_convert(q_cmd)

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
        self.tip_vz_pre[:, self._substep] = self._get_tip_velocity()[..., 2]   # 이번 물리 스텝 직전 스틱 끝 수직 속도 (로봇 상태 = 직전 물리 스텝 끝)
        self.wrist_vel_pre[:, self._substep] = -self.robot_interface.get_joint_vel(self.robot)[:, self.wrist_ids]   # 이번 물리 스텝 직전 손목 각속도, 내려치는 방향 + (로봇 좌표계 손목 각도는 드는 쪽이 +라서 부호 반대)
        self._substep += 1

        ctrl_joint_ids = self.robot_interface.get_ctrl_joint_ids()
        if self.delay_max > 0:
            # 순환 버퍼에 이번 목표를 넣고, env마다 D 물리 스텝 전 목표를 꺼내 적용 (D = 0이면 이번 목표)
            n = self.delay_max + 1
            self._delay_ptr = (self._delay_ptr + 1) % n
            self.target_buf[self._delay_ptr] = self.target_joint_pos
            idx = (self._delay_ptr - self.env_delay) % n                    # (N,) env별 읽을 칸
            target = self.target_buf[idx, self._env_arange]                  # (N, 9)
        else:
            target = self.target_joint_pos
        self.robot.set_joint_position_target(target, joint_ids=ctrl_joint_ids)
        if self.cfg.waist_dist_std > 0:
            renew = torch.rand(self.num_envs, 1, device=self.device) < self.physics_dt / self.cfg.waist_dist_hold_s
            self.waist_dist = torch.where(renew, self.cfg.waist_dist_std * torch.randn_like(self.waist_dist), self.waist_dist)
            self.robot.set_joint_effort_target(self.waist_dist, joint_ids=[self.waist_usd_id])
            # renew_yaw = torch.rand(self.num_envs, 2, device=self.device) < self.physics_dt / self.cfg.waist_dist_hold_s   # 관절마다 따로 다시 뽑음
            # self.yaw_dist = torch.where(renew_yaw, self.cfg.waist_dist_std * torch.randn_like(self.yaw_dist), self.yaw_dist)
            # self.yaw_dist = torch.where(renew_yaw, self.cfg.yaw_dist_std * torch.randn_like(self.yaw_dist), self.yaw_dist)
            # self.robot.set_joint_effort_target(self.yaw_dist, joint_ids=self.yaw_usd_ids)
            renew_arm = torch.rand(self.num_envs, 6, device=self.device) < self.physics_dt / self.cfg.waist_dist_hold_s   # 관절마다 따로 다시 뽑음
            self.arm_dist = torch.where(renew_arm, self.cfg.arm_dist_std * torch.randn_like(self.arm_dist), self.arm_dist)
            self.robot.set_joint_effort_target(self.arm_dist, joint_ids=self.arm_usd_ids)
            renew_wrist = torch.rand(self.num_envs, 2, device=self.device) < self.physics_dt / self.cfg.waist_dist_hold_s # 관절마다 따로 다시 뽑음
            self.wrist_dist = torch.where(renew_wrist, self.cfg.wrist_dist_std * torch.randn_like(self.wrist_dist), self.wrist_dist)
            self.robot.set_joint_effort_target(self.wrist_dist, joint_ids=self.wrist_usd_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # episode time out
        self.steps = self.steps + 1
        time_out = self.steps >= self.episode_length_step - 1
        died = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        # 정렬만 학습: 실제 타격 판정 대신 가상 타격 (음표 시각에 플래너가 배정한 손이 쳤다고 간주)
        tip_pos = self.robot_interface.get_body_pos(self.robot)                                # (N, 2, 3) 플래너 입력
        virtual = self._virtual_hits()                                                          # (N, 2, M)

        # 실제 타격 판정 (타격 보상용): 물리 스텝별 접촉력·접촉 지점 -> 정상 타격 / 잘못된 접촉 + 악보 매칭
        self.contact_from_tip[:, -1] = self._get_contact_from_tip()                             # 마지막 물리 스텝 접촉 지점
        hit = self.hit_detector.step(
            contact_force=self._get_drum_contact_force(),
            contact_from_tip=self.contact_from_tip,
            steps=self.steps,
            tip_vz_pre=self.tip_vz_pre,
            wrist_vel_pre=self.wrist_vel_pre,
        )

        # 자기 충돌: [왼 상완, 왼 전완, 왼 손목] x [오른 상완, 오른 전완, 오른 손목], 이번 정책 스텝의 물리 스텝 중 한 번이라도 닿으면 True
        self_contact = torch.stack(
            [(s.data.force_matrix_w_history.norm(dim=-1) > 1e-3).any(dim=1)[:, 0] for s in self.self_contacts], dim=1
        )   # (N, 3, 3)

        # 손 배정: 이번 스텝 (가상) 타격까지 반영한 뒤 플래너 실행 -> 손별 다음 음표, 목표별 담당 손
        next_hits = self.rds.get_next_hits(self.steps)
        self.last_hit_step = torch.where(virtual.any(dim=-1), self.steps[:, None].expand(-1, 2), self.last_hit_step)
        plan = self.planner.schedule(self.steps, next_hits, tip_pos, self._get_tip_velocity(), self.drum_pos, self.last_hit_step)
        ev_arm, ev_off = plan[5], plan[6]
        self.hit_detector.set_target_arm(self.steps, ev_arm, ev_off)                           # 다음 음표들의 담당 손 (가상 타격에 사용)
        # self.arm_drum, self.arm_t, self.arm_valid = self._arm_next_notes(ev_arm, next_hits)
        self.arm_drum, self.arm_t, self.arm_valid, self.arm_grade = self._arm_next_notes(ev_arm, next_hits)
        # 손별 다음 음표를 이미 맞혔는지 (일찍 맞힌 뒤 음표 시각까지 근접·진척 끄기용): 음표 스텝 = 현재 + 남은 시간 × (미리보기 - 1)
        det = self.hit_detector
        note_step = (self.steps.long()[:, None] + torch.round(self.arm_t * (det.env.max_lookahead_step - 1)).long()).clamp(0, self.episode_length_step - 1)   # (N, 2)
        arm_matched = det.matched_target[det.env_arange[:, None], note_step, self.arm_drum] & self.arm_valid                      # (N, 2)

        # 보상: 허리·어깨 정렬
        joint_pos = self.robot_interface.get_joint_pos(self.robot)
        joint_vel = self.robot_interface.get_joint_vel(self.robot)
        reward, reward_log = self.reward_computer.compute(
            hit_per_arm=virtual,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            drum_pos=self.drum_pos,
            arm_drum=self.arm_drum,
            arm_t=self.arm_t,
            arm_valid=self.arm_valid,
            actions=self.actions,
            hit=hit,
            self_contact=self_contact,
            tip_pos=tip_pos,
            arm_matched=arm_matched,
            waist_q=self._real_feedback_pos(joint_pos)[:, self.waist_ctrl_id],   # 양자화된 허리 q0 (어깨 yaw 끝점 θk* 계산용, 실기 측정값과 같은 엔코더 칸)
        )

        # 로그 출력
        self.logger.add(reward_log)
        self.logger.maybe_flush()
        self.extras["log"] = {k: v.float().mean() for k, v in reward_log.items()}   # 텐서보드: skrl 이 "Info / 키" 로 기록 (전체 env 평균)

        # 시각화 (가상 타격 표시, 드럼 색상 변경)
        # self.visualizer.step(tip_pos=tip_pos, next_hits=self.next_hits, hit_per_arm=virtual)
        self.visualizer.step(tip_pos=tip_pos, next_hits=self.next_hits, hit_per_arm=hit.hit_per_arm)

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
        # self.rds.reset(env_ids=env_ids, score_ratio=0.0, selection_strength=0.0)  # 무작위 악보만 (배포 모델 환경, MIDI 섞을 때는 0.3)
        self.rds.reset(env_ids=env_ids, score_ratio=1.0, selection_strength=0.0)  # MIDI 악보만 (음표별 세기 1~7 이 있음)

        # 로봇 자세 리셋
        init_robot_pos, init_drum = self.robot_initializer.reset_init_pos(env_ids)
        joint_pos, joint_vel = self.robot_interface.reset(robot=self.robot, env_ids=env_ids, init_robot_pos=init_robot_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.wrist_seeded[env_ids] = False      # 손목 속도 유한차분 다시 시작

        # 지연 버퍼를 리셋 자세로 채우고 env별 지연 다시 뽑기 (안 하면 새 에피소드 첫 스텝에 이전 에피소드 목표가 적용됨)
        if self.delay_max > 0:
            self.target_buf[:, env_ids, :] = joint_pos[:, self.robot_interface.get_ctrl_joint_ids()]
            self.env_delay[env_ids] = torch.randint(self.delay_min, self.delay_max + 1, (len(env_ids),), device=self.device)

        # 누적 목표 = 시작 자세 (관절 제한 안)
        low, high = self.robot_interface.get_limit()
        self.target_q[env_ids] = torch.max(torch.min(init_robot_pos, high), low)
        self.waist_dist[env_ids] = self.cfg.waist_dist_std * torch.randn(len(env_ids), 1, device=self.device)
        # self.yaw_dist[env_ids] = self.cfg.waist_dist_std * torch.randn(len(env_ids), 2, device=self.device)
        # self.yaw_dist[env_ids] = self.cfg.yaw_dist_std * torch.randn(len(env_ids), 2, device=self.device)
        self.arm_dist[env_ids] = self.cfg.arm_dist_std * torch.randn(len(env_ids), 6, device=self.device)
        self.wrist_dist[env_ids] = self.cfg.wrist_dist_std * torch.randn(len(env_ids), 2, device=self.device)

        # 타격 감지 리셋 (rds.reset 이후: 새 악보로 매칭 범위 계산)
        self.hit_detector.reset(env_ids=env_ids, rds=self.rds.get_rds())

        # 보상 리셋 (드럼 위치 리셋 이후: 드럼 쌍별 목표 허리 표 계산)
        self.reward_computer.reset(env_ids=env_ids, drum_pos=self.drum_pos, init_drum=init_drum)
        self.reward_computer.alignment.ref_yaw[env_ids] = self.target_q[env_ids][:, self.yaw_ctrl_ids]   # 시작 θk_d = 시작 어깨 yaw (리셋 직후 첫 관측도 이 값)

        # 손 배정 리셋
        self.planner.reset(env_ids)                     # 확정된 배정 지움
        self.last_hit_step[env_ids] = -10000
        self.arm_valid[env_ids] = False                 # 리셋 직후 첫 관측은 "배정 없음"
        self.arm_t[env_ids] = 1.0
        self.arm_grade[env_ids] = 0.0

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
        # K = self.rds.cfg.num_hits                    # 다음 타격 관측 뺌

        self.obs_dim_joint_pos = 9                     # ctrl 관절 수
        self.obs_dim_joint_vel = 9
        self.obs_dim_lead      = 9                     # 목표 앞섬 (누적 목표 - 측정 각도)
        self.obs_dim_tip       = 2 * 3                 # 양손 * xyz
        self.obs_dim_drum      = M * 3                 # 드럼 * xyz
        # self.obs_dim_next      = K * (M + 2)           # lookahead K개 * (one-hot M + time + valid)  (다음 타격 관측 뺌)
        self.obs_dim_arm       = 2 * (M + 1)           # 손별 다음 음표 (플래너 배정): 드럼 one-hot M + 남은 시간
        # self.obs_dim_arm       = 2 * (M + 2)           # 손별 다음 음표 (플래너 배정): 드럼 one-hot M + 세기/7 + 남은 시간
        # self.obs_dim_align     = 3                     # 정렬 오차 (허리, 오른 yaw, 왼 yaw)  (θ0*로 바꿈)
        self.obs_dim_waist_des = 1                     # 목표 허리 θ0*
        self.obs_dim_waist_vel_des = 1                 # 목표 허리 속도 θ̇0_d
        self.obs_dim_yaw_des   = 2                     # 목표 어깨 yaw θ1*, θ2*
        self.obs_dim_yaw_vel_des = 2                   # 목표 어깨 yaw 속도 θ̇1_d, θ̇2_d
        # self.obs_dim_waist_phase = 1                   # θ0* 바뀐 뒤 진행률 s/D
        # self.obs_dim_waist_phase = 0                   # 진행률 칸 뺌
        self.obs_dim_last_act  = 9                     # 직전 행동

        obs_dim_total = (
            self.obs_dim_joint_pos + self.obs_dim_joint_vel + self.obs_dim_lead
            + self.obs_dim_tip + self.obs_dim_drum
            # + self.obs_dim_next                       # 다음 타격 관측 뺌
            # + self.obs_dim_align                      # θ0*로 바꿈
            # + self.obs_dim_arm + self.obs_dim_waist_des + self.obs_dim_waist_phase
            # + self.obs_dim_arm + self.obs_dim_waist_des + self.obs_dim_waist_vel_des + self.obs_dim_waist_phase
            # + self.obs_dim_arm + self.obs_dim_waist_des + self.obs_dim_waist_vel_des
            # + self.obs_dim_arm + self.obs_dim_waist_des + self.obs_dim_waist_vel_des + self.obs_dim_yaw_des
            + self.obs_dim_arm + self.obs_dim_waist_des + self.obs_dim_waist_vel_des + self.obs_dim_yaw_des + self.obs_dim_yaw_vel_des
            + self.obs_dim_last_act
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
            # next_hits: torch.Tensor,      # 다음 타격 관측 뺌
            arm_next: torch.Tensor,
            # align_err: torch.Tensor,      # θ0*로 바꿈
            waist_des: torch.Tensor,
            waist_vel_des: torch.Tensor,
            yaw_des: torch.Tensor,
            yaw_vel_des: torch.Tensor,
            # waist_phase: torch.Tensor,
            last_action: torch.Tensor,
        ) -> torch.Tensor:
        # joint_pos, joint_vel, lead: (N, 9)
        # tip_pos: (N, 2, 3)
        # drum_pos: (N, M, 3)
        # # next_hits: (N, K, M+2)        (다음 타격 관측 뺌)
        # arm_next: (N, 2, M+1) 손별 다음 음표 (이미 0~1)
        # # align_err: (N, 3) 정렬 오차 [허리, 오른 yaw, 왼 yaw] (이미 -1~1)  (θ0*로 바꿈)
        # waist_des: (N, 1) 목표 허리 θ0* (관절 각도와 같은 정규화, 이미 -1~1)
        # # waist_phase: (N, 1) θ0* 바뀐 뒤 진행률 s/D (이미 0~1)
        # last_action: (N, 9) 직전 행동 (clamp, 고정 관절 0 적용 후, 이미 -1~1)

        # normalize
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -10.0, 10.0)

        waist_vel_des_n = torch.clamp(waist_vel_des / self.joint_vel_scale, -10.0, 10.0)   # θ̇0_d, 관절 속도와 같은 정규화
        yaw_vel_des_n = torch.clamp(yaw_vel_des / self.joint_vel_scale, -10.0, 10.0)       # θ̇1_d, θ̇2_d, 관절 속도와 같은 정규화
        # waist_vel_des_n = waist_vel_des                 # θ̇0_d [rad/s] 그대로 (정규화 안 함)
        # yaw_vel_des_n = yaw_vel_des                     # θ̇1_d, θ̇2_d [rad/s] 그대로 (정규화 안 함)

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
                # next_hits.reshape(self.num_envs, self.obs_dim_next),   # 다음 타격 관측 뺌
                arm_next.reshape(self.num_envs, self.obs_dim_arm),
                # align_err,                    # θ0*로 바꿈
                waist_des,
                waist_vel_des_n,
                yaw_des,
                yaw_vel_des_n,
                # waist_phase,
                last_action,
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
        v = d.body_link_lin_vel_w[:, self.wrist_body_ids]   # 링크 원점 속도 (body_lin_vel_w는 질량중심 속도라 팁보다 0.111 m 바깥 점이 됨)
        w = d.body_ang_vel_w[:, self.wrist_body_ids]
        r = math_utils.quat_apply(d.body_quat_w[:, self.wrist_body_ids], self.stick_tip_off.expand_as(v))
        return v + torch.cross(w, r, dim=-1)

    def _arm_next_notes(self, ev_arm: torch.Tensor, next_hits: torch.Tensor):
        # 플래너 배정 (이벤트별 드럼 담당 손) -> 손별 가장 이른 음표: 드럼 (N, 2), 남은 시간 (N, 2), 있음 (N, 2)
        N, K, M = ev_arm.shape
        drum = torch.zeros((N, 2), dtype=torch.long, device=self.device)
        t = torch.ones((N, 2), device=self.device)
        valid = torch.zeros((N, 2), dtype=torch.bool, device=self.device)
        grade = torch.zeros((N, 2), device=self.device)                                 # 손별 다음 음표 악보 세기 (없으면 0)
        for a in (0, 1):
            for k in reversed(range(K)):                    # 뒤에서부터 덮어써서 가장 이른 음표가 남음
                mask = ev_arm[:, k] == a                    # (N, M) 이벤트 k에서 손 a가 맡은 드럼
                has = mask.any(dim=-1)
                drum[:, a] = torch.where(has, mask.int().argmax(dim=-1), drum[:, a])
                t[:, a] = torch.where(has, next_hits[:, k, M], t[:, a])
                grade[:, a] = torch.where(has, (next_hits[:, k, :M] * mask.float()).amax(dim=-1), grade[:, a])   # next_hits 앞 M칸 = 악보 값 = 세기
                valid[:, a] |= has
        # return drum, t, valid
        return drum, t, valid, grade

    def _virtual_hits(self) -> torch.Tensor:
        # 가상 타격: 이번 스텝 음표를 플래너가 확정한 담당 손이 쳤다고 간주 -> (N, 2, M) [왼손, 오른손]
        # (정렬만 학습: 실제로 치지 않아도 손별 직전 드럼, 마지막 타격 시각이 실제 연주 순서대로 바뀌게)
        det, ar = self.hit_detector, self.hit_detector.env_arange
        s = self.steps.clamp(max=self.episode_length_step - 1)
        owner = det.target_arm[ar, s].long()                                   # (N, M) 담당 손 (플래너가 0.2초 전에 확정)
        owner = torch.where(owner < 0, torch.ones_like(owner), owner)         # 미배정이면 오른손 (실제로는 안 생김)
        return det.target[ar, s][:, None] & (owner[:, None] == det.arm_idx)

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

        # 직전 행동 리셋 (새 에피소드 첫 관측에 이전 에피소드 행동이 안 들어가게, 보상 prev_a 리셋과 같음)
        self.actions[env_ids] = 0.0
