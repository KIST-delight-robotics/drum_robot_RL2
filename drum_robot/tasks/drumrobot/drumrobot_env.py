from __future__ import annotations
                                                                # pyright ignore 는 패키지 누락이 아니라 Isaac Sim이 런타임에 import path를 추가하는 구조 때문
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
from .components.reward import RewardComputerCfg, RewardComputer, ARM_IDLE, ARM_FREE
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

        # 전역 정책 스텝 카운터 (MIDI 커리큘럼 램프용, 매 _get_dones 에서 +1)
        self.global_step = 0

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
        rds_cfg = RDSCfg(
            eval_midi_file=self.cfg.eval_midi_file,   # 평가용 (기본 "" 이면 학습과 동일)
            eval_full_song=self.cfg.eval_full_song,
            slow_factor=self.cfg.eval_slow_factor,    # play1 에서 override 가능
        )
        max_lookahead_step = int(rds_cfg.max_lookahead_s / self.step_dt)
        self.max_lookahead_step = max_lookahead_step             # 스케줄러: 정규화 time 을 스텝으로 환산
        self.hit_window_step = hit_detector_cfg.hit_window_step  # 스케줄러: 재장전 t_rearm = 2W+1
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

        # sim 물리 관절한계를 specs.joint_limit 값으로 재설정 (전 9관절).
        #   USD 기본 한계는 사실상 무한이라 관절이 목표를 관성으로 넘어갈 수 있다. PhysX 하드 한계로 막는다.
        #   robot 프레임 한계를 USD 프레임으로 부호 변환한다 (dir=-1 이면 [lo,hi] 가 [-hi,-lo] 로 뒤집힘).
        ri = self.robot_interface
        _d = ri.dir_tensor.float()                                  # (1,9) +1 또는 -1
        _usd_a = _d * ri.joint_low                                  # robot 프레임을 usd 프레임으로
        _usd_b = _d * ri.joint_high
        _usd_low  = torch.minimum(_usd_a, _usd_b)                   # (1,9)
        _usd_high = torch.maximum(_usd_a, _usd_b)
        _limits = torch.stack([_usd_low, _usd_high], dim=-1)        # (1,9,2)
        _limits = _limits.expand(self.num_envs, -1, -1).contiguous()
        self.robot.write_joint_position_limit_to_sim(
            _limits, joint_ids=ri.get_ctrl_joint_ids(), warn_limit_violation=False,
        )
        print("[LIMIT] sim 물리 관절한계 = specs.joint_limit 로 재설정 (전 9관절)")

        # 관절별 action_scale (손목만 상한 크게). ctrl_joint_names 는 resolve 된 순서
        ctrl_joint_names = self.robot_interface.get_ctrl_joint_name()
        scale_list = [
            self.cfg.wrist_action_scale if "wrist" in name else self.cfg.action_scale
            for name in ctrl_joint_names
        ]
        self.action_scale_vec = torch.tensor(
            scale_list, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # 제어 지연 버퍼 (물리 스텝 단위, 순환 인덱스). max=0 이면 미사용.
        #   _apply_action 이 물리 스텝마다 호출되므로 SIM_DT 해상도로 지연된다.
        #   env 별로 [min, max] 에서 균등 추첨하고 리셋마다 재추첨한다.
        self.delay_min = int(self.cfg.action_delay_steps_min)
        self.delay_max = int(self.cfg.action_delay_steps_max)
        if not (0 <= self.delay_min <= self.delay_max):
            raise ValueError(
                f"action_delay_steps_min({self.delay_min}) <= max({self.delay_max}) 이고 둘 다 >=0 이어야 함"
            )
        self._env_arange = torch.arange(self.num_envs, device=self.device)
        self.target_buf = torch.zeros(
            (self.delay_max + 1, self.num_envs, len(ctrl_joint_names)),
            device=self.device, dtype=torch.float32,
        )
        self._delay_ptr = 0
        # env 별 지연 [스텝]. min==max 이면 상수 텐서 = 고정 지연
        self.env_delay = torch.randint(
            self.delay_min, self.delay_max + 1, (self.num_envs,), device=self.device, dtype=torch.long,
        )
        print(f"[DELAY] action delay = [{self.delay_min}, {self.delay_max}] 물리스텝 "
              f"= [{self.delay_min * self.cfg.sim.dt * 1e3:.1f}, {self.delay_max * self.cfg.sim.dt * 1e3:.1f}] ms"
              + ("  (고정)" if self.delay_min == self.delay_max else "  (env별 DR, 리셋마다 재추첨)"))

        # 관절 한계 초과 로그용 짧은 이름 (ctrl 순서)
        _short = {
            "waist_joint": "waist",
            "left_shoulder_1": "Lsh1", "left_shoulder_2": "Lsh2", "left_elbow": "Lelb",
            "right_shoulder_1": "Rsh1", "right_shoulder_2": "Rsh2", "right_elbow": "Relb",
            "left_wrist": "Lwr", "right_wrist": "Rwr",
        }
        self.limit_log_names = [_short.get(n, n) for n in ctrl_joint_names]

        # 손목 관절 인덱스 [왼, 오]. hit_per_arm 팔 축(0=왼, 1=오)과 맞춤
        name_to_idx = {name: i for i, name in enumerate(ctrl_joint_names)}
        self.wrist_joint_ids = torch.tensor(
            [name_to_idx["left_wrist"], name_to_idx["right_wrist"]],
            device=self.device, dtype=torch.long,
        )
        # 리바운드 주입용: 드럼별 반발계수 (M,) 와 env 별 DR 값 (N,M). 손목 USD 관절 인덱스는 속도를 USD 프레임에서
        #   그대로 뒤집으므로 부호 변환이 필요 없다
        _cymbals = {"hihat", "ride", "crash_r", "crash_l"}
        self.rebound_e = torch.tensor(
            [self.cfg.rebound_e_cymbal if n in _cymbals else self.cfg.rebound_e_drum
             for n in self.instruments.all.keys()],
            device=self.device, dtype=torch.float32,
        )                                                                   # (M,)
        self.rebound_e_env = self.rebound_e.unsqueeze(0).repeat(self.num_envs, 1)   # (N,M)
        self.wrist_usd_ids = [self.robot_interface.get_ctrl_joint_ids()[i] for i in self.wrist_joint_ids.tolist()]
        if self.cfg.rebound_enabled:
            print(f"[REBOUND] e per drum = {dict(zip(self.instruments.all.keys(), self.rebound_e.tolist()))}"
                  f"  jitter={self.cfg.rebound_e_jitter}")
        # [sim2real 점검] 손목 속도 유한차분용 버퍼 (cfg.wrist_vel_finite_diff).
        #   _wrist_fd_seeded: 리셋 직후 첫 obs 는 차분이 무의미하므로 실기와 같이 참값 사용
        self.prev_wrist_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._wrist_fd_seeded = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        # 팔꿈치 관절 인덱스 [왼, 오]. 세기별 리프트 목표 매칭용
        self.elbow_joint_ids = torch.tensor(
            [name_to_idx["left_elbow"], name_to_idx["right_elbow"]],
            device=self.device, dtype=torch.long,
        )
        # 팔별 gross(위치이동) 관절 인덱스 [왼3, 오3] = 어깨1, 어깨2, 팔꿈치. 어깨 과꺾임 진단용
        self.arm_gross_ids = torch.tensor([
            [name_to_idx["left_shoulder_1"],  name_to_idx["left_shoulder_2"],  name_to_idx["left_elbow"]],
            [name_to_idx["right_shoulder_1"], name_to_idx["right_shoulder_2"], name_to_idx["right_elbow"]],
        ], device=self.device, dtype=torch.long)   # (2, 3)
        # [진단] 허리 인덱스 (타격 순간 몸통 회전량 관찰)
        self.waist_idx = name_to_idx["waist_joint"]
        # 리프트 baseline(직전 타격 팔꿈치각) + 스윙 중 리프트 apex(최대) 추적 버퍼
        self.elbow_baseline = torch.zeros((self.num_envs, 2), device=self.device)
        self.elbow_apex_buf = torch.zeros((self.num_envs, 2), device=self.device)
        # 스케줄러가 정한 h0 담당(왼팔=True). 보상 배정에 주입 (1스텝 지연으로 사용)
        self.use_left_h0 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.use_case1_h0 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # 동시타 배정 (왼팔이 idx0 담당이면 True)
        # 팔별 마지막 실제 타격 스텝 (스케줄러 재장전 시드용). 초기값은 먼 과거 = 시작은 양팔 자유
        self.last_hit_step = torch.full((self.num_envs, 2), -10000, dtype=torch.int64, device=self.device)
        # 팔 상태 (로그와 노는 팔 접근 보상용). 판단은 보상 계산(RewardComputer.compute)이 관측에 있는 값으로 하고 결과를 돌려준다
        #   arm_state: ARM_STROKE, ARM_PREP, ARM_IDLE(맡은 노트 있음), ARM_FREE(맡은 노트 없음)
        #   arm_idle:  노는 팔 또는 자유 팔
        self.arm_state = torch.full((self.num_envs, 2), ARM_FREE, dtype=torch.int64, device=self.device)
        self.arm_idle = torch.ones((self.num_envs, 2), dtype=torch.bool, device=self.device)
        # 팔별 마지막으로 친 드럼 위치 (진단용)
        self.last_hit_drum_pos = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        # 노는 팔 대기점 기준(다음 노트 위치) 변화 진단용 직전 값 (팔별)
        self.prev_home_p = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        # [진단] 배정 변화 측정용 직전 스텝 값
        self.prev_use_left_h0 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_arm_tgt0 = torch.zeros((self.num_envs, 2, 3), device=self.device)
        # 채점용 담당 팔 기록 (N, T, M). 악보 칸(절대 스텝, 드럼)마다 판정 창이 열리기 직전의 담당 팔 (0=왼, 1=오른, -1=기록 없음).
        #   한 번 기록하면 바꾸지 않는다. 관측의 배정은 스케줄러가 매 스텝 새로 정하므로 이 기록과 다를 수 있다
        self.assign_frozen = torch.full((self.num_envs, self.episode_length_step, len(self.instruments.all)), -1,
                                        dtype=torch.int8, device=self.device)
        # [진단] 기록된 담당 팔과 지금 관측의 담당 팔이 다른 노트 수 / 기록된 노트 수 (관측 생성 때 세고 다음 보상 때 로그)
        self.assign_mm_num = torch.zeros(self.num_envs, device=self.device)
        self.assign_mm_den = torch.zeros(self.num_envs, device=self.device)

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

    def _compute_arm_schedule(self, next_hits: torch.Tensor, tip_pos: torch.Tensor, drum_pos: torch.Tensor):
        """팔별 담당 스케줄 (reachability, 시간순). 현재 팁 위치로 시드.
        각 타격을 현재위치 + 이동가용시간(window) + 재장전(busy) 로 배정한다.
          window = t_k - 그 팔 자유시각,  req_speed = travel/window,  busy_pen = BIG * relu(t_rearm - window)
          score = req_speed + busy_pen 의 argmin. 방금 친 팔은 busy_pen 이 커서 빠른 클러스터는 양팔에 분배된다.
        동시 2타 이벤트는 두 드럼을 양 팔에 분배한다 (min-cost split).
        Returns: use_left_h0 (N,) [case1 h0 왼팔?], use_case1_h0 (N,) [case2: 왼팔이 idx0?],
          per_arm_pos (N,2,NOTES,3), per_arm_time (N,2,NOTES), per_arm_valid (N,2,NOTES). arm0=왼, arm1=오른."""
        N = self.num_envs
        M = len(self.instruments.all)
        K = self.rds.cfg.num_hits
        NOTES = self.sched_notes
        L = float(self.max_lookahead_step)
        t_rearm = float(2 * self.hit_window_step + 1)                     # 한 손 최소 재타 간격(스텝)
        BIG = 1.0e3

        multihot = next_hits[:, :, :M]                                    # (N,K,M)
        time_norm = next_hits[:, :, M]                                    # (N,K) 정규화 time_to_hit
        valid = next_hits[:, :, M + 1] > 0.5                              # (N,K)
        # 이벤트당 상위 2개 드럼 (보상 case2 의 topk 와 동일 정렬: 드럼 인덱스 큰 순)
        arange_m = torch.arange(M, device=self.device).view(1, 1, M).expand(N, K, M)
        masked_idx = torch.where(multihot > 0.5, arange_m, torch.full_like(arange_m, -1))
        top2 = torch.topk(masked_idx, k=2, dim=-1).values                 # (N,K,2)
        idx0 = top2[:, :, 0].clamp(min=0); idx1 = top2[:, :, 1].clamp(min=0)
        has2 = top2[:, :, 1] >= 0                                         # 둘째 드럼 존재 = 동시 2타
        pos0 = torch.gather(drum_pos, 1, idx0.unsqueeze(-1).expand(N, K, 3))  # (N,K,3) 높은 드럼
        pos1 = torch.gather(drum_pos, 1, idx1.unsqueeze(-1).expand(N, K, 3))  # (N,K,3) 둘째 드럼
        t_step = time_norm * max(L - 1.0, 1.0)                            # (N,K) 정규화 시간을 스텝으로

        proj = tip_pos.clone()                                            # (N,2,3) 팔 투영위치 (시드=현재 팁)
        # 팔 자유시각 (스텝, now=0 기준). 방금 친 팔은 재장전 남은 만큼 양수 = busy
        free = ((self.last_hit_step - self.steps.unsqueeze(1)).float() + t_rearm).clamp(min=0.0)  # (N,2)
        use_left_h0 = torch.ones(N, dtype=torch.bool, device=self.device)
        use_case1_h0 = torch.ones(N, dtype=torch.bool, device=self.device)   # case2: True = 왼팔이 idx0
        per_arm_pos = drum_pos.new_zeros((N, 2, NOTES, 3))
        per_arm_time = time_norm.new_ones((N, 2, NOTES))                  # 기본 1.0 = 없음/멀리
        per_arm_valid = torch.zeros((N, 2, NOTES), dtype=torch.bool, device=self.device)
        count = torch.zeros((N, 2), dtype=torch.long, device=self.device)
        # 이벤트별 드럼 담당 팔 (N,K,M). 0=왼, 1=오른, -1=해당 없음 (채점용 기록에 쓴다)
        ev_arm = torch.full((N, K, M), -1, dtype=torch.int8, device=self.device)
        arm_left = torch.zeros(N, dtype=torch.int8, device=self.device)
        arm_right = torch.ones(N, dtype=torch.int8, device=self.device)
        arm_none = torch.full((N,), -1, dtype=torch.int8, device=self.device)

        CROSS = self.cfg.sched_cross_penalty

        def arm_score(a, p, tk):
            w = tk - free[:, a]                                           # free 이후 여유. 음수면 아직 재장전 중
            # 교차 침범량 [m]: 왼팔(a=0)이 우측(x>0), 오른팔(a=1)이 좌측(x<0) 드럼을 맡을 때만 양수.
            #   드럼 x 와 팔 정체성만으로 결정되므로 현재 팁 위치와 무관하다
            px = p[:, 0]
            cross = torch.relu(px) if a == 0 else torch.relu(-px)         # (N,)
            travel = torch.linalg.norm(proj[:, a] - p, dim=-1) + CROSS * cross   # 교차를 추가 이동거리로 환산
            return travel / w.clamp(min=1.0) + BIG * torch.relu(-w)       # 이동필요속도 + 미복구 페널티

        for k in range(K):
            v = valid[:, k]
            two = v & has2[:, k]                                          # 동시 2타
            one = v & (~has2[:, k])                                       # 단일
            p0 = pos0[:, k]; p1 = pos1[:, k]; tk = t_step[:, k]; tn = time_norm[:, k]
            sL0 = arm_score(0, p0, tk); sR0 = arm_score(1, p0, tk)
            sL1 = arm_score(0, p1, tk); sR1 = arm_score(1, p1, tk)
            # 단일: p0 를 어느 팔에. k=0 은 히스테리시스 적용 (직전 담당팔을 유지하는 상대 마진).
            #   두 팁이 등거리이면 승자가 쉽게 뒤집히고 proj 체인 때문에 양팔 큐 전체가 스왑되므로 k=0 에서 잡는다
            if k == 0:
                h = self.cfg.sched_hysteresis
                choose_left_one = torch.where(
                    self.prev_use_left_h0,
                    sL0 <= sR0 * (1.0 + h),      # 직전=왼팔: 유지 쪽에 관대
                    sL0 * (1.0 + h) <= sR0,      # 직전=오른팔: 왼팔이 h 만큼 확실히 나아야 뒤집힘
                )
            else:
                choose_left_one = sL0 <= sR0
            useA = (sL0 + sR1) <= (sL1 + sR0)                             # 2타: A(왼 p0, 오른 p1) vs B(swap)
            if k == 0:
                use_left_h0 = torch.where(one, choose_left_one, use_left_h0)
                use_case1_h0 = torch.where(two, useA, use_case1_h0)
            # 팔별 이번 이벤트 타겟: 단일 = 담당팔만 p0, 2타 = 양팔 각자 split
            left_gets = (one & choose_left_one) | two
            right_gets = (one & (~choose_left_one)) | two
            # 이벤트 k 의 드럼별 담당 팔 기록. 단일 = 고른 팔이 idx0, 2타 = A 면 왼팔 idx0 / 오른팔 idx1, B 면 반대.
            #   2타가 아니면 idx1 자리에 idx0 를 같은 값으로 써서, idx1 이 0 으로 잘린 칸을 덮어쓰지 않게 한다
            a0 = torch.where(two, torch.where(useA, arm_left, arm_right),
                             torch.where(choose_left_one, arm_left, arm_right))
            a1 = torch.where(useA, arm_right, arm_left)
            i0 = idx0[:, k:k + 1]
            i1 = torch.where(two, idx1[:, k], idx0[:, k]).unsqueeze(1)
            ev_k = torch.full((N, M), -1, dtype=torch.int8, device=self.device)
            ev_k.scatter_(1, i1, torch.where(v, torch.where(two, a1, a0), arm_none).unsqueeze(1))
            ev_k.scatter_(1, i0, torch.where(v, a0, arm_none).unsqueeze(1))
            ev_arm[:, k] = ev_k
            left_pos = torch.where(two.unsqueeze(-1), torch.where(useA.unsqueeze(-1), p0, p1), p0)
            right_pos = torch.where(two.unsqueeze(-1), torch.where(useA.unsqueeze(-1), p1, p0), p0)
            for a, gets, gp in ((0, left_gets, left_pos), (1, right_gets, right_pos)):
                can = gets & (count[:, a] < NOTES)
                rows = torch.where(can)[0]
                if rows.numel() > 0:
                    s = count[rows, a]
                    per_arm_pos[rows, a, s] = gp[rows]
                    per_arm_time[rows, a, s] = tn[rows]
                    per_arm_valid[rows, a, s] = True
                    count[rows, a] = count[rows, a] + 1
            new_free = tk + t_rearm
            proj[:, 0] = torch.where(left_gets.unsqueeze(-1), left_pos, proj[:, 0])
            proj[:, 1] = torch.where(right_gets.unsqueeze(-1), right_pos, proj[:, 1])
            free[:, 0] = torch.where(left_gets, new_free, free[:, 0])
            free[:, 1] = torch.where(right_gets, new_free, free[:, 1])
        ev_off = t_step.round().long()                                    # (N,K) 이벤트까지 남은 스텝
        return use_left_h0, use_case1_h0, per_arm_pos, per_arm_time, per_arm_valid, ev_arm, ev_off

    def _get_observations(self) -> dict:
        # 로봇의 관절 위치
        joint_pos = self.robot_interface.get_joint_pos(self.robot)

        # 로봇의 관절 속도
        joint_vel = self.robot_interface.get_joint_vel(self.robot)

        # [sim2real 점검] 손목 2축만 실기와 같은 유한차분으로 대체 (obs 전용)
        if self.cfg.wrist_vel_finite_diff:
            _w = self.wrist_joint_ids
            _q = joint_pos[:, _w]
            _fd = (_q - self.prev_wrist_pos) / self.step_dt
            _fd = torch.where(self._wrist_fd_seeded.unsqueeze(1), _fd, joint_vel[:, _w])
            self.prev_wrist_pos = _q.clone()
            self._wrist_fd_seeded[:] = True
            joint_vel = joint_vel.clone()
            joint_vel[:, _w] = _fd

        # 위치, 로봇 상태
        tip_pos, hit_armed = self.hit_detector.get_value_for_obs()
        drum_pos = self.drum_pos

        # 다음 타격
        self.next_hits = self.rds.get_next_hits(self.steps)

        # 팔별 담당 스케줄 (reachability). obs 와 보상 h0 배정(use_left)에 사용.
        # 1스텝 지연으로 다음 _get_rewards 에서 사용 (arm_role 과 동일한 지연)
        (self.use_left_h0, self.use_case1_h0, self.per_arm_pos, self.per_arm_time, self.per_arm_valid,
         ev_arm, ev_off) = self._compute_arm_schedule(self.next_hits, tip_pos, drum_pos)

        # 채점용 담당 팔 기록: 노트까지 W+1 스텝 이하로 처음 들어온 순간의 담당 팔을 적고 이후 바꾸지 않는다.
        #   관측은 보상 계산 뒤에 만들어지므로 W+1 에서 적어야 판정 창 첫 스텝의 타격도 기록으로 채점된다.
        #   스케줄러가 그 뒤 배정을 바꿔도 관측만 바뀌고 기록은 그대로다 (그 빈도를 assign_mismatch 로 센다)
        W = self.hit_window_step
        T = self.episode_length_step
        self.assign_mm_num.zero_()
        self.assign_mm_den.zero_()
        for k in range(ev_arm.shape[1]):
            ev = ev_arm[:, k]                                                         # (N,M)
            open_k = ((ev_off[:, k] <= W + 1) & (self.steps + ev_off[:, k] < T)).unsqueeze(-1)   # (N,1)
            tau = (self.steps + ev_off[:, k]).clamp(max=T - 1)                        # (N,) 노트의 절대 스텝
            cur = self.assign_frozen[self._env_arange, tau]                           # (N,M)
            seen = open_k & (cur >= 0) & (ev >= 0)
            self.assign_mm_num += (seen & (ev != cur)).float().sum(dim=-1)
            self.assign_mm_den += seen.float().sum(dim=-1)
            write = open_k & (cur < 0) & (ev >= 0)
            self.assign_frozen[self._env_arange, tau] = torch.where(write, ev, cur)

        obs = self._normalize_and_pack_obs(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            tip_pos=tip_pos,
            drum_pos=drum_pos,
            next_hits=self.next_hits,
            hit_armed=hit_armed.float(),
            arm_role=self.arm_role,
            per_arm_pos=self.per_arm_pos,
            per_arm_time=self.per_arm_time,
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
        if self.delay_max > 0:
            # 순환 버퍼에 지금 목표를 넣고, env 별 D 스텝 전 목표를 꺼내 적용 (순수 정수 지연)
            n = self.delay_max + 1
            self._delay_ptr = (self._delay_ptr + 1) % n
            self.target_buf[self._delay_ptr] = self.target_joint_pos
            idx = (self._delay_ptr - self.env_delay) % n                  # (N,) env 별 읽기 슬롯
            target = self.target_buf[idx, self._env_arange]                # (N, 9)
        else:
            target = self.target_joint_pos
        self.robot.set_joint_position_target(target, joint_ids=ctrl_joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 전역 스텝 (MIDI 커리큘럼 램프)
        self.global_step += 1

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
            xy_in_any, xyz_in_any,
        ) = self.hit_detector.detect_hit(tip_pos=tip_pos, drum_pos=self.drum_pos)

        # 담당 팔 확인: 각 드럼의 판정 창(+-W) 안 타겟에 기록된 담당 팔. 타겟이 없거나 기록 전이면 -1 (어느 팔이든 인정).
        #   같은 드럼의 노트 간격은 2W+1 스텝 이상이라 한 드럼의 +-W 안 타겟은 하나뿐이다
        N = self.num_envs
        M = self.drum_pos.shape[1]
        W = self.hit_window_step
        T = self.episode_length_step
        rds = self.rds.get_rds()
        assigned_now = torch.full((N, M), -1, dtype=torch.int8, device=self.device)
        for o in range(-W, W + 1):
            tau = self.steps + o
            tau_ok = (tau >= 0) & (tau < T)
            tau_c = tau.clamp(0, T - 1)
            is_tgt = (rds[self._env_arange, tau_c] > 0) & tau_ok.unsqueeze(-1)                  # (N,M)
            assigned_now = torch.where(is_tgt, self.assign_frozen[self._env_arange, tau_c], assigned_now)
        arm_id = torch.arange(2, device=self.device, dtype=torch.int8).view(1, 2, 1)
        arm_ok = (assigned_now.unsqueeze(1) < 0) | (assigned_now.unsqueeze(1) == arm_id)          # (N,2,M)
        hit_ok_per_arm = hit_per_arm & arm_ok        # 성공 후보 (담당 팔의 타격)
        hit_bad_per_arm = hit_per_arm & (~arm_ok)    # 담당 아닌 팔이 판정 창 안 타겟을 침 (오타)

        # 관절각, 관절 속도 (손목 각속도 계산 위해 먼저 가져옴)
        robot_pos = self.robot_interface.get_joint_pos(self.robot)
        robot_vel = self.robot_interface.get_joint_vel(self.robot)
        robot_torque = self.robot_interface.get_joint_torque(self.robot)

        # 리바운드 주입: 팁이 헤드 면을 이번 스텝에 아래로 통과한 팔의 손목 각속도를 -e 배로 반전 (다음 물리 스텝부터 반영).
        #   트리거는 타격 판정과 독립. 직전 샘플이 면 위, 이번 샘플이 면 아래이면 그 사이에 면을 지난 것이다.
        #   보상/판정에 쓰는 robot_vel 은 반전 전 값이다 (타격 순간 손목 속도 = 세기)
        rebound_arm = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
        if self.cfg.rebound_enabled:
            z_plane = self.drum_pos[:, :, 2] + self.cfg.rebound_plane_offset                       # (N,M) 헤드 면 높이
            dz_now  = tip_pos[:, :, 2].unsqueeze(-1)      - z_plane.unsqueeze(1)                  # (N,2,M)
            dz_prev = prev_tip_pos[:, :, 2].unsqueeze(-1) - z_plane.unsqueeze(1)                  # (N,2,M)
            dxy_sq  = ((tip_pos[:, :, :2].unsqueeze(2) - self.drum_pos[:, :, :2].unsqueeze(1)) ** 2).sum(dim=-1)  # (N,2,M)
            cross = (dz_prev > 0.0) & (dz_now <= 0.0) & (dxy_sq <= self.cfg.rebound_xy_radius ** 2)  # (N,2,M) 면 통과
            rebound_arm = cross.any(dim=-1)                                                        # (N,2)
            if bool(rebound_arm.any()):
                e_arm = (cross.float() * self.rebound_e_env.unsqueeze(1)).max(dim=-1).values       # (N,2) 통과한 드럼의 e
                usd_wrist_vel = self.robot.data.joint_vel[:, self.wrist_usd_ids]                   # (N,2) USD 프레임
                new_wrist_vel = torch.where(rebound_arm, -e_arm * usd_wrist_vel, usd_wrist_vel)
                self.robot.write_joint_velocity_to_sim(new_wrist_vel, joint_ids=self.wrist_usd_ids)

        # 타격 순간 드럼별 하강 속도 (담당 팔의 타격 중 max). 세기 평가와 성공 매칭은 담당 팔의 타격만 쓴다
        down_vel = torch.clamp(-tip_vel[:, :, 2], 0.0)                         # (N, 2)
        hit_vel_per_drum = (hit_ok_per_arm.float() * down_vel[:, :, None]).max(dim=1).values  # (N, M)

        # 타격 순간 드럼별 손목 각속도 (그 드럼을 친 담당 팔의 손목)
        wrist_vel_abs = robot_vel[:, self.wrist_joint_ids].abs()                # (N, 2)
        hit_wrist_per_drum = (hit_ok_per_arm.float() * wrist_vel_abs[:, :, None]).max(dim=1).values  # (N, M)

        # 팔꿈치 리프트: baseline(직전 타격 팔꿈치각) 대비 |현재각|. 스윙 중 apex(최대)를 추적한다
        elbow_angle = robot_pos[:, self.elbow_joint_ids]                        # (N, 2)
        elbow_lift = (elbow_angle - self.elbow_baseline).abs()                  # (N, 2)
        self.elbow_apex_buf = torch.maximum(self.elbow_apex_buf, elbow_lift)    # (N, 2) running max
        hit_elbow_apex_per_drum = (hit_ok_per_arm.float() * self.elbow_apex_buf[:, :, None]).max(dim=1).values  # (N, M)

        # 타격 시간/속도/손목각속도/팔꿈치리프트 apex 기록 (담당 팔의 타격만 성공 매칭에 쓰인다)
        self.rds.set_rds_visit(
            steps=self.steps, hit_mask=hit_ok_per_arm.any(dim=1),
            hit_vel=hit_vel_per_drum, hit_wrist=hit_wrist_per_drum, hit_elbow_apex=hit_elbow_apex_per_drum,
            xy_in=xy_in_any, xyz_in=xyz_in_any,
        )

        # 다음 사이클 위해 타격한 팔의 baseline 갱신 + apex 리셋 (리프트 계산 후에)
        hit_arm_any = hit_per_arm.any(dim=-1)                                   # (N, 2)
        self.elbow_baseline = torch.where(hit_arm_any, elbow_angle, self.elbow_baseline)
        self.elbow_apex_buf = torch.where(hit_arm_any, torch.zeros_like(self.elbow_apex_buf), self.elbow_apex_buf)
        # 팔별 마지막 타격 스텝 갱신 (스케줄러가 재장전 중인 팔을 알도록). self.steps 는 이미 증가된 현재 스텝
        self.last_hit_step = torch.where(hit_arm_any, self.steps.unsqueeze(1), self.last_hit_step)
        # 팔별 마지막 친 드럼 위치 갱신 (동시에 여러 드럼이면 인덱스 작은 쪽)
        hit_drum_idx = hit_per_arm.float().argmax(dim=-1)                                    # (N,2)
        picked_drum = torch.gather(self.drum_pos, 1, hit_drum_idx.unsqueeze(-1).expand(-1, -1, 3))
        self.last_hit_drum_pos = torch.where(hit_arm_any.unsqueeze(-1), picked_drum, self.last_hit_drum_pos)

        # 결과
        rds_visit = self.rds.get_rds_visit()
        rds_visit_vel = self.rds.get_rds_visit_vel()
        rds_visit_wrist = self.rds.get_rds_visit_wrist()
        rds_visit_elbow_apex = self.rds.get_rds_visit_elbow_apex()
        rds_visit_xy = self.rds.get_rds_visit_xy()
        rds_visit_xyz = self.rds.get_rds_visit_xyz()
        rds_T = self.rds.get_rds_T()
        (success, wrong_hit, missed_target, time_error, success_impact_vel, success_wrist_ang_vel,
         target_done, target_velocity, success_elbow_apex, target_T, time_error_signed,
         miss_reach, miss_zpos, miss_dir,
         dir_armed, dir_down, dir_slow, dir_arb,
         reach_near, reach_mid, reach_far,
         zpos_above, zpos_below, zpos_both, zpos_gap) = self.hit_detector.get_result(
            hit_mask=hit_ok_per_arm.any(dim=1),
            hit_mask_all=hit_mask,
            wrong_arm=hit_bad_per_arm.any(dim=1),
            steps=self.steps,
            rds=rds,
            rds_visit=rds_visit,
            rds_visit_vel=rds_visit_vel,
            rds_visit_wrist=rds_visit_wrist,
            rds_visit_elbow_apex=rds_visit_elbow_apex,
            rds_visit_xy=rds_visit_xy,
            rds_visit_xyz=rds_visit_xyz,
            rds_T=rds_T,
        )

        joint_low, joint_high = self.robot_interface.get_limit()

        wrist_pos = self.robot_interface.get_wrist_pos(self.robot)  # (N, 2, 3)
        elbow_pos = self.robot_interface.get_elbow_pos(self.robot)  # (N, 2, 3)

        reward, log_terms, rate_log_terms, active_arm, arm_state = self.reward_computer.compute(
            success=success,
            wrong_hit=wrong_hit,
            missed_target=missed_target,
            time_error=time_error,
            tip_pos=tip_pos,
            prev_tip_pos=prev_tip_pos,
            drum_pos=self.drum_pos,
            next_hits=self.next_hits,
            tip_vel=tip_vel,
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
            target_velocity=target_velocity,
            success_elbow_apex=success_elbow_apex,
            target_T=target_T,
            assigned_use_left=self.use_left_h0,
            assigned_use_case1=self.use_case1_h0,
            idle_tgt_pos=self.per_arm_pos[:, :, 0, :],
            idle_tgt_valid=self.per_arm_valid[:, :, 0],
            idle_tgt_time=self.per_arm_time[:, :, 0],      # 관측의 팔별 노트 칸과 같은 값 (팔 상태 판정용)
        )

        # 팔 상태: 보상 계산이 관측에 있는 값으로 판정한 결과. 노는 팔 접근 보상과 로그가 같은 판단을 쓴다
        self.arm_state = arm_state                                                             # (N,2)
        self.arm_idle = (arm_state == ARM_IDLE) | (arm_state == ARM_FREE)                      # (N,2)

        # 팔별 역할 저장 (다음 스텝 관측에 사용). 다음 노트 담당 팔이며 관측 의미는 그대로
        self.arm_role = active_arm

        # 노는 팔 접근(progress) 보상: 자기 담당 다음 드럼으로 팁 거리를 좁힐 때만 (prev_d - curr_d).
        # per_arm_* 는 직전 _get_observations 에서 계산 (1스텝 지연). 노는 팔은 self.arm_idle
        idle_approach = torch.zeros(self.num_envs, device=self.device)
        for a in range(2):
            tgt = self.per_arm_pos[:, a, 0, :]                                   # (N,3) 그 팔 담당 첫 노트
            gate = self.per_arm_valid[:, a, 0] & self.arm_idle[:, a]             # 노는 팔 & 유효 타겟
            curr_d = torch.linalg.norm(tip_pos[:, a, :] - tgt, dim=-1)          # (N,)
            prev_d = torch.linalg.norm(prev_tip_pos[:, a, :] - tgt, dim=-1)
            idle_approach = idle_approach + torch.where(gate, prev_d - curr_d, torch.zeros_like(curr_d))
        reward = reward + self.reward_computer.cfg.w_idle_approach * idle_approach

        # 가중치 업데이트
        self.reward_computer.update_difficulty_weights(success=success, missed_target=missed_target)

        # 타이밍 정밀도 진단: 성공 타격의 타겟 시각 오차 (ms). time_error 는 |스텝차이| (매칭 안되면 -1)
        te_valid = (time_error >= 0).float()                                # (N, M)
        te_ms = time_error.float() * (self.step_dt * 1000.0)                # 스텝을 ms 로
        te_sum = (te_ms * te_valid).sum(dim=-1)                             # (N,) 유효분 합
        te_cnt = te_valid.sum(dim=-1)                                       # (N,) 유효 개수
        rate_log_terms["avg_time_error_ms"] = torch.stack([te_sum, te_cnt], dim=-1)  # 합/개수 비율

        # [진단] 조기/정시/지연 비율과 부호 있는 평균 오차 [ms] (음수면 전체적으로 일찍 치는 것)
        _tm = time_error_signed > -99                                         # (N,M) 성공 매칭 셀
        _n_t = _tm.float().sum(dim=-1)                                        # (N,)
        for _tag, _c in (("t_early", time_error_signed < 0),
                         ("t_ontime", time_error_signed == 0),
                         ("t_late",  (time_error_signed > 0) & _tm)):
            rate_log_terms[_tag] = torch.stack([(_c & _tm).float().sum(dim=-1), _n_t], dim=-1)
        _ts = torch.where(_tm, time_error_signed.float(), torch.zeros_like(_tm, dtype=torch.float32))
        rate_log_terms["t_signed_ms"] = torch.stack(
            [(_ts * self.step_dt * 1000.0).sum(dim=-1), _n_t], dim=-1)

        # [진단] miss 3분할: 팁이 타겟 윈도우 동안 어디까지 갔나 (분모 = miss 수)
        #   reach = xy 반경에 못 옴 / zpos = xy 는 왔는데 z 밖 / dir = xyz 박스는 왔는데 미등록
        missed_f = missed_target.float()
        missed_sum = missed_f.sum(dim=-1)                                    # (N,) 이번 스텝 총 miss 수
        rate_log_terms["miss3_reach"] = torch.stack([miss_reach.float().sum(dim=-1), missed_sum], dim=-1)
        rate_log_terms["miss3_zpos"]  = torch.stack([miss_zpos.float().sum(dim=-1),  missed_sum], dim=-1)
        rate_log_terms["miss3_dir"]   = torch.stack([miss_dir.float().sum(dim=-1),   missed_sum], dim=-1)

        # [진단] miss3_dir 원인 4분할 (분모 = miss3_dir 수). 상호배타가 아니라 합이 1 이 아니다
        #   armed = 재장전 안 됨 / down = 상승 중 통과 / slow = 하강이지만 min_impact_velocity 미달 / arb = 최근접 중재에 밀림
        dir_sum = miss_dir.float().sum(dim=-1)                               # (N,)
        for _tag, _m in (("armed", dir_armed), ("down", dir_down), ("slow", dir_slow), ("arb", dir_arb)):
            rate_log_terms[f"dir_{_tag}"] = torch.stack([_m.float().sum(dim=-1), dir_sum], dim=-1)

        # [진단] miss3_reach 세분화: 타겟 구간 중 그 드럼에 가장 가까웠던 XY 거리로 3분할 (near/mid/far, 합 1)
        reach_sum = miss_reach.float().sum(dim=-1)                           # (N,)
        for _tag, _m in (("near", reach_near), ("mid", reach_mid), ("far", reach_far)):
            rate_log_terms[f"reach_{_tag}"] = torch.stack([_m.float().sum(dim=-1), reach_sum], dim=-1)

        # [진단] miss3_zpos 세분화: above = 드럼 위에 떠 있었다 / below = 드럼면 아래로 지나갔다 / both = 둘 다
        #   zpos_gap = XY 반경 안에서의 최소 |z 오차| [m]
        zpos_sum = miss_zpos.float().sum(dim=-1)                             # (N,)
        for _tag, _m in (("above", zpos_above), ("below", zpos_below), ("both", zpos_both)):
            rate_log_terms[f"zpos_{_tag}"] = torch.stack([_m.float().sum(dim=-1), zpos_sum], dim=-1)
        rate_log_terms["zpos_gap_m"] = torch.stack([zpos_gap.sum(dim=-1), zpos_sum], dim=-1)

        # [진단] 타격 순간 어깨 각도: 친 팔의 어깨1(방위축)/어깨2(앙각축)가 가동범위의 몇 % 지점인가 (1.0 = 한계)
        sh1_ids = self.arm_gross_ids[:, 0]                                  # (2,) [Lsh1,Rsh1]
        sh2_ids = self.arm_gross_ids[:, 1]                                  # (2,) [Lsh2,Rsh2]
        rng1 = (joint_high[..., sh1_ids] - joint_low[..., sh1_ids]).clamp(min=1e-6)
        rng2 = (joint_high[..., sh2_ids] - joint_low[..., sh2_ids]).clamp(min=1e-6)
        sh1_frac = (robot_pos[:, sh1_ids] - joint_low[..., sh1_ids]) / rng1  # (N,2) 0~1
        sh2_frac = (robot_pos[:, sh2_ids] - joint_low[..., sh2_ids]) / rng2  # (N,2)
        hit_any_arm = hit_per_arm.any(dim=2)                                # (N,2) 이 스텝 그 팔이 침
        hit_cnt = hit_any_arm.float().sum(dim=1)                            # (N,)
        rate_log_terms["hit_sh1_frac"] = torch.stack([(hit_any_arm.float() * sh1_frac).sum(dim=1), hit_cnt], dim=-1)
        rate_log_terms["hit_sh2_frac"] = torch.stack([(hit_any_arm.float() * sh2_frac).sum(dim=1), hit_cnt], dim=-1)
        any_hit = hit_any_arm.any(dim=1).float()                            # (N,) 이 스텝 타격 발생
        hL = hit_any_arm[:, 0].float(); hR = hit_any_arm[:, 1].float()

        # [진단] 몸통 비틀림: 두 어깨1 각도의 합이 180deg 에서 벗어난 정도.
        #   두 shoulder_1 은 같은 기준축이라 각각 90deg 면 그 팔이 몸통 정면, 합이 180deg 면 좌우 대칭.
        RAD2DEG = 180.0 / 3.141592653589793
        sh1_dev = (robot_pos[:, sh1_ids] - 1.5707963267948966) * RAD2DEG     # (N,2) 각 팔의 정면 대비 편차 [deg]
        sh1_sum_dev = sh1_dev.sum(dim=-1)                                    # (N,) = (sh1_L + sh1_R) - 180deg
        rate_log_terms["hit_sh1_sumdev"] = torch.stack([sh1_sum_dev * any_hit, any_hit], dim=-1)
        rate_log_terms["hit_sh1_devL"] = torch.stack([sh1_dev[:, 0] * hL, hL], dim=-1)  # 왼팔 타격 시 왼팔 편차
        rate_log_terms["hit_sh1_devR"] = torch.stack([sh1_dev[:, 1] * hR, hR], dim=-1)  # 오른팔 타격 시 오른팔 편차
        # [진단] 실제 타격 중 왼팔이 친 비율 (0.5 근처면 양손 균형, 한쪽으로 쏠리면 한손 연주)
        rate_log_terms["hit_left_share"] = torch.stack([hL, hL + hR], dim=-1)
        # 팔 벌어짐 = sh1_L - sh1_R [deg]. 0 = 둘 다 정면, 음수 = 안쪽으로 모임
        arm_open = sh1_dev[:, 0] - sh1_dev[:, 1]                             # (N,)
        rate_log_terms["hit_arm_open"] = torch.stack([arm_open * any_hit, any_hit], dim=-1)
        # 허리 사용량 (타격 순간 |허리각|)
        waist_abs = robot_pos[:, self.waist_idx].abs()                       # (N,) |허리각| [rad]
        rate_log_terms["hit_waist_abs"] = torch.stack([waist_abs * any_hit, any_hit], dim=-1)

        # [진단] 배정 변화
        #   assign_flip: h0 담당팔이 직전 스텝과 뒤바뀐 비율
        #   idle_tgt_jump: 노는 팔의 첫 타겟이 5cm 이상 순간이동한 비율
        flip = (self.use_left_h0 != self.prev_use_left_h0).float()               # (N,)
        ones_like_flip = torch.ones_like(flip)
        rate_log_terms["assign_flip"] = torch.stack([flip, ones_like_flip], dim=-1)
        #   assign_ok: 담당 팔이 기록된 판정 창 안 타겟을 친 타격 중 담당 팔이 친 비율
        has_assign = (assigned_now >= 0).unsqueeze(1)                                        # (N,1,M)
        rate_log_terms["assign_ok"] = torch.stack([(hit_ok_per_arm & has_assign).float().sum(dim=(1, 2)),
                                                   (hit_per_arm & has_assign).float().sum(dim=(1, 2))], dim=-1)
        #   assign_mismatch: 채점용으로 기록된 노트 중 지금 관측의 담당 팔이 기록과 다른 비율 (직전 관측 생성 때 센 값).
        #   크면 관측과 채점이 자주 어긋난다는 뜻이라 스케줄러가 기록을 따르게 하는 고정이 필요하다
        rate_log_terms["assign_mismatch"] = torch.stack([self.assign_mm_num, self.assign_mm_den], dim=-1)
        tgt_move = torch.linalg.norm(self.per_arm_pos[:, :, 0, :] - self.prev_arm_tgt0, dim=-1)  # (N,2)
        idle_valid = self.arm_idle & self.per_arm_valid[:, :, 0]                 # (N,2) 노는 팔 & 유효 타겟
        jumped = ((tgt_move > 0.05) & idle_valid).any(dim=1).float()             # (N,)
        rate_log_terms["idle_tgt_jump"] = torch.stack([jumped, ones_like_flip], dim=-1)
        # 노는 팔(맡은 노트 있음) 대기점 기준 진단 (팔별, 분모 = 노는 팔 수): 대기점 기준인 다음 노트 위치가 5cm 넘게 바뀌었나.
        #   대기점이 옮겨 간 비율이며, 이때의 이동은 보상에서 정지 벌점이 거리 조건으로 꺼져 있다
        idle_note = (self.arm_state == ARM_IDLE)                                 # (N,2)
        home_p = self.per_arm_pos[:, :, 0, :]                                    # (N,2,3)
        home_move = ((torch.linalg.norm(home_p - self.prev_home_p, dim=-1) > 0.05) & idle_note).float()   # (N,2)
        rate_log_terms["idle_home_move"] = torch.stack([home_move.sum(dim=1), idle_note.float().sum(dim=1)], dim=-1)
        self.prev_home_p = home_p.clone()

        self.prev_use_left_h0 = self.use_left_h0.clone()
        self.prev_arm_tgt0 = self.per_arm_pos[:, :, 0, :].clone()

        # 관절 한계 초과 진단 (실제 관절각 robot_pos vs specs 하드한계). 한 관절이라도 넘은 env 비율
        over_hi = (robot_pos - joint_high).clamp(min=0.0)          # (N,9) 상한 초과분 (rad)
        over_lo = (joint_low - robot_pos).clamp(min=0.0)           # (N,9) 하한 초과분 (rad)
        over = over_hi + over_lo                                   # (N,9) 총 초과분 (rad), 범위 안 = 0
        exceeded = (over > 1e-3).float()                           # (N,9) 초과 여부
        ones_n = torch.ones(self.num_envs, device=self.device)
        rate_log_terms["lim_exceed_any"] = torch.stack([(exceeded.sum(dim=-1) > 0).float(), ones_n], dim=-1)

        # 리바운드 적용 횟수 (스텝당 팔 수). 면 통과 기준이라 타격 사건 수와 다를 수 있다. 0 에 가까우면 정책이 면까지 안 내려가는 것
        if self.cfg.rebound_enabled:
            rate_log_terms["rebound_applied"] = torch.stack([rebound_arm.float().sum(dim=1), ones_n], dim=-1)

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

        # RDS 리셋. MIDI 커리큘럼: 초반 랜덤 위주, 후반 실제 MIDI 비중 램프인
        score_ratio = self._midi_curriculum_ratio()
        self.rds.reset(env_ids=env_ids, score_ratio=score_ratio, selection_strength=self.cfg.midi_selection_strength)

        # 로봇 자세 리셋
        init_robot_pos = self.robot_initializer.reset_init_pos(env_ids)
        joint_pos, joint_vel = self.robot_interface.reset(robot=self.robot, env_ids=env_ids, init_robot_pos=init_robot_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 손목 유한차분 시딩 해제. 새 에피소드 첫 obs 는 참값 사용
        self._wrist_fd_seeded[env_ids] = False

        # 지연 버퍼도 리셋 자세로 채움 (안 하면 새 에피소드 첫 스텝에 이전 에피소드 목표가 적용됨)
        #   env 별 지연 재추첨 (DR). min==max 이면 같은 값이 다시 들어가 고정 지연 유지
        if self.delay_max > 0:
            self.target_buf[:, env_ids, :] = joint_pos[:, self.robot_interface.get_ctrl_joint_ids()]
            if self.delay_max > self.delay_min:
                self.env_delay[env_ids] = torch.randint(
                    self.delay_min, self.delay_max + 1, (len(env_ids),),
                    device=self.device, dtype=torch.long,
                )

        # 리바운드 반발계수 DR (env 별 재추첨)
        if self.cfg.rebound_enabled and self.cfg.rebound_e_jitter > 0.0:
            _j = (torch.rand((len(env_ids), 1), device=self.device) * 2.0 - 1.0) * self.cfg.rebound_e_jitter
            self.rebound_e_env[env_ids] = (self.rebound_e.unsqueeze(0) + _j).clamp(0.0, 1.0)

        # tip 리셋
        tip_pos = self.robot_interface.get_body_pos(self.robot)
        self.hit_detector.reset(env_ids=env_ids, tip_pos=tip_pos)

        # 팔꿈치 baseline 리셋: 직전 타격 없음 = 현재 팔꿈치각 (리프트 0 에서 시작), apex 0
        elbow_angle = self.robot_interface.get_joint_pos(self.robot)[:, self.elbow_joint_ids]
        self.elbow_baseline[env_ids] = elbow_angle[env_ids]
        self.elbow_apex_buf[env_ids] = 0.0
        self.use_left_h0[env_ids] = True   # 스케줄러 h0 배정 초기화
        self.use_case1_h0[env_ids] = True
        self.last_hit_step[env_ids] = -10000   # 재장전 상태 초기화 (시작은 자유)
        self.arm_state[env_ids] = ARM_FREE     # 팔 상태 로그용 초기화 (첫 보상 계산에서 다시 판정됨)
        self.arm_idle[env_ids] = True
        # 마지막 친 드럼 시드: 시작 자세(드럼 위 hover) 아래 드럼 높이 (진단용)
        seed_drum = tip_pos[env_ids].clone()
        seed_drum[:, :, 2] = seed_drum[:, :, 2] - self.reward_computer.cfg.ready_height
        self.last_hit_drum_pos[env_ids] = seed_drum
        self.prev_use_left_h0[env_ids] = True  # 히스테리시스 기준 초기화 (use_left_h0 와 동일하게)
        self.prev_arm_tgt0[env_ids] = 0.0      # 진단용 직전 타겟 초기화
        self.prev_home_p[env_ids] = 0.0        # 진단용 직전 대기점 초기화

        # 텐서 변수들 리셋
        self._reset_tensors(env_ids)

        # 시각화 리셋
        self.visualizer.reset(self.drum_pos)

    # ============================================================
    # [Custom Functions]
    # Internal utility methods (NOT called by RL engine directly)
    # ============================================================

    def _midi_curriculum_ratio(self) -> float:
        # 평가: MIDI 비율 고정 (커리큘럼 무시)
        if self.cfg.eval_midi_ratio >= 0.0:
            return self.cfg.eval_midi_ratio

        # 성공률(EMA) 기반: start 전엔 순수 랜덤, start~full 선형 증가, 그 뒤 final 고정.
        # 래치: 한 번 올라간 비율은 성공률이 떨어져도 내려가지 않는다 (악보 분포가 성공률을 따라 흔들리지 않게)
        ema = self.reward_computer.success_ema
        start = self.cfg.midi_success_start
        frac = (ema - start) / max(1e-6, self.cfg.midi_success_full - start)
        frac = min(max(frac, 0.0), 1.0)
        self.midi_frac_latch = max(getattr(self, "midi_frac_latch", 0.0), frac)
        return self.cfg.midi_final_ratio * self.midi_frac_latch

    """ init """
    def _init_obs(self):
        # obs 차원 계산
        M = len(self.instruments.all)
        K = self.rds.cfg.num_hits

        self.obs_dim_joint_pos = 9                     # ctrl 관절 수
        self.obs_dim_joint_vel = 9
        self.obs_dim_tip       = 2 * 3                 # 양손 * xyz
        self.obs_dim_drum      = M * 3                 # 드럼 * xyz
        self.obs_dim_next      = K * (M + 3)           # lookahead K개 * (one-hot M + time + valid + 목표세기 u)
        self.obs_dim_armed     = 2 * M                 # 양손 * 드럼
        self.obs_dim_role      = 2                     # 팔별 역할 (활성=1, 노는팔=0)
        self.sched_notes       = 2                     # 팔별 담당 노트 개수 (forward-scan)
        self.obs_dim_per_arm   = 2 * self.sched_notes * 4   # 2팔 * NOTES * (pos xyz 3 + time 1)

        obs_dim_total = (
            self.obs_dim_joint_pos + self.obs_dim_joint_vel
            + self.obs_dim_tip + self.obs_dim_drum
            + self.obs_dim_next + self.obs_dim_armed
            + self.obs_dim_role + self.obs_dim_per_arm
        )

        # 팔별 역할 버퍼 (보상 함수 배정에서 갱신, 1스텝 지연으로 관측에 사용)
        self.arm_role = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        # 팔별 담당 스케줄 버퍼 (_get_observations 에서 갱신, _get_rewards 에서 사용)
        self.per_arm_pos = torch.zeros((self.num_envs, 2, self.sched_notes, 3), device=self.device, dtype=torch.float32)
        self.per_arm_time = torch.ones((self.num_envs, 2, self.sched_notes), device=self.device, dtype=torch.float32)
        self.per_arm_valid = torch.zeros((self.num_envs, 2, self.sched_notes), device=self.device, dtype=torch.bool)

        if obs_dim_total != self.cfg.observation_space:
            raise RuntimeError("observation space dim mismatch.")

        # joint
        joint_low, joint_high = self.robot_interface.get_limit()
        self.joint_center = 0.5 * (joint_low + joint_high)
        self.joint_half_range = 0.5 * (joint_high - joint_low) + 1e-6

        self.joint_vel_scale = torch.tensor(self.cfg.joint_vel_scale, device=self.device, dtype=torch.float32)

        # drum
        #   정규화 상수는 obs 에 들어가는 값과 같은 프레임이어야 한다. default_drum_pos 는 허리 기준이지만
        #   drum_pos / tip_pos 는 env 프레임(_reset_drum 에서 z 에 robot_waist_joint_offset_z 를 더함)이므로
        #   상수에도 같은 오프셋을 더한다. 프레임이 다르면 z 채널이 clamp 에 포화돼 높이를 관측하지 못한다.
        drum_min = self.default_drum_pos.min(dim=0).values.clone()
        drum_max = self.default_drum_pos.max(dim=0).values.clone()
        drum_min[2] = drum_min[2] + self.cfg.robot_waist_joint_offset_z
        drum_max[2] = drum_max[2] + self.cfg.robot_waist_joint_offset_z

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
            arm_role: torch.Tensor,
            per_arm_pos: torch.Tensor,
            per_arm_time: torch.Tensor,
        ) -> torch.Tensor:
        # joint_pos, joint_vel: (N, 9)
        # tip_pos: (N, 2, 3)
        # drum_pos: (N, M, 3)
        # next_hits: (N, K, M+3)  (targets M + time + valid + 목표세기 u)
        # hit_armed: (N, 2, M)
        # arm_role: (N, 2)  활성=1, 노는팔=0
        # per_arm_pos: (N, 2, NOTES, 3)  팔별 담당 다음 노트 드럼위치
        # per_arm_time: (N, 2, NOTES)    팔별 담당 노트 time_to_hit (없음=1.0)

        # normalize
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -10.0, 10.0)

        tip_pos_n = (tip_pos - self.drum_center) / self.drum_half_range
        tip_pos_n = torch.clamp(tip_pos_n, -1.5, 1.5)

        drum_pos_n = (drum_pos - self.drum_center) / self.drum_half_range
        drum_pos_n = torch.clamp(drum_pos_n, -1.5, 1.5)

        # 팔별 담당 노트: 위치는 드럼과 동일 정규화, time 은 이미 [0,1]. (N,2,NOTES,3+1)
        per_arm_pos_n = (per_arm_pos - self.drum_center.unsqueeze(1)) / self.drum_half_range.unsqueeze(1)
        per_arm_pos_n = torch.clamp(per_arm_pos_n, -1.5, 1.5)
        per_arm_feat = torch.cat([per_arm_pos_n, per_arm_time.unsqueeze(-1)], dim=-1)  # (N,2,NOTES,4)

        obs = torch.cat(
            [
                joint_pos_n,
                joint_vel_n,
                tip_pos_n.reshape(self.num_envs, self.obs_dim_tip),
                drum_pos_n.reshape(self.num_envs, self.obs_dim_drum),
                next_hits.reshape(self.num_envs, self.obs_dim_next),
                hit_armed.reshape(self.num_envs, self.obs_dim_armed),
                arm_role.reshape(self.num_envs, self.obs_dim_role),
                per_arm_feat.reshape(self.num_envs, self.obs_dim_per_arm),
            ],
            dim=-1
        )

        return obs

    """ func (_reset_idx) """
    def _reset_drum(self, env_ids: torch.Tensor):
        # 각 env/각 에피소드에서 악기 위치
        drum_pos = self.default_drum_pos.unsqueeze(0).repeat(len(env_ids), 1, 1)  # (M, 3) 에서 (1, M, 3) 으로 차원 추가 후 (N, M, 3) 으로 복제
        drum_pos = drum_pos + self.instruments.drum_noise_scale * torch.randn_like(drum_pos)
        drum_pos[:, :, 2] = drum_pos[:, :, 2] + self.cfg.robot_waist_joint_offset_z

        self.drum_pos[env_ids] = drum_pos

    def _reset_tensors(self, env_ids: torch.Tensor):
        # 스텝 리셋
        self.steps[env_ids] = 0

        # 채점용 담당 팔 기록 리셋 (새 악보)
        self.assign_frozen[env_ids] = -1
        self.assign_mm_num[env_ids] = 0.0
        self.assign_mm_den[env_ids] = 0.0

        # 팔별 역할 리셋 (에피소드 시작 시 stale 값 방지)
        self.arm_role[env_ids] = 0.0
        self.per_arm_pos[env_ids] = 0.0
        self.per_arm_time[env_ids] = 1.0
        self.per_arm_valid[env_ids] = False
