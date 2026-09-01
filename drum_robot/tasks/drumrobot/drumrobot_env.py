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

        # 전역 정책 스텝 카운터 (MIDI 커리큘럼 램프용 — 매 _get_dones에서 +1)
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
            eval_midi_file=self.cfg.eval_midi_file,   # 평가용 (기본 "" → 학습과 동일)
            eval_full_song=self.cfg.eval_full_song,
            slow_factor=self.cfg.eval_slow_factor,    # 기본 1.5=학습과 동일. play1에서 override 가능
        )
        max_lookahead_step = int(rds_cfg.max_lookahead_s / self.step_dt)
        self.max_lookahead_step = max_lookahead_step             # 스케줄러: 정규화 time→스텝 환산용
        self.hit_window_step = hit_detector_cfg.hit_window_step  # 스케줄러: 재장전 t_rearm=2W+1
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

        # ===== sim 물리 관절한계를 specs.joint_limit 값으로 재설정 (전 9관절) =====
        #   USD 기본 한계가 ±180(사실상 무한)이라 실제 관절이 하드클램프 target을 관성 오버슈트(특히 compliant 손목).
        #   PhysX 하드 한계를 specs 값으로 → 물리적 초과 불가. specs.py 어느 값 바꿔도 전 관절 자동 반영.
        #   robot 프레임 한계 → USD 프레임(dir 부호변환; dir=-1이면 [lo,hi]→[-hi,-lo]로 구간 뒤집힘).
        ri = self.robot_interface
        _d = ri.dir_tensor.float()                                  # (1,9) ±1
        _usd_a = _d * ri.joint_low                                  # robot→usd
        _usd_b = _d * ri.joint_high
        _usd_low  = torch.minimum(_usd_a, _usd_b)                   # (1,9)
        _usd_high = torch.maximum(_usd_a, _usd_b)
        _limits = torch.stack([_usd_low, _usd_high], dim=-1)        # (1,9,2)
        _limits = _limits.expand(self.num_envs, -1, -1).contiguous()
        self.robot.write_joint_position_limit_to_sim(
            _limits, joint_ids=ri.get_ctrl_joint_ids(), warn_limit_violation=False,
        )
        print("[LIMIT] sim 물리 관절한계 = specs.joint_limit 로 재설정 (전 9관절)")

        # 관절별 action_scale (손목만 상한 크게 → 손목 스냅 가능). ctrl_joint_names는 resolve된 순서
        ctrl_joint_names = self.robot_interface.get_ctrl_joint_name()
        scale_list = [
            self.cfg.wrist_action_scale if "wrist" in name else self.cfg.action_scale
            for name in ctrl_joint_names
        ]
        self.action_scale_vec = torch.tensor(
            scale_list, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 9)

        # 제어 지연 버퍼 (물리 스텝 단위, 순환 인덱스). 0이면 미사용 — 기존 동작과 완전히 동일.
        #   _apply_action 이 물리 스텝마다 호출되므로 여기서 밀면 1/120초 해상도가 나옴.
        self.delay_steps = int(self.cfg.action_delay_steps)
        self.target_buf = torch.zeros(
            (self.delay_steps + 1, self.num_envs, len(ctrl_joint_names)),
            device=self.device, dtype=torch.float32,
        )
        self._delay_ptr = 0

        # 관절 한계-초과 로그용 짧은 이름 (ctrl 순서). lim/<이름> 으로 출력됨
        _short = {
            "waist_joint": "waist",
            "left_shoulder_1": "Lsh1", "left_shoulder_2": "Lsh2", "left_elbow": "Lelb",
            "right_shoulder_1": "Rsh1", "right_shoulder_2": "Rsh2", "right_elbow": "Relb",
            "left_wrist": "Lwr", "right_wrist": "Rwr",
        }
        self.limit_log_names = [_short.get(n, n) for n in ctrl_joint_names]

        # 손목 관절 인덱스 [왼, 오] (손목 각속도 계산용). hit_per_arm 팔 축(0=왼,1=오)과 맞춤
        name_to_idx = {name: i for i, name in enumerate(ctrl_joint_names)}
        self.wrist_joint_ids = torch.tensor(
            [name_to_idx["left_wrist"], name_to_idx["right_wrist"]],
            device=self.device, dtype=torch.long,
        )
        # 팔꿈치 관절 인덱스 [왼, 오] — 세기별 리프트 목표 매칭용 (자연스러운 "들었다 내리치기") [[intensity-teaching-design]]
        self.elbow_joint_ids = torch.tensor(
            [name_to_idx["left_elbow"], name_to_idx["right_elbow"]],
            device=self.device, dtype=torch.long,
        )
        # 팔별 gross(위치이동) 관절 인덱스 [왼3, 오3] = 어깨1·어깨2·팔꿈치. 손목/허리 제외.
        #   어깨 과꺾임 진단(hit_sh1_frac/hit_sh2_frac)에 사용
        self.arm_gross_ids = torch.tensor([
            [name_to_idx["left_shoulder_1"],  name_to_idx["left_shoulder_2"],  name_to_idx["left_elbow"]],
            [name_to_idx["right_shoulder_1"], name_to_idx["right_shoulder_2"], name_to_idx["right_elbow"]],
        ], device=self.device, dtype=torch.long)   # (2, 3)
        # [진단] 허리 인덱스 (타격 순간 몸통 회전량 관찰 — "허리 고정 → 어깨 과꺾임" 가설 확인용)
        self.waist_idx = name_to_idx["waist_joint"]
        # 리프트 baseline(직전 타격 팔꿈치각) + 스윙 중 리프트 apex(최대) 추적 버퍼
        self.elbow_baseline = torch.zeros((self.num_envs, 2), device=self.device)
        self.elbow_apex_buf = torch.zeros((self.num_envs, 2), device=self.device)
        # 스케줄러가 정한 h0 담당(왼팔=True). 보상 배정에 주입 (1스텝 지연으로 사용)
        self.use_left_h0 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.use_case1_h0 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # 동시타 case2 배정 (왼팔→idx0?)
        # 팔별 마지막 실제 타격 스텝 (스케줄러 재장전 시드용). 초기값=먼 과거 → 시작은 양팔 자유
        self.last_hit_step = torch.full((self.num_envs, 2), -10000, dtype=torch.int64, device=self.device)
        # 팔별 마지막으로 친 드럼 위치 — 스케줄러 배정이 없을 때 ready_pose 대기점 (움직이지 않는 기준)
        self.last_hit_drum_pos = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        # ready_pose 대기점의 안정성 진단용 (배정노트 ↔ 마지막친드럼 전환 점프까지 포착)
        self.prev_home_p = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # [진단] 배정 churn 측정용 직전 스텝 값 (노는 팔 헛스윙 원인: 배정 채터링 vs 파워페널티 완화 판별)
        self.prev_use_left_h0 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_arm_tgt0 = torch.zeros((self.num_envs, 2, 3), device=self.device)

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
        각 타격을 '현재위치 + 이동가용시간(window) + 재장전(busy)'로 배정 → 거리·간격·부하가 자동 통합.
          window = t_k − 그 팔 자유시각,  req_speed = travel/window,  busy_pen = BIG·relu(t_rearm − window)
          score = req_speed + busy_pen  → argmin. 방금 친 팔은 busy_pen↑ → 빠른 클러스터는 양팔 분배.
        동시 2타 이벤트는 두 드럼을 양 팔에 분배(min-cost split) → MIDI 화음도 두 손이 제대로 갈림.
        Returns: use_left_h0 (N,)[case1 h0 왼팔?], use_case1_h0 (N,)[case2: 왼팔→idx0(높은드럼)?],
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
        # 이벤트당 상위 2개 드럼 (보상 case2의 topk와 동일 정렬: 드럼 인덱스 큰 순)
        arange_m = torch.arange(M, device=self.device).view(1, 1, M).expand(N, K, M)
        masked_idx = torch.where(multihot > 0.5, arange_m, torch.full_like(arange_m, -1))
        top2 = torch.topk(masked_idx, k=2, dim=-1).values                 # (N,K,2)
        idx0 = top2[:, :, 0].clamp(min=0); idx1 = top2[:, :, 1].clamp(min=0)
        has2 = top2[:, :, 1] >= 0                                         # 둘째 드럼 존재 = 동시 2타
        pos0 = torch.gather(drum_pos, 1, idx0.unsqueeze(-1).expand(N, K, 3))  # (N,K,3) 높은 드럼
        pos1 = torch.gather(drum_pos, 1, idx1.unsqueeze(-1).expand(N, K, 3))  # (N,K,3) 둘째 드럼
        t_step = time_norm * max(L - 1.0, 1.0)                            # (N,K) 정규화→스텝

        proj = tip_pos.clone()                                            # (N,2,3) 팔 투영위치(시드=현재 팁)
        # 팔 자유시각(스텝, now=0 기준). 방금 실제로 친 팔은 재장전 남은만큼 +값 → busy로 잡혀 클러스터에서 교대 유도.
        free = ((self.last_hit_step - self.steps.unsqueeze(1)).float() + t_rearm).clamp(min=0.0)  # (N,2)
        use_left_h0 = torch.ones(N, dtype=torch.bool, device=self.device)
        use_case1_h0 = torch.ones(N, dtype=torch.bool, device=self.device)   # case2: True=왼팔→idx0
        per_arm_pos = drum_pos.new_zeros((N, 2, NOTES, 3))
        per_arm_time = time_norm.new_ones((N, 2, NOTES))                  # 기본 1.0 = 없음/멀리
        per_arm_valid = torch.zeros((N, 2, NOTES), dtype=torch.bool, device=self.device)
        count = torch.zeros((N, 2), dtype=torch.long, device=self.device)

        CROSS = self.cfg.sched_cross_penalty

        def arm_score(a, p, tk):
            w = tk - free[:, a]                                           # free 이후 여유. 음수면 아직 재장전 중
            # 교차 침범량[m]: 왼팔(a=0)이 우측(x>0) / 오른팔(a=1)이 좌측(x<0) 드럼을 맡을 때만 >0.
            #   드럼 x와 팔 정체성만으로 결정 → 현재 팁 위치 무관 → '이미 교차해 travel=0' 자기강화 고리를 끊음.
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
            # 단일: p0를 어느 팔에. k=0은 히스테리시스 적용 — 직전 담당팔을 유지하는 상대 마진.
            #   배정엔 관성이 없어 두 팁이 등거리면 mm 단위로 승자가 뒤집히고, proj 체인 때문에 양팔 큐 전체가 스왑됨.
            #   그게 idle 팔 타겟을 점프시켜 idle_approach가 왕복을 양방향 보상 → 헛스윙. k=0만 잡으면 뒤도 안정.
            if k == 0:
                h = self.cfg.sched_hysteresis
                choose_left_one = torch.where(
                    self.prev_use_left_h0,
                    sL0 <= sR0 * (1.0 + h),      # 직전=왼팔 → 유지 쪽에 관대
                    sL0 * (1.0 + h) <= sR0,      # 직전=오른팔 → 왼팔이 h만큼 확실히 나아야 뒤집힘
                )
            else:
                choose_left_one = sL0 <= sR0
            useA = (sL0 + sR1) <= (sL1 + sR0)                             # 2타: A(왼→p0,오른→p1) vs B(swap)
            if k == 0:
                use_left_h0 = torch.where(one, choose_left_one, use_left_h0)
                use_case1_h0 = torch.where(two, useA, use_case1_h0)
            # 팔별 이번 이벤트 타겟: 단일=담당팔만 p0 / 2타=양팔 각자 split
            left_gets = (one & choose_left_one) | two
            right_gets = (one & (~choose_left_one)) | two
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
        return use_left_h0, use_case1_h0, per_arm_pos, per_arm_time, per_arm_valid

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

        # 팔별 담당 스케줄 (reachability). obs + 노는팔 접근보상 + 보상 h0 배정(use_left)에 사용.
        # 1스텝 지연으로 다음 _get_rewards에서 사용 (arm_role과 동일한 지연 — 60Hz라 무시 가능).
        self.use_left_h0, self.use_case1_h0, self.per_arm_pos, self.per_arm_time, self.per_arm_valid = self._compute_arm_schedule(
            self.next_hits, tip_pos, drum_pos)

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
        if self.delay_steps > 0:
            # 순환 버퍼에 지금 목표를 넣고, D 스텝 전 목표를 꺼내 적용 (순수 정수 지연)
            n = self.delay_steps + 1
            self._delay_ptr = (self._delay_ptr + 1) % n
            self.target_buf[self._delay_ptr] = self.target_joint_pos
            target = self.target_buf[(self._delay_ptr - self.delay_steps) % n]
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

        # 팔꿈치 리프트: baseline(직전 타격 팔꿈치각) 대비 |현재각|. 스윙 중 apex(최대) 추적 → "얼마나 높이 들었나"
        elbow_angle = robot_pos[:, self.elbow_joint_ids]                        # (N, 2)
        elbow_lift = (elbow_angle - self.elbow_baseline).abs()                  # (N, 2)
        self.elbow_apex_buf = torch.maximum(self.elbow_apex_buf, elbow_lift)    # (N, 2) running max
        hit_elbow_apex_per_drum = (hit_per_arm.float() * self.elbow_apex_buf[:, :, None]).max(dim=1).values  # (N, M)

        # 타격 시간/속도/손목각속도/팔꿈치리프트apex 기록
        self.rds.set_rds_visit(
            steps=self.steps, hit_mask=hit_mask,
            hit_vel=hit_vel_per_drum, hit_wrist=hit_wrist_per_drum, hit_elbow_apex=hit_elbow_apex_per_drum,
            xy_in=xy_in_any, xyz_in=xyz_in_any,
        )

        # 다음 사이클 위해 타격한 팔의 baseline 갱신 + apex 리셋 (리프트 계산 후에)
        hit_arm_any = hit_per_arm.any(dim=-1)                                   # (N, 2)
        self.elbow_baseline = torch.where(hit_arm_any, elbow_angle, self.elbow_baseline)
        self.elbow_apex_buf = torch.where(hit_arm_any, torch.zeros_like(self.elbow_apex_buf), self.elbow_apex_buf)
        # 팔별 마지막 타격 스텝 갱신 (스케줄러가 재장전 중인 팔을 알도록). self.steps는 이미 증가된 현재 스텝
        self.last_hit_step = torch.where(hit_arm_any, self.steps.unsqueeze(1), self.last_hit_step)
        # 팔별 마지막 친 드럼 위치 갱신 (동시에 여러 드럼이면 인덱스 작은 쪽)
        hit_drum_idx = hit_per_arm.float().argmax(dim=-1)                                    # (N,2)
        picked_drum = torch.gather(self.drum_pos, 1, hit_drum_idx.unsqueeze(-1).expand(-1, -1, 3))
        self.last_hit_drum_pos = torch.where(hit_arm_any.unsqueeze(-1), picked_drum, self.last_hit_drum_pos)

        # 결과
        rds = self.rds.get_rds()
        rds_visit = self.rds.get_rds_visit()
        rds_visit_vel = self.rds.get_rds_visit_vel()
        rds_visit_wrist = self.rds.get_rds_visit_wrist()
        rds_visit_elbow_apex = self.rds.get_rds_visit_elbow_apex()
        rds_visit_xy = self.rds.get_rds_visit_xy()
        rds_visit_xyz = self.rds.get_rds_visit_xyz()
        rds_T = self.rds.get_rds_T()
        success, wrong_hit, missed_target, time_error, success_impact_vel, success_wrist_ang_vel, target_done, target_velocity, success_elbow_apex, target_T, miss_reach, miss_zpos, miss_dir = self.hit_detector.get_result(
            hit_mask=hit_mask,
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

        reward, log_terms, rate_log_terms, active_arm = self.reward_computer.compute(
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
            idle_tgt_time=self.per_arm_time[:, :, 0],
            last_drum_pos=self.last_hit_drum_pos,
        )

        # 팔별 역할 저장 (다음 스텝 관측에 사용 — 보행 위상클럭 격)
        self.arm_role = active_arm

        # 노는 팔 접근(progress) 보상: 자기 담당 다음 드럼으로 팁 거리를 좁힐 때만(prev_d - curr_d). hover 불가.
        # per_arm_* 는 직전 _get_observations에서 계산(1스텝 지연). active_arm<0.5 = 노는 팔.
        idle_approach = torch.zeros(self.num_envs, device=self.device)
        for a in range(2):
            tgt = self.per_arm_pos[:, a, 0, :]                                   # (N,3) 그 팔 담당 첫 노트
            gate = self.per_arm_valid[:, a, 0] & (active_arm[:, a] < 0.5)        # 노는 팔 & 유효 타겟
            curr_d = torch.linalg.norm(tip_pos[:, a, :] - tgt, dim=-1)          # (N,)
            prev_d = torch.linalg.norm(prev_tip_pos[:, a, :] - tgt, dim=-1)
            idle_approach = idle_approach + torch.where(gate, prev_d - curr_d, torch.zeros_like(curr_d))
        reward = reward + self.reward_computer.cfg.w_idle_approach * idle_approach

        # 가중치 업데이트
        self.reward_computer.update_difficulty_weights(success=success, missed_target=missed_target)

        # 타이밍 정밀도 진단: 성공 타격의 타겟 시각 오차 (ms). time_error는 |스텝차이|(매칭 안되면 -1)
        te_valid = (time_error >= 0).float()                                # (N, M)
        te_ms = time_error.float() * (self.step_dt * 1000.0)                # 스텝 → ms
        te_sum = (te_ms * te_valid).sum(dim=-1)                             # (N,) 유효분 합
        te_cnt = te_valid.sum(dim=-1)                                       # (N,) 유효 개수
        rate_log_terms["avg_time_error_ms"] = torch.stack([te_sum, te_cnt], dim=-1)  # 합/개수 비율

        # 동시타(case2) 성공률 진단: 같은 스텝에 2+ 드럼이 함께 마감된 그룹의 드럼별 성공률.
        # finalized = 이번 스텝 마감 타겟(성공∪미스). 한 env에서 2개↑ 마감 = 동시타 그룹.
        # n(=count)은 동시타 그룹 드럼 개수 → 빈도도 같이 보임. (3+ 동시타는 손2개라 최대 2/3라 rate 상한 있음)
        finalized = success.bool() | missed_target.bool()                   # (N, M)
        is_double = (finalized.sum(dim=-1, keepdim=True) >= 2)              # (N, 1) 동시 2+ 타겟 그룹
        dbl_succ = (success.bool() & is_double).sum(dim=-1).float()         # (N,)
        dbl_tgt  = (finalized & is_double).sum(dim=-1).float()             # (N,)
        rate_log_terms["double_success_rate"] = torch.stack([dbl_succ, dbl_tgt], dim=-1)

        # ===== [진단] miss 3분할: 팁이 타겟 윈도우 동안 어디까지 갔나 (분모=miss 수) =====
        #   reach = xy반경에 못 옴        → 진짜 리치/이동시간   → 처방: 궤적계획/속도
        #   zpos  = xy는 왔는데 z밖       → 접근 각도/타점높이   → 처방: 타점높이/접근각(한계무관)
        #   dir   = xyz박스는 왔는데 미등록 → 하강X/armed X/중재밀림 → 처방: 접근방향/재장전(한계무관)
        missed_f = missed_target.float()
        missed_sum = missed_f.sum(dim=-1)                                    # (N,) 이번 스텝 총 miss 수
        rate_log_terms["miss3_reach"] = torch.stack([miss_reach.float().sum(dim=-1), missed_sum], dim=-1)
        rate_log_terms["miss3_zpos"]  = torch.stack([miss_zpos.float().sum(dim=-1),  missed_sum], dim=-1)
        rate_log_terms["miss3_dir"]   = torch.stack([miss_dir.float().sum(dim=-1),   missed_sum], dim=-1)

        # ===== [진단] 타격 순간 어깨 과꺾임 =====
        #   sh1_frac(방위축) / sh2_frac(앙각축): 친 팔의 각도가 가동범위의 몇 % 지점인가 (1.0 = 한계에 붙음)
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

        # ===== [진단] 몸통 비틀림: 두 어깨1 각도의 합이 180°에서 벗어난 정도 =====
        #   IK 기하상 두 shoulder_1은 같은 기준(+x)에서 재므로 각각 90°면 그 팔이 몸통 정면.
        #   따라서 sh1_L + sh1_R = 180° = 두 팔이 몸통 기준 좌우 대칭 = 몸통이 팔들의 평균 방향을 봄.
        #   실측 확인(프로브): sideways(팔이 정면에서 벗어난 각) = sh1 − 90°, 허리 1rad = 몸통 1rad 회전.
        #   → 허리를 δ 돌리면 두 편차가 각각 −δ, 합은 −2δ 이동 ⇒ 허리는 '합편차의 절반'만 돌면 대칭 회복.
        #   개별 팔이 안쪽으로 뻗는 건 자연(어깨가 몸중심에서 26cm 밖) — '한쪽 쏠림'만 문제.
        RAD2DEG = 180.0 / 3.141592653589793
        sh1_dev = (robot_pos[:, sh1_ids] - 1.5707963267948966) * RAD2DEG     # (N,2) 각 팔의 정면 대비 편차[°]
        sh1_sum_dev = sh1_dev.sum(dim=-1)                                    # (N,) = (sh1_L+sh1_R) − 180°
        rate_log_terms["hit_sh1_sumdev"] = torch.stack([sh1_sum_dev * any_hit, any_hit], dim=-1)
        rate_log_terms["hit_sh1_devL"] = torch.stack([sh1_dev[:, 0] * hL, hL], dim=-1)  # 왼팔 타격 시 왼팔 편차
        rate_log_terms["hit_sh1_devR"] = torch.stack([sh1_dev[:, 1] * hR, hR], dim=-1)  # 오른팔 타격 시 오른팔 편차
        # 팔 벌어짐 = sh1_L − sh1_R [°].  0=둘 다 정면, 음수=안쪽으로 모임(스틱 근접 자세).
        #   waist_twist는 '합'만 봐서 대칭 모임/벌림을 전혀 감지 못함 — 이게 그 맹점을 메움
        arm_open = sh1_dev[:, 0] - sh1_dev[:, 1]                             # (N,)
        rate_log_terms["hit_arm_open"] = torch.stack([arm_open * any_hit, any_hit], dim=-1)
        # 허리 사용량 — 도달의 핵심 자유도(어깨가 0.26·sinθ 만큼 이동). ≈0이면 허리를 안 쓰는 것
        waist_abs = robot_pos[:, self.waist_idx].abs()                       # (N,) |허리각|[rad]
        rate_log_terms["hit_waist_abs"] = torch.stack([waist_abs * any_hit, any_hit], dim=-1)

        # ===== [진단] 배정 churn — 노는 팔 헛스윙의 원인 판별 =====
        #   assign_flip : h0 담당팔이 직전 스텝과 뒤바뀐 비율. 배정엔 히스테리시스가 없어 등거리 근처에서 채터링 가능.
        #                 k=0이 뒤집히면 proj 체인 때문에 양팔 큐 전체가 스왑됨.
        #   idle_tgt_jump: 노는 팔의 첫 타겟이 5cm 이상 순간이동한 비율. idle_approach(진척보상)는 타겟 고정일 때만
        #                 왕복이 상쇄되므로, 이 값이 크면 왕복이 양방향 모두 보상받아 헛스윙이 '학습'됨.
        #   둘 다 ~0 이면 배정은 안정 → 원인은 파워페널티 완화(0.1→0.05) 쪽.
        flip = (self.use_left_h0 != self.prev_use_left_h0).float()               # (N,)
        ones_like_flip = torch.ones_like(flip)
        rate_log_terms["assign_flip"] = torch.stack([flip, ones_like_flip], dim=-1)
        tgt_move = torch.linalg.norm(self.per_arm_pos[:, :, 0, :] - self.prev_arm_tgt0, dim=-1)  # (N,2)
        idle_valid = (active_arm < 0.5) & self.per_arm_valid[:, :, 0]            # (N,2) 노는 팔 & 유효 타겟
        jumped = ((tgt_move > 0.05) & idle_valid).any(dim=1).float()             # (N,)
        rate_log_terms["idle_tgt_jump"] = torch.stack([jumped, ones_like_flip], dim=-1)
        # ready_pose 대기점 진단: 스케줄러 배정을 받았나 / 대기점 자체가 튀었나
        #   idle_tgt_jump는 per_arm_pos만 봐서 has_tgt 전환(배정노트↔마지막친드럼) 점프를 놓침
        idle_has = torch.where(self.use_left_h0, self.per_arm_valid[:, 1, 0], self.per_arm_valid[:, 0, 0])
        sched_p  = torch.where(self.use_left_h0.unsqueeze(-1), self.per_arm_pos[:, 1, 0, :], self.per_arm_pos[:, 0, 0, :])
        last_p   = torch.where(self.use_left_h0.unsqueeze(-1), self.last_hit_drum_pos[:, 1], self.last_hit_drum_pos[:, 0])
        home_p   = torch.where(idle_has.unsqueeze(-1), sched_p, last_p)
        home_move = (torch.linalg.norm(home_p - self.prev_home_p, dim=-1) > 0.05).float()
        rate_log_terms["idle_has_tgt"]   = torch.stack([idle_has.float(), ones_like_flip], dim=-1)
        rate_log_terms["idle_home_move"] = torch.stack([home_move, ones_like_flip], dim=-1)
        self.prev_home_p = home_p.clone()

        self.prev_use_left_h0 = self.use_left_h0.clone()
        self.prev_arm_tgt0 = self.per_arm_pos[:, :, 0, :].clone()

        # ===== 관절 한계 초과 진단 (실제 관절각 robot_pos vs specs 하드한계) =====
        #   lim_exceed_any = 한 관절이라도 한계 넘은 env 비율 (top-line; 0이면 없음)
        #   lim/Lwr·Rwr    = 손목 한계 초과 비율 (compliant 손목만; stiff 관절은 항상 0이라 생략)
        over_hi = (robot_pos - joint_high).clamp(min=0.0)          # (N,9) 상한 초과분(rad)
        over_lo = (joint_low - robot_pos).clamp(min=0.0)           # (N,9) 하한 초과분(rad)
        over = over_hi + over_lo                                   # (N,9) 총 초과분(rad), 범위 안=0
        exceeded = (over > 1e-3).float()                           # (N,9) 초과 여부
        ones_n = torch.ones(self.num_envs, device=self.device)
        rate_log_terms["lim_exceed_any"] = torch.stack([(exceeded.sum(dim=-1) > 0).float(), ones_n], dim=-1)

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

        # RDS 리셋 — MIDI 커리큘럼: 초반 랜덤 위주(기본 타격 습득) → 후반 실제 MIDI 비중 램프인
        score_ratio = self._midi_curriculum_ratio()
        self.rds.reset(env_ids=env_ids, score_ratio=score_ratio, selection_strength=self.cfg.midi_selection_strength)

        # 로봇 자세 리셋
        init_robot_pos = self.robot_initializer.reset_init_pos(env_ids)
        joint_pos, joint_vel = self.robot_interface.reset(robot=self.robot, env_ids=env_ids, init_robot_pos=init_robot_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 지연 버퍼도 리셋 자세로 채움 — 안 하면 새 에피소드 첫 스텝에 '이전 에피소드 목표'가 적용됨
        if self.delay_steps > 0:
            self.target_buf[:, env_ids, :] = joint_pos[:, self.robot_interface.get_ctrl_joint_ids()]

        # tip 리셋
        tip_pos = self.robot_interface.get_body_pos(self.robot)
        self.hit_detector.reset(env_ids=env_ids, tip_pos=tip_pos)

        # 팔꿈치 baseline 리셋: 직전 타격 없음 → 현재 팔꿈치각으로 (리프트 0에서 시작), apex 0
        elbow_angle = self.robot_interface.get_joint_pos(self.robot)[:, self.elbow_joint_ids]
        self.elbow_baseline[env_ids] = elbow_angle[env_ids]
        self.elbow_apex_buf[env_ids] = 0.0
        self.use_left_h0[env_ids] = True   # 스케줄러 h0 배정 초기화
        self.use_case1_h0[env_ids] = True
        self.last_hit_step[env_ids] = -10000   # 재장전 상태 초기화 (시작은 자유)
        # 대기점 시드: 시작 자세(드럼 위 hover)를 그대로 대기점으로 → 에피소드 초반 ready_pose 과도현상 없음
        seed_drum = tip_pos[env_ids].clone()
        seed_drum[:, :, 2] = seed_drum[:, :, 2] - self.reward_computer.cfg.ready_height
        self.last_hit_drum_pos[env_ids] = seed_drum
        self.prev_use_left_h0[env_ids] = True  # 히스테리시스 기준 초기화 (use_left_h0와 동일하게)
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

        # 성공률(EMA) 기반: start 전엔 순수 랜덤 → start~full 선형 증가 → final 고정.
        # reward_computer가 유지하는 전체 성공률 EMA 사용. MIDI가 성공률 끌어내리면 비율도 자동 감소(자기조절).
        ema = self.reward_computer.success_ema
        start = self.cfg.midi_success_start
        frac = (ema - start) / max(1e-6, self.cfg.midi_success_full - start)
        frac = min(max(frac, 0.0), 1.0)
        return self.cfg.midi_final_ratio * frac

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
        self.obs_dim_role      = 2                     # 팔별 역할 (활성=1/노는팔=0) — 보행 위상클럭 격
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
        # 팔별 담당 스케줄 버퍼 (_get_observations에서 갱신, _get_rewards에서 노는팔 접근보상에 사용)
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
        # ⚠️ 프레임 정합 (2026-08-31 수정): default_drum_pos 는 허리 기준이지만, obs 에 실제로 들어가는
        #   drum_pos / tip_pos 는 env 프레임(_reset_drum 에서 z 에 robot_waist_joint_offset_z 를 더함)이다.
        #   상수를 허리 기준으로 만들면 z 분자가 half_range 의 ~5배가 되어 8개 드럼이 전부 clamp(+1.5)에 포화됐다.
        #   그 결과 z 채널 14개(tip 2 + drum 8 + per_arm 4)가 상수가 되어 정책이 높이를 관측하지 못했고,
        #   드럼 위치 DR(±2cm)의 z 성분도 clamp 에 묻혀 버려졌다. → 상수를 값과 같은 프레임으로 올린다.
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
        # arm_role: (N, 2)  활성=1/노는팔=0
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

        # 팔별 담당 노트: 위치는 드럼과 동일 정규화, time은 이미 [0,1]. (N,2,NOTES,3+1)
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
        drum_pos = self.default_drum_pos.unsqueeze(0).repeat(len(env_ids), 1, 1)  # (M, 3) -차원 추가-> (1, M, 3) -복제-> (N, M, 3)
        drum_pos = drum_pos + self.instruments.drum_noise_scale * torch.randn_like(drum_pos)
        drum_pos[:, :, 2] = drum_pos[:, :, 2] + self.cfg.robot_waist_joint_offset_z

        self.drum_pos[env_ids] = drum_pos

    def _reset_tensors(self, env_ids: torch.Tensor):
        # 스텝 리셋
        self.steps[env_ids] = 0

        # 팔별 역할 리셋 (에피소드 시작 시 stale 값 방지)
        self.arm_role[env_ids] = 0.0
        self.per_arm_pos[env_ids] = 0.0
        self.per_arm_time[env_ids] = 1.0
        self.per_arm_valid[env_ids] = False
