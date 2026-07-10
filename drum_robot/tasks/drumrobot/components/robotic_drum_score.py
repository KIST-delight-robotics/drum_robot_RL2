"""
목표 악보(RDS: robotic drum score) 클래스
"""

from __future__ import annotations

from dataclasses import dataclass, field
                            # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
from mido import MidiFile   # pyright: ignore[reportMissingImports]
import torch                # pyright: ignore[reportMissingImports]
from pathlib import Path
import random
import time

from .specs import EnvRuntimeSpec, Instruments 

GENERAL_MIDI_PERCUSSION_KEY_MAP = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    59: "Ride Cymbal 2",
    22: "Tom(22)",
    26: "Tom(26)",
    58: "58",
}

@dataclass
class RDSCfg:
    midi_folder_path: str = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/MIDIs"

    slow_factor: float = 1.5        # slow_factor=2 -> 0.5배속 / slow_factor=0.5 -> 2배속
    start_offset_steps: int = 20

    # 랜덤 생성 시 동시 타격(좌우 쌍) 비율. 0.0=단일만(1단계), 0.2~=2단계
    double_hit_ratio: float = 0.1

    note_to_drum_idx: dict = field(default_factory=lambda: {
        "Acoustic Snare": 0,
        "Electric Snare": 0,
        "Side Stick": 0,    # 임시로 스네어 타격

        "Low Floor Tom": 1,
        "High Floor Tom": 1,
        
        "Low Tom": 2,
        "Low-Mid Tom": 2,

        "Hi Mid Tom": 3,
        "High Tom": 3,

        "Closed Hi Hat": 4,
        "Open Hi-Hat": 4,
        
        "Ride Cymbal 1": 5,
        "Ride Cymbal 2": 5,
        "Ride Bell": 5,     # 임시로 라이드 타격

        "Crash Cymbal 1": 7,    # 왼쪽 크러시
        "Crash Cymbal 2": 7,
        "Chinese Cymbal": 6,    # 임시로 오른쪽 크러시 타격
        "Splash Cymbal": 6,

        # 알 수 없는 거
        "Tom(22)": 3,
        "Tom(26)": 3,   # 확인 필요
    })

    pedal_instruments: set = field(default_factory=lambda: {
        # 페달로 타격하는 경우 예외 처리
        "Acoustic Bass Drum",
        "Bass Drum 1",
        "Pedal Hi-Hat",
        "58",   # 뭔지 모르겠음
    })

    # 타격 관측
    max_lookahead_s: float = 1.0    # 최대 관측 범위 (초)
    num_hits: int = 3               # 최대 관측 타격 개수

class RDS:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RDSCfg,
            env: EnvRuntimeSpec,
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.env = env

        # drum
        instruments = Instruments()
        self.num_drums = len(instruments.all)

        # 동시 타격용 좌우 드럼 쌍 (왼쪽 x<0 × 오른쪽 x>=0, 교차 불필요)
        xs = [inst.position[0] for inst in instruments.all.values()]
        left_drums = [i for i, x in enumerate(xs) if x < 0.0]
        right_drums = [i for i, x in enumerate(xs) if x >= 0.0]
        pairs = [(l, r) for l in left_drums for r in right_drums]
        self.double_pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)  # (P, 2)

        # midi
        random.seed(time.time_ns())
        self.rng = random.Random()

        self.midi_files = self._glob_midi_files()
        self.rds_dataset = self._build_rds_dataset(self.midi_files)
        self.score = self._compute_score(self.rds_dataset)

        # 목표 악보을 저장할 텐서
        N = env.num_envs
        T = env.episode_length_step
        M = self.num_drums

        self.rds = torch.zeros((N, T, M), device=self.device, dtype=torch.int64)
        self.rds_visit = torch.zeros((N, T, M), device=self.device, dtype=torch.bool)
        self.rds_visit_vel = torch.zeros((N, T, M), device=self.device, dtype=torch.float32)  # 타격 순간 하강 속도 기록

        # util
        self.env_arange = torch.arange(self.env.num_envs, device=self.device)
    
    def get_next_hits(self, step: torch.Tensor) -> torch.Tensor:
        """
        현재 step 이후 max_lookahead_step 안에 있는 다음 K개 타격 이벤트를 반환.

        Args:
            rds:  (N, T, M)  # RDS, time x drum multi-hot
            step: (N,)       # 현재 step

        Returns:
            next_hits_obs: (N, K, M + 2)
                [:, :, :M]     = target drum multi-hot
                [:, :, M]      = normalized time_to_hit, 0.0 ~ 1.0
                [:, :, M + 1]  = valid flag, 1이면 유효 이벤트, 0이면 없음
        """
        N, T, M = self.rds.shape
        L = self.env.max_lookahead_step
        K = self.cfg.num_hits

        # 현재 step부터 L step 이후까지의 index 생성
        offsets = torch.arange(L, device=self.device, dtype=torch.long)  # (L,) # offset=0을 포함
        idx = step.unsqueeze(1) + offsets.unsqueeze(0)              # (N, L)

        valid_time = (idx >= 0) & (idx < T)                         # (N, L)
        idx_clamped = idx.clamp(0, T - 1)

        # 미래 RDS window 추출
        future_rds = self.rds.gather(
            dim=1,
            index=idx_clamped.unsqueeze(-1).expand(-1, -1, M)
        )   # (N, L, M)

        # T 범위 밖은 0 처리
        future_rds = future_rds * valid_time.unsqueeze(-1).to(future_rds.dtype)

        # 해당 timestep에 하나 이상의 드럼 hit가 있으면 event
        event_mask = future_rds.sum(dim=-1) > 0  # (N, L)

        # 각 env별 event 순서 번호
        # event가 아닌 위치도 값은 생기지만 event_mask로 다시 걸러냄
        event_order = torch.cumsum(event_mask.to(torch.long), dim=1) - 1    # (N, L)  # torch.cumsum: dim 방향으로 누적 합

        # K개까지만 선택
        selected = event_mask & (event_order >= 0) & (event_order < K)  # (N, L)

        # 출력 버퍼 생성
        next_targets = torch.zeros((N, K, M), device=self.device, dtype=torch.float32)
        next_times = torch.ones((N, K, 1), device=self.device, dtype=torch.float32)
        next_valid = torch.zeros((N, K, 1), device=self.device, dtype=torch.float32)

        if selected.any():
            env_idx, time_idx = torch.where(selected)      # 선택된 event 위치  # (num_selected,)
            hit_idx = event_order[env_idx, time_idx]       # 몇 번째 hit인지, 0 ~ K-1

            # target multi-hot
            next_targets[env_idx, hit_idx] = future_rds[env_idx, time_idx].to(torch.float32)

            # normalized time_to_hit
            # offset 0이면 지금, offset L이면 horizon 끝
            time_norm = offsets[time_idx].to(torch.float32) / float(L - 1)
            time_norm = torch.clamp(time_norm, 0.0, 1.0)

            next_times[env_idx, hit_idx, 0] = time_norm
            next_valid[env_idx, hit_idx, 0] = 1.0

        next_hits = torch.cat(
            [next_targets, next_times, next_valid],
            dim=-1
        )   # (N, K, M+2)

        return next_hits
    
    def set_rds_visit(self, steps: torch.Tensor, hit_mask: torch.Tensor, hit_vel: torch.Tensor):
        self.rds_visit[self.env_arange, steps, :] = hit_mask        # 타격한 시간에 방문 처리
        self.rds_visit_vel[self.env_arange, steps, :] = hit_vel     # 그 시간의 타격 속도 기록

    def get_rds(self) -> torch.Tensor:
        return self.rds

    def get_rds_visit(self) -> torch.Tensor:
        return self.rds_visit

    def get_rds_visit_vel(self) -> torch.Tensor:
        return self.rds_visit_vel

    def reset(self, env_ids: torch.Tensor, score_ratio: float=0.5, selection_strength: float=0.5):
        N = len(env_ids)

        num_score = int(N * score_ratio)

        perm = torch.randperm(N, device=self.device)    # 0~N 사이의 정수를 무작위로 섞어서 텐서 만들기
        score_env_ids = env_ids[perm[:num_score]]
        rand_env_ids = env_ids[perm[num_score:]]

        outputs = []

        # score-based
        if num_score > 0:
            out_score = self._reset_midi(
                score_env_ids,
                selection_strength=selection_strength
            )
            outputs.append(out_score)

        # random-based
        if N - num_score > 0:
            out_rand = self._reset_random(rand_env_ids)
            outputs.append(out_rand)

        # reset
        robotic_drum_score = torch.cat(outputs, dim=0)
        self.rds[env_ids] = robotic_drum_score
        self.rds_visit[env_ids] = False
        self.rds_visit_vel[env_ids] = 0.0

    # ===== INIT =====
    def _glob_midi_files(self) -> list:
        folder = Path(self.cfg.midi_folder_path)
        midi_files = list(folder.glob("*.mid"))

        if not midi_files:
            raise ValueError("MIDI 파일이 없습니다!")

        return midi_files
    
    def _build_rds_dataset(self, midi_files: list) -> torch.Tensor:
        rds_list = []

        for file_path in midi_files:
            bpm, events = self._read_midi_file(file_path)

            end_event = False
            prev_t = 0.0

            while not end_event:
                rds, prev_t, end_event = self._generate_rds_from_midi(bpm, events, prev_t)

                if torch.any(rds > 0):
                    rds_list.append(rds)

        if len(rds_list) == 0:
            raise RuntimeError("No RDS segments were generated from MIDI files.")

        rds_tensors = torch.stack(rds_list, dim=0).to(self.device)   # (N, T, M)
        
        return rds_tensors

    def _read_midi_file(self, file_path) -> tuple[float, list]:
        mid = MidiFile(file_path)
        ticks_per_beat = mid.ticks_per_beat

        default_tempo = 500000  # 120 BPM
        tempo = default_tempo
        bpm = 60_000_000 / tempo

        events = []

        # 먼저 전체 트랙에서 템포 찾기
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    bpm = 60_000_000 / tempo
                    break
            if tempo != default_tempo:
                break

        # 이벤트 파싱
        for i, track in enumerate(mid.tracks):  # 대부분 단일 트랙인거 같으니 드럼인지 확인하고 하나만 읽자
            current_time = 0
            tempo = default_tempo  # 트랙별 초기화

            for msg in track:
                current_time += msg.time

                if msg.type == 'set_tempo':
                    tempo = msg.tempo

                if msg.type == 'note_on' and msg.velocity > 0 and msg.channel == 9:
                    note = msg.note
                    instrument = GENERAL_MIDI_PERCUSSION_KEY_MAP.get(note, f"Unknown({note})")

                    time_sec = (current_time * tempo) / (ticks_per_beat * 1_000_000)

                    events.append((time_sec, instrument))

        events.sort()   # 멀티 트랙인 경우 시간 순으로 정렬

        return bpm, events
    
    def _generate_rds_from_midi(
            self,
            bpm: float,
            events: list[tuple[float, str]],
            prev_t: float,
    ) -> tuple[torch.Tensor, float, bool]:
        T = self.env.episode_length_step
        M = self.num_drums

        seconds_per_beat = 60.0 / bpm
        measure_duration = 4 * seconds_per_beat  # 4/4

        robotic_drum_score = torch.zeros((T, M), device=self.device, dtype=torch.int64)

        first = True
        end = True
        next_t = 0.0

        for t, note in events:
            if t < prev_t:
                continue

            if first:
                first_t = t
                first = False

            if t > first_t + measure_duration:
                end = False
                next_t = t
                break

            # instrument → index
            if note not in self.cfg.note_to_drum_idx:
                if note not in self.cfg.pedal_instruments:
                    print(f"Warning: Unknown instrument {note}, skipping.")
                continue
            drum_idx = self.cfg.note_to_drum_idx[note]

            # 시간 → index
            time_idx = int(round(self.cfg.slow_factor * (t - first_t) / self.env.step_dt)) + self.cfg.start_offset_steps
            safe_T = T - self.env.hit_window_step   # episode 마지막 W step은 판정하지 않음으로 타격으로 채우지 않음
            if time_idx >= safe_T:
                continue  # episode 길이 초과 이벤트는 무시

            # 모든 env에 이벤트 적용
            robotic_drum_score[time_idx, drum_idx] = 1

        return robotic_drum_score, next_t, end

    def _compute_score(self, rds_tensors: torch.Tensor) -> torch.Tensor:
        # rds_tensors: (N, T, M), 0/1
        N, T, M = rds_tensors.shape

        # 1. 사용된 드럼 개수
        num_drum_hit = rds_tensors.sum(dim=1)   # (N, M)
        drum_hit_mask = num_drum_hit > 0.5
        num_target = drum_hit_mask.float().sum(dim=1)   # (N,)

        # 2. 드럼 변경 횟수
        switch_count = torch.zeros((N,), device=self.device)

        prev_note = torch.zeros((N, M), device=self.device)
        for i in range(T):
            curr_note = rds_tensors[:, i, :]  # (N, M)
            curr_hit = curr_note.sum(dim=1) > 0.5  # (N,)

            # 이전과 현재가 다르면 switch
            diff = (curr_note != prev_note).any(dim=1)  # (N,)
            switch = curr_hit & diff
            switch_count += switch.float()

            # 현재 타격이 있을 때만 prev 갱신
            prev_note = torch.where(curr_hit.unsqueeze(1), curr_note, prev_note)

        # 3. 스네어 비율
        num_total_hit = num_drum_hit.sum(dim=1).clamp(min=1e-6)
        snare_rate = num_drum_hit[:, 0] / num_total_hit

        # 4. 크러시
        crash_count = num_drum_hit[:, 6] + num_drum_hit[:, 7]
        crash_exist = (crash_count > 0).float()

        # 5. 최종 score
        w1, w2, w3, w4 = 1.0, 1.0, 1.0, 0.0
        score = w1 * num_target + w2 * switch_count + w3 * (1 - snare_rate) + w4 * crash_exist

        return score
    
    # ===== RESET =====
    def _reset_midi(self, env_ids: torch.Tensor, selection_strength: float=0.5) -> torch.Tensor:
        weights = self.score.clone()

        # selection_strength 랜덤성 조절 (0 <= selection_strength < 1)
        # 값이 클수록 스코어 기준으로 선별, 값이 0이면 완전 랜덤
        weights = torch.pow(weights, selection_strength)

        probs = weights / (weights.sum() + 1e-6)

        # probs를 기반으로 인덱스를 랜덤 샘플링
        idx = torch.multinomial(
            probs,
            num_samples=len(env_ids),
            replacement=True    # 같은 인덱스가 여러 번 뽑힐 수 있음
        )

        return self.rds_dataset[idx]

    def _reset_random(self, env_ids: torch.Tensor) -> torch.Tensor:
        N = len(env_ids)
        T = self.env.episode_length_step
        M = self.num_drums

        rds_rand = torch.zeros((N, T, M), device=self.device, dtype=torch.int64)

        s = self.cfg.start_offset_steps
        e = T - self.env.hit_window_step

        k = 10
        ### 윈도우 겹침 문제 해결 - 새로운 변수 정의############################
        W = self.env.hit_window_step
        min_gap = 2*W+1
        hit_zone_width = max(1, (e - s - (k - 1) * min_gap) // k)
        #################################################################

        for i in range(k):
            #si = (int)(s + (2 * i) * (e - s) / (2 * k - 1))
            #ei = (int)(s + (2 * i + 1) * (e - s) / (2 * k - 1)) 
            #time_rand = torch.randint(si, ei, (N,), device=self.device) #기존코드

            ###########################윈도우 겹침 문제 해결###########################
            si = s + i * (hit_zone_width + min_gap)
            ei = si + hit_zone_width
            if ei > e:
                break
            time_rand = torch.randint(si, max(si + 1, ei), (N,), device=self.device)
            ########################################################################

            env = torch.arange(N, device=self.device)

            # env별 단일/동시 결정
            is_double = torch.rand(N, device=self.device) < self.cfg.double_hit_ratio

            # 단일 타격: 아무 드럼 하나
            drum_single = torch.randint(0, M, (N,), device=self.device)
            single_mask = ~is_double
            rds_rand[env[single_mask], time_rand[single_mask], drum_single[single_mask]] = 1

            # 동시 타격: 좌우 쌍에서 하나 뽑아 두 드럼
            P = self.double_pairs.shape[0]
            if P > 0:
                pair_idx = torch.randint(0, P, (N,), device=self.device)
                pair = self.double_pairs[pair_idx]   # (N, 2)
                rds_rand[env[is_double], time_rand[is_double], pair[is_double, 0]] = 1
                rds_rand[env[is_double], time_rand[is_double], pair[is_double, 1]] = 1

        return rds_rand
