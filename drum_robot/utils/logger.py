"""
로그 출력 클래스 

path: source/extensions/drum_robot/drum_robot/utils/logger.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
                        # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch            # pyright: ignore[reportMissingImports]
from tqdm import tqdm   # type: ignore


@dataclass
class LoggerCfg:
    interval: int = 2000              # n step마다 출력
    sample_env_id: Optional[int] = None  # None: 전체 mean, 정수: 특정 env만


class EnvLogger:

    def __init__(self, num_envs: int, device: torch.device | str, cfg: LoggerCfg):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self._reset()

    @torch.no_grad()
    def _reset(self):
        self.step = 0
        self._sum: Dict[str, torch.Tensor] = {}
        self._p_sum: Dict[str, torch.Tensor] = {}
        self._count = 0

    @staticmethod
    @torch.no_grad()
    def _to_env_vector(x: torch.Tensor) -> torch.Tensor:
        # (N,) -> (N,)
        if x.ndim == 1:
            return x
        # (N, k, ...) -> (N,)
        return x.mean(dim=tuple(range(1, x.ndim)))

    @torch.no_grad()
    def add(self, terms: Dict[str, torch.Tensor]):
        for k, v in terms.items():
            if not torch.is_tensor(v):
                continue

            v = self._to_env_vector(v)

            # (N,) 강제
            if v.ndim != 1 or v.shape[0] != self.num_envs:
                continue

            if k not in self._sum:
                self._sum[k] = torch.zeros(self.num_envs, device=self.device)
            self._sum[k] += v

        self._count += 1
        self.step += 1
    
    @torch.no_grad()
    def add_probability(self, terms: Dict[str, torch.Tensor]):
        # add 한번 호출 후 반드시 add_probability가 한번 호출된다고 가정
        for k, v in terms.items():
            if not torch.is_tensor(v):
                continue

            # (N,2) 강제
            if v.ndim != 2 or v.shape[0] != self.num_envs or v.shape[1] != 2:
                continue

            if k not in self._p_sum:
                self._p_sum[k] = torch.zeros((self.num_envs, 2), device=self.device)
            self._p_sum[k] += v

    @torch.no_grad()
    def maybe_flush(self, prefix: str = ""):
        if self.cfg.interval <= 0:
            return
        if self.step % self.cfg.interval != 0:
            return

        denom = float(self._count) + 1e-6

        # value 출력
        keys = self._sum.keys()
        parts = []

        for k in keys:
            avg_env = self._sum[k] / denom  # (N,)
            if self.cfg.sample_env_id is None:
                val = avg_env.mean()
            else:
                val = avg_env[int(self.cfg.sample_env_id)]
            parts.append(f"{k}={val.item():.3f}")

        # probability 출력
        p_keys = self._p_sum.keys()
        p_parts = []

        for k in p_keys:
            p_sum = self._p_sum[k]  # (N,2)
            num = p_sum[:,0].sum() / denom
            den = p_sum[:,1].sum() / denom
            n_total = p_sum[:,1].sum()   # 이 지표가 평균낸 총 표본 수(분모 누적, window 전체·전 env 합)
            p_parts.append(f"{k}={num/(den+1e-6):.3f}(n{n_total.item():.0f})")

        head = f"\n[STEP {self.step}]"
        if prefix:
            head += f" {prefix}"

        all_parts = parts + p_parts
        tqdm.write(head + " " + " | ".join(all_parts))

        # window reset
        for v in self._sum.values():
            v.zero_()
        for v in self._p_sum.values():
            v.zero_()
        self._count = 0

        