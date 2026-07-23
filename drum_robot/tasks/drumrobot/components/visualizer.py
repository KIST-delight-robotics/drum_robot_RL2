"""
시각화 클래스 
"""

from __future__ import annotations

from dataclasses import dataclass
                                    # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch                        # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils    # pyright: ignore[reportMissingImports]
import omni.usd                     # pyright: ignore[reportMissingImports]
from pxr import UsdGeom, Gf         # pyright: ignore[reportMissingImports]

from .specs import EnvRuntimeSpec, Instruments

@dataclass
class VisualizerCfg:
    """ 시각화 설정 """
    enable_visualization: bool = True

    drum_radius: float = 0.1
    drum_height: float = 0.01

    drum_near_color: tuple[float, float, float] = (1.0, 1.0, 0.0)
    drum_far_color: tuple[float, float, float] = (0.2, 0.2, 0.0)
    drum_base_color: tuple[float, float, float] = (0.2, 0.2, 0.2)

    hit_marker_radius: float = 0.05
    hit_marker_hidden_position: tuple[float, float, float] = (0.0, 0.0, -1.0)

    hit_marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)

class Visualizer():
    def __init__(
            self,
            device: torch.device | str,
            cfg: VisualizerCfg,
            env: EnvRuntimeSpec,
    ):
        self.device = device
        self.cfg = cfg
        self.env = env

        instruments = Instruments()
        self.num_drums = len(instruments.all)

    # =========================================================
    # Public Interface
    # =========================================================
    def init_visualization(self, drum_names: list):
        if self.cfg.enable_visualization:
            self._init_drum(drum_names)
            self._init_hit_marker()
    
    def step(self, tip_pos: torch.Tensor, next_hits: torch.Tensor, hit_per_arm: torch.Tensor):
        if self.cfg.enable_visualization:
            self._update_drum_color(next_hits)
            self._translate_hit_marker(tip_pos, hit_per_arm)
    
    def reset(self, drum_pos: torch.Tensor):
        if self.cfg.enable_visualization:
            self._translate_drum(drum_pos)
        
    # =========================================================
    # Domain Logic  (드럼로봇 시각화 요소별 업데이트)
    # =========================================================
    def _init_drum(self, drum_names):
        # 고속 업데이트를 위한 TranslateOp 캐시
        self._drum_translate_ops = []

        # 고속 업데이트를 위한 색 Primvar 캐시
        self._drum_color_ops = []

        for _, name in enumerate(drum_names):
            t_op, c_op = self._create_cylinder(
                node_name=name,
                radius=self.cfg.drum_radius,
                height=self.cfg.drum_height,
                color=self.cfg.drum_base_color,
            )

            self._drum_translate_ops.append(t_op)
            self._drum_color_ops.append(c_op)

    def _init_hit_marker(self):
        # 고속 업데이트를 위한 TranslateOp 캐시
        self._hit_marker_translate_ops = []

        for i in range(2):
            name = "hit_marker_" + str(i)

            t_op, _ = self._create_sphere(
                node_name=name,
                radius=self.cfg.hit_marker_radius,
                color=self.cfg.hit_marker_color,
            )

            self._hit_marker_translate_ops.append(t_op)

    def _update_drum_color(self, next_hits: torch.Tensor):
        M = self.num_drums
        L = self.env.max_lookahead_step
        W = self.env.hit_window_step

        hits = next_hits[:, :, :M]
        times = next_hits[:, :, M]
        valid = next_hits[:, :, M + 1] > 0.5

        for i in range(M):

            hit_mask = (hits[:, :, i] > 0.5) & valid            # (N, K)
            
            time_threshold = W / (L - 1)
            time_mask = times <= time_threshold                 # (N, K)

            near_mask = torch.any(hit_mask & time_mask, dim=1)  # (N,)
            far_mask = torch.any(hit_mask & ~time_mask, dim=1)  # (N,)

            # 우선순위: near > mid > far
            c_op = self._drum_color_ops[i]

            near_ids = [op for idx, op in enumerate(c_op) if near_mask[idx]]
            far_ids = [op for idx, op in enumerate(c_op) if ~near_mask[idx] & far_mask[idx]]
            none_ids = [op for idx, op in enumerate(c_op) if ~near_mask[idx] & ~far_mask[idx]]

            near_color = torch.tensor(self.cfg.drum_near_color, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(len(near_ids), 1)
            far_color = torch.tensor(self.cfg.drum_far_color, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(len(far_ids), 1)
            base_color = torch.tensor(self.cfg.drum_base_color, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(len(none_ids), 1)
        
            self._color(near_ids, near_color)
            self._color(far_ids, far_color)
            self._color(none_ids, base_color)

    def _translate_hit_marker(self, tip_pos_per_arm: torch.Tensor, hit_per_arm: torch.Tensor):
        # tip_pos:      (N, 2, 3)
        # hit_per_arm:  (N, 2, M)
        hit_mask_per_arm = torch.any(hit_per_arm, dim=2)    # (N, 2)

        for i in range(2):
            tip_pos = tip_pos_per_arm[:, i, :]
            hit_mask = hit_mask_per_arm[:, i]

            t_op = self._hit_marker_translate_ops[i]

            hit_ids = [op for idx, op in enumerate(t_op) if hit_mask[idx]]
            hidden_ids = [op for idx, op in enumerate(t_op) if ~hit_mask[idx]]

            hidden_pos = torch.tensor(
                self.cfg.hit_marker_hidden_position,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(len(hidden_ids), 1)

            self._translate(hidden_ids, hidden_pos)
            self._translate(hit_ids, tip_pos[hit_mask])

    def _translate_drum(self, drum_pos: torch.Tensor):
        for i in range(self.num_drums):
            p = drum_pos[:, i, :]
            self._translate(self._drum_translate_ops[i], p)

    # =========================================================
    # USD Primitives  (IsaacSim USD Stage 저수준 조작)
    # =========================================================
    def _create_sphere(self, node_name: str, radius: float, color: tuple[float, float, float]) -> tuple[list, list]:
        # 고속 업데이트를 위한 오퍼레이터 캐시
        translate_ops = []
        color_ops = []

        # sphere 설정
        sphere_cfg = sim_utils.SphereCfg(
            radius=radius,
        )

        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.env.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드

            xform_path = f"{viz_root}/{node_name}"
            stage.DefinePrim(xform_path, "Xform")

            sphere_path = f"{xform_path}/sphere"
            # IsValid 체크 후 sphere prim 생성
            if not stage.GetPrimAtPath(sphere_path).IsValid():
                sphere_cfg.func(sphere_path, sphere_cfg)

            # TranslateOp를 1회 생성하고 캐싱
            prim = stage.GetPrimAtPath(xform_path)
            xf = UsdGeom.Xformable(prim)    # prim이 transform 연산을 가질 수 있도록 감싸는 wrapper
            ops = xf.GetOrderedXformOps()   # 현재 들어있는 transform 연산 목록 가져오기
            if len(ops) > 0 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:    # 첫 번째 연산이 TranslateOp
                t_op = ops[0]
            else:
                # Clear 후 TranslateOp 추가
                xf.ClearXformOpOrder()
                t_op = xf.AddTranslateOp()

            translate_ops.append(t_op)

            # 색상 설정
            sphere_prim = stage.GetPrimAtPath(sphere_path)
            gprim = UsdGeom.Gprim(sphere_prim)
            pv = gprim.GetDisplayColorPrimvar()
            if not pv:
                pv = gprim.CreateDisplayColorPrimvar()
            pv.Set([color_vec])

            # 색 Primvar 캐싱
            color_ops.append(pv)

        return translate_ops, color_ops

    def _create_cylinder(self, node_name: str, radius: float, height: float, color: tuple[float, float, float]) -> tuple[list, list]:
        # 고속 업데이트를 위한 오퍼레이터 캐시
        translate_ops = []
        color_ops = []

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.env.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드

            xform_path = f"{viz_root}/{node_name}"
            stage.DefinePrim(xform_path, "Xform")

            cylinder_path = f"{xform_path}/cylinder"
            # IsValid 체크 후 cylinder prim 생성
            if not stage.GetPrimAtPath(cylinder_path).IsValid():
                stage.DefinePrim(cylinder_path, "Cylinder")
                cyl = UsdGeom.Cylinder(stage.GetPrimAtPath(cylinder_path))
                cyl.CreateRadiusAttr().Set(radius)
                cyl.CreateHeightAttr().Set(height)

            # TranslateOp를 1회 생성하고 캐싱
            prim = stage.GetPrimAtPath(xform_path)
            xf = UsdGeom.Xformable(prim)    # prim이 transform 연산을 가질 수 있도록 감싸는 wrapper
            ops = xf.GetOrderedXformOps()   # 현재 들어있는 transform 연산 목록 가져오기
            if len(ops) > 0 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:    # 첫 번째 연산이 TranslateOp
                t_op = ops[0]
            else:
                # Clear 후 TranslateOp 추가
                xf.ClearXformOpOrder()
                t_op = xf.AddTranslateOp()

            translate_ops.append(t_op)

            # 색상 설정
            cylinder_prim = stage.GetPrimAtPath(cylinder_path)
            gprim = UsdGeom.Gprim(cylinder_prim)
            pv = gprim.GetDisplayColorPrimvar()
            if not pv:
                pv = gprim.CreateDisplayColorPrimvar()
            pv.Set([color])

            # 색 Primvar 캐싱
            color_ops.append(pv)

        return translate_ops, color_ops

    def _translate(self, t_op: list, pos: torch.Tensor):
        n = len(pos[:,0])

        for i in range(n):
            p = pos[i]
            t_op[i].Set(
                Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
            )

    def _color(self, c_op: list, color: torch.Tensor):
        n = len(color[:,0])

        for i in range(n):
            c = color[i]
            c_op[i].Set(
                [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))]
            )