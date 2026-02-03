# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging

from backproject import backproject
from project import project
from monocular_cameras.init import init_fc, init_qt
from utils3d.torch import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
)

from dataclasses import dataclass


@dataclass
class CameraConfig:
    lr_q: float | None = 1e-4
    lr_t: float | None = 1e-4
    lr_f: float | None = 1e-4
    lr_c: float | None = 0.0


class MonocularCameras(nn.Module):
    def __init__(
        self,
        n_time_steps: int,
        default_H: int,
        default_W: int,
        fxfycxcy: Optional[
            torch.Tensor
        ] = None,  # intr init 1, fxfy in deg, cxcy in ratio [53.1, 53.1, 0.5, 0.5]
        K: Optional[torch.Tensor] = None,  # intr init 2, K is 3x3 mat
        #
        delta_flag: bool = True,
        init_camera_pose: Optional[
            torch.Tensor
        ] = None,  # either T_wc in indep model; or T_i(i+1) in delta model
        # * cam flag
        iso_focal: bool = False,
    ) -> None:
        # super().__init__()
        self.T = n_time_steps
        self.delta_flag = nn.Buffer(torch.tensor(delta_flag))
        self.iso_focal = nn.Buffer(torch.tensor(iso_focal))

        rel_focal, cxcy_ratio = init_fc(
            fxfycxcy, K, width=float(self.default_W), height=float(self.default_H)
        )
        self._rel_focal = nn.Parameter(rel_focal)  # ! both rel to short side=2
        self.cxcy_ratio = nn.Parameter(cxcy_ratio)  # ! separate ratio for H=1, W=1

        param_cam_q, param_cam_t = init_qt(self.T, init_camera_pose, delta_flag)
        self.q_wc = nn.Parameter(param_cam_q)
        self.t_wc = nn.Parameter(param_cam_t)

        self.default_H = nn.Buffer(torch.tensor(default_H))
        self.default_W = nn.Buffer(torch.tensor(default_W))
        self.summary()

    @property
    def rel_focal(self):
        if self.iso_focal:
            return self._rel_focal[0].repeat(2)
        else:
            return self._rel_focal

    def __len__(self):
        return self.T

    @classmethod
    def load_from_ckpt(cls, ckpt):
        logging.info("Load camera from checkpoint")
        H = ckpt["default_H"]
        W = ckpt["default_W"]
        delta_flag = ckpt["delta_flag"]
        if "iso_focal" not in ckpt.keys():
            logging.warning(
                "Load the old ckpt, by default in the old version, iso-focal is False"
            )
            ckpt["iso_focal"] = torch.tensor(False)
        if "_rel_focal" not in ckpt.keys():
            logging.warning("Load the old ckpt, use rel_focal instead of _rel_focal")
            ckpt["_rel_focal"] = ckpt["rel_focal"]
            del ckpt["rel_focal"]
        T = len(ckpt["q_wc"])
        cams = cls(n_time_steps=T, default_H=H, default_W=W, delta_flag=delta_flag)
        cams.load_state_dict(ckpt, strict=True)
        return cams

    def summary(self):
        logging.info(
            f"MonoCam Summary: T={self.T}, delta={self.delta_flag}, H={self.default_H}, W={self.default_W}"
        )

    ################################################################################
    # * Extrinsic
    ################################################################################
    def forward_T(self, until: Optional[int] = None):
        # until is not included
        if until is None:
            until = self.T
        dR = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
        dt = self.t_wc
        dT_list = torch.cat([dR, dt[..., None]], -1)
        bottom = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(dt)[None].expand(self.T, -1)
        dT_list = torch.cat([dT_list, bottom[:, None]], 1)
        ret = [dT_list[0]]  # ! the first frame can also be optimized
        for dT in dT_list[1:until]:
            # T = dT @ ret[-1] # old
            # ! different from the old version, here the order changed
            T = ret[-1] @ dT  # The saved is T[0] = T_w0, T[i] = T_(i-1)_i
            ret.append(T)
        ret = torch.stack(ret)
        return ret  # ->until,4,4

    def T_wc(self, ind: int):
        # * core function
        assert ind >= 0 and ind < self.T, f"Invalid index {ind} for T={self.T}"
        if self.delta_flag:
            T_list = self.forward_T(ind)
            assert len(T_list) == ind + 1
            return T_list[-1].float()
        else:
            R = quaternion_to_matrix(F.normalize(self.q_wc[ind : ind + 1], dim=-1))[0]
            t = self.t_wc[ind]
            T = torch.eye(4).to(R)
            T[:3, :3] = R
            T[:3, 3] = t
            return T.float()

    def T_cw(self, ind: int):
        T_wc = self.T_wc(ind)
        return torch.inverse(T_wc)

    def Rt_wc(self, ind: int):
        T = self.T_wc(ind)
        return T[:3, :3], T[:3, -1]

    def Rt_cw(self, ind: int):
        R_wc, t_wc = self.Rt_wc(ind)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw

    def T_wc_list(self):
        # * core function
        if self.delta_flag:
            return self.forward_T().float()
        else:
            R = quaternion_to_matrix(F.normalize(self.q_wc, dim=-1))
            t = self.t_wc
            ret = torch.cat([R, t[..., None]], -1)
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(ret)
            ret = torch.cat([ret, bottom[None, None, :].expand(len(R), -1, -1)], -2)
            return ret.float()

    def T_cw_list(self):
        T_wc = self.T_wc_list()
        T_cw = torch.inverse(T_wc)
        return T_cw

    def Rt_wc_list(self):
        T_wc_list = self.T_wc_list()
        R_wc = T_wc_list[:, :3, :3]
        t_wc = T_wc_list[:, :3, -1]
        return R_wc, t_wc

    def Rt_cw_list(self):
        R_wc, t_wc = self.Rt_wc_list()
        R_cw = R_wc.transpose(1, 2)
        t_cw = -torch.einsum("bij,bj->bi", R_cw, t_wc)
        return R_cw, t_cw

    def Rt_ij(self, i: int, j: int):
        T_wi = self.T_wc(i)
        T_wj = self.T_wc(j)
        T_ij = T_wi.inverse() @ T_wj
        R_ij = T_ij[:3, :3]
        t_ij = T_ij[:3, 3]
        return R_ij, t_ij

    @torch.no_grad()
    def disable_delta(self):
        if not self.delta_flag:
            logging.warning("Already in non-delta mode, skip")
            return

        T_wc_list = self.T_wc_list()
        param_cam_q = matrix_to_quaternion(T_wc_list[:, :3, :3])
        param_cam_t = T_wc_list[:, :3, -1]
        # ! directly modify ...
        self.q_wc.data = param_cam_q
        self.t_wc.data = param_cam_t
        self.delta_flag = ~self.delta_flag
        logging.info("Switch to independent mode")
        return

    ################################################################################
    # * Smoothness
    ################################################################################
    def smoothness_loss(
        self, t_list=None, shift_list=[1], weight_list=[1.0], square_flag=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # identify which frame to use
        if t_list is None:
            t_list = torch.arange(self.T).to(self.q_wc)
        else:
            t_list = torch.as_tensor(t_list).to(self.q_wc)
        t_list = t_list.sort()[0].long()
        T_wc_list = self.T_wc_list()
        T_wc_list = T_wc_list[t_list]
        assert len(shift_list) == len(weight_list)
        loss_rot, loss_trans = torch.tensor(0.0), torch.tensor(0.0)
        for delta_t, w in zip(shift_list, weight_list):
            T_a = T_wc_list[:-delta_t]
            T_b = T_wc_list[delta_t:]
            T_a_inv = T_a.inverse()
            dT = torch.einsum("tij,tjk->tik", T_a_inv, T_b)
            dR = dT[:, :3, :3]
            ang_vel = matrix_to_axis_angle(dR).norm(dim=-1)
            transl_vel = dT[:, :3, -1].norm(dim=-1)
            if square_flag:
                ang_vel = ang_vel**2
                transl_vel = transl_vel**2
            loss_rot = loss_rot + w * ang_vel.sum()
            loss_trans = loss_trans + w * transl_vel.sum()
        return loss_rot, loss_trans

    ################################################################################
    # * Intrinsic
    ################################################################################
    def K(self, H=None, W=None):
        if H is None and W is None:
            return self.default_K
        else:
            assert H is not None and W is not None, "H and W must be both provided"
        L = min(H, W)  # ! the rel means to rel to the short side
        fx = self.rel_focal[0] * L / 2.0
        fy = self.rel_focal[1] * L / 2.0
        cx = W * self.cxcy_ratio[0]
        cy = H * self.cxcy_ratio[1]
        K = torch.eye(3).to(self.rel_focal)
        K[0, 0] = K[0, 0] * 0 + fx
        K[1, 1] = K[1, 1] * 0 + fy
        K[0, 2] = K[0, 2] * 0 + cx
        K[1, 2] = K[1, 2] * 0 + cy
        return K

    @property
    def default_K(self):
        return self.K(self.default_H, self.default_W)

    def homo(self, H=None, W=None):
        if H is None:
            H = int(self.default_H)
        if W is None:
            W = int(self.default_W)
        uv = __get_homo_coordinate_map__(H, W)
        uv = torch.from_numpy(uv).float().to(self.rel_focal.device)
        return uv

    @property
    def fov(self):
        focal = self.rel_focal.detach().cpu().numpy()
        half_angle = np.arctan(1.0 / focal)
        angle = np.rad2deg(half_angle * 2.0)
        return angle

    def get_optimizable_list(self, cfg: CameraConfig):
        ret = []
        if cfg.lr_q is not None and cfg.lr_q > 0:
            ret.append({"params": [self.q_wc], "lr": cfg.lr_q, "name": "R"})
        if cfg.lr_t is not None and cfg.lr_t > 0:
            ret.append({"params": [self.t_wc], "lr": cfg.lr_t, "name": "t"})
        if cfg.lr_f is not None and cfg.lr_f > 0:
            ret.append({"params": [self._rel_focal], "lr": cfg.lr_f, "name": "f"})
        if cfg.lr_c is not None and cfg.lr_c > 0:
            ret.append({"params": [self.cxcy_ratio], "lr": cfg.lr_c, "name": "cxcy"})
        return ret

    def backproject(self, uv: torch.Tensor, d: torch.Tensor):
        return backproject(uv, d, self)

    def project(self, xyz: torch.Tensor, th: float = 1e-5):
        return project(xyz, self, th=th)

    def trans_pts_to_world(self, tid: int, pts_c: torch.Tensor):
        assert pts_c.shape[-1] == 3  # and pts_c.ndim == 2
        R, t = self.Rt_wc(tid)
        original_shape = pts_c.shape
        pts_w = torch.einsum("ij,nj->ni", R, pts_c.reshape(-1, 3)) + t
        return pts_w.reshape(*original_shape)

    def trans_pts_to_cam(self, tid: int, pts_w: torch.Tensor):
        assert pts_w.shape[-1] == 3  # and pts_w.ndim == 2
        R, t = self.Rt_cw(tid)
        original_shape = pts_w.shape
        pts_c = torch.einsum("ij,nj->ni", R, pts_w.reshape(-1, 3)) + t
        return pts_c.reshape(*original_shape)

    def get_homo_coordinate_map(
        self, height: Optional[int] = None, width: Optional[int] = None
    ):
        u_range = [0, 0]
        height_val = height if height is not None else int(self.default_H.item())
        width_val = width if width is not None else int(self.default_W.item())

        # the grid take the short side has (-1,+1)
        assert height_val is not None
        assert width_val is not None
        if height_val > width_val:
            u_range = [-1.0, 1.0]
            v_range = [-float(height_val) / width_val, float(height_val) / width_val]
        else:  # H<=W
            u_range = [-float(width_val) / height_val, float(width_val) / height_val]
            v_range = [-1.0, 1.0]
        # make uv coordinate
        u, v = torch.meshgrid(
            torch.linspace(u_range[0], u_range[1], width_val),
            torch.linspace(v_range[0], v_range[1], height_val),
        )
        uv = torch.stack([u, v], dim=-1)  # H,W,2
        return uv.to(self.rel_focal).to(self.rel_focal.device)


def __get_homo_coordinate_map__(H: int, W: int):
    # the grid take the short side has (-1,+1)
    if H > W:
        u_range = [-1.0, 1.0]
        v_range = [-float(H) / W, float(H) / W]
    else:  # H<=W
        u_range = [-float(W) / H, float(W) / H]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = np.meshgrid(
        np.linspace(u_range[0], u_range[1], W), np.linspace(v_range[0], v_range[1], H)
    )
    uv = np.stack([u, v], axis=-1)  # H,W,2
    return uv
