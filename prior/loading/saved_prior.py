# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
from dataclasses import dataclass, field
import logging
import os
import os.path as osp
from glob import glob
from typing import Literal
import torch
import numpy as np
from torch import nn
import imageio
from tqdm import tqdm

from prior.filters import (
    filter_track,
    laplacian_filter_depth,
    LaplacianDepthFilterConfig,
    TrackFilterConfig,
)
from .epi import load_epi_error

from .RGB2INST import RGB2INST


@dataclass
class Saved2DConfig:
    laplacian_filter: LaplacianDepthFilterConfig = field(
        default_factory=LaplacianDepthFilterConfig
    )
    track_filter: TrackFilterConfig = field(default_factory=TrackFilterConfig)
    rgb_directory: str = "images"
    depth_directory: str = "depths"
    depth_model: Literal["metric3d", "unidepth", "depthcrafter", "depth_anything_3"] = (
        "depth_anything_3"
    )
    depth_min: float = 1e-3
    depth_max: float = 1000.0
    depth_median: float = 1.0
    depth_boundary_threshold: float = 1.0
    epi_directory: str = "epi"
    epi_model: Literal["RAFT"] = "RAFT"
    track_directory: str = "tracks"
    track_model: Literal["bootstapir", "spatracker", "spatracker2", "cotracker"] = (
        "bootstapir"
    )
    track_min_valid_count: int | None = 4


class Saved2D(nn.Module):
    def __init__(self, ws: str, cfg: Saved2DConfig) -> None:
        super().__init__()
        self.ws = ws
        self.cfg = cfg
        self.load_rgb()
        self.load_homo_coordinate_map()
        self.load_epi()
        self.load_dep()
        self.normalize_depth()
        self.recompute_dep_mask()
        self.load_track()
        # self.rescale_perframe_depth_from_bundle(
        #     bundle_pth_fn=osp.join(log_path, "bundle", "bundle.pth")
        # )
        # self.load_vos() TODO

    @torch.no_grad
    def load_rgb(self):
        img_dir = osp.join(self.ws, self.cfg.rgb_directory)
        img_npz = img_dir + ".npz"
        if osp.exists(img_npz):
            images = torch.load(img_npz)["images"].float()  # ! in [0,255]
            images = torch.from_numpy(images).float() / 255.0  # T,H,W,3
            img_names = [f"{i:05d}" for i in range(images.shape[0])]
        elif osp.exists(img_dir):
            img_fns = [
                f
                for f in os.listdir(img_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
            img_fns.sort()
            img_names = [osp.splitext(f)[0] for f in img_fns]
            images = [
                imageio.v3.imread(osp.join(img_dir, img_fn)) for img_fn in img_fns
            ]
            images = torch.Tensor(np.stack(images)) / 255.0  # T,H,W,3
        else:
            raise ValueError(f"Cannot find images in {img_dir}")
        # assign
        self.frame_names = img_names
        images = images[..., :3]
        self.rgb = nn.Buffer(images, requires_grad=False)
        return self

    @property
    def T(self):
        return self.rgb.shape[0]

    @property
    def H(self):
        return self.rgb.shape[1]

    @property
    def W(self):
        return self.rgb.shape[2]

    # load more data
    # todo: register the dyn mask on 2D
    @torch.no_grad()
    def load_epi(self):
        epi_dir = osp.join(self.ws, self.cfg.epi_directory, self.cfg.epi_model)
        if not osp.exists(epi_dir):
            logging.warning(
                f"Calling 2D EPI loading, but {epi_dir} not found, usually this means use track epi not optical flow epi, so skip!"
            )
            return self
        epipolar_errs = load_epi_error(epi_dir).float()  # T,H,W
        self.epi = nn.Buffer(epipolar_errs, requires_grad=False)
        self.has_epi = True
        return self

    @torch.no_grad()
    def load_dep(
        self,
        depth_min: float = 1e-3,
        depth_max: float = 1000.0,
        mask_depth_flag: bool = True,
    ):
        dep_dir = osp.join(self.ws, self.cfg.depth_directory, self.cfg.depth_model)
        if osp.isdir(dep_dir):
            dep = [
                np.load(osp.join(dep_dir, f"{img_name}.npz"))["dep"]
                for img_name in self.frame_names
            ]
            dep = np.stack(dep)  # T,H,W
        else:
            dep_fn = dep_dir + ".npz"
            assert osp.exists(dep_fn)
            dep = np.load(dep_fn)["dep"]
        if mask_depth_flag:
            dep_mask, _ = laplacian_filter_depth(dep, self.cfg.laplacian_filter)
            logging.info(f"Dep Boundary Mask {dep_mask.mean() * 100:.2f}%")
            dep_mask = (
                dep_mask * (dep > self.cfg.depth_min) * (dep < self.cfg.depth_max)
            )
            dep = np.clip(
                dep,
                self.cfg.depth_min,
                self.cfg.depth_max,
            )
        else:
            dep_mask = np.ones_like(dep) > 0

        # assign
        self.dep = nn.Buffer(torch.Tensor(dep), requires_grad=False)
        self.dep_mask = nn.Buffer(torch.Tensor(dep_mask).bool(), requires_grad=False)
        return self

    @torch.no_grad()
    def replace_depth(self, dep, dep_mask):
        dep_old = self.dep
        assert dep_old is not None
        assert dep.shape == dep_old.shape, f"{dep.shape} vs {dep_old.shape}"
        assert dep_mask.shape == self.dep_mask.shape, (
            f"{dep_mask.shape} vs {self.dep_mask.shape}"
        )
        self.dep = nn.Buffer(dep, requires_grad=False)
        self.dep_mask = nn.Buffer(dep_mask, requires_grad=False)
        return self

    @torch.no_grad()
    def recompute_dep_mask(self):
        dep = self.dep.cpu().numpy()
        dep_mask, _ = laplacian_filter_depth(dep, self.cfg.laplacian_filter)
        logging.info(f"Dep Boundary Mask {dep_mask.mean() * 100:.2f}%")
        dep_mask = dep_mask * (dep > self.cfg.depth_min) * (dep < self.cfg.depth_max)
        # dep[~dep_mask] = np.inf
        dep = np.clip(dep, self.cfg.depth_min, self.cfg.depth_max)
        self.dep = nn.Buffer(torch.Tensor(dep), requires_grad=False)
        self.dep_mask = nn.Buffer(torch.Tensor(dep_mask).bool(), requires_grad=False)
        return self

    @torch.no_grad()
    def load_track(self):
        assert hasattr(self, "dep_mask"), "Load DEP before TAP!"
        # * load from saved files
        long_track_fns = glob(
            osp.join(self.ws, self.cfg.track_directory, f"*{self.cfg.track_model}*.npz")
        )
        logging.info(f"Loading TAP from {long_track_fns}...")
        assert len(long_track_fns) > 0, "no TAP found!"
        track, track_mask = [], []
        for track_fn in long_track_fns:
            track_data = np.load(track_fn, allow_pickle=True)
            track.append(track_data["tracks"])
            track_mask.append(track_data["visibility"])
        # ! explicitly round to long
        track = torch.from_numpy(np.concatenate(track, 1)).float()  # T,N,2/3
        track[:, :, :2] = track[:, :, :2].long().float()
        track_mask = torch.from_numpy(np.concatenate(track_mask, 1)).bool()  # T,N
        track_mask = track_mask * (track[..., 0] >= 0) * (track[..., 1] >= 0)
        track_mask = track_mask * (track[..., 0] < self.W) * (track[..., 1] < self.H)
        assert track.shape[:2] == track_mask.shape
        assert len(track) == self.T
        # * filter the load tracks
        if (
            self.cfg.track_min_valid_count is not None
            and self.cfg.track_min_valid_count > 0
        ):
            track, track_mask = filter_track(
                track, track_mask, self.dep_mask.to(track.device), self.cfg.track_filter
            )

        # append the tracks
        if hasattr(self, "track"):
            self.track = torch.cat([self.track, track.to(self.track.device)], 1)
            self.track_mask = torch.cat(
                [self.track_mask, track_mask.to(self.track_mask.device)], 1
            )
        else:
            self.track = nn.Buffer(track.float(), requires_grad=False)  # 2D/3D track
        self.track_mask = nn.Buffer(track_mask, requires_grad=False)
        # ! warning, the track static mask is not saved here
        # assert (
        #     self.track.dtype == torch.long
        # ), "Must use Long type for track! to avoid later roudning error, otherwise the system will fail."

        # re-scale the 3rd depth if the depth is rescaled
        if self.track.shape[-1] == 3 and hasattr(self, "scale_nw"):
            logging.info("Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * self.scale_nw

        return self

    @torch.no_grad()
    def load_vos(self, vos_dirname: str = "vos_deva/Annotations"):
        vos_dir = osp.join(self.ws, vos_dirname)
        if not osp.exists(vos_dir):
            logging.warning(
                f"Calling 2D VOS loading, but {vos_dir} not found, usually this means the data is not available, so skip!"
            )
            return self
        logging.info(f"loading vos results from {vos_dirname}...")
        id_mask_list = []
        fn_list = sorted(os.listdir(vos_dir))
        for fn in fn_list:
            seg_fn = osp.join(vos_dir, fn)
            seg = imageio.imread(seg_fn)
            id_map = RGB2INST(seg)
            id_mask_list.append(id_map)
        id_mask_list = np.stack(id_mask_list, 0)
        unique_id = np.unique(id_mask_list)
        # remove 0 from unique id, which is unknown
        unique_id = unique_id[unique_id != 0]
        logging.info(
            f"loaded {len(unique_id)} unique ids with {len(id_mask_list)} frames."
        )
        self.vos = nn.Buffer(torch.from_numpy(id_mask_list), requires_grad=False)
        self.vos_id = nn.Buffer(torch.from_numpy(unique_id), requires_grad=False)
        self.has_vos = True
        return self

    @torch.no_grad()
    def load_flow(self, flow_dirname="flow_raft"):
        flow_list, flow_mask_list, src_t_list, dst_t_list = [], [], [], []
        flow_ij_to_listind_dict = {}
        flow_dir = osp.join(self.ws, flow_dirname)
        if not osp.exists(flow_dir):
            logging.warning(
                f"Calling 2D Flow loading, but {flow_dir} not found, usually this means the data is not available, so skip!"
            )
            return self
        for fn in tqdm(sorted(os.listdir(flow_dir))):
            if not fn.endswith(".npz"):
                continue
            flow_fn = osp.join(flow_dir, fn)
            flow_data = np.load(flow_fn, allow_pickle=True)
            flow, mask = flow_data["flow"], flow_data["mask"]
            src_t, dst_t = fn[:-4].split("_to_")
            src_t = self.frame_names.index(src_t[:-4])
            dst_t = self.frame_names.index(dst_t[:-4])
            flow_list.append(flow)
            flow_mask_list.append(mask)
            src_t_list.append(src_t)
            dst_t_list.append(dst_t)
            flow_ij_to_listind_dict[(src_t, dst_t)] = len(flow_list) - 1
        flow_list = np.stack(flow_list, 0)
        flow_mask_list = np.stack(flow_mask_list, 0)
        src_t_list = np.array(src_t_list)
        dst_t_list = np.array(dst_t_list)
        self.flow = nn.Buffer(torch.from_numpy(flow_list), requires_grad=False)
        self.flow_mask = nn.Buffer(
            torch.from_numpy(flow_mask_list), requires_grad=False
        )
        self.flow_src_t = nn.Buffer(torch.from_numpy(src_t_list), requires_grad=False)
        self.flow_dst_t = nn.Buffer(torch.from_numpy(dst_t_list), requires_grad=False)
        self.flow_ij_to_listind_dict = flow_ij_to_listind_dict
        return self

    @torch.no_grad()
    def register_2d_identification(self, static_2d_mask, dynamic_2d_mask):
        device = self.rgb.device
        self.sta_mask = nn.Buffer(
            static_2d_mask.float().to(device), requires_grad=False
        )
        self.dyn_mask = nn.Buffer(dynamic_2d_mask.float().to(device))
        logging.info(
            f"Saved2d register 2d identification: {static_2d_mask.sum()} static, {dynamic_2d_mask.sum()} dynamic; Unused: {(~static_2d_mask & ~dynamic_2d_mask).sum()}"
        )

    @torch.no_grad()
    def register_track_indentification(
        self, static_track_mask: torch.Tensor, dynamic_track_mask
    ):
        assert hasattr(self, "track"), "Load track before identify!"
        assert len(static_track_mask) == self.track.shape[1]
        device = self.track.device
        self.static_track_mask = nn.Buffer(static_track_mask.bool().to(device))
        logging.warning(
            f"Saved2d register track identification: {static_track_mask.sum()} static, {dynamic_track_mask.sum()} dynamic; Unused: {(~static_track_mask & ~dynamic_track_mask).sum()}"
        )
        return self

    @torch.no_grad()
    def get_mask_by_key(self, key):
        if key == "all":
            return torch.ones_like(self.dep_mask)
        elif key == "static":
            return self.sta_mask
        elif key == "dynamic":
            return self.dyn_mask
        elif key == "dep":
            return self.dep_mask
        else:
            raise ValueError(f"Unknown key={key}")

    @torch.no_grad()
    def rescale_depth(self, dep_scale):
        dep_scale = dep_scale.to(self.dep.device)
        assert len(dep_scale) == self.T, f"dep_scale:{len(dep_scale)} vs T:{self.T}"
        self.dep = self.dep.clone() * dep_scale[:, None, None]
        if self.track.shape[-1] == 3:
            logging.info("Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * dep_scale[:, None]
        return

    @torch.no_grad()
    def rescale_perframe_depth_from_bundle(self, bundle_pth_fn=None):
        if bundle_pth_fn is None:
            bundle_pth_fn = osp.join(self.ws, "bundle", "bundle.pth")
        bundle_data = torch.load(bundle_pth_fn)
        dep_scale = bundle_data["dep_scale"]
        logging.info(f"Rescale depth with {bundle_pth_fn} and scale={dep_scale}")
        self.rescale_depth(dep_scale)
        return self

    @torch.no_grad()
    def normalize_depth(self):
        # align the median of the depth to 1.0
        if self.cfg.depth_median < 0:
            scale_nw = 1.0
        else:
            world_depth = self.dep.clone()
            world_depth_fg = world_depth[self.dep_mask]
            world_depth_median = torch.median(world_depth_fg)
            scale_nw = (
                self.cfg.depth_median / world_depth_median
            )  # depth_normalized = scale_nw * depth_world
        self.scale_nw = float(scale_nw)
        assert hasattr(self, "dep_mask"), "Load DEP before noramlization!"
        # assert hasattr(self, "track"), "Load TAP before noramlization!"

        self.dep.mul(scale_nw)
        if hasattr(self, "track") and self.track.shape[-1] == 3:
            logging.info("Also align the 3D track with the depth scale")
            self.track[:, :, 2] = self.track[:, :, 2].clone() * scale_nw
        return self

    def load_homo_coordinate_map(self):
        # the grid take the short side has (-1,+1)
        if self.H > self.W:
            u_range = [-1.0, 1.0]
            v_range = [-float(self.H) / self.W, float(self.H) / self.W]
        else:  # H<=W
            u_range = [-float(self.W) / self.H, float(self.W) / self.H]
            v_range = [-1.0, 1.0]
        # make uv coordinate
        u, v = torch.meshgrid(
            torch.linspace(u_range[0], u_range[1], self.W),
            torch.linspace(v_range[0], v_range[1], self.H),
        )
        uv = torch.stack([u, v], dim=-1)  # H,W,2
        self.homo_map = nn.Buffer(uv, requires_grad=False)
