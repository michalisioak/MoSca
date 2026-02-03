from typing import Optional
import torch
from utils3d.torch import matrix_to_quaternion


def init_qt(T: int, init_camera_pose: Optional[torch.Tensor], delta_flag: bool):
    if init_camera_pose is not None:
        if delta_flag:
            # T_w0, [T_01, T_12, ...]
            assert len(init_camera_pose) == T - 1
            delta_q0 = matrix_to_quaternion(torch.eye(3)[None])
            delta_q = matrix_to_quaternion(init_camera_pose[:, :3, :3])
            param_cam_q = torch.cat([delta_q0, delta_q], 0)
            delta_t0 = torch.zeros(3)[None]
            delta_t = init_camera_pose[:, :3, -1]
            param_cam_t = torch.cat([delta_t0, delta_t], 0)
        else:
            # construct independent: T_wc
            init_camera_pose = torch.as_tensor(init_camera_pose)
            param_cam_q = matrix_to_quaternion(init_camera_pose[:, :3, :3])
            param_cam_t = init_camera_pose[:, :3, -1]
    else:
        param_cam_q = torch.zeros(T, 4)
        param_cam_q[:, 0] = 1.0
        param_cam_t = torch.zeros(T, 3)
    return param_cam_q.float(), param_cam_t.float()
