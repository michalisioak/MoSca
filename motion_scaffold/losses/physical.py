import torch
from utils3d.torch import matrix_to_axis_angle

from motion_scaffold import MoSca, q2R


def compute_velocity_acceleration_loss(
    scf: MoSca, time_indices=None, detach_mask=None, reduce_type="mean", square=False
):
    if time_indices is None:
        time_indices = torch.arange(scf.total_frames).to(scf.device)
    assert time_indices.max() <= scf.total_frames - 1

    xyz_coordinates = scf.node_xyz[time_indices]
    rotation_world_to_inertial = q2R(scf._node_rotation[time_indices])

    if detach_mask is not None:
        detach_mask = detach_mask.float()[:, None, None]
        xyz_coordinates = xyz_coordinates.detach() * detach_mask + xyz_coordinates * (
            1 - detach_mask
        )
        rotation_world_to_inertial = (
            rotation_world_to_inertial.detach() * detach_mask[..., None]
            + rotation_world_to_inertial * (1 - detach_mask)[..., None]
        )

    xyz_velocity, angular_velocity, xyz_acceleration, angular_acceleration = (
        compute_vel_acc(xyz_coordinates, rotation_world_to_inertial)
    )

    if square:
        xyz_velocity, angular_velocity, xyz_acceleration, angular_acceleration = (
            xyz_velocity**2,
            angular_velocity**2,
            xyz_acceleration**2,
            angular_acceleration**2,
        )

    if reduce_type == "mean":
        loss_position_velocity, loss_orientation_velocity = (
            xyz_velocity.mean(),
            angular_velocity.mean(),
        )
        loss_position_acceleration, loss_orientation_acceleration = (
            xyz_acceleration.mean(),
            angular_acceleration.mean(),
        )
    elif reduce_type == "sum":
        loss_position_velocity, loss_orientation_velocity = (
            xyz_velocity.sum(),
            angular_velocity.sum(),
        )
        loss_position_acceleration, loss_orientation_acceleration = (
            xyz_acceleration.sum(),
            angular_acceleration.sum(),
        )
    else:
        raise NotImplementedError()

    return (
        loss_position_velocity,
        loss_orientation_velocity,
        loss_position_acceleration,
        loss_orientation_acceleration,
    )


def compute_vel_acc(xyz: torch.Tensor, R_wi: torch.Tensor):
    xyz_vel = (xyz[1:] - xyz[:-1]).norm(dim=-1)
    xyz_acc = (xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]).norm(dim=-1)

    delta_R = torch.einsum("tnij,tnkj->tnik", R_wi[1:], R_wi[:-1])
    ang_vel = matrix_to_axis_angle(delta_R).norm(dim=-1)
    ang_acc_mag = abs(ang_vel[1:] - ang_vel[:-1])
    return xyz_vel, ang_vel, xyz_acc, ang_acc_mag
