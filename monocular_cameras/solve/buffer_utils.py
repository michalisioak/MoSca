import torch


def query_buffers_by_track(
    image_buffer: torch.Tensor,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    default_value=0.0,
):
    # image_buffer: T,H,W,C; track: T,N,2
    assert image_buffer.ndim == 4 and tracks.ndim == 3 and visibility.ndim == 2
    assert len(image_buffer) == len(tracks) == len(visibility)
    T, H, W, C = image_buffer.shape
    N = tracks.shape[1]
    ret_buffer = torch.ones(T, N, C).to(image_buffer) * default_value

    for i in range(T):
        _uv = tracks[i][..., :2]
        _int_uv, _inside_mask = round_int_coordinates(_uv, H, W)
        _value = query_image_buffer_by_pix_int_coord(image_buffer[i].clone(), _int_uv)
        ret_buffer[i] = _value
    return ret_buffer


def round_int_coordinates(coord: torch.Tensor, H: int, W: int):
    ret = coord.round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask


def query_image_buffer_by_pix_int_coord(
    buffer: torch.Tensor, pixel_int_coordinate: torch.Tensor
):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    print(f"buffer shape: {buffer.shape}")
    print(f"pixel_int_coordinate: {pixel_int_coordinate}")
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    # if isinstance(ret, np.ndarray):
    #     ret = ret.copy()
    return ret


def prepare_track_homo_dep_rgb_buffers(
    rgb: torch.Tensor,
    dep: torch.Tensor,
    track: torch.Tensor,
    track_mask: torch.Tensor,
):
    # track: T,N,2, track_mask: T,N
    time, height, width, _ = rgb.shape
    homo_map = get_homo_coordinate_map(height, width)
    device = track.device
    homo_list, ori_dep_list, rgb_list = [], [], []
    for tid in range(time):
        _uv = track[tid]
        _int_uv, _inside_mask = round_int_coordinates(_uv, height, width)
        _dep = query_image_buffer_by_pix_int_coord(dep[tid].clone().to(device), _int_uv)
        _homo = query_image_buffer_by_pix_int_coord(
            homo_map.clone().to(device), _int_uv
        )
        ori_dep_list.append(_dep.to(device))
        homo_list.append(_homo.to(device))
        # for viz purpose
        _rgb = query_image_buffer_by_pix_int_coord(rgb[tid].clone().to(device), _int_uv)
        rgb_list.append(_rgb.to(device))
    rgb_list = torch.stack(rgb_list, 0)
    ori_dep_list = torch.stack(ori_dep_list, 0)
    homo_list = torch.stack(homo_list)
    ori_dep_list[~track_mask] = -1
    homo_list[~track_mask] = 0.0
    return homo_list, ori_dep_list, rgb_list


def get_homo_coordinate_map(height: int, width: int):
    # the grid take the short side has (-1,+1)
    if height > width:
        u_range = [-1.0, 1.0]
        v_range = [-float(height) / width, float(height) / width]
    else:  # H<=W
        u_range = [-float(width) / height, float(width) / height]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = torch.meshgrid(
        torch.linspace(u_range[0], u_range[1], width),
        torch.linspace(v_range[0], v_range[1], height),
    )
    uv = torch.stack([u, v], dim=-1)  # H,W,2
    return uv
