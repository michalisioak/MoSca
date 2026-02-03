import numpy as np
import torch


def sample_random_points(time: int, height: int, width: int, num_points: int):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, time, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def get_uniform_random_queries(
    time: int,
    height: int,
    width: int,
    points: int,
    mask_list: np.ndarray | None = None,
    interval: int = 1,
    shift: int = 0,
):
    queries = []
    key_inds = [i for i in range(shift, time) if i % interval == 0]
    if shift == 0 and time - 1 not in key_inds:
        key_inds.append(time - 1)
    if shift == 0 and 0 not in key_inds:
        key_inds = [0] + key_inds

    T = len(key_inds)

    mask_weight = None
    if mask_list is not None:
        mask_list = mask_list[key_inds]
        _count = (mask_list.reshape(T, -1) > 0).sum(-1)
        mask_weight = _count / _count.sum()
    for i, t in enumerate(key_inds):
        mask = np.ones((height, width), dtype=np.uint8)
        target_num = points / T
        if mask_list is not None:
            mask = mask_list[i]
            assert mask_weight is not None
            target_num = points * mask_weight[i]
        q = tracker_get_query_uv(
            mask,
            fid=t,
            num=int(target_num) * 3,
        )
        queries.append(q)
    queries = torch.cat(queries, 0)
    choice = torch.randperm(queries.shape[0])[:points]
    queries = queries[choice]
    return queries  # N,3


def tracker_get_query_uv(
    mask: np.ndarray,
    fid: int,
    num=1024,
):
    mask = mask
    H, W = mask.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # u=W ind, H ind
    uv = np.stack([u, v], -1)  # H,W,2
    fg_uv = uv[mask > 0]
    weight = mask[mask > 0]
    if num < fg_uv.shape[0] and num > 0:  # set num to -1 if use all
        choice = np.random.choice(len(weight), size=num, replace=False)
        # choice = np.random.choice(dyn_uv.shape[0], num, replace=False)
        fg_uv = fg_uv[choice]
    queries = torch.tensor(fg_uv).float()  # N,2
    queries = torch.cat([torch.ones(queries.shape[0], 1) * fid, queries], -1)  # N,3
    return queries
