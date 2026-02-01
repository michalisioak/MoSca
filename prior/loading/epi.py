import os.path as osp
import glob
import torch


def load_epi_error(root_dir: str):
    saved_dir = osp.join(root_dir, "error")
    epi_fns = glob.glob(osp.join(saved_dir, "*.npy"))
    epi_fns.sort()
    epi_list: list[torch.Tensor] = []
    for fn in epi_fns:
        epi = torch.load(fn)
        epi_list.append(epi)
    epi = torch.stack(epi_list, dim=0)
    return epi
