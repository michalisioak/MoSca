import numpy as np
from prior.optical_flow import OpticalFlowModel
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms as T


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            # T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch


@torch.no_grad()
class RAFT(OpticalFlowModel):
    def __init__(self):
        super().__init__()
        self.model = raft_large(pretrained=True, progress=False)
        self.model = self.model.eval()

    def forward(self, img_list: np.ndarray):
        res: torch.Tensor = self.model(torch.from_numpy(img_list))  # (N, 2, H, W)
        res = res.permute(2, 3, 1, 0).squeeze(-1)  # (H, W, 2)
        return res.cpu().numpy()
