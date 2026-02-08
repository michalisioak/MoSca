import random
import torch
import numpy as np
from loguru import logger

SEED = 1234


def seed_everything(seed: int = SEED):
    logger.info(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
