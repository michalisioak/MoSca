from tapnet.torch import tapir_model
import torch.nn.functional as F
import torch


def preprocess_frames(frames: torch.Tensor):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


# def sample_random_points(frame_max_idx, height, width, num_points):
#   """Sample random points with (time, height, width) order."""
#   y = np.random.randint(0, height, (num_points, 1))
#   x = np.random.randint(0, width, (num_points, 1))
#   t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
#   points = np.concatenate((t, y, x), axis=-1).astype(
#       np.int32
#   )  # [num_points, 3]
#   return points


def postprocess_occlusions(occlusions: torch.Tensor, expected_dist: torch.Tensor):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles


def inference(frames: torch.Tensor, query_points, model: tapir_model.TAPIR):
    assert frames.dim() == 4, frames.shape
    assert frames.shape[-1] == 3, frames.shape
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    outputs = model(frames, query_points)
    tracks: torch.Tensor
    tracks, occlusions, expected_dist = (
        outputs["tracks"][0],
        outputs["occlusion"][0],
        outputs["expected_dist"][0],
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles
