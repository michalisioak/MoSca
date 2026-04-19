import matplotlib.pyplot as plt
import imageio
import numpy as np


def save_error_video_colormap(error_array, output_path, cmap="inferno"):
    """
    Save error maps as colored videos using matplotlib colormaps.
    """
    error_array = np.asarray(error_array, dtype=np.float32)
    error_array = np.nan_to_num(error_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to 0-1 range
    vmin, vmax = error_array.min(), error_array.max()
    if vmax - vmin > 1e-8:
        normalized = (error_array - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(error_array)

    # Apply colormap to each frame
    colormap = plt.colormaps[cmap]
    colored_frames = []

    for i in range(normalized.shape[0]):
        frame = colormap(normalized[i])  # Returns RGBA (H, W, 4)
        frame_rgb = (frame[:, :, :3] * 255).astype(
            np.uint8
        )  # Remove alpha, convert to uint8
        colored_frames.append(frame_rgb)

    colored_frames = np.array(colored_frames)
    imageio.mimsave(output_path, list(colored_frames))
    print(f"Saved colored video to {output_path}")
