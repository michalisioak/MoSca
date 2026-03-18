#!/usr/bin/env python3
"""
Simple script to copy camera JSON files with fixed subsampling factor of 15
(assuming original 30fps to target 2fps)
"""

import os
import json
import re
from pathlib import Path

# List of scenes to process
SCENES = ["spin", "teddy", "wheel", "apple", "block", "paper-windmill", "space-out"]

# If original video is 30fps and we want 2fps, keep 1 frame every 15 frames
SUBSAMPLE_FACTOR = 15


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_camera_files(cameras_dir):
    """Get all camera files sorted by camera index and frame number"""
    camera_files = []
    pattern = re.compile(r"(\d+)_(\d+)\.json$")

    for filename in os.listdir(cameras_dir):
        if filename.endswith(".json"):
            match = pattern.search(filename)
            if match:
                camera_idx = int(match.group(1))
                frame_num = int(match.group(2))
                camera_files.append((camera_idx, frame_num, filename))

    return sorted(camera_files, key=lambda x: (x[0], x[1]))


def copy_subsampled_cameras(source_dir, dest_dir):
    """Copy camera files, keeping every Nth frame based on subsample factor"""

    if not os.path.exists(source_dir):
        print(f"❌ Source not found: {source_dir}")
        return False

    ensure_directory(dest_dir)

    # Group files by camera
    cameras = {}
    for camera_idx, frame_num, filename in get_camera_files(source_dir):
        if camera_idx not in cameras:
            cameras[camera_idx] = []
        cameras[camera_idx].append((frame_num, filename))

    print(f"  Found {len(cameras)} camera(s)")

    total_copied = 0

    for camera_idx in sorted(cameras.keys()):
        frames = sorted(cameras[camera_idx], key=lambda x: x[0])

        # Keep every SUBSAMPLE_FACTOR-th frame
        kept_frames = []
        for i, (frame_num, filename) in enumerate(frames):
            if i % SUBSAMPLE_FACTOR == 0:
                kept_frames.append((i // SUBSAMPLE_FACTOR, frame_num, filename))

        print(f"  Camera {camera_idx}: {len(frames)} → {len(kept_frames)} frames")

        # Copy with new sequential numbering
        for new_idx, old_frame_num, filename in kept_frames:
            new_filename = f"{camera_idx}_{new_idx:05d}.json"
            src = os.path.join(source_dir, filename)
            dst = os.path.join(dest_dir, new_filename)

            # Copy and optionally update frame references
            try:
                with open(src, "r") as f:
                    data = json.load(f)

                # Update frame indices if they exist in the JSON
                if "frame_id" in data:
                    data["frame_id"] = new_idx
                if "frame_idx" in data:
                    data["frame_idx"] = new_idx

                with open(dst, "w") as f:
                    json.dump(data, f, indent=2)

                total_copied += 1
            except Exception as e:
                print(f"    Error copying {filename}: {e}")

    print(f"  ✅ Copied {total_copied} files to {dest_dir}")
    return True


def main():
    source_base = "./data/iphone"
    dest_base = "./data/iphone2"

    print("🎬 Copying subsampled camera files")
    print(
        f"Subsample factor: {SUBSAMPLE_FACTOR} (keep 1 frame every {SUBSAMPLE_FACTOR})"
    )
    print("-" * 50)

    successful = 0
    for scene in SCENES:
        source_cameras = os.path.join(source_base, scene, "cameras")
        dest_cameras = os.path.join(dest_base, scene, "cameras")

        print(f"\nScene: {scene}")
        if copy_subsampled_cameras(source_cameras, dest_cameras):
            successful += 1

    print(f"\n{'=' * 50}")
    print(f"Complete: {successful}/{len(SCENES)} scenes processed")


if __name__ == "__main__":
    main()
