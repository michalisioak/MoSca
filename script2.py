#!/usr/bin/env python3
"""
Extract frames from videos in ./data/iphone2/<scene_name>/input.mp4
and save them as images in ./data/iphone2/<scene_name>/images/ with sequential numbering
"""

import os
import subprocess
import sys
from pathlib import Path

# List of scenes to process
SCENES = ["spin", "teddy", "wheel", "apple", "block", "paper-windmill", "space-out"]


def ensure_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_frames_to_images(input_video_path, output_images_dir, start_number=1):
    """
    Extract all frames from a video and save them as sequentially numbered PNG images
    """
    if not os.path.exists(input_video_path):
        print(f"❌ Video file not found: {input_video_path}")
        return False

    # Ensure output directory exists
    ensure_directory(output_images_dir)

    # Check if there are already images in the directory
    existing_images = list(Path(output_images_dir).glob("*.png"))
    if existing_images:
        print(
            f"⚠️  Output directory {output_images_dir} already contains {len(existing_images)} images."
        )
        response = input("Do you want to overwrite them? (y/n): ")
        if response.lower() != "y":
            print("Skipping...")
            return False

    # FFmpeg command to extract frames as PNG images
    # Using %06d.png for 6-digit numbering (000001.png, 000002.png, etc.)
    output_pattern = os.path.join(output_images_dir, f"%06d.png")

    cmd = [
        "ffmpeg",
        "-i",
        input_video_path,  # Input video
        "-start_number",
        str(start_number),  # Starting number for sequence
        "-vf",
        "fps=2",  # Use video's original frame rate (extract all frames)
        "-pix_fmt",
        "rgb24",  # RGB pixel format for PNG
        "-vframes",
        "1000",  # Limit to first 1000 frames (remove for all frames)
        "-y",  # Overwrite output files
        output_pattern,
    ]

    # Alternative: extract ALL frames (remove the -vframes limit)
    cmd_all_frames = [
        "ffmpeg",
        "-i",
        input_video_path,
        "-start_number",
        str(start_number),
        "-vf",
        "fps=2",  # Keep this to respect the 2fps we set earlier
        "-pix_fmt",
        "rgb24",
        "-y",
        output_pattern,
    ]

    # Ask user how many frames to extract
    print(f"\n🎬 Video: {input_video_path}")
    print("Options:")
    print("1. Extract ALL frames")
    print("2. Extract first 100 frames")
    print("3. Extract first 500 frames")
    print("4. Extract first 1000 frames")
    print("5. Custom number of frames")

    choice = input("Select option (1-5): ").strip()

    if choice == "1":
        cmd = cmd_all_frames
        frame_limit = "all"
    elif choice == "2":
        cmd[cmd.index("-vframes") + 1] = "100" if "-vframes" in cmd else None
        frame_limit = "100"
    elif choice == "3":
        cmd[cmd.index("-vframes") + 1] = "500" if "-vframes" in cmd else None
        frame_limit = "500"
    elif choice == "4":
        cmd[cmd.index("-vframes") + 1] = "1000" if "-vframes" in cmd else None
        frame_limit = "1000"
    elif choice == "5":
        custom = input("Enter number of frames to extract: ").strip()
        if custom.isdigit():
            if "-vframes" in cmd:
                cmd[cmd.index("-vframes") + 1] = custom
            else:
                cmd.insert(cmd.index("-y"), "-vframes")
                cmd.insert(cmd.index("-vframes") + 1, custom)
            frame_limit = custom
        else:
            print("Invalid number, using all frames")
            cmd = cmd_all_frames
            frame_limit = "all"
    else:
        print("Invalid choice, using all frames")
        cmd = cmd_all_frames
        frame_limit = "all"

    try:
        print(f"🔄 Extracting {frame_limit} frames from: {input_video_path}")
        print(f"📁 Saving to: {output_images_dir}/000001.png, 000002.png, ...")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Count extracted frames
        extracted_files = list(Path(output_images_dir).glob("*.png"))
        print(
            f"✅ Successfully extracted {len(extracted_files)} frames to {output_images_dir}"
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting frames from {input_video_path}:")
        if e.stderr:
            print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def extract_all_frames_simple(input_video_path, output_images_dir):
    """
    Simplified version - extract all frames without asking options
    """
    if not os.path.exists(input_video_path):
        print(f"❌ Video file not found: {input_video_path}")
        return False

    ensure_directory(output_images_dir)

    output_pattern = os.path.join(output_images_dir, f"%06d.png")

    cmd = [
        "ffmpeg",
        "-i",
        input_video_path,
        "-start_number",
        "1",
        "-vf",
        "fps=2",  # Use the video's frame rate (should be 2fps from previous step)
        "-pix_fmt",
        "rgb24",
        "-y",
        output_pattern,
    ]

    try:
        print(f"🔄 Extracting all frames from: {input_video_path}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        extracted_files = list(Path(output_images_dir).glob("*.png"))
        print(
            f"✅ Successfully extracted {len(extracted_files)} frames to {output_images_dir}"
        )

        # Show first few and last few filenames
        if extracted_files:
            files_sorted = sorted(extracted_files)
            print(f"   First: {files_sorted[0].name}")
            if len(files_sorted) > 1:
                print(f"   Last:  {files_sorted[-1].name}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting frames from {input_video_path}:")
        if e.stderr:
            print(e.stderr[:500] + "..." if len(e.stderr) > 500 else e.stderr)
        return False


def main():
    # Base directory
    base_dir = "./data/iphone2"

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg is not installed or not in PATH.")
        sys.exit(1)

    print("🎬 Frame Extraction Tool")
    print("========================")
    print(f"Base directory: {base_dir}")
    print(f"Scenes found: {', '.join(SCENES)}")
    print()

    # Ask for extraction mode
    print("Extraction modes:")
    print("1. Interactive mode (ask for each scene)")
    print("2. Batch mode (extract all frames from all scenes)")

    mode = input("Select mode (1-2): ").strip()

    successful = 0
    failed = 0
    skipped = 0

    for scene in SCENES:
        video_path = os.path.join(base_dir, scene, "input.mp4")
        images_dir = os.path.join(base_dir, scene, "images")

        print(f"\n{'=' * 50}")
        print(f"Scene: {scene}")
        print(f"{'=' * 50}")

        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            skipped += 1
            continue

        if mode == "1":
            # Interactive mode
            if extract_frames_to_images(video_path, images_dir):
                successful += 1
            else:
                failed += 1
        else:
            # Batch mode - extract all frames
            if extract_all_frames_simple(video_path, images_dir):
                successful += 1
            else:
                failed += 1

    print(f"\n{'=' * 50}")
    print("✅ Extraction complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Skipped: {skipped}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
