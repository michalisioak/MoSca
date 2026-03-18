#!/usr/bin/env python3
"""
Convert videos from ./data/iphone/<scene_name>/input.mp4 to 2fps
and save them to ./data/iphone2/<scene_name>/input.mp4
Uses mpeg4 encoder instead of libx264 for compatibility
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


def check_available_encoders():
    """Check which encoders are available"""
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        encoders = result.stdout

        if "libx264" in encoders:
            return "libx264"
        elif "mpeg4" in encoders:
            return "mpeg4"
        elif "libx265" in encoders:
            return "libx265"
        else:
            # Try to find any H.264 encoder
            for encoder in ["h264", "h264_v4l2m2m", "h264_vaapi"]:
                if encoder in encoders:
                    return encoder
            return None
    except:
        return None


def convert_video_to_2fps(input_path, output_path, encoder="mpeg4"):
    """
    Convert video to 2fps using ffmpeg
    """
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))

    # Base FFmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        input_path,  # Input file
        "-vf",
        "fps=2",  # Set framerate to 2fps
        "-y",  # Overwrite output file if exists
        output_path,
    ]

    # Add encoder-specific options
    if encoder == "libx264":
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            "fps=2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-y",
            output_path,
        ]
    elif encoder == "mpeg4":
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            "fps=2",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",  # Quality (1-31, lower is better)
            "-pix_fmt",
            "yuv420p",
            "-y",
            output_path,
        ]
    elif encoder == "libx265":
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            "fps=2",
            "-c:v",
            "libx265",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-y",
            output_path,
        ]

    try:
        print(f"🔄 Converting: {input_path} -> {output_path} (using {encoder} encoder)")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Successfully converted: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {input_path}:")
        if e.stderr:
            print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    # Base directories
    source_base = "./data/iphone"
    dest_base = "./data/iphone2"

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg is not installed or not in PATH.")
        sys.exit(1)

    # Check available encoders
    encoder = check_available_encoders()
    if not encoder:
        print("❌ No suitable video encoder found in your FFmpeg installation.")
        print("\nAvailable encoders in your FFmpeg:")
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True
            )
            print(result.stdout[:500] + "...")  # Show first 500 chars
        except:
            pass
        sys.exit(1)

    print(f"🎬 Starting video conversion to 2fps...")
    print(f"Source directory: {source_base}")
    print(f"Destination directory: {dest_base}")
    print(f"Using encoder: {encoder}")
    print(f"Scenes to process: {', '.join(SCENES)}")
    print("-" * 50)

    successful = 0
    failed = 0

    for scene in SCENES:
        input_path = os.path.join(source_base, scene, "input.mp4")
        output_path = os.path.join(dest_base, scene, "input.mp4")

        if convert_video_to_2fps(input_path, output_path, encoder):
            successful += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"✅ Conversion complete! Successful: {successful}, Failed: {failed}")

    if failed > 0:
        print("\n⚠️  Some conversions failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
