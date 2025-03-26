import argparse
import os
import sys
from pathlib import Path

import cv2
import loguru


def extract_frames(video_path, output_dir, file_prefix="frame_"):
    """Extract frames from a video and save them as images.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where the frames will be saved.
        file_prefix (str, optional): Prefix for the output filenames. Defaults to "frame_".

    Returns:
        int: Number of frames extracted or -1 if failed.
    """
    # Check if video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.", file=sys.stderr)
        return -1

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        loguru.logger.error(f"Error: Could not open video file '{video_path}'.", file=sys.stderr)
        return -1

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    loguru.logger.info(f"Video FPS: {fps}")
    loguru.logger.info(f"Total frames: {frame_count}")
    loguru.logger.info(f"Duration: {duration:.2f} seconds")

    # Extract frames
    frame_number = 0
    extracted_count = 0

    while True:
        # Read next frame
        success, frame = video.read()

        # Break the loop if no more frames
        if not success:
            break

        # Save the frame as an image
        output_file = output_path / f"{file_prefix}{frame_number:06d}.png"
        cv2.imwrite(str(output_file), frame)

        frame_number += 1
        extracted_count += 1

        # Print progress every 100 frames
        if frame_number % 100 == 0:
            loguru.logger.info(f"Extracted {frame_number} frames...")

    # Release the video capture object
    video.release()

    loguru.logger.info(f"Extraction complete. Saved {extracted_count} frames to {output_dir}")
    return extracted_count


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video_path", help="Path to the video file.")
    parser.add_argument("--output_dir", help="Directory where the frames will be saved.")
    parser.add_argument("--prefix", default="frame_", help="Prefix for the output filenames.")

    args = parser.parse_args()

    # Extract frames
    extract_frames(args.video_path, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
