"""
Main Entry Point for the ANPR System
"""

import argparse
from anpr_system import ANPRSystem


def main():

    parser = argparse.ArgumentParser(description="ANPR System using Computer Vision")
    parser.add_argument("-i", "--image", help="Input image path")
    parser.add_argument("-d", "--directory", help="Input directory containing images")
    parser.add_argument("-v", "--video", help="Input video path")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show processing steps",
    )
    parser.add_argument("--lane", action="store_true", help="Enable lane detection")

    args = parser.parse_args()

    lane_type = "curved" if args.lane else None
    anpr = ANPRSystem(debug_mode=args.debug, lane_type=lane_type)

    if args.image:
        anpr.process_image(args.image)
    elif args.directory:
        anpr.process_directory(args.directory)
    elif args.video:
        anpr.process_video(args.video)
    else:
        print("Please provide input: -i for image, -d for directory, or -v for video")
        parser.print_help()


if __name__ == "__main__":
    main()
