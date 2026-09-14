import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import sys
import time
from satark.utils.video import infer_video


def progress_callback(data, *args):
    if isinstance(data, dict):
        cur_frame = data['current_frame']
        total_frames = data['total_frames']
        pct = data['progress_pct']
        cur_count = data['current_count']
        fps_proc = data.get('fps_processed', 0.0)
        eta = data.get('eta_sec', 0.0)
        zone = data.get('zone', '')
        headgear = data.get('per_class', {})
        hg_str = ' | '.join(f"{k.capitalize()}: {int(round(v))}" for k, v in headgear.items() if k != 'head count')
        extra = f" | {hg_str}" if hg_str else ""
        sys.stdout.write(f"\r[SATARK Live] Frame: {cur_frame}/{total_frames} ({pct:.1f}%) | Count: {cur_count:.1f} [{zone}]{extra} | Speed: {fps_proc:.1f} fps | ETA: {eta:.1f}s   ")
    else:
        cur_frame = data
        total_frames = args[0] if len(args) > 0 else 1
        cur_count = args[1] if len(args) > 1 else 0
        pct = (cur_frame / max(total_frames, 1)) * 100
        sys.stdout.write(f"\r[SATARK Processing] Frame: {cur_frame}/{total_frames} ({pct:.1f}%) | Count: {cur_count:.1f}   ")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="SATARK Continuous Video Crowd Counter")
    parser.add_argument("--video-path", required=True, help="Path to input video file")
    parser.add_argument("--model-path", default="checkpoints/satark_best.pth", help="Trained model checkpoint path")
    parser.add_argument("--output-dir", default="outputs/inference", help="Directory to save output video & telemetry")
    parser.add_argument("--stride", default="auto", help="Process every N-th frame or 'auto' for ~2 FPS sampling")
    parser.add_argument("--max-dim", type=int, default=1000, help="Maximum frame dimension for inference (default: 1000)")
    parser.add_argument("--temporal-window", type=int, default=1, help="Odd-numbered median window (default: 1 for instantaneous image accuracy)")
    parser.add_argument("--extract-frames", action="store_true", default=True, help="Extract key frames as individual images with head counts (default: True)")
    parser.add_argument("--no-extract-frames", action="store_false", dest="extract_frames", help="Disable frame extraction")
    parser.add_argument("--extract-fps", type=float, default=1.0, help="Extracted key frames per second (default: 1.0)")
    args = parser.parse_args()

    # Parse stride argument
    stride = 'auto' if args.stride == 'auto' else int(args.stride)

    print("=" * 64)
    print("SATARK — Continuous Video Crowd Counting & Frame Extraction Engine")
    print("=" * 64)
    print(f"  Input Video   : {args.video_path}")
    print(f"  Model Path    : {args.model_path}")
    print(f"  Output Dir    : {args.output_dir}")
    print(f"  Frame Stride  : {args.stride}")
    print(f"  Extract Frames: {args.extract_frames} (rate: {args.extract_fps} fps)")
    print("-" * 64)

    t_start = time.time()
    try:
        result = infer_video(
            video_path=args.video_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
            frame_stride=stride,
            max_dim=args.max_dim,
            temporal_window=args.temporal_window,
            extract_frames=args.extract_frames,
            extract_fps=args.extract_fps,
            progress_callback=progress_callback,
        )
        print("\n" + "-" * 64)
        elapsed = time.time() - t_start
        print(f"[SUCCESS] Processing completed in {elapsed:.1f}s!")
        print(f"  Total Frames    : {result['total_frames']}")
        print(f"  Video Length    : {result['duration_sec']} seconds")
        print(f"  Peak Crowd      : {result['peak_count']} people (at {result['peak_time_sec']}s)")
        print(f"  Average Crowd   : {result['avg_count']} people")
        print(f"  Peak Zone       : {result['zone']}")
        print(f"  Extracted Frames: {len(result.get('extracted_frames', []))} frames saved to disk")
        print(f"  Annotated Video : {result['output_path']}")
        print(f"  Telemetry JSON  : {result['analytics_file']}")
        print("=" * 64)
    except Exception as e:
        print(f"\n[ERROR] Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
