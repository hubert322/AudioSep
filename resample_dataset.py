import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Try to import tqdm for progress bar, fallback to simple print if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable

def resample_file(task):
    src_path, dst_path, target_sr = task
    
    # Create destination directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if file already exists
    if dst_path.exists():
        return False

    # ffmpeg command for high-quality resampling
    # -ar: set audio rate
    # -v error: only show errors
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src_path),
        "-ar", str(target_sr),
        str(dst_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Resample a dataset in parallel.")
    parser.add_argument("--src", type=str, required=True, help="Source directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--sr", type=int, required=True, help="Target sample rate (e.g., 16000 or 32000)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    
    if not src_root.exists():
        print(f"Error: Source directory {src_root} does not exist.")
        return

    print(f"Scanning for files in {src_root}...")
    audio_files = list(src_root.rglob("*.wav"))
    print(f"Found {len(audio_files)} files.")
    
    tasks = []
    for src_path in audio_files:
        # Maintain relative path structure
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        tasks.append((src_path, dst_path, args.sr))
    
    print(f"Starting resampling to {args.sr}Hz using {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Convert map results to list to trigger execution
        list(tqdm(executor.map(resample_file, tasks), total=len(tasks)))

    print(f"Finished resampling to {args.sr}Hz.")

if __name__ == "__main__":
    main()
