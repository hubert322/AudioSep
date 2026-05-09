#!/bin/bash
#SBATCH --job-name=resample_fsd50k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=base
#SBATCH --output=resample_%j.out

# Load ffmpeg (Standard on NYU Greene)
module load ffmpeg

# Activate your conda environment
# Replace 'AudioSep' with your actual environment name if different
source activate AudioSep

# Define Root Directories
FSD_ROOT="/scratch/th3622/machine-listening/fsdk-download"

# 1. Resample DEV AUDIO
SRC_DEV="$FSD_ROOT/FSD50K.dev_audio"
DST_DEV_16K="$FSD_ROOT/FSD50K.dev_audio_16k"
DST_DEV_32K="$FSD_ROOT/FSD50K.dev_audio_32k"

# 2. Resample EVAL AUDIO (if it exists)
SRC_EVAL="$FSD_ROOT/FSD50K.eval_audio"
DST_EVAL_16K="$FSD_ROOT/FSD50K.eval_audio_16k"
DST_EVAL_32K="$FSD_ROOT/FSD50K.eval_audio_32k"

echo "Launching Resampling Tasks in Parallel..."

# Launch 16K tasks
if [ -d "$SRC_DEV" ]; then
    python resample_dataset.py --src "$SRC_DEV" --dst "$DST_DEV_16K" --sr 16000 --workers 16 &
fi
if [ -d "$SRC_EVAL" ]; then
    python resample_dataset.py --src "$SRC_EVAL" --dst "$DST_EVAL_16K" --sr 16000 --workers 16 &
fi

# Launch 32K tasks
if [ -d "$SRC_DEV" ]; then
    python resample_dataset.py --src "$SRC_DEV" --dst "$DST_DEV_32K" --sr 32000 --workers 16 &
fi
if [ -d "$SRC_EVAL" ]; then
    python resample_dataset.py --src "$SRC_EVAL" --dst "$DST_EVAL_32K" --sr 32000 --workers 16 &
fi

# Wait for all background processes to finish
wait

echo "All resampling tasks complete."
