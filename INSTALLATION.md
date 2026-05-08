# AudioSep: Installation & Setup Guide

This guide covers how to set up the environment, configure logging, and run the training pipeline.

## 🛠 1. Environment Setup

Run these commands in order. These specific versions are strictly required for compatibility with the pre-trained weights and audio processing logic.

```bash
# 1. Core ML Stack (CUDA 11.6)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 2. Lightning & Transformers (Specific versions to avoid CLAP encoder issues)
pip install lightning==2.0.0 transformers==4.28.1

# 3. Audio & Dataset Utilities
pip install torchlibrosa==0.1.0 librosa==0.10.0.post2 soundfile==0.12.1
pip install ftfy braceexpand webdataset wget h5py

# 4. Monitoring & Logging
pip install wandb tensorboard
```

---

## 📈 2. Monitoring & Logging

The pipeline is set up for **Dual Logging** (WandB + TensorBoard).

### Weights & Biases (WandB)
- **Enabled by default**.
- **On HPC**: You must set your API key before running to avoid the script hanging:
  ```bash
  export WANDB_API_KEY=your_key_here
  ```
- **To Disable**: Use the `--no_wandb` flag in your training command.

### TensorBoard
- Local logs are saved to `workspace/tf_logs/`.
- View them using: `tensorboard --logdir workspace/tf_logs`

---

## 🚀 3. Running Training

Each user should use their own **`--workspace`** directory to avoid overwriting checkpoints.

### Standard Training (Base Model)
Use this for the standard AudioSep architecture.
```bash
python train.py \
    --workspace workspace/YourName \
    --config_yaml config/audiosep_base.yaml \
    --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt
```

### Transformer Training (Advanced)
Use this if you want to enable the Transformer bottleneck for better semantic alignment.
```bash
python train.py \
    --workspace workspace/YourName_Transformer \
    --config_yaml config/audiosep_transformer.yaml \
    --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt
```

### Training without WandB
```bash
python train.py \
    --workspace workspace/YourName \
    --config_yaml config/audiosep_base.yaml \
    --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt \
    --no_wandb
```

### Smoke Test (Verify Setup)
```bash
python train.py \
    --workspace workspace/smoke_test \
    --config_yaml config/audiosep_smoke_test.yaml \
    --resume_checkpoint_path '' \
    --fast_dev_run
```

## 📂 4. Preparing Datafiles

The configuration files expect dataset JSONs with **absolute paths** to the audio files. If you have a JSON with relative paths, you can use the `fix_json_paths.py` script to generate a version compatible with the HPC.

### Running the path fixer:
```bash
python fix_json_paths.py \
    --input datafiles/fsd50k_dev_auto_caption.json \
    --base_dir /scratch/th3622/path/to/fsd50k/audio/ \
    --suffix _abs
```
This will create `datafiles/fsd50k_dev_auto_caption_abs.json` with the correct absolute paths.

### Update your Config:
After generating the `_abs.json` file, make sure to update your YAML configuration (e.g., `config/audiosep_base.yaml`) to point to the new file:
```yaml
data:
    datafiles:
        - 'datafiles/fsd50k_dev_auto_caption_abs.json'
```

> [!IMPORTANT]
> **HPC Permissions**: If you are sharing datasets across accounts on NYU Greene, ensure the dataset directory is readable by your teammates:
> `chmod -R 755 /scratch/your_netid/path/to/datasets`

## 💾 5. Checkpoints
- **Frequency**: Checkpoints are saved every **2,000 steps** (configured in your YAML config). This is optimized for 4-8 hour GPU sessions.
- **Location**: Found in `workspace/checkpoints/train/`.
