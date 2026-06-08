# ARC Setup Guide

This guide describes how to run the HECKTOR segmentation baseline on the University of Calgary ARC cluster from a clean checkout.

The expected ARC layout is:

```text
/home/syedessamuddin.khawa/
├── EssamProjects/
│   └── HECKTOR2026/
└── HECKTOR 2026 Training Data/
    ├── CHUM-001/
    │   ├── CHUM-001__CT.nii.gz
    │   ├── CHUM-001__PT.nii.gz
    │   └── CHUM-001.nii.gz
    └── ...
```

The training code automatically looks for the home-level data directory:

```bash
../../HECKTOR 2026 Training Data
```

You can override this with `HECKTOR_DATA_ROOT` or `--data-root`.

## 1. Connect To ARC

From your Mac terminal:

```bash
ssh syedessamuddin.khawa@arc.ucalgary.ca
```

If you are off campus, connect to the UCalgary VPN first.

## 2. Pull The Latest Code

```bash
cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
git reset --hard
git pull
```

## 3. Recreate The Python Environment

Use a fresh Linux virtual environment on ARC. Do not reuse a venv copied from your Mac.

```bash
cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
rm -rf venv

module avail python
module load python/3.11

python --version
python -c "import ctypes; print('ctypes ok')"
python -m venv venv
source venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Use Python 3.10, 3.11, or 3.12. Avoid Python 3.6 because PyTorch 2.x will not install. Avoid Python 3.14 on ARC if `import ctypes` fails with `_ctypes` missing.
Install a PyTorch CUDA 12.x build on ARC. CUDA 13 wheels may install successfully but fail at runtime if the cluster GPU driver is older.

Verify the key packages:

```bash
python -m pip --version
python -c "import torch, monai, nibabel, SimpleITK; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('monai', monai.__version__)"
```

On the login node, `cuda available` may be `False`. That is expected. Check CUDA from a GPU allocation or batch job.

## 4. Verify The Data Path

```bash
cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
ls "../../HECKTOR 2026 Training Data" | head
ls "../../HECKTOR 2026 Training Data/CHUM-001"
```

Expected files:

```text
CHUM-001__CT.nii.gz
CHUM-001__PT.nii.gz
CHUM-001.nii.gz
```

If the data lives somewhere else, export the path:

```bash
export HECKTOR_DATA_ROOT="/path/to/HECKTOR 2026 Training Data"
```

## 5. Run A CPU Smoke Test

This confirms imports, config, splits, and data paths. Run it only as a quick check on the login node. Stop with `Ctrl+C` once it starts loading or training.

```bash
cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
source venv/bin/activate

python Task/Segmentation/scripts/train.py \
  --config unet3d \
  --device cpu \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --cache-rate 0
```

## 6. Run An Interactive GPU Test

Use this for a short manual test on a GPU node.

```bash
salloc --mem=32G -t 01:00:00 -p gpu-v100 --gres=gpu:1
```

Then, inside the allocation:

```bash
cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
source venv/bin/activate

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python Task/Segmentation/scripts/train.py \
  --config unet3d \
  --device cuda \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 4 \
  --cache-rate 0
```

Exit the allocation when finished:

```bash
exit
```

## 7. Submit A Batch Training Job

Create a Slurm job file on ARC:

```bash
cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
nano train_segmentation.slurm
```

Paste:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=hecktor-seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

set -euo pipefail

cd /home/syedessamuddin.khawa/EssamProjects/HECKTOR2026
source venv/bin/activate

export HECKTOR_DATA_ROOT="${HECKTOR_DATA_ROOT:-../../HECKTOR 2026 Training Data}"

python Task/Segmentation/scripts/train.py \
  --config unet3d \
  --fold 0 \
  --data-root "${HECKTOR_DATA_ROOT}" \
  --epochs 350 \
  --batch-size 2 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --cache-rate 0.25 \
  --device cuda
```

Submit:

```bash
sbatch train_segmentation.slurm
```

Monitor:

```bash
squeue -u "$USER"
tail -f hecktor-seg-JOBID.out
```

Replace `JOBID` with the job number printed by `sbatch`.

## Useful Overrides

Run fewer epochs:

```bash
python Task/Segmentation/scripts/train.py --config unet3d --epochs 5
```

Use an explicit data path:

```bash
python Task/Segmentation/scripts/train.py \
  --config unet3d \
  --data-root "/home/syedessamuddin.khawa/HECKTOR 2026 Training Data"
```

Train a different segmentation model:

```bash
python Task/Segmentation/scripts/train.py --config segresnet
python Task/Segmentation/scripts/train.py --config unetr
python Task/Segmentation/scripts/train.py --config swinunetr
```
