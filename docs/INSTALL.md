# Installation

This guide sets up the environment used for all CSRFormer experiments.

## Tested environment

| Component | Version |
|---|---|
| OS | Ubuntu (WSL2 on Windows) |
| GPU | 1× NVIDIA RTX 3090 (24 GB) |
| Python | 3.8.20 |
| PyTorch / torchvision | 2.4.0 / 0.19.0 (CUDA 12.1) |
| detectron2 | 0.6 (built from source) |
| timm | 1.0.24 |
| numpy | 1.24.4 |

Other versions may work, but the versions above are the ones used to produce the reported results.

## 1. Create the environment

```bash
conda create -n csrformer python=3.8 -y
conda activate csrformer
```

## 2. Install PyTorch

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. Install detectron2 from source

```bash
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
```

Building detectron2 requires a CUDA toolkit (`nvcc`) matching the PyTorch CUDA version (12.1). If `nvcc` is not found, set `CUDA_HOME` to your CUDA installation, e.g. `export CUDA_HOME=/usr/local/cuda-12.1`.

## 4. Clone CSRFormer and install its dependencies

```bash
git clone https://github.com/nsohpeter/CSRFormer.git
cd CSRFormer/Mask2Former
pip install -r requirements.txt
pip install numpy==1.24.4
```

`requirements.txt` installs `cython`, `scipy`, `shapely`, `timm`, `h5py`, `submitit` and `scikit-image`.

## 5. Compile the multi-scale deformable attention ops

Mask2Former's pixel decoder uses a custom CUDA operator that must be compiled once:

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

## 6. Pretrained backbone

CSRFormer is initialised from Swin-B pretrained on ImageNet-22K at 384×384, converted to detectron2 format:

```bash
mkdir -p pretrained_weights
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py \
  swin_base_patch4_window12_384_22k.pth \
  pretrained_weights/swin_base_patch4_window12_384_22k.pkl
```

The training configs expect the converted file at `Mask2Former/pretrained_weights/swin_base_patch4_window12_384_22k.pkl`.

## 7. Verify the installation

Run this from `CSRFormer/Mask2Former`. It builds the model on CPU and checks that the CSR module is present:

```bash
python - <<'EOF'
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.modeling import build_model
cfg = get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
cfg.merge_from_file('custom_configs/training/csfd_v1_run2_swinb_90k.yaml')
cfg.MODEL.DEVICE = 'cpu'
m = build_model(cfg)
print('CSR params:', sum(p.numel() for n, p in m.named_parameters() if 'csfd' in n))
EOF
```

Expected output ends with:

```
CSR params: 179457
```

Datasets are not needed for this check. Next, prepare the datasets as described in [DATASETS.md](DATASETS.md).

## Troubleshooting

**`ModuleNotFoundError: MultiScaleDeformableAttention`.** The deformable attention ops were not compiled, or were compiled for a different PyTorch/CUDA version. Re-run step 5 inside the active environment.

**Harmless warnings.** The following messages appear during normal runs and can be ignored:

- `torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument`
- `torch.cuda.amp.autocast(args...) is deprecated`
- `Importing from timm.models.layers is deprecated`
- `Loading config ... with yaml.unsafe_load`

**Logged learning rate looks 10× too small.** detectron2 logs the learning rate of the most common parameter group, which is the backbone (`BASE_LR × BACKBONE_MULTIPLIER = 1e-5`). The decoder and CSR module train at `1e-4` as configured.