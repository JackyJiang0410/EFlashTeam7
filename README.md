<h1 align="center" style="font-size: 2.0em; font-weight: bold; margin-bottom: 0; border: none; border-bottom: none;">eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Miscrostructures</h1>

##### <p align="center"> [Venkatesh Pattabiraman](https://venkyp.com), [Zizhou Huang](https://huangzizhou.github.io/), [Daniele Panozzo](https://cims.nyu.edu/gcl/daniele.html), [Denis Zorin](https://cims.nyu.edu/gcl/denis.html), [Lerrel Pinto](https://www.lerrelpinto.com/) and [Raunaq Bhirangi](https://raunaqbhirangi.github.io/)</p>
##### <p align="center"> New York University </p>

<!-- <p align="center">
  <img src="assets/eflesh.gif">
 </p> -->

#####
<div align="center">
    <a href="https://e-flesh.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue"></a> &ensp;
    <a href="https://arxiv.org/abs/2506.09994"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red"></a> &ensp; 
    <a href="https://github.com/notvenky/eFlesh/blob/main/microstructure/README.md"><img src="https://img.shields.io/static/v1?label=CAD2eFlesh&message=Tool&color=lightblue"></a> &ensp;
    <a href="mailto:venkatesh.p@nyu.edu">
      <img src="https://img.shields.io/static/v1?label=Questions?&amp;message=Reach%20Out&amp;color=purple">
    </a>
    <!-- <a href="https://github.com/notvenky/eFlesh/tree/main/characterization/datasets"><img src="https://img.shields.io/static/v1?label=Characterization&message=Datasets&color=blue"></a> &ensp; -->
    
</div>

#####

## Getting Started
```bash
git clone --recurse-submodules https://github.com/notvenky/eFlesh.git
cd eFlesh
conda env create -f env.yml
conda activate eflesh
```

---

## Repository Layout

```
Database/
  Data/
    single_touch_data/          # position_*.csv (single-contact experiments)
    multi_touch_data/           # multi_touch_*.csv (simultaneous contacts)
  result/
    localization_single_*/      # generated checkpoints + summaries
    localization_multi_*/       # generated checkpoints + summaries
    classifier_single_multi_*/  # classifier checkpoints + summaries
```

Each new training run creates a timestamped directory in `Database/result/` containing:

- `checkpoint.pt` – model parameters and normalization statistics
- `summary.txt` – the final training/validation/test metrics (mirrors console output)

## Datasets and CSV Fields

- `timestamp` – capture time (seconds)
- `x_pos`, `y_pos` – labelled grid position (single-touch files)
- `mag{0-4}_{x,y,z}` – tri-axial magnetometer channels (model inputs)
- `fz` – normal force; used to filter multi-touch samples
- `ty` – torque about the y-axis (informational)
- `pos{i}_{x,y}` – labelled positions for touch index `i` (multi-touch files)

Rows with all-zero magnetometer readings are discarded. Multi-touch samples additionally require `|fz| > min_force` (default 1 N).

## Models, Inputs, and Outputs

| Model | Script & Mode | Inputs (`X`) | Outputs (`Y`) |
|-------|---------------|--------------|---------------|
| Single-touch localization | `characterization/train.py --mode single_touch` | 15 magnetometer readings (`mag0_x`…`mag4_z`) | `[x, y, z]` contact location (z can be fixed per capture) |
| Multi-touch localization | `characterization/train.py --mode multi_touch` | 15 magnetometer readings | Flattened coordinates `[pos1_x, pos1_y, pos2_x, pos2_y, …]` |
| Touch classifier | `characterization/touch_classifier.py` | 15 magnetometer readings | Binary label (`0` single-touch sample, `1` multi-touch sample`) |

All datasets use z-score normalization computed from the training split; the statistics are bundled with each checkpoint for inference.

## Training Commands

```bash
# Single-touch localization
python characterization/train.py \
  --mode single_touch \
  --folder Database/Data/single_touch_data \
  --epochs 500 --batch_size 128 --lr 1e-3

# Multi-touch localization
python characterization/train.py \
  --mode multi_touch \
  --folder Database/Data/multi_touch_data \
  --epochs 500 --batch_size 128 --lr 1e-3

# Touch classifier (single vs multi)
python characterization/touch_classifier.py \
  --single-dir Database/Data/single_touch_data \
  --multi-dir Database/Data/multi_touch_data \
  --epochs 200 --batch_size 128 --lr 1e-3
```

Use `--device auto` (default) or `--device cuda` to train on GPU when available. Additional flags let you override file patterns (`--pattern`, `--multi_touch_pattern`) and force thresholds (`--multi_touch_min_force`, `--min-force`).

## Inspecting Results

```bash
ls Database/result/localization_multi_*/summary.txt
cat Database/result/localization_multi_20251111_172916/summary.txt
```

To load a checkpoint:

```python
import torch
from characterization.models import MLP  # 3-layer MLP used for localization

run_dir = "Database/result/localization_single_20251111_101010"
ckpt = torch.load(f"{run_dir}/checkpoint.pt")

model = MLP(in_dim=15, out_dim=ckpt["out_dim"], hidden=128)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

`ckpt["out_dim"]` equals 3 for single-touch runs and `2 * touches` for multi-touch runs.

## Primary References
eFlesh draws upon these prior works:

1. [Cut-Cell Microstructures for Two-scale Structural Optimization](https://cims.nyu.edu/gcl/papers/2024-cutcells.pdf)
2. [Learning Precise, Contact-Rich Manipulation through Uncalibrated Tactile Skins](https://visuoskin.github.io)
3. [AnySkin: Plug-and-play Skin Sensing for Robotic Touch](https://any-skin.github.io)
4. [ReSkin: versatile, replaceable, lasting tactile skins](https://reskin.dev)

## Cite 
If you build on our work or find it useful, please cite it using the following bibtex
```
@article{pattabiraman2025eflesh,
  title={eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures},
  author={Pattabiraman, Venkatesh and Huang, Zizhou and Panozzo, Daniele and Zorin, Denis and Pinto, Lerrel and Bhirangi, Raunaq},
  journal={arXiv preprint arXiv:2506.09994},
  year={2025}
}
```

