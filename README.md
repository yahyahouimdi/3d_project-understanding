# 3D Structural Change Detection

Prototypes for detecting **structural changes between two 3D scans of the same structure** taken at different dates, for example a missing, displaced or added member on a bridge, and turning them into a grounded inspection report.

The long-term pipeline:

```
scan at T1 ─┐
            ├─► 3D reconstruction ─► alignment ─► change detection ─► change records ─► report
scan at T2 ─┘   (3D Gaussian Splatting) (RANSAC + ICP)  (distances + clustering)  (JSON)       (every claim cites a record)
```

Each prototype builds one part of it.

## Prototypes

| Folder | What it does | Runs on | Status |
|---|---|---|---|
| [`prototype_1/`](prototype_1/) | End-to-end change detection on a **synthetic truss bridge**: alignment, detection, structured records, evaluation and a grounded report | Any laptop (CPU, a few seconds) | Working: 3 / 3 changes found, F1 = 1.00 |
| [`prototype_2/`](prototype_2/) | **3D Gaussian Splatting** reconstruction of a real scene (Tanks & Temples "train") with Nerfstudio, the step that will replace the synthetic clouds | Free cloud GPU (Kaggle / Colab T4) | Working: PSNR 20.8 dB, ~29 FPS |

### Prototype 1: change detection pipeline

Two point clouds of a labelled truss bridge, with three injected changes (a removed diagonal, a displaced vertical and a new object on the deck). The pipeline aligns them (FPFH + RANSAC, then ICP on stable points), finds changes with two-way cloud-to-cloud distances and DBSCAN, writes one JSON record per change, scores itself against the ground truth, and writes a severity-ranked report.

```
- HIGH · The north diagonal 4 (about 1.36 m) is missing in the second scan [C-01], confidence 1.0.
- MEDIUM · The south vertical 4 appears displaced by about 0.41 m [C-02], confidence 1.0.
- LOW · A new object of about 0.87 x 0.86 x 1.26 m appears near the floor beam 3 [C-03], confidence 0.8.
```

→ Details and setup: [prototype_1/README.md](prototype_1/README.md)

### Prototype 2: 3D Gaussian Splatting reconstruction

A Jupyter notebook that installs Nerfstudio, trains `splatfacto` for 15,000 steps on one T4 GPU (~21 min), evaluates it on 38 held-out views, renders a fly-through video and exports the scene as `splat.ply`.

| PSNR | SSIM | LPIPS | Gaussians |
|---|---|---|---|
| 20.78 dB | 0.790 | 0.203 | 499,057 |

→ Details and setup: [prototype_2/README.md](prototype_2/README.md)

## Quick start

**Prototype 1** (local, Python 3.10–3.12):

```powershell
cd prototype_1
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python change_detection_demo.py        # results in prototype_1/outputs/
```

**Prototype 2** (cloud GPU): open [`prototype_2/3dgs_tanks_and_temples_train.ipynb`](prototype_2/3dgs_tanks_and_temples_train.ipynb) on Kaggle or Colab with a T4 GPU and run the cells in order. The first run takes about an hour, mostly the one-time CUDA compile and training.

## Repository layout

```
.
├── prototype_1/
│   ├── change_detection_demo.py     # the full pipeline in one script
│   ├── requirements.txt
│   ├── outputs/                     # generated: pipeline.png, change_records.json, metrics.json, report.md
│   └── README.md
└── prototype_2/
    ├── 3dgs_tanks_and_temples_train.ipynb
    ├── my_3dgs_results/results/     # eval.json, train_flythrough.mp4 (splat.ply is git-ignored, ~124 MB)
    └── README.md
```

Files over 100 MB (`*.ply`, `*.zip`, checkpoints) are git-ignored, because GitHub rejects them. Re-create them by running the notebook.

## Roadmap

1. **Real data in the change-detection pipeline.** Reconstruct the same structure at two dates with 3DGS (prototype 2) and feed the results into prototype 1's alignment and detection.
2. **More robust distances.** Use M3C2 instead of cloud-to-cloud distances, to handle noise and uneven point density.
3. **Render-then-compare.** Render both 3DGS models from identical cameras and compare deep image features to find changed components.
4. **Component naming.** Name changed parts with a vision-language model on rendered crops, or match them against a BIM / design model.

## References

- Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* SIGGRAPH 2023.
- [Open3D](https://www.open3d.org/) · [Nerfstudio](https://docs.nerf.studio/) · [gsplat](https://github.com/nerfstudio-project/gsplat)
- Lague, Brodu, Leroux. *Accurate 3D comparison of complex topography with terrestrial laser scanner: application to the Rangitikei canyon (N-Z).* ISPRS J. 2013 (M3C2).
- Tanks and Temples dataset, as packaged by the [3DGS authors](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
