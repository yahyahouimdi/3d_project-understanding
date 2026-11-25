# 3D Gaussian Splatting on Tanks & Temples

Training a **3D Gaussian Splatting** model with Nerfstudio (`splatfacto`) on a free cloud GPU (NVIDIA T4), then evaluating it, rendering a fly-through video and exporting the scene. It's a first step toward **structural change detection**: reconstruct a structure at two dates, then compare the two 3D models.

▶ **Notebook:** [`3dgs_tanks_and_temples_train.ipynb`](3dgs_tanks_and_temples_train.ipynb)

<!-- Add a short GIF or screenshot of the fly-through here, e.g. ![Fly-through](media/flythrough.gif) -->

## Results

Scene: Tanks & Temples **“train”**, 15,000 training steps, evaluated on 38 held-out views.

| Metric | Value |
|---|---|
| PSNR | **20.78 dB** |
| SSIM | **0.790** |
| LPIPS | **0.203** |
| Rendering speed | **~29 FPS** |
| Gaussians | **499,057** |
| Training time | **~21 min** on one T4 (~84 ms/step) |

A solid first result rather than state of the art: the original paper trains 30,000 steps on a stronger GPU.

## How to run

1. Open the notebook on **Kaggle** (Settings → Accelerator → *GPU T4*) or **Colab** (Runtime → *T4 GPU*).
2. Run the cells in order. Expect about 10 min of setup, about 30 min for the one-time CUDA compile, and about 21 min of training.
3. Download `3dgs_results.zip` from the link in the last cell: metrics, video and `splat.ply`.
4. View `splat.ply` in the free [SuperSplat editor](https://superspl.at/editor).

## Problems solved along the way

| Problem | Fix |
|---|---|
| Images are half resolution, but the COLMAP cameras describe full-size photos | `--downscale-factor 2` + an `images_2/` folder |
| 1959 / 2 = 979.5 rounded down to 979 px instead of 980 | `--downscale-rounding-mode ceil` |
| The gsplat CUDA build ran out of RAM on 4 CPUs and stalled | `MAX_JOBS=2` |
| PyTorch 2.6 refuses to load the checkpoint (`weights_only`) | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, for self-trained checkpoints only |

The full troubleshooting table is at the end of the notebook.

## Next steps

- Train two models of the same structure at two dates and align them (RANSAC + ICP on stable regions).
- Render both from identical cameras and compare deep features to find changed components.
- Generate a grounded inspection report from structured change records.

## References

- Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* SIGGRAPH 2023.
- [Nerfstudio](https://docs.nerf.studio/) · [gsplat](https://github.com/nerfstudio-project/gsplat)
- Dataset: Tanks and Temples, as packaged by the [3DGS authors](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
