# ECE792 (Fall 2026): Adaptive Safety Filter + MPPI

| Folder | Contents |
|---|---|
| `AdaptiveSF/` | Trained models (SDF VAE + latent-conditioned HJ value network), examples, method notes (`document/method.tex`). See `AdaptiveSF/README.md`. |
| `SimValidation/` | Simulation validation. **Hub: `SimValidation/index.html`** (progress, conventions, results, session log). |
| `SimValidation/notes/stage3_plan.md` | Detailed plan for Stage 3 (MPPI / log-MPPI reproduction with PyCUDA, our nominal + baseline, mp4s). |

Keep `AdaptiveSF/` and `SimValidation/` side by side: `SimValidation/simenv/sensing.py` imports `../AdaptiveSF`.

## Setup (GPU machine)

```bash
pip install torch numpy scipy matplotlib pycuda jinja2   # pycuda needs the CUDA toolkit (nvcc) on PATH
# ffmpeg is needed for the mp4 renders
cd SimValidation
python scripts/s1_generate_forest.py --seed 0
python scripts/s1_check_value_function.py --device cuda   # stage-1 scripts default to mps
```
