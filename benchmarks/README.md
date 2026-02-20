# GPU vs CPU Benchmark Suite

Compares CPU (PCG+DIC) vs GPU (OGLPCG via Ginkgo) performance on OpenFOAM cases.

## Quick Start — Mobius MIX 200L on A100

### 1. Clone and checkout

```bash
git clone https://github.com/gogipav14/OpenFOAM-13.git
cd OpenFOAM-13
git checkout ogl-gpu-solvers
```

### 2. Build the Docker image

The Dockerfile already targets SM 80 (A100). Build for A100 only to speed up compilation:

```bash
docker build --build-arg CUDA_ARCHS="80" -t mixfoam:latest .
```

Full build (A100 + Ada + Blackwell):

```bash
docker build -t mixfoam:latest .
```

### 3. Generate the Mobius MIX 200L case

```bash
# Check available configs
python -m mixfoam \
    --archive "mixing_cases/MixIT Reactors.mdata" \
    info Mobius_MIX 200L

# Generate the case (serial, default 3mm cell size)
python -m mixfoam \
    --archive "mixing_cases/MixIT Reactors.mdata" \
    setup Mobius_MIX 200L \
    --volume 23 --rpm 20 \
    --nprocs 1 \
    -o ./mix200_case
```

For a finer mesh (2mm, ~3x more cells):

```bash
python -m mixfoam \
    --archive "mixing_cases/MixIT Reactors.mdata" \
    setup Mobius_MIX 200L \
    --volume 23 --rpm 20 \
    --cell-size 0.002 \
    --nprocs 1 \
    -o ./mix200_fine
```

### 4. Run CPU vs GPU comparison

```bash
cd benchmarks

python benchmark.py compare ../mix200_case \
    --docker \
    --image mixfoam:latest \
    --timesteps 20 \
    --timeout 7200 \
    --precision FP64 \
    --cpu-variant cpu_pcg \
    --results-dir ./compare_results_mix200 \
    -v
```

This will:
- Copy the case into `compare_results_mix200/cpu/` and `compare_results_mix200/gpu/`
- Swap pressure solver: CPU gets PCG+DIC, GPU gets OGLPCG+BlockJacobi (FP64)
- Run blockMesh + snappyHexMesh + foamRun inside Docker for each variant
- Print speedup, NFE cost ratio, and residual comparison

## Benchmark Commands

### `compare` — Single case CPU vs GPU

```bash
python benchmark.py compare <case_path> --docker [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | 50 | Number of timesteps to run |
| `--timeout` | 3600 | Timeout per variant (seconds) |
| `--precision` | FP64 | GPU precision: FP32, FP64, MIXED |
| `--cpu-variant` | cpu_pcg | `cpu` (original solver) or `cpu_pcg` (PCG+DIC) |
| `--results-dir` | ./compare_results | Output directory |
| `-v` | off | Verbose output |

### `scaling` — Mesh scaling study (blockMesh cases)

```bash
python benchmark.py scaling <case_path> --docker [options]
```

Scales hex blocks by refinement factors, adjusts deltaT with CFL control (maxCo=0.5).

| Flag | Default | Description |
|------|---------|-------------|
| `--factors` | 1,2,4,8 | Comma-separated refinement factors |
| `--timesteps` | 20 | Timesteps per mesh size |
| `--timeout` | 1200 | Timeout per case (seconds) |

### `run` — Batch tutorial benchmarks

```bash
python benchmark.py run --docker [options]
```

Scans the tutorials directory and runs all GPU-compatible cases.

### `scan` — List GPU-compatible tutorials

```bash
python benchmark.py scan --list-all
```

## Plotting

```bash
python plot_scaling.py --csv scaling_results_cavity/scaling_results.csv
```

Generates 4 publication-quality figures:
1. `scaling_speedup.png` — GPU speedup vs mesh size
2. `scaling_walltime.png` — Wall time per step (log-log)
3. `scaling_nfe_cost.png` — NFE cost ratio and iteration ratio
4. `scaling_residuals.png` — Convergence quality comparison

## Cavity Scaling Results (RTX 5060)

CPU: PCG+DIC | GPU: OGLPCG+BlockJacobi | FP64 | maxCo=0.5

| Mesh | Cells | CPU s/step | GPU s/step | Speedup | NFE Cost Ratio |
|------|-------|------------|------------|---------|----------------|
| x1 | 400 | 0.0007 | 0.0243 | 0.03x | 2.73 |
| x4 | 6,400 | 0.0145 | 0.1370 | 0.11x | 2.68 |
| x16 | 102,400 | 0.8403 | 0.8082 | 1.04x | 0.32 |
| x32 | 409,600 | 9.8552 | 2.9851 | 3.30x | 0.10 |
| x50 | 1,000,000 | 42.7444 | 9.2009 | 4.65x | 0.07 |

GPU crossover at ~100K cells. At 1M cells each GPU iteration costs 7% of a CPU iteration.

## Architecture

```
benchmarks/
  benchmark.py        — CLI: scan, run, scaling, compare
  case_modifier.py    — Creates CPU/GPU case variants (swaps fvSolution)
  runner.py           — Docker/native execution (blockMesh, snappyHexMesh, solver)
  log_parser.py       — Parses OpenFOAM solver logs for metrics
  report_generator.py — Comparison metrics, CSV/JSON output
  plot_scaling.py     — Matplotlib publication-quality plots
```
