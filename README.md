# American Options Pricing — CUDA & OpenMP

Reimplementation of *Cvetanoska & Stojanovski, "Using High Performance Computing and Monte Carlo Simulation for Pricing American Options"*, extended with true Quasi-Monte Carlo (QMC) backends sharing one math core:

- **Serial CPU** — reference implementation (LCG Pseudo-Random)
- **OpenMP** — multi-core parallel (LCG)
- **CUDA** — GPU parallel (LCG)
- **QMC OpenMP** — True low-discrepancy QMC (Sobol + Brownian Bridge)
- **QMC CUDA** — True low-discrepancy QMC on GPU

## Algorithm

American calls are priced as Bermudan options with `m` exercise points using Quasi-Monte Carlo:

1. Forward-simulate `N` GBM paths using LCG + Moro inverse CND for normal shocks.
2. Backward-induct: at each node, value = max(intrinsic, discounted continuation).
3. Average and discount the time-zero values.

See `build.md` for the full mathematical derivation.

## Repository layout

```
src/
  core/       # shared math (BS formula, LCG, Moro, Sobol, Halton, BB, Scrambling)
  cpu/        # serial pseudo-random reference
  openmp/     # OpenMP parallel (both standard & QMC)
  cuda/       # CUDA kernels and host launcher (both standard & QMC)
  benchmark/  # benchmark main()
tests/
  validate.cpp
CMakeLists.txt
```

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

If CUDA is not installed, the CUDA executable is skipped automatically.

> Note: the default CUDA arch is `sm_80` (A100). Change `CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs (e.g. `75` for Turing, `86` for Ampere consumer, `89` for Ada, `90` for Hopper).

## Validate

```bash
./validate
```

Checks:
- Black-Scholes against Hull textbook value
- Moro inverse CND at 0.975
- American call ≥ European call
- Monotonic convergence as `m` increases

## Benchmark

```bash
./american_serial
./american_omp
./american_qmc_omp
./american_cuda       # only if CUDA was built
./american_qmc_cuda   # only if CUDA was built
```

Each prints a table over `N ∈ {10, 100, 1k, 10k, 100k, 200k, 300k, 500k, 1M}` paths with timing and price.

## Notes

- The CUDA kernel uses a per-thread stack array of size 64, supporting `m ≤ 63`. For larger `m`, switch to shared/global memory storage.
- LCG seeds are `(path_id + 1) * 1234567u` so every path is independent and reproducible.
- All three backends should agree on the price within MC noise (~0.5 absolute on N=100k).