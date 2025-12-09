# Qwen3-8B Performance Benchmark Results - 4 GPU Configuration

## Test Configuration

**Model:** Qwen3-8B (Q5_K_M quantization)  
**Hardware:** ORCA Supercluster  
**Framework:** llama.cpp  
**Date:** Tue Dec  2 09:01:14 PM PST 2025
**Node:** orcaga19
**Nodes Used:** 1
**GPUs per Node:** 4

## System Information

```
CPU Info:
CPU(s):                                  64
On-line CPU(s) list:                     0-63
Model name:                              AMD EPYC 9534 64-Core Processor
Thread(s) per core:                      1
Core(s) per socket:                      64
CPU(s) scaling MHz:                      70%
NUMA node0 CPU(s):                       0-15
NUMA node1 CPU(s):                       16-31
NUMA node2 CPU(s):                       32-47
NUMA node3 CPU(s):                       48-63

GPU Info:
index, name, memory.total [MiB], driver_version
0, NVIDIA A30, 24576 MiB, 580.105.08
1, NVIDIA A30, 24576 MiB, 580.105.08
2, NVIDIA A30, 24576 MiB, 580.105.08
3, NVIDIA A30, 24576 MiB, 580.105.08

CUDA Toolkit Version:
Cuda compilation tools, release 12.9, V12.9.41
```

## Benchmark Configurations

| Configuration | GPUs Used | Tensor Split | CPU Threads | Description |
|---------------|-----------|--------------|-------------|-------------|
| Single GPU | 1 | N/A | 64 | Baseline: All layers on GPU 0 |
| Dual GPU | 2 | 8,8,0,0 | 64 | Split across GPU 0 and 1 |
| Quad GPU Balanced | 4 | 4,4,4,4 | 64 | Evenly distributed across all 4 GPUs |
| Quad GPU Custom | 4 | 5,5,3,3 | 64 | Weighted distribution (testing variation) |

---

## Test 1: Single GPU (Baseline)

**Configuration:** All layers on GPU 0

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA A30, compute capability 8.0, VMM: yes
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 |           pp512 |       2419.90 ± 5.97 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 |           tg128 |         74.62 ± 0.10 |

build: 8c32d9d96 (7199)
```

## Test 2: Dual GPU (2 GPUs)

**Configuration:** Tensor split 8,8,0,0 (using GPU 0 and 1)

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 1: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 2: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 3: NVIDIA A30, compute capability 8.0, VMM: yes
| model                          |       size |     params | backend    | threads | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------ | --------------: | -------------------: |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 8.00         |           pp512 |       2404.57 ± 4.35 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 8.00         |           tg128 |         74.81 ± 0.22 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 8.00         |           pp512 |       2421.04 ± 4.47 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 8.00         |           tg128 |         75.17 ± 0.09 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 0.00         |           pp512 |      1762.55 ± 65.84 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 0.00         |           tg128 |         54.69 ± 1.37 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 0.00         |           pp512 |      1672.77 ± 21.98 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 0.00         |           tg128 |         54.34 ± 0.28 |

build: 8c32d9d96 (7199)
```

## Test 3: Quad GPU - Balanced Distribution

**Configuration:** Tensor split 4,4,4,4 (evenly distributed)

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 1: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 2: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 3: NVIDIA A30, compute capability 8.0, VMM: yes
| model                          |       size |     params | backend    | threads | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------ | --------------: | -------------------: |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           pp512 |       2408.69 ± 8.58 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           tg128 |         74.58 ± 0.13 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           pp512 |      2412.39 ± 20.56 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           tg128 |         74.55 ± 0.03 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           pp512 |       2399.46 ± 7.29 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           tg128 |         74.45 ± 0.03 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           pp512 |      2395.45 ± 13.51 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 4.00         |           tg128 |         74.52 ± 0.09 |

build: 8c32d9d96 (7199)
```

## Test 4: Quad GPU - Custom Distribution

**Configuration:** Tensor split 5,5,3,3 (weighted distribution)

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 4 CUDA devices:
  Device 0: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 1: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 2: NVIDIA A30, compute capability 8.0, VMM: yes
  Device 3: NVIDIA A30, compute capability 8.0, VMM: yes
| model                          |       size |     params | backend    | threads | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------ | --------------: | -------------------: |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 5.00         |           pp512 |       2393.53 ± 9.46 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 5.00         |           tg128 |         73.75 ± 0.06 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 5.00         |           pp512 |      2384.61 ± 11.85 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 5.00         |           tg128 |         73.79 ± 0.06 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 3.00         |           pp512 |      2387.33 ± 23.13 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 3.00         |           tg128 |         73.95 ± 0.08 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 3.00         |           pp512 |       2381.21 ± 9.00 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | CUDA,BLAS  |      64 | 3.00         |           tg128 |         74.03 ± 0.07 |

build: 8c32d9d96 (7199)
```

## Performance Summary

### Multi-GPU Scaling Analysis

| Configuration | GPUs | Prompt Processing (pp512) | Text Generation (tg128) | Speedup vs 1 GPU (pp) | Speedup vs 1 GPU (tg) |
|---------------|------|---------------------------|-------------------------|-----------------------|-----------------------|
| Single GPU | 1 | - | - | 1.00x | 1.00x |
| Dual GPU | 2 | - | - | - | - |
| Quad GPU (Balanced) | 4 | - | - | - | - |
| Quad GPU (Custom) | 4 | - | - | - | - |

*Note: Fill in actual values from results above and calculate speedups*

## Observations

### Prompt Processing (pp512)
- **Single GPU Performance:** 
- **Dual GPU Scaling:** 
- **Quad GPU Scaling:** 

### Text Generation (tg128)
- **Single GPU Performance:** 
- **Dual GPU Scaling:** 
- **Quad GPU Scaling:** 

### Multi-GPU Efficiency
- **Linear Scaling?:** 
- **Communication Overhead:** 
- **Optimal Configuration:** 

## Build Configuration

```
CMAKE_BUILD_TYPE:STRING=Release
GGML_BLAS:BOOL=ON
GGML_BLAS_VENDOR:STRING=OpenBLAS
GGML_CUDA:BOOL=ON
GGML_CUDA_COMPRESSION_MODE:STRING=size
GGML_CUDA_FA:BOOL=ON
GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF
GGML_CUDA_FORCE_CUBLAS:BOOL=OFF
GGML_CUDA_FORCE_MMQ:BOOL=OFF
GGML_CUDA_GRAPHS:BOOL=OFF
GGML_CUDA_NO_PEER_COPY:BOOL=OFF
GGML_CUDA_NO_VMM:BOOL=OFF
GGML_CUDA_PEER_MAX_BATCH_SIZE:STRING=128
//STRINGS property for variable: CMAKE_BUILD_TYPE
CMAKE_BUILD_TYPE-STRINGS:INTERNAL=Debug;Release;MinSizeRel;RelWithDebInfo
//STRINGS property for variable: GGML_CUDA_COMPRESSION_MODE
GGML_CUDA_COMPRESSION_MODE-STRINGS:INTERNAL=none;speed;balance;size
```

---
*Generated by automated 4-GPU benchmark script*
*Stored in: /home/aaronw/perf-analysis-modeling-project/measurements/aaron/benchmark_results_4gpu_20251202_210114.md*
