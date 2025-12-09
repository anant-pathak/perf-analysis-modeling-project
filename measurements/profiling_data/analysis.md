# llama-bench Profiling Analysis

## Summary

This document contains the results of CPU and GPU profiling of llama-bench with the Qwen3 8B Q5_K_M.

**Key Findings:**

- **CPU Execution:** Dominated by BLAS operations (46.3% in sgemm_kernel_COOPERLAKE)
- **GPU Execution:** Kernel operations consume 67.5% of GPU time in quantized matrix multiplication

---

## 1. CPU Profiling Analysis

### 1.1 Overall Statistics

- **Total Samples:** 222,442
- **Profiling Method:** Statistical sampling (using gperftools)
- **Data File:** `llama32_CPU_profile.txt` (18KB report from 18MB raw data)

### 1.2 Top CPU Hotspots

| Function                  | Samples | % Time | Cumulative % |
| ------------------------- | ------- | ------ | ------------ |
| `sgemm_kernel_COOPERLAKE` | 102,913 | 46.3%  | 46.3%        |
| `ggml_gemm_q4_K_8x8_q8_K` | 32,239  | 14.5%  | 60.8%        |
| `ggml_gemv_q4_K_8x8_q8_K` | 28,312  | 12.7%  | 73.5%        |
| `do_spin` (inline)        | 23,960  | 10.8%  | 84.3%        |
| `ggml_vec_dot_q6_K_q8_K`  | 17,025  | 7.7%   | 91.9%        |

### 1.3 Key Observations

**1. BLAS Dominance (46.3%)**

- `sgemm_kernel_COOPERLAKE` is the single largest hotspot
- Optimized for Intel Xeon Cooper Lake microarchitecture
- Utilizes AVX-512 instructions (confirmed by `__memset_avx512_unaligned_erms`)
- BLAS copy operations (`sgemm_incopy_COOPERLAKE`, `sgemm_oncopy_COOPERLAKE`) add 2.3% overhead

**2. Quantized Operations (35.6% combined)**

- Q4_K and Q6_K quantization formats dominate:
  - `ggml_gemm_q4_K_8x8_q8_K`: 14.5% (8x8 block GEMM)
  - `ggml_gemv_q4_K_8x8_q8_K`: 12.7% (GEMV variant)
  - `ggml_vec_dot_q6_K_q8_K`: 7.7% (vector dot product)
- Indicates heavy use of quantized weights for memory efficiency

**3. Synchronization Overhead (10.8%)**

- `do_spin`: 10.8% indicates significant thread synchronization time
- OpenMP barriers visible (`gomp_barrier_wait_start`: 0.3%)
- Thread pool management (`ggml_threadpool_chunk_add`: 0.3%)
- Suggests potential for optimization in parallel workload distribution

**4. Memory Operations**

- AVX-512 memory operations present (`__memset_avx512_unaligned_erms`, `__memmove_avx512_unaligned_erms`)
- Repack operations for quantized data: `repack_q4_K_to_q4_K_8_bl` (0.9%)
- Memory bandwidth may be a limiting factor

**5. Computational Operations (Top 5 account for 91.9%)**

- High concentration in top functions shows well-optimized hot paths
- Matrix multiplication variants dominate the workload
- Minimal overhead from framework code (< 3%)

---

## 2. GPU Profiling Analysis

### 2.1 Overall Statistics

- **Profiling Tool:** NVIDIA Nsight Systems
- **Data File:** `llama-gpu-profile.nsys-rep` (2.1MB)
- **GPU Architecture:** NVIDIA Ampere (based on kernel names)

### 2.2 CUDA Kernel Performance

#### Top Kernels by Execution Time

| Kernel                       | Time (%) | Total Time (ns) | Instances | Avg (ns)  |
| ---------------------------- | -------- | --------------- | --------- | --------- |
| `mul_mat_q<Q4_K, 128>`       | 67.5%    | 849,234,613     | 1,284     | 661,397.7 |
| `mul_mat_q<Q6_K, 128>`       | 11.9%    | 149,189,519     | 210       | 710,426.3 |
| `soft_max_f32`               | 3.4%     | 43,253,843      | 216       | 200,249.3 |
| `convert_unary<float, half>` | 2.3%     | 29,212,058      | 432       | 67,620.5  |
| `quantize_mmq_q8_1`          | 1.7%     | 21,385,785      | 1,284     | 16,655.6  |

### 2.3 Key GPU Observations

**1. Quantized Matrix Multiplication Dominance (79.4%)**

- Q4_K kernels: 67.5% of GPU time
- Q6_K kernels: 11.9% of GPU time
- Clear indication that quantization is central to the workload
- Batch size of 128 used consistently

**2. Attention Mechanism (3.4%)**

- Softmax operations: 3.4% (216 instances)
- RMS normalization: 2.8% combined
- RoPE (Rotary Position Embedding): 0.7% combined

**3. Data Type Conversions (2.7%)**

- FP32 ↔ FP16 conversions: 2.3% + 0.4% = 2.7%
- Necessary for mixed precision computation
- 432 instances suggest frequent conversions between layers

**4. Ampere Tensor Core Utilization**

- `ampere_s1688gemm_fp16_256x128`: 1.2%
- `ampere_h16816gemm_128x128`: 0.6%
- Tensor cores being used for FP16 GEMM operations
- Relatively small contribution suggests quantized paths are preferred

**5. Memory-bound Operations**

- `quantize_mmq_q8_1`: 3.4% combined (on-the-fly quantization)
- Set rows and copy operations: 1.9% combined
- Suggests some memory movement overhead

### 2.4 Memory Transfer Analysis

#### GPU Memory Operations

| Operation      | Time (%) | Total Time (ns) | Count | Avg (ns) |
| -------------- | -------- | --------------- | ----- | -------- |
| Host-to-Device | 94.1%    | 257,295,635     | 4,280 | 60,115.8 |
| Device-to-Host | 5.9%     | 16,029,916      | 647   | 24,775.8 |
| Memset         | 0.0%     | 128,576         | 2     | 64,288.0 |

### 2.5 CUDA API Performance

| API Call                | Time (%) | Total Time (ns) | Calls | Avg (ns)  |
| ----------------------- | -------- | --------------- | ----- | --------- |
| `cudaStreamSynchronize` | 90.0%    | 8,644,272,105   | 9,464 | 913,384.6 |
| `cudaLaunchKernel`      | 5.5%     | 527,196,211     | 9,426 | 55,930.0  |
| `cudaMemcpyAsync`       | 2.7%     | 261,330,684     | 4,963 | 52,655.8  |

---

## 3. CPU vs GPU Comparison

### 3.1 Architectural Differences

| Aspect                 | CPU                     | GPU                        |
| ---------------------- | ----------------------- | -------------------------- |
| **Top Hotspot**        | sgemm_kernel (46.3%)    | mul_mat_q Q4_K (67.5%)     |
| **Quantization Focus** | Mixed Q4_K/Q6_K (35.6%) | Primarily Q4_K (79.4%)     |
| **Synchronization**    | Thread barriers (10.8%) | Stream sync (90% of API)   |
| **Parallelism**        | OpenMP threading        | Thousands of CUDA threads  |
| **Memory Pattern**     | Shared system memory    | Explicit H2D/D2H transfers |

---

## Appendices

### Appendix A: Data Collection Methodology

**CPU Profiling:**

- Tool: gperftools
- Sampling rate: 1000 samples/s

**GPU Profiling:**

- Tool: NVIDIA Nsight Systems
- Trace file: 2.1 MB .nsys-rep

### Appendix B: File Inventory

```
profiling_data/
├── cpu/
│   ├── llama32.prof              (18 MB - raw perf data)
│   └── llama32_CPU_profile.txt   (18 KB - processed report)
├── gpu/
│   ├── llama-gpu-profile.nsys-rep (2.1 MB - NSight trace)
│   ├── full-stats.txt            (19 KB - all statistics)
│   ├── kernel-summary.txt        (5.4 KB - kernel details)
│   └── memory-summary.txt        (852 B - memory operations)
├── flamegraphs/
│   └── llama.svg                 (flamegraph visualization)
└── scripts/
    └── (empty for now)
```

### Appendix C: Key Metrics Summary

| Metric               | CPU   | GPU                      |
| -------------------- | ----- | ------------------------ |
| Top Hotspot %        | 46.3% | 67.5%                    |
| Sync Overhead        | 10.8% | 90% (API time)           |
| Quantization Time    | 35.6% | 79.4%                    |
| Memory Transfer Time | N/A   | 257 ms (94.1% of memops) |
| Kernel Launch Count  | N/A   | 9,426                    |
| Memory Transferred   | N/A   | 5.95 GB (H2D+D2H)        |

### Appendix D: Glossary

- **GEMM:** General Matrix Multiply
- **GEMV:** General Matrix-Vector Multiply
- **Q4_K/Q6_K/Q8_K:** Quantization formats (4/6/8-bit with K-means clustering)
- **RMS Norm:** Root Mean Square Normalization
- **RoPE:** Rotary Position Embedding
- **H2D/D2H:** Host-to-Device / Device-to-Host memory transfer
- **Ampere:** NVIDIA GPU architecture (RTX 30xx, A100, etc.)
- **AVX-512:** Advanced Vector Extensions (Intel SIMD instruction set)
- **Cooper Lake:** Intel Xeon microarchitecture

---
