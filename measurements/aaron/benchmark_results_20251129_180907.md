# Qwen3-8B Performance Benchmark Results

## Test Configuration

**Model:** Qwen3-8B (Q5_K_M quantization)  
**Hardware:** ORCA Supercluster  
**Framework:** llama.cpp  
**Date:** Sat Nov 29 06:09:07 PM PST 2025
**Node:** orcaga01
**Nodes Used:** 1
**GPUs per Node:** 1

## System Information

```
CPU Info:
CPU(s):                                  64
On-line CPU(s) list:                     0-63
Model name:                              AMD EPYC 9534 64-Core Processor
Thread(s) per core:                      1
Core(s) per socket:                      64
CPU(s) scaling MHz:                      71%
NUMA node0 CPU(s):                       0-15
NUMA node1 CPU(s):                       16-31
NUMA node2 CPU(s):                       32-47
NUMA node3 CPU(s):                       48-63

GPU Info:
NVIDIA L40S, 46068 MiB, 580.105.08
NVIDIA L40S, 46068 MiB, 580.105.08
NVIDIA L40S, 46068 MiB, 580.105.08
NVIDIA L40S, 46068 MiB, 580.105.08

CUDA Toolkit Version:
Cuda compilation tools, release 12.9, V12.9.41
```

## Benchmark Configurations

| Configuration | Layers on GPU | CPU Threads | Description |
|---------------|---------------|-------------|-------------|
| CPU-Only | 0 | 64 | Pure CPU processing with OpenBLAS |
| GPU Partial | 10 | 64 | Hybrid: 10 layers on GPU, rest on CPU |
| GPU Full | 99 (all) | 64 | All layers offloaded to GPU |

---

## Test 1: CPU-Only (64 threads)

```
