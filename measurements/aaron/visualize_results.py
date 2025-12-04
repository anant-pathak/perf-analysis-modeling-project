#!/usr/bin/env python3
"""
Dynamic Performance Visualization Script for Qwen3-8B Benchmarks
Automatically parses analysis markdown and generates visualizations
CS 431/531 Performance Project
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Configuration
BENCHMARK_DIR = Path.home() / "perf-analysis-modeling-project/measurements/aaron"
OUTPUT_DIR = BENCHMARK_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

def find_latest_analysis():
    """Find the most recent analysis file"""
    analysis_files = sorted(BENCHMARK_DIR.glob("analysis_*.md"))
    if not analysis_files:
        raise FileNotFoundError(f"No analysis files found in {BENCHMARK_DIR}")
    return analysis_files[-1]

def parse_analysis_file(filepath):
    """Parse the analysis markdown file to extract data"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {
        'cpu_baseline': None,
        'configurations': [],
        'hardware_types': {}
    }
    
    # Extract CPU baseline
    cpu_match = re.search(r'CPU Baseline \((\w+)\):\s+Prompt Processing: ([\d.]+) t/s\s+Text Generation:\s+([\d.]+) t/s', content)
    if cpu_match:
        data['cpu_baseline'] = {
            'node': cpu_match.group(1),
            'pp512': float(cpu_match.group(2)),
            'tg128': float(cpu_match.group(3))
        }
    
    # Extract configuration data from the comparison table
    table_pattern = r'\| (\w+)\s+\| ([\w\s]+?)\s+\| ([\w\s()]+?)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|'
    matches = re.finditer(table_pattern, content)
    
    for match in matches:
        node = match.group(1).strip()
        gpu_type = match.group(2).strip()
        config_name = match.group(3).strip()
        pp512 = float(match.group(4))
        tg128 = float(match.group(5))
        
        config_data = {
            'node': node,
            'gpu_type': gpu_type,
            'name': config_name,
            'pp512': pp512,
            'tg128': tg128,
            'is_cpu': 'CPU-Only' in config_name
        }
        
        data['configurations'].append(config_data)
        
        # Track hardware types
        if not config_data['is_cpu']:
            if gpu_type not in data['hardware_types']:
                data['hardware_types'][gpu_type] = []
            data['hardware_types'][gpu_type].append(config_data)
    
    return data

def create_cpu_vs_gpu_comparison(data):
    """Figure 1: CPU vs GPU configurations"""
    # Get CPU baseline
    cpu_config = next((c for c in data['configurations'] if c['is_cpu']), None)
    
    # Get GPU configurations - FIXED LOGIC
    gpu_configs = [c for c in data['configurations'] 
                   if not c['is_cpu'] and ('Single GPU' in c['name'] or 'GPU Full' in c['name'] or 'GPU Partial' in c['name'])]
    
    print(f"  DEBUG: Found CPU config: {cpu_config is not None}")
    print(f"  DEBUG: Found {len(gpu_configs)} GPU configs: {[c['name'] for c in gpu_configs]}")
    
    if not cpu_config:
        print("  ⚠ No CPU configuration found")
        return
    
    if not gpu_configs:
        print("  ⚠ No GPU configurations found")
        return
    
    # Prepare data
    configs = ['CPU-Only']
    pp_data = [cpu_config['pp512']]
    tg_data = [cpu_config['tg128']]
    
    for config in gpu_configs[:3]:  # Limit to 3 GPU configs
        configs.append(config['name'].replace(' ', '\n'))
        pp_data.append(config['pp512'])
        tg_data.append(config['tg128'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CPU vs GPU Offloading Performance', fontsize=16, fontweight='bold')
    
    # Prompt Processing
    bars1 = ax1.bar(configs, pp_data, color=colors[:len(configs)], edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Text Generation
    bars2 = ax2.bar(configs, tg_data, color=colors[:len(configs)], edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_cpu_vs_gpu_comparison.png', dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: 1_cpu_vs_gpu_comparison.png')
    plt.close()

def create_speedup_analysis(data):
    """Figure 2: Speedup analysis"""
    cpu_baseline = data['cpu_baseline']
    if not cpu_baseline:
        print("  ⚠ Skipping speedup analysis - no CPU baseline")
        return
    
    gpu_configs = [c for c in data['configurations'] if not c['is_cpu']]
    
    if not gpu_configs:
        print("  ⚠ Skipping speedup analysis - no GPU configurations")
        return
    
    configs = []
    pp_speedups = []
    tg_speedups = []
    
    for config in gpu_configs[:4]:  # Limit to 4 configs
        configs.append(config['name'].replace(' ', '\n'))
        pp_speedups.append(config['pp512'] / cpu_baseline['pp512'])
        tg_speedups.append(config['tg128'] / cpu_baseline['tg128'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pp_speedups, width, label='Prompt Processing',
                   color=colors[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, tg_speedups, width, label='Text Generation',
                   color=colors[1], edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Speedup (vs CPU-Only)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Acceleration Speedup over CPU Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_speedup_analysis.png', dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: 2_speedup_analysis.png')
    plt.close()

def create_multi_gpu_scaling(data):
    """Figure 3: Multi-GPU scaling analysis"""
    # Find single and multi-GPU configs
    single_gpu = [c for c in data['configurations'] if 'Single GPU' in c['name'] or ('GPU Full' in c['name'] and not c['is_cpu'])]
    multi_gpu = [c for c in data['configurations'] if 'Dual GPU' in c['name'] or 'Quad GPU' in c['name']]
    
    if not single_gpu or not multi_gpu:
        print("  ⚠ Skipping multi-GPU scaling - insufficient data")
        return
    
    all_configs = single_gpu[:1] + multi_gpu  # Take first single GPU + all multi
    configs = [c['name'].replace(' ', '\n') for c in all_configs]
    pp_data = [c['pp512'] for c in all_configs]
    tg_data = [c['tg128'] for c in all_configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Multi-GPU Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Prompt Processing
    bars1 = ax1.bar(configs, pp_data, color=colors[:len(configs)], edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=single_gpu[0]['pp512'], color='red', linestyle='--', linewidth=2, label='Single GPU Baseline')
    ax1.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Text Generation
    bars2 = ax2.bar(configs, tg_data, color=colors[:len(configs)], edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=single_gpu[0]['tg128'], color='red', linestyle='--', linewidth=2, label='Single GPU Baseline')
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_multi_gpu_scaling.png', dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: 3_multi_gpu_scaling.png')
    plt.close()

def create_hardware_comparison(data):
    """Figure 4: Hardware comparison"""
    if len(data['hardware_types']) < 2:
        print("  ⚠ Skipping hardware comparison - need at least 2 GPU types")
        return
    
    hardware_names = []
    pp_avgs = []
    tg_avgs = []
    
    for gpu_type, configs in data['hardware_types'].items():
        # Get best single GPU config for each hardware type
        single_configs = [c for c in configs if 'Single GPU' in c['name'] or 'GPU Full' in c['name']]
        if single_configs:
            best = max(single_configs, key=lambda x: x['pp512'])
            hardware_names.append(gpu_type.replace('NVIDIA ', ''))
            pp_avgs.append(best['pp512'])
            tg_avgs.append(best['tg128'])
    
    if len(hardware_names) < 2:
        print("  ⚠ Skipping hardware comparison - insufficient single GPU data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hardware Comparison', fontsize=16, fontweight='bold')
    
    # Prompt Processing
    bars1 = ax1.bar(hardware_names, pp_avgs, color=colors[:len(hardware_names)], edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax1.set_title('Prompt Processing (pp512)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} t/s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    if len(pp_avgs) == 2:
        speedup = max(pp_avgs) / min(pp_avgs)
        ax1.text(0.5, max(pp_avgs) * 0.5, f'{speedup:.2f}x faster',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Text Generation
    bars2 = ax2.bar(hardware_names, tg_avgs, color=colors[:len(hardware_names)], edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax2.set_title('Text Generation (tg128)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} t/s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    if len(tg_avgs) == 2:
        speedup = max(tg_avgs) / min(tg_avgs)
        ax2.text(0.5, max(tg_avgs) * 0.5, f'{speedup:.2f}x faster',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_hardware_comparison.png', dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: 4_hardware_comparison.png')
    plt.close()

def create_summary_dashboard(data):
    """Figure 5: Summary dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Qwen3-8B Performance Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    cpu_baseline = data['cpu_baseline']
    if not cpu_baseline:
        print("  ⚠ Cannot create dashboard without CPU baseline")
        return
    
    # Get best GPU config
    gpu_configs = [c for c in data['configurations'] if not c['is_cpu']]
    best_gpu = max(gpu_configs, key=lambda x: x['pp512']) if gpu_configs else None
    
    if not best_gpu:
        print("  ⚠ Cannot create dashboard without GPU configs")
        return
    
    # Subplot 1: CPU vs GPU
    ax1 = fig.add_subplot(gs[0, 0])
    configs = ['CPU', 'GPU']
    pp_data = [cpu_baseline['pp512'], best_gpu['pp512']]
    bars = ax1.bar(configs, pp_data, color=[colors[2], colors[0]], edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Tokens/Second', fontweight='bold')
    ax1.set_title('Prompt Processing: CPU vs GPU', fontweight='bold', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Speedup
    ax2 = fig.add_subplot(gs[0, 1])
    speedup_pp = best_gpu['pp512'] / cpu_baseline['pp512']
    speedup_tg = best_gpu['tg128'] / cpu_baseline['tg128']
    speedup_data = [speedup_pp, speedup_tg]
    speedup_labels = ['Prompt\nProcessing', 'Text\nGeneration']
    bars = ax2.bar(speedup_labels, speedup_data, color=[colors[0], colors[1]], edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Speedup (vs CPU)', fontweight='bold')
    ax2.set_title('GPU Acceleration Speedup', fontweight='bold', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}x',
                ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: Multi-GPU comparison (if available)
    multi_gpu = [c for c in data['configurations'] if 'GPU' in c['name'] and not c['is_cpu']]
    if len(multi_gpu) > 1:
        ax3 = fig.add_subplot(gs[1, :])
        configs = [c['name'].replace(' ', '\n') for c in multi_gpu[:4]]
        pp_data = [c['pp512'] for c in multi_gpu[:4]]
        bars = ax3.bar(configs, pp_data, color=colors[:len(configs)], edgecolor='black', linewidth=1.2)
        ax3.set_ylabel('Tokens/Second', fontweight='bold')
        ax3.set_title('GPU Configuration Comparison: Prompt Processing', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Subplot 4: Hardware comparison (if available)
    if len(data['hardware_types']) >= 2:
        ax4 = fig.add_subplot(gs[2, 0])
        hw_names = []
        hw_pp = []
        for gpu_type, configs in data['hardware_types'].items():
            single = [c for c in configs if 'Single' in c['name'] or 'Full' in c['name']]
            if single:
                hw_names.append(gpu_type.replace('NVIDIA ', ''))
                hw_pp.append(max(c['pp512'] for c in single))
        
        bars = ax4.bar(hw_names, hw_pp, color=colors[:len(hw_names)], edgecolor='black', linewidth=1.2)
        ax4.set_ylabel('Tokens/Second', fontweight='bold')
        ax4.set_title('Hardware Comparison (Prompt)', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Subplot 5: Key findings
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    findings_text = f"""
KEY FINDINGS

✓ GPU acceleration: {speedup_pp:.0f}x speedup
  for prompt processing

✓ Best GPU: {best_gpu['gpu_type']}
  {best_gpu['pp512']:.0f} t/s (prompt)

✓ CPU Baseline: {cpu_baseline['pp512']:.1f} t/s
  Text generation: {cpu_baseline['tg128']:.1f} t/s

RECOMMENDATION:
Single GPU with full layer offload
for maximum inference performance
"""
    ax5.text(0.1, 0.9, findings_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(f'{OUTPUT_DIR}/5_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: 5_summary_dashboard.png')
    plt.close()

def main():
    print("="*70)
    print("DYNAMIC VISUALIZATION GENERATION")
    print("="*70)
    print(f"\nSearching for analysis files in: {BENCHMARK_DIR}")
    
    try:
        analysis_file = find_latest_analysis()
        print(f"✓ Found: {analysis_file.name}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    print("\nParsing analysis data...")
    data = parse_analysis_file(analysis_file)
    
    print(f"✓ Found {len(data['configurations'])} configurations")
    if data['cpu_baseline']:
        print(f"✓ CPU baseline: {data['cpu_baseline']['pp512']:.2f} t/s")
    print(f"✓ Hardware types: {', '.join(data['hardware_types'].keys())}")
    
    print(f"\nGenerating visualizations in: {OUTPUT_DIR}")
    print("-" * 70)
    
    create_cpu_vs_gpu_comparison(data)
    create_speedup_analysis(data)
    create_multi_gpu_scaling(data)
    create_hardware_comparison(data)
    create_summary_dashboard(data)
    
    print("-" * 70)
    print("\n✅ VISUALIZATION GENERATION COMPLETE!")
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nGenerated visualizations:")
    for i, f in enumerate(sorted(OUTPUT_DIR.glob("*.png")), 1):
        print(f"  {i}. {f.name}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
