import matplotlib.pyplot as plt
import numpy as np
import os

# Set output directory to ../paper
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set scientific style
# specialized style parameters since seaborn-paper might not be installed or behave differently
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# =======================
# Figure 1: Performance Degradation
# =======================
def plot_fig1():
    print("Generating Figure 1...")
    # Data derived from paper context
    attack_intensity = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Data points (SR%)
    gpt4v_tvsc = [19.1, 18.8, 18.5, 18.0, 17.8]
    gemini_tvsc = [16.8, 16.5, 16.1, 15.8, 15.9]
    gpt4v_base = [18.5, 14.0, 10.0, 8.0, 6.8]
    gemini_base = [16.2, 12.0, 8.0, 6.5, 5.5]

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot lines
    ax.plot(attack_intensity, gpt4v_tvsc, 'o-', label='GPT-5 + TVSC (Ours)', color='#d62728', linewidth=2)
    ax.plot(attack_intensity, gemini_tvsc, 's-', label='Gemini 2.5 Pro + TVSC (Ours)', color='#ff7f0e', linewidth=2)
    ax.plot(attack_intensity, gpt4v_base, 'o--', label='GPT-5 (Base)', color='#1f77b4', alpha=0.7)
    ax.plot(attack_intensity, gemini_base, 's--', label='Gemini 2.5 Pro (Base)', color='#2ca02c', alpha=0.7)

    ax.set_xlabel('Attack Intensity ($\epsilon$)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Performance Degradation under Increasing Attack')
    ax.legend()
    ax.set_ylim(0, 22)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'Figure1_Robustness.pdf')
    plt.savefig(output_path)
    print(f"Saved to {output_path}")
    # Also save a png for quick viewing if needed
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300)

# =======================
# Figure 2: Latency vs Robustness
# =======================
def plot_fig2():
    print("Generating Figure 2...")
    # Data from Table 11
    methods = [
        {'name': 'GPT-5 Base', 'lat': 360, 'sr': 6.8, 'type': 'base'},
        {'name': 'Base + Heuristics', 'lat': 400, 'sr': 8.5, 'type': 'base'}, 
        {'name': 'Vis + OCR', 'lat': 440, 'sr': 12.1, 'type': 'ablation'},
        {'name': 'TVSC-Fast', 'lat': 440, 'sr': 14.2, 'type': 'ours'},
        {'name': 'TVSC-Full', 'lat': 530, 'sr': 16.5, 'type': 'ours'}
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw Pareto frontier approximation first (bottom layer)
    ax.plot([360, 440, 530], [6.8, 14.2, 16.5], '--', color='gray', alpha=0.3, zorder=0)

    for m in methods:
        color = '#d62728' if m['type'] == 'ours' else ('#1f77b4' if m['type'] == 'base' else 'gray')
        marker = '*' if m['type'] == 'ours' else 'o'
        size = 200 if m['type'] == 'ours' else 80
        
        ax.scatter(m['lat'], m['sr'], c=color, s=size, marker=marker, label=m['name'] if m['type'] == 'ours' else "", zorder=10)
        
        # Adjust text position to avoid overlap
        xytext = (5, 5)
        if m['name'] == 'TVSC-Fast':
            xytext = (5, -15)
        
        ax.annotate(m['name'], (m['lat'], m['sr']), xytext=xytext, textcoords='offset points', fontsize=9)

    ax.set_xlabel('Latency (ms/action)')
    ax.set_ylabel('Success Rate under Hybrid Attack (%)')
    ax.set_title('Latency vs. Robustness Trade-off')
    ax.set_xlim(300, 600)
    ax.set_ylim(4, 18)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'Figure2_Tradeoff.pdf')
    plt.savefig(output_path)
    print(f"Saved to {output_path}")
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300)

if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
