import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os


def plot_errors_vs_M(M_values, errors, title, k, T, output_dir="plots"):
    """Plot errors vs M with dots and connecting lines"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, errors, 'o-', markersize=8, linewidth=2)  # dots with lines
    #plt.yscale('log')
    plt.xlabel("M (Number of Streams)")
    plt.ylabel("Relative L2 Error")
    plt.title(f"{title} moment (T = {T}, k = {k})")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_errors_vs_T(T_values, errors_dict, M_values, k):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'x', '*']
    
    for i, M in enumerate(M_values):
        # small horizontal shift so lines don’t overlap
        jitter = (i - len(M_values)/2) * 0.05  
        T_jittered = [t + jitter for t in T_values]
        
        plt.plot(
            T_jittered,
            errors_dict[M],
            marker=markers[i % len(markers)],
            linestyle='-',
            linewidth=1,
            label=f'M = {M}'
        )

    #plt.yscale('log')
    plt.xlabel("Time T")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Error vs Time (k = {k})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def print_results_summary(M_values, errors_initial, errors_final, T):
    """Print summary of results"""
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"\nTime T = {T}")
    print("\nInitial (vs Exact) errors:")
    for M, error in zip(M_values, errors_initial):
        if not np.isnan(error):
            print(f"M = {M:2d}: L2 error = {error:.6e}")

    print("\nFinal (vs 2D Vlasov) errors:")
    for M, error in zip(M_values, errors_final):
        if not np.isnan(error):
            print(f"M = {M:2d}: L2 error = {error:.6e}")