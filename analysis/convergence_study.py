import logging
import numpy as np
import sys
sys.path.append('..')
from params import P
# Import our modules
from data_loader import load_2d_vlasov_data, load_multistream_data
from moment_calculator import compute_2d_moment, compute_multistream_moment, compute_exact_moment
from moment_transfer import create_line_mesh, transfer_to_line
from error_analysis import analyze_convergence
from plotting import plot_errors_vs_M, plot_errors_vs_T

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
H, k, A = P.H, P.k, P.A

def analyze_fixed_time(M_values, T=0.5):
    """Analyze convergence for fixed time T, varying M"""
    logger.info(f"Analyzing convergence for T = {T}")
    
    # Get line mesh
    mesh_1d, _, _ = load_multistream_data(2, T, "initial")
    line = create_line_mesh(mesh_1d)
    
    # References
    m_exact = compute_exact_moment(line)
    mesh_2d, fn, _ = load_2d_vlasov_data(T, "final")
    m_2d = compute_2d_moment(mesh_2d, fn)
    m_2d_line = transfer_to_line(m_2d, line)
    
    # Collect moments
    moments_initial = []
    moments_final = []
    
    for M in M_values:
        # Initial vs exact
        _, q_init, u_init = load_multistream_data(M, T, "initial")
        if q_init:
            m_init = compute_multistream_moment(q_init, u_init, mesh_1d)
            moments_initial.append(transfer_to_line(m_init, line))
        else:
            moments_initial.append(None)
        
        # Final vs 2D Vlasov
        _, q_final, u_final = load_multistream_data(M, T, "final")
        if q_final:
            m_final = compute_multistream_moment(q_final, u_final, mesh_1d)
            moments_final.append(transfer_to_line(m_final, line))
        else:
            moments_final.append(None)
    
    # Compute errors
    errors_initial = analyze_convergence(M_values, m_exact, moments_initial)
    errors_final = analyze_convergence(M_values, m_2d_line, moments_final)
    
    # Plot
    plot_errors_vs_M(M_values, errors_initial, "Multistream vs Exact", k, T)
    #plot_errors_vs_M(M_values, errors_final, "Multistream vs 2D Vlasov", k, T)
    
    return errors_initial, errors_final

def analyze_varying_time(M_values, T_values):
    
    logger.info("Analyzing error evolution vs time")
    errors_dict = {int(M): [] for M in M_values} 
    
    for T in T_values:
        logger.info(f"Processing T = {T}")
        errors_init, errors_final = analyze_fixed_time(M_values, T)
        for i, M in enumerate(M_values):
            M_key = int(M)  
            if i < len(errors_final) and not np.isnan(errors_final[i]):
                errors_dict[M_key].append(errors_final[i])
            else:
                errors_dict[M_key].append(np.nan)
        
    print(errors_dict)
    print(f"Available M values in errors_dict: {list(errors_dict.keys())}")
    print(f"M_values to plot: {M_values}")
    
    # Check data completeness
    for M in list(errors_dict.keys()):
        print(f"M={M} has {len(errors_dict[M])} error values: {errors_dict[M]}")
    
    plot_errors_vs_T(T_values, errors_dict, M_values, k)
    
    return errors_dict

if __name__ == "__main__":
    M_values = np.arange(2,10)
    #T_values = np.arange(11,20)
    
    analyze_fixed_time(M_values, )
    # Time evolution analysis
    analyze_varying_time(M_values, T_values)