from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import roots_hermite
from multistream_func import run_multistream
import os
import logging
from params import P

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
results_dir = "results"
H = P.H
k = P.k
A = P.A



def w(u):
    """Weight function for moment calculation"""
    return u**4

def compute_2d_moment(mesh_2d, fn):
    """Compute moment from 2D Vlasov distribution"""
    global H
    V_2d = FunctionSpace(mesh_2d, 'DQ', 1)
    Wbar_2d = FunctionSpace(mesh_2d, 'CG', 1, vfamily='R', vdegree=0)
    x, v = SpatialCoordinate(mesh_2d)
    m_trial = TrialFunction(Wbar_2d)
    r_test = TestFunction(Wbar_2d)
    m_2d = Function(Wbar_2d)
    a_moment = r_test * m_trial * dx
    L_moment = H * r_test * w(v) * fn * dx
    moment_problem = LinearVariationalProblem(a_moment, L_moment, m_2d)
    moment_solver = LinearVariationalSolver(moment_problem)
    moment_solver.solve()
    return m_2d

def compute_moment(q_list, u_list, V):
    """Compute moment from multistream distribution"""
    moment = Function(V, name="moment")
    moment_expr = sum([w(ui[0]) * qi for ui, qi in zip(u_list, q_list)])
    moment.interpolate(moment_expr)
    return moment

logger.info("Loading 1D mesh...")
sample_checkpoint = f"results/M_2/multistream_M2_initial_checkpoint.h5"
if not os.path.exists(sample_checkpoint):
    logger.info("Running M=2 simulation to create mesh...")
    run_multistream(2,T, k, output_dir=f"results/M_2")

try: 
    with CheckpointFile(sample_checkpoint, 'r') as afile:
        mesh_1d = afile.load_mesh("1d_mesh")
except FileNotFoundError:
    logger.error("Checkpoint file not found!")

V_dg_1d = FunctionSpace(mesh_1d, "DG", 1)
V_cg_1d = FunctionSpace(mesh_1d, "CG", 1)
W_cg_1d = VectorFunctionSpace(mesh_1d, "CG", 1)

# Create immersed mesh
logger.info("Creating immersed mesh...")
x, = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)
V_dg = FunctionSpace(line, 'DG', 1)

x = SpatialCoordinate(line)[0]  # Note: using the line mesh
B = Constant(0.05)
V_dg = FunctionSpace(line, 'DG', 1)
m_exact = Function(V_dg, name="m_exact")
m_exact.interpolate(1 + B*cos(k*x))

# Storage for results
results_initial = []
results_final = []
m_list_initial = []
m_list_final = []

# Process both checkpoint types
for checkpoint_type in ["initial", "final"]:
    logger.info(f"\n{'='*40}")
    logger.info(f"Processing {checkpoint_type.upper()} checkpoints")
    logger.info(f"{'='*40}")
    
    # Load 2D Vlasov data
    logger.info(f"Loading 2D Vlasov {checkpoint_type} data...")
    try:
        with CheckpointFile(f"vlasov_{checkpoint_type}_checkpoint.h5", 'r') as afile:
            mesh_2d = afile.load_mesh("2d_mesh")
            fn = afile.load_function(mesh_2d, "fn")
            phi_2d = afile.load_function(mesh_2d, "phi")
    except FileNotFoundError:
        logger.error(f"2D Vlasov {checkpoint_type} checkpoint not found!")
        continue
    
    # Compute 2D moment
    logger.info(f"Computing 2D Vlasov {checkpoint_type} moment...")
    m_2d = compute_2d_moment(mesh_2d, fn)
    logger.info(f"Norm of 2D vlasov moment: {norm(m_2d)}")
    
    # Transfer to line mesh
    logger.info("Transferring 2D moment to line mesh...")
    m2d_line = Function(V_dg, name=f"m2d_line_{checkpoint_type}")
    m2d_line.assign(assemble(interpolate(m_2d, V_dg)))
    logger.info(f"Norm after transfer: {norm(m2d_line)}")
    
    # Store reference moment
    if checkpoint_type == "initial":
        m_list_initial.append(m2d_line)
    else:
        m_list_final.append(m2d_line)
    
    # Ensure all simulations exist
    for M in M_values:
        checkpoint_file = os.path.join(results_dir, f"M_{M}", 
                                      f"multistream_M{M}_{checkpoint_type}_checkpoint.h5")
        if not os.path.exists(checkpoint_file):
            logger.info(f"Running simulation for M = {M}...")
            run_multistream(M,T, k, output_dir=os.path.join(results_dir, f"M_{M}"))
    
    # Analyse each M value
    for M in M_values:
        q_list = []
        u_list = []
        logger.info(f"Analyzing M = {M}...")
        checkpoint_file = os.path.join(results_dir, f"M_{M}", 
                                      f"multistream_M{M}_{checkpoint_type}_checkpoint.h5")
        try:
            with CheckpointFile(checkpoint_file, 'r') as afile:
                phi = afile.load_function(mesh_1d, "phi")
                
                for i in range(M):
                    q = afile.load_function(mesh_1d, f"q_{i+1}")
                    u = afile.load_function(mesh_1d, f"u_{i+1}")
                    q_list.append(q)
                    u_list.append(u)
            
            # Compute multistream moment
            m_s = compute_moment(q_list, u_list, V_dg_1d)
            ms_line = Function(V_dg)
            ms_line.assign(assemble(interpolate(m_s, V_dg)))
            
            # Store moment
            if checkpoint_type == "initial":
                m_list_initial.append(ms_line)
            else:
                m_list_final.append(ms_line)
            
            # Compute error
            #error = norm(ms_line - m2d_line) / norm(m2d_line)
            error = norm(ms_line-m2d_line)/norm(m2d_line)
            
            if checkpoint_type == "initial":
                results_initial.append(error)
            else:
                results_final.append(error)
                
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
            
# Save VTK outputs
outfile_initial = VTKFile("analysis_initial.pvd")
outfile_initial.write(*m_list_initial)

outfile_final = VTKFile("analysis_final.pvd")  
outfile_final.write(*m_list_final)
with PdfPages(f"plots/moments_convergence_k_{k}.pdf") as pdf:
    # Plot 1: Initial checkpoint
    plt.figure(figsize=(10, 6))
    plt.scatter(M_values, results_initial, s=20, color='blue', zorder=5)
    plt.semilogy(M_values, results_initial, 'b-', alpha=0.7)
    plt.xlabel("M (Number of Streams)")
    plt.ylabel("Relative L2 Error")
    #plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.title(f"Multistream vs 2D Vlasov: Initial Checkpoint (k = {k})")
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight', dpi=300)  # Save to PDF
    plt.show()

    # Plot 2: Final checkpoint
    plt.figure(figsize=(10, 6))
    plt.scatter(M_values, results_final, s=20, color='red', zorder=5)
    plt.semilogy(M_values, results_final, 'r-', alpha=0.7)
    plt.xlabel("M (Number of Streams)")
    plt.ylabel("Relative L2 Error")
    #plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.title(f"Multistream vs 2D Vlasov: Final Checkpoint (k = {k})")
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight', dpi=300)  # Save to PDF
    plt.show()

# Print summary
logger.info("\n" + "="*40)
logger.info("FINAL RESULTS")
logger.info("="*40)
logger.info("\nInitial checkpoint errors:")
for M, error in zip(M_values, results_initial):
    if not np.isnan(error):
        logger.info(f"M = {M:2d}: L2 error = {error:.6e}")

logger.info("\nFinal checkpoint errors:")
for M, error in zip(M_values, results_final):
    if not np.isnan(error):
        logger.info(f"M = {M:2d}: L2 error = {error:.6e}")

logger.info(f"\nPlots saved to:")
logger.info(f" - plots/moments_convergence_k_{k}.pdf")