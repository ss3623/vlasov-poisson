"""
This is an analysis script to compare multistream data with the 2d vlasov data.
We compute the L2 error norms between the moment calculations from both methods.

"""

from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite
from multistream_func import run_multistream
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
results_dir = "results"
checkpoint_type = "final"  # Easy to change this line

#Constants
H = 10
results = []
m_list = []

def w(u):
    "Weight function for moment calculation"
    return u**2

def compute_2d_moment(mesh_2d,fn):
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

def compute_moment(q_list, u_list,V):
    moment = Function(V, name="moment")
    moment_expr = sum([w(ui[0]) * qi for ui, qi in zip(u_list, q_list)])
    moment.interpolate(moment_expr)
    return moment

logger.info("Loading 2D Vlasov data...")
try:
    with CheckpointFile(f"vlasov_{checkpoint_type}_checkpoint.h5", 'r') as afile:
        mesh_2d = afile.load_mesh("2d_mesh")
        fn = afile.load_function(mesh_2d, "fn")
        phi_2d = afile.load_function(mesh_2d, "phi")
except FileNotFoundError:
    logger.error("Checkpoint file not found!")

logger.info("Computing 2D Vlasov moment...")
m_2d = compute_2d_moment(mesh_2d,fn)
logger.info(f"Norm of 2d vlasov moment: {norm(m_2d)}")

logger.info("Loading 1D mesh...")
sample_checkpoint = f"results/M_2/multistream_M2_{checkpoint_type}_checkpoint.h5"
try: 
    with CheckpointFile(sample_checkpoint, 'r') as afile:
        mesh_1d = afile.load_mesh("1d_mesh")
except:
    logger.error("Checkpoint file not found!")

V_dg_1d = FunctionSpace(mesh_1d, "DG", 1)
V_cg_1d = FunctionSpace(mesh_1d, "CG", 1)
W_cg_1d = VectorFunctionSpace(mesh_1d, "CG", 1)
breakpoint()

logger.info("Creating immersed mesh...")
x, = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)
V_dg = FunctionSpace(line, 'DG', 1)

logger.info("Transferring 2D moment to line mesh...")
m2d_line = Function(V_dg,name = "m2d_line")
m2d_line.assign(assemble(interpolate(m_2d, V_dg)))
m_list.append(m2d_line)
logger.info(f"norm of 2d vlasov moment after transfer: {norm(m2d_line)}")

M_values = np.arange(2,51)  

for M in M_values:
    checkpoint_file = os.path.join(results_dir, f"M_{M}", f"multistream_M{M}_{checkpoint_type}_checkpoint.h5")
    if not os.path.exists(checkpoint_file):
        logger.info(f"Running simulation for M = {M}...")
        run_multistream(M, output_dir=os.path.join(results_dir, f"M_{M}"))

for M in M_values:
    logger.info(f"\nAnalyzing M = {M}...")
    checkpoint_file = os.path.join(results_dir, f"M_{M}", f"multistream_M{M}_{checkpoint_type}_checkpoint.h5")
    try:
        with CheckpointFile(checkpoint_file, 'r') as afile:
            phi = afile.load_function(mesh_1d, "phi")
            q_list = []
            u_list = []
            for i in range(M):
                q = afile.load_function(mesh_1d, f"q_{i+1}")
                u = afile.load_function(mesh_1d, f"u_{i+1}")
                q_list.append(q)
                u_list.append(u)                
        m_s = compute_moment(q_list, u_list,V_dg_1d) 
        ms_line = Function(V_dg)
        ms_line.assign(assemble(interpolate(m_s, V_dg))) 
        m_list.append(ms_line)
        error = (ms_line - m2d_line)
        results.append((norm(error))/norm(m2d_line))
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_file}")
        continue

outfile = VTKFile("analysis_new.pvd")
outfile.write(*m_list)

logger.info("Creating convergence plot...")
plt.figure(figsize=(10, 6))
plt.scatter(M_values, results, s=20, color='red', zorder=5)
plt.plot(M_values, results, 'b-', alpha=0.7)
plt.xlabel("M (Number of Streams)")
plt.title("Multistream vs 2D Vlasov: L2 Norm of Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/moments_convergence_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"\nFinal results:")
for M, error in zip(M_values, results):
    logger.info(f"M = {M:2d}: L2 error = {error}")
logger.info("Plot saved to plots/moments_convergence_analysis.png")



