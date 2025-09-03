from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite
from 1d1v_multistream import run_multistream
import os

results = []

def w(u):
    return u**2

H = 1

# Load (x,v) vlasov data ONCE
print("Loading 2D Vlasov data...")
with CheckpointFile("vlasov_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn = afile.load_function(mesh_2d, "fn")
    phi_2d = afile.load_function(mesh_2d, "phi")

# 2d Vlasov Moment Calculation
print("Computing 2D Vlasov moment...")
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

moment_value = assemble(m_2d * dx)
print(f"2D Vlasov moment: {moment_value}")

# Load 1D mesh ONCE (from any checkpoint file - they're all the same)
print("Loading 1D mesh...")
sample_checkpoint = "results/M_2/multistream_M2_final_checkpoint.h5"
with CheckpointFile(sample_checkpoint, 'r') as afile:
    mesh_1d = afile.load_mesh("1d_mesh")

# Set up 1D function spaces ONCE
V_dg_1d = FunctionSpace(mesh_1d, "DG", 1)
V_cg_1d = FunctionSpace(mesh_1d, "CG", 1)
W_cg_1d = VectorFunctionSpace(mesh_1d, "CG", 1)

# Create immersed mesh ONCE
print("Creating immersed mesh...")
x, = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)

V_dg = FunctionSpace(line, "DG", 1)
V_cg = FunctionSpace(line, "CG", 1)
W_cg = VectorFunctionSpace(line, "CG", 1)

# Transfer 2D moment to line mesh ONCE
print("Transferring 2D moment to line mesh...")
m2d_line = Function(V_dg)
m2d_line.assign(assemble(interpolate(m_2d, V_dg)))

def compute_moment(q_list, u_list):
    '''Compute moment Σ w(u_i) * q_i'''
    moment = Function(V_dg_1d, name="moment")
    moment_expr = sum([w(ui[0]) * qi for ui, qi in zip(u_list, q_list)])
    moment.interpolate(moment_expr)
    return moment

# Make sure results directory and run simulations if needed
M_values = [5, 10, 15]
results_dir = "results"

# Run simulations if they don't exist
for M in M_values:
    checkpoint_file = os.path.join(results_dir, f"M_{M}", f"multistream_M{M}_final_checkpoint.h5")
    if not os.path.exists(checkpoint_file):
        print(f"Running simulation for M = {M}...")
        run_multistream(M, output_dir=os.path.join(results_dir, f"M_{M}"))

# Now analyze each M value
for M in M_values:
    print(f"\nAnalyzing M = {M}...")
    
    # Load multistream results from checkpoint
    checkpoint_file = os.path.join(results_dir, f"M_{M}", f"multistream_M{M}_final_checkpoint.h5")
    
    with CheckpointFile(checkpoint_file, 'r') as afile:
        # Load functions onto our pre-existing mesh
        q_list = []
        u_list = []
        
        for i in range(M):
            q = afile.load_function(mesh_1d, f"q_{i+1}")
            u = afile.load_function(mesh_1d, f"u_{i+1}")
            q_list.append(q)
            u_list.append(u)
        
        phi = afile.load_function(mesh_1d, "phi")
    
    # Compute multistream moment
    m_s = compute_moment(q_list, u_list)
    m_s_int = assemble(m_s * dx)
    print(f"Multistream moment (M={M}): {m_s_int}")
    
    # Transfer multistream moment to line mesh
    ms_line = Function(V_dg)
    ms_line.assign(assemble(interpolate(m_s, V_dg)))
    
    # Compute error
    error = abs(ms_line - m2d_line)
    error_norm = norm(error)
    results.append(error_norm)
    print(f"L2 error norm for M={M}: {error_norm}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(M_values, results, s=100, color='red', zorder=5)
plt.plot(M_values, results, 'b-', alpha=0.7)
plt.xlabel("M (Number of Streams)")
plt.ylabel("L2 Error Norm")
plt.xticks(M_values)
plt.title("Multistream vs 2D Vlasov: L2 Norm of Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/multistream_convergence_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal results:")
for M, error in zip(M_values, results):
    print(f"M = {M:2d}: L2 error = {error:.6e}")