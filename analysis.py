from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite
from multistream_func import run_multistream
import os

results = []
def w(u):
    return u**2
H = 10

print("Loading 2D Vlasov data...")

with CheckpointFile("vlasov_final_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn = afile.load_function(mesh_2d, "fn")
    phi_2d = afile.load_function(mesh_2d, "phi")

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
print(f"norm of 2d vlasov moment: {norm(m_2d)}")


print("Loading 1D mesh...")
sample_checkpoint = "results/M_2/multistream_M2_final_checkpoint.h5"

with CheckpointFile(sample_checkpoint, 'r') as afile:
    mesh_1d = afile.load_mesh("1d_mesh")

V_dg_1d = FunctionSpace(mesh_1d, "DG", 1)
V_cg_1d = FunctionSpace(mesh_1d, "CG", 1)
W_cg_1d = VectorFunctionSpace(mesh_1d, "CG", 1)

def compute_moment(q_list, u_list):
    moment = Function(V_dg_1d, name="moment")
    moment_expr = sum([w(ui[0]) * qi for ui, qi in zip(u_list, q_list)])
    moment.interpolate(moment_expr)
    return moment

print("Creating immersed mesh...")
x, = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)
V_dg = FunctionSpace(line, 'DG', 1)

print("Transferring 2D moment to line mesh...")

m2d_line = Function(V_dg)
m2d_line.assign(assemble(interpolate(m_2d, V_dg)))
print(f"norm of 2d vlasov moment after transfer: {norm(m2d_line)}")

M_values = [2,3,4,5,6,7,8,9,10,15,20,23,27,30,37,40]

results_dir = "results"

for M in M_values:
    checkpoint_file = os.path.join(results_dir, f"M_{M}", f"multistream_M{M}_final_checkpoint.h5")
    if not os.path.exists(checkpoint_file):
        print(f"Running simulation for M = {M}...")
        run_multistream(M, output_dir=os.path.join(results_dir, f"M_{M}"))

for M in M_values:
    print(f"\nAnalyzing M = {M}...")
    checkpoint_file = os.path.join(results_dir, f"M_{M}", f"multistream_M{M}_final_checkpoint.h5")
    
    with CheckpointFile(checkpoint_file, 'r') as afile:
        q_list = []
        u_list = []
        for i in range(M):
            q = afile.load_function(mesh_1d, f"q_{i+1}")
            u = afile.load_function(mesh_1d, f"u_{i+1}")
            q_list.append(q)
            u_list.append(u)
            phi = afile.load_function(mesh_1d, "phi")
    
    m_s = compute_moment(q_list, u_list) 
    ms_line = Function(V_dg)
    ms_line.assign(assemble(interpolate(m_s, V_dg))) 
    error = (ms_line - m2d_line)
    results.append((norm(error)))

plt.figure(figsize=(10, 6))
plt.scatter(M_values, results, s=50, color='red', zorder=5)
plt.plot(M_values, results, 'b-', alpha=0.7)
plt.xlabel("M (Number of Streams)")
plt.xticks(M_values)
plt.title("Multistream vs 2D Vlasov: L2 Norm of Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/multistream_convergence_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal results:")
for M, error in zip(M_values, results):
    print(f"M = {M:2d}: L2 error = {error}")




