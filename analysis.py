from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite
from multistream_function import run_multistream_simulation

def w(u):
    return (u**2+2*u)

H = 1

# Load (x,v) vlasov data
with CheckpointFile("vlasov_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn = afile.load_function(mesh_2d, "fn")
    phi_2d = afile.load_function(mesh_2d, "phi")

# 2d Vlasov Moment Calculation
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
print(f"2D moment on immersed mesh: {moment_value}")

M_values = [5, 10, 20]
results = {}  # Storage for all results

for M in M_values:
    q_list = []
    u_list = []
    with CheckpointFile("multistream_checkpoint.h5", 'r') as afile:
        mesh_1d = afile.load_mesh("1d_mesh")
        for i in range(M):
            q_list.append(afile.load_function(mesh_1d, f"q_{i+1}"))
            u_list.append(afile.load_function(mesh_1d, f"u_{i+1}"))
        phi_1d = afile.load_function(mesh_1d, "phi")

    # Multistream moment
    V_dg_1d = FunctionSpace(mesh_1d, "DG", 1)
    
    def compute_moment(q_list, u_list):
        moment = Function(V_dg_1d, name="moment")
        moment_expr = sum([w(ui[0]) * qi for ui, qi in zip(u_list, q_list)])
        moment.interpolate(moment_expr)
        return moment

    m_s = compute_moment(q_list, u_list)
    m_s_int = assemble(m_s * dx)
    print(f"Multistream moment (M={M}): {m_s_int}")

    # Creating immersed mesh (only do once)
    if M == M_values[0]:  # Only create mesh for first M
        x, = SpatialCoordinate(mesh_1d)
        coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
        new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
        line = Mesh(new_coord)
        V_dg = FunctionSpace(line, "DG", 1)
        
        # Transfer 2D moment to line mesh (only once)
        m2d_line = Function(V_dg)
        m2d_line.assign(assemble(interpolate(m_2d, V_dg)))
        m2d_values = m2d_line.dat.data[:]
        x_coords = line.coordinates.dat.data[:, 0]

    # Transfer multistream moment to line mesh
    ms_line = Function(V_dg)
    ms_line.assign(assemble(interpolate(m_s, V_dg)))
    ms_values = ms_line.dat.data[:]
    
    # Store results for this M
    results[M] = ms_values.copy()

# Plot all results together
plt.figure()
for M, ms_values in results.items():
    plt.plot(x_coords, abs(m2d_values - ms_values), label=f'M={M} streams')

plt.xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi, 6*np.pi, 7*np.pi, 8*np.pi],
           ['0', 'π', '2π', '3π', '4π', '5π', '6π', '7π', '8π'])
plt.title("Moment Difference: Multistream vs 2D Vlasov")
plt.xlabel("x")
plt.ylabel("|2D Vlasov - Multistream|")
plt.legend()
plt.grid()
plt.show()