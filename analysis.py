from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
from scipy.special import roots_hermite
from config import M

print(f"Using M = {M} streams")

def w(u):
    return (u**0)

H = 10.0
q_list = []
u_list = []

with CheckpointFile("multistream_checkpoint.h5", 'r') as afile:
    mesh_1d = afile.load_mesh("1d_mesh")
    for i in range(M):
        q_list.append(afile.load_function(mesh_1d, f"q_{i+1}"))
        u_list.append(afile.load_function(mesh_1d, f"u_{i+1}"))
    phi_1d = afile.load_function(mesh_1d, "phi")

#load (x,v) vlasov data
with CheckpointFile("vlasov_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn = afile.load_function(mesh_2d, "fn")
    phi_2d = afile.load_function(mesh_2d, "phi")

#---------------- Multistream moment------------------

V_dg_1d = FunctionSpace(mesh_1d, "DG", 1)     # for charge density q
V_cg_1d = FunctionSpace(mesh_1d, "CG", 1)     # for potential phi
W_cg_1d = VectorFunctionSpace(mesh_1d, "CG", 1)
def compute_moment(q_list,u_list):
    '''Compute moment Σ w(u_i) * q_i'''
    moment = Function(V_dg_1d, name = "moment")
    moment_expr = sum([w(ui[0]) * qi for ui,qi in zip(u_list,q_list)])
    moment.interpolate(moment_expr)
    return moment
m_s = compute_moment(q_list,u_list)
m_s_int = assemble(m_s* dx)

print(f"initial multistream moment: ",m_s_int)
breakpoint()
#------------------------------------------------------------------------

# CREATING IMMERSED MESH

x,  = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)

V_dg = FunctionSpace(line, "DG", 1)     # for charge density q
V_cg = FunctionSpace(line, "CG", 1)     # for potential phi
W_cg = VectorFunctionSpace(line, "CG", 1)  # for velocity u

outfile = VTKFile("analysis_multistream.pvd")
#2d (x,v) to immersed

Wbar_cg = FunctionSpace(line, "CG", 1)
f_line = Function(V_dg)
f_line.assign(assemble(interpolate(fn, V_dg)))

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
moment_value = assemble(m_2d* dx)
print(f"2D moment on immersed mesh: {moment_value}")

