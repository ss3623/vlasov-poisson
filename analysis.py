from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
from scipy.special import roots_hermite
from config import M

print(f"Using M = {M} streams")

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

# CREATING IMMERSED MESH

x,  = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)

V_dg = FunctionSpace(line, "DG", 1)     # for charge density q
V_cg = FunctionSpace(line, "CG", 1)     # for potential phi
W_cg = VectorFunctionSpace(line, "CG", 1)  # for velocity u


#---------------- Multistream to immersed mesh------------------
#weight function

def w(u):
    return (u[0]**1)

def compute_moment(q_list,u_list):
    '''Compute moment Σ w(u_i) * q_i'''
    moment = Function(V_dg, name = "moment")
    moment_expr = sum([w(ui) * qi for ui,qi in zip(u_list,q_list)])
    moment.interpolate(moment_expr)
    return moment

nm = compute_moment(q_list,u_list)
new_moment = assemble(nm * dx)

print(f"Final multistream moment: ",new_moment)
print("I have successfully transferred 1d moments to the immersed mesh!")
#------------------------------------------------------------------------
#2d (x,v) to immersed

Wbar_cg = FunctionSpace(line, "CG", 1)
f_line = Function(V_dg)
f_line.assign(assemble(interpolate(fn, V_dg)))

m_trial = TrialFunction(Wbar_cg) 
r_test = TestFunction(Wbar_cg) 
m_line = Function(Wbar_cg)

a_moment = r_test * m_trial * dx
L_moment = H * r_test * 1.0 * f_line * dx

moment_problem = LinearVariationalProblem(a_moment, L_moment, m_line)
moment_solver = LinearVariationalSolver(moment_problem)
moment_solver.solve()

moment_value = assemble(m_line * dx)
print(f"2D moment on immersed mesh: {moment_value}")

