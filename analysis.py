from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
M = 4
q_list = []
u_list = []
#open file 1: load 1d multistream data
with CheckpointFile("multistream_checkpoint.h5", 'r') as afile:
    #create lists 
    mesh_1d = afile.load_mesh("1d_mesh")
    for i in range(M):
        q_list.append(afile.load_function(mesh_1d, f"q_{i+1}"))
        u_list.append(afile.load_function(mesh_1d, f"u_{i+1}"))
    phi_1d = afile.load_function(mesh_1d, "phi")

 #load 2D vlasov data

with CheckpointFile("vlasov_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn = afile.load_function(mesh_2d, "fn")
    phi_2d = afile.load_function(mesh_2d, "phi")
# mesh 2d to 1d immersed

x,  = SpatialCoordinate(mesh_1d)
coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
line = Mesh(new_coord)

#from the firedrake website. Will edit later?
V_dg = FunctionSpace(line, "DG", 1)     # for charge density q
V_cg = FunctionSpace(line, "CG", 1)     # for potential phi
W_cg = VectorFunctionSpace(line, "CG", 1)  # for velocity u


q_line = []
u_line = []

for i in range(M):
    f = q_list[i]
    g = Function(functionspaceimpl.WithGeometry.create(f.function_space(), line),
             val=f.topological)
    q_line.append(g)
    f = u_list[i]
    g = Function(functionspaceimpl.WithGeometry.create(f.function_space(), line),
             val=f.topological)
    u_line.append(g)

f = Function(V_dg)
f.assign(assemble(interpolate(fn, V_dg)))

outfile = VTKFile("analysis_vlasov.pvd")
outfile.write(*q_line,*u_line,f)


#---------------- the moments stuff ------------------
#weight function w(v) = v (for now)

def w(u):
    return u[0]

def compute_moment(q_list,u_list):
    '''Compute moment Σ w(u_i) * q_i'''
    moment = Function(V_dg, name = "moment")
    moment_expr = sum([w(ui) * qi for ui,qi in zip(u_list,q_list)])
    moment.interpolate(moment_expr)
    return moment
om = compute_moment(q_list, u_list)
old_moment = assemble(om * dx)
nm = compute_moment(q_line,u_line)
new_moment = assemble(nm * dx)
print(f"Old moment (w(v) = v) : ",old_moment)
print(f"New moment (w(v) = v) : ",new_moment)


#currently in the other two scripts. check those two, then continue writing here
'''next tasks:
- transfer 1d moments to immersed mesh (replacing mesh coordinates)
- interpolate 2d moments to the 1d immersed mesh''' 
