from firedrake import *
from firedrake.__future__ import interpolate

#open file 1: load 1d multistream data
with CheckpointFile("multistream_checkpoint.h5", 'r') as afile:
    mesh_1d = afile.load_mesh("1d_mesh")
    q1 = afile.load_function(mesh_1d, "q_1")
    q2 = afile.load_function(mesh_1d, "q_2")
    u1 = afile.load_function(mesh_1d, "u_1")
    u2 = afile.load_function(mesh_1d, "u_2")
    phi_1d = afile.load_function(mesh_1d, "phi")

 #load 2D vlasov data

 with CheckpointFile("vlasov_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn = afile.load_function(mesh_2d, "fn")
    phi_2d = afile.load_function(mesh_2d, "phi")
# mesh 2d to 1d immersed

cells = np.asarray([[0, 1]])
vertex_coords = np.asarray([[0.0, 0.0], [1.0, 1.0]])
plex = mesh_2d.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
line = mesh.Mesh(plex, dim=2)
#from the firedrake website. Will edit later?

#---------------- the moments stuff ------------------

#currently in the other two scripts. check those two, then continue writing here 