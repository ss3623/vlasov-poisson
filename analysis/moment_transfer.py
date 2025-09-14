from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np

def create_line_mesh(mesh_1d):
    """Create immersed line mesh for comparison"""
    x, = SpatialCoordinate(mesh_1d)
    coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
    new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
    line = Mesh(new_coord)
    return line

def transfer_to_line(moment_function, line_mesh):
    """Transfer any moment to the line mesh"""
    V_line = FunctionSpace(line_mesh, 'DG', 1)
    moment_line = Function(V_line, name="moment_line")
    moment_line.assign(assemble(interpolate(moment_function, V_line)))
    return moment_line
