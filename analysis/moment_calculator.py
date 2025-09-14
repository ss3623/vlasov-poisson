from firedrake import *
import sys
sys.path.append('..')
from params import P
H,A,k = P.H,P.A,P.k

def w(v):
       """Weight function for moment calculation"""
       return v**2

def compute_multistream_moment(q_list, u_list, mesh):
    V = FunctionSpace(mesh, "DG", 1)
    moment = Function(V, name="moment")
    moment_expr = sum(w(ui[0]) * qi for ui,qi in zip(u_list,q_list))
    moment.interpolate(moment_expr)
    return moment

def compute_2d_moment(mesh_2d, fn):
    """Compute moment from 2D Vlasov"""
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


def compute_exact_moment(mesh_or_line):
    """Compute exact analytical moment"""
    V = FunctionSpace(mesh_or_line, 'DG', 1)
    x = SpatialCoordinate(mesh_or_line)[0]    
    m_exact = Function(V, name="m_exact")
    m_exact.interpolate(1 + A*cos(k*x))  
    
    return m_exact

def create_line_mesh(mesh_1d):
    """Create immersed line mesh for comparison"""
    x, = SpatialCoordinate(mesh_1d)
    coord_fs = VectorFunctionSpace(mesh_1d, "DG", 1, dim=2)
    new_coord = assemble(interpolate(as_vector([x, 0]), coord_fs))
    line = Mesh(new_coord)
    return line