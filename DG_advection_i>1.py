from firedrake import *
import math

def w(u):
    w1 = Function(u.function_space())
    w_expr = inner(u,u)
    w1.interpolate(w_expr)   
    return w1

ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)

# --- Time-stepping parameters ---
T = 3.0
dt = T/500.0
T = dt*5.0
dtc = Constant(dt)
mass = Constant(1.0)
# ----------------------------------
# Function spaces
V   = FunctionSpace(mesh, "DG", 1)
W   = VectorFunctionSpace(mesh, "CG", 1)
Vcg = FunctionSpace(mesh, "CG", 1)

x, = SpatialCoordinate(mesh)

# Number of streams
m = 2

# Initialize lists for q and u for each stream
q_list  = [Function(V, name=f"q_{i}") for i in range(m)]
q1_list = [Function(V, name=f"q1_{i}") for i in range(m)]
q2_list = [Function(V, name=f"q2_{i}") for i in range(m)]
u_list  = [Function(W, name=f"u_{i}") for i in range(m)]
u1_list = [Function(W, name=f"u1_{i}") for i in range(m)]
u2_list = [Function(W, name=f"u2_{i}") for i in range(m)]

# Shared initial profiles: Gaussian density and constant velocity
q0_expr = exp(-((x - 0.5)**2) / (0.2**2/2))/2    # Gaussian bump
u0_expr = as_vector([0.0])
m = Function(V)
m_exp = w(u_list[0])*q_list[0]
for i in range (1,m):
    m_exp += w(u_list[i])*q_list[i]


# Apply the same initialization to every stream
for qi, ui in zip(q_list, u_list):
    qi.interpolate(q0_expr)
    ui.interpolate(u0_expr)

# Placeholder functions for the solver
q  = Function(V)
us = Function(W)

# 1. RK3 advection residual setup
psi      = TestFunction(V)
dq_trial = TrialFunction(V)
a        = psi * dq_trial * dx
n        = FacetNormal(mesh)
un       = 0.5*(dot(us, n) + abs(dot(us, n)))

L1 = dtc*(inner(us, grad(psi))*q*dx
          - (psi('+') - psi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)
q1 = Function(V)
q2 = Function(V)
L2 = replace(L1, {q: q1})
L3 = replace(L1, {q: q2})

dq = Function(V)  # increment for q

# Create advection solvers
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

# 2. Poisson solve for phi
phi_sol = TrialFunction(Vcg)
v = TestFunction(Vcg)
phi = Function(Vcg, name="phi")
nullspace = VectorSpaceBasis(constant=True)

a_phi = inner(grad(phi_sol), grad(v))*dx
Paphi = phi_sol*v*dx + a_phi

# Sum densities for each RK stage
q_total  = sum(q_list)
q1_total = sum(q1_list)
q2_total = sum(q2_list)

# Define Poisson solvers for each stage
phi_problem   = LinearVariationalProblem(a_phi, q_total*v*dx, phi, aP=Paphi)
phi_solver    = LinearVariationalSolver(phi_problem, nullspace=nullspace,
                     solver_parameters={'ksp_type':'gmres','ksp_atol':1.0e-11})
phi_problem_1 = LinearVariationalProblem(a_phi, q1_total*v*dx, phi, aP=Paphi)
phi_solver_1  = LinearVariationalSolver(phi_problem_1, nullspace=nullspace,
                     solver_parameters={'ksp_type':'gmres','ksp_atol':1.0e-11})
phi_problem_2 = LinearVariationalProblem(a_phi, q2_total*v*dx, phi, aP=Paphi)
phi_solver_2  = LinearVariationalSolver(phi_problem_2, nullspace=nullspace,
                     solver_parameters={'ksp_type':'gmres','ksp_atol':1.0e-11})

# Velocity update solver
du_trial = TrialFunction(W)
u_test = TestFunction(W)
a       = inner(u_test, du_trial)*dx
L1 = -dtc/mass*inner(u_test, grad(phi))*dx
du = Function(W)
du_prob = LinearVariationalProblem(a, L1, du)
du_solv = LinearVariationalSolver(du_prob)

# Time-stepping loop setup
t = 0.0
step = 0
output_freq = 20
outfile = VTKFile("advection_multiple_streams.pvd")
projected = VTKFile("proj_output_multiple_strms.pvd", project_output=True)
# Initial write
phi_solver.solve()
outfile.write(*q_list, phi, *u_list)
projected.write(*q_list, phi, *u_list)

# Main RK3 loop
while t < T - 0.5*dt:
    # Stage 1
    phi_solver.solve()
    for i in range(m):
        us.assign(u_list[i])
        q.assign(q_list[i])
        solv1.solve()
        du_solv.solve()
        q1_list[i].assign(q_list[i] + dq)
        u1_list[i].assign(u_list[i] + du)

    # Stage 2
    phi_solver_1.solve()
    for i in range(m):
        us.assign(u1_list[i])
        q1.assign(q1_list[i])
        solv2.solve()
        du_solv.solve()
        q2_list[i].assign(0.75*q_list[i] + 0.25*(q1_list[i] + dq))
        u2_list[i].assign(0.75*u_list[i] + 0.25*(u1_list[i] + du))
   
    # Stage 3
    phi_solver_2.solve()
    for i in range(m):
        us.assign(u2_list[i])
        q2.assign(q2_list[i])
        solv3.solve()
        du_solv.solve()
        q_list[i].assign((1.0/3.0)*q_list[i] + (2.0/3.0)*(q2_list[i] + dq))
        u_list[i].assign((1.0/3.0)*u_list[i] + (2.0/3.0)*(u2_list[i] + du))

    step += 1
    t += dt

    # Output
    if step % output_freq == 0:
        outfile.write(*q_list, phi, *u_list)
        projected.write(*q_list, phi, *u_list)
        print("t=", t)