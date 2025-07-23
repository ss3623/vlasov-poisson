from firedrake import *
import math

# Mesh setup
ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)

# Number of streams
M = 2

# Function spaces
V_dg = FunctionSpace(mesh, "DG", 1)     # for charge density q
V_cg = FunctionSpace(mesh, "CG", 1)     # for potential phi
W_cg = VectorFunctionSpace(mesh, "CG", 1)  # for velocity u

def w(u):
    return u[0]

# Initialize lists for each stream
q_list = [Function(V_dg, name=f"q{i+1}") for i in range(M)]
u_list = [Function(W_cg, name=f"u{i+1}") for i in range(M)]
phi = Function(V_cg, name="phi")


# Coordinate
x = SpatialCoordinate(mesh)[0]

# Initial conditions - different for each stream
for i in range(M):
    center = 0.3 + i * 0.4  # stream 1 at x=0.3, stream 2 at x=0.7
    q_list[i].interpolate(exp(-(x-center)**2/(0.05**2)))
    u_list[i].interpolate(as_vector([0.1]))        # SAME initial velocity for all!


# Time stepping
T = 3.0
dt = T/500.0
t = 0.0

# Constants
dtc = Constant(dt)
mass = Constant(1.0)

# RK3 stage functions
q1_list = [Function(V_dg, name=f"q1_{i+1}") for i in range(M)]
q2_list = [Function(V_dg, name=f"q2_{i+1}") for i in range(M)]
u1_list = [Function(W_cg, name=f"u1_{i+1}") for i in range(M)]
u2_list = [Function(W_cg, name=f"u2_{i+1}") for i in range(M)]

# Placeholder functions for solvers
q_temp = Function(V_dg)   # temporary q for solver
u_temp = Function(W_cg)   # temporary u for solver

# Advection solver setup: ∂q/∂t + ∇·(uq) = 0
dq_trial = TrialFunction(V_dg)
psi = TestFunction(V_dg)
a_adv = psi * dq_trial * dx

n = FacetNormal(mesh)
un = 0.5 * (dot(u_temp, n) + abs(dot(u_temp, n)))  # upwind flux

L1 = dtc * (inner(u_temp, grad(psi)) * q_temp * dx
           - (psi('+') - psi('-')) * (un('+') * q_temp('+') - un('-') * q_temp('-')) * dS)

# RK3 stage forms
q1_temp = Function(V_dg)
q2_temp = Function(V_dg)
L2 = replace(L1, {q_temp: q1_temp})
L3 = replace(L1, {q_temp: q2_temp})

dq = Function(V_dg)  # increment for q

# Advection solvers
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
advection_prob1 = LinearVariationalProblem(a_adv, L1, dq)
advection_solv1 = LinearVariationalSolver(advection_prob1, solver_parameters=params)
advection_prob2 = LinearVariationalProblem(a_adv, L2, dq)
advection_solv2 = LinearVariationalSolver(advection_prob2, solver_parameters=params)
advection_prob3 = LinearVariationalProblem(a_adv, L3, dq)
advection_solv3 = LinearVariationalSolver(advection_prob3, solver_parameters=params)

# Poisson solver: -∇²φ = Σ q_i
phi_trial = TrialFunction(V_cg)
phi_test = TestFunction(V_cg)
phi = Function(V_cg, name="phi")
nullspace = VectorSpaceBasis(constant=True)

a_phi = inner(grad(phi_trial), grad(phi_test)) * dx
Paphi = phi_trial * phi_test * dx + a_phi

# Total charge for each RK stage
q_total = sum(q_list)
q1_total = sum(q1_list)
q2_total = sum(q2_list)

# Poisson solvers for each RK stage
poisson_problem = LinearVariationalProblem(a_phi, q_total * phi_test * dx, phi, aP=Paphi)
poisson_solver = LinearVariationalSolver(poisson_problem, nullspace=nullspace,
                    solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1.0e-11})

poisson_problem1 = LinearVariationalProblem(a_phi, q1_total * phi_test * dx, phi, aP=Paphi)
poisson_solver1 = LinearVariationalSolver(poisson_problem1, nullspace=nullspace,
                     solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1.0e-11})

poisson_problem2 = LinearVariationalProblem(a_phi, q2_total * phi_test * dx, phi, aP=Paphi)
poisson_solver2 = LinearVariationalSolver(poisson_problem2, nullspace=nullspace,
                     solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1.0e-11})

# Velocity solver: ∂u/∂t = -∇φ/m
du_trial = TrialFunction(W_cg)
u_test = TestFunction(W_cg)
a_vel = inner(u_test, du_trial) * dx
L_vel = -dtc/mass * inner(u_test, grad(phi)) * dx
du = Function(W_cg)
velocity_problem = LinearVariationalProblem(a_vel, L_vel, du)
velocity_solver = LinearVariationalSolver(velocity_problem)

# Output setup
step = 0
output_freq = 20
outfile = VTKFile("multistream_vlasov.pvd")

# Initial solve and output
poisson_solver.solve()
q_total = Function(V_dg, name="q_total")
q_total.interpolate(sum(q_list))  # initial q1 + q2
outfile.write(*q_list, phi, *u_list,q_total)
#moment function
def compute_moment(q_list,u_list):
    '''Compute moment Σ w(u_i) * q_i'''
    moment = Function(V_dg, name = "moment")
    moment_expr = sum([w(ui) * qi for ui,qi in zip(u_list,q_list)])
    moment.interpolate(moment_expr)
    return moment
#SSP-RK3 time loop
while t < T - 0.5*dt:
    
    # Stage 1
    poisson_solver.solve()
    for i in range(M):
        u_temp.assign(u_list[i])
        q_temp.assign(q_list[i])
        advection_solv1.solve()
        velocity_solver.solve()
        q1_list[i].assign(q_list[i] + dq)
        u1_list[i].assign(u_list[i] + du)

    # Stage 2
    poisson_solver1.solve()
    for i in range(M):
        u_temp.assign(u1_list[i])
        q1_temp.assign(q1_list[i])
        advection_solv2.solve()
        velocity_solver.solve()
        q2_list[i].assign(0.75*q_list[i] + 0.25*(q1_list[i] + dq))
        u2_list[i].assign(0.75*u_list[i] + 0.25*(u1_list[i] + du))
   
    # Stage 3
    poisson_solver2.solve()
    for i in range(M):
        u_temp.assign(u2_list[i])
        q2_temp.assign(q2_list[i])
        advection_solv3.solve()
        velocity_solver.solve()
        q_list[i].assign((1.0/3.0)*q_list[i] + (2.0/3.0)*(q2_list[i] + dq))
        u_list[i].assign((1.0/3.0)*u_list[i] + (2.0/3.0)*(u2_list[i] + du))

    step += 1
    t += dt

    # Output
    if step % output_freq == 0:
        moment = compute_moment(q_list, u_list) #compute moment inside loop
        q_total.interpolate(sum(q_list))  # update q_total
        outfile.write(*q_list, phi, *u_list, q_total)
        print(f"Step: {step}, Time: {t:.3f}")

print(f"Simulation complete! Total steps: {step}")
#write checkpoint
with Checkpointfile("multistream_checkpoint.h5",'w') as afile:
    for i in range(M):
        afile.save_function(q_list[i], name=f"q_{i+1}")
        afile.save_function(u_list[i], name=f"u_{i+1}")
    
    afile.save_function(phi, name="phi")
    