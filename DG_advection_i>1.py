from firedrake import *
import math

ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)

# --- Time-stepping parameters ---

T = 3.0
dt = T/500.0
dtc = Constant(dt)
mass = Constant(1.0) #JUST renaming m to mass for clarity

V = FunctionSpace(mesh, "DG", 1)         
W = VectorFunctionSpace(mesh, "CG", 1) 
Vcg = FunctionSpace(mesh, "CG", 1)

x, = SpatialCoordinate(mesh)

m = 2 # number of streams

q_list = [] 
for i in range(m):
    q = Function(V)
    q_list.append(q)
q = Function(V) #input to solver

for i, q in enumerate(q_list):
    # Example: Shift center for each stream
    q.interpolate(exp(-((x-0.5-0.1*i)**2)/(0.2**2/2)))
q1_list = [Function(V) for _ in range(m)]
q2_list = [Function(V) for _ in range(m)]

u_list = []
for i in range(m):
    u = Function(W)
    u_list.append(u)
u = Function(W) #input to solver
for i, u in enumerate(u_list):
    # Example: Change frequency for each stream
    u.interpolate(as_vector([0.5*(1 + sin(2*pi*(i+1)*x))]))
u1_list = [Function(W) for _ in range(m)]
u2_list = [Function(W) for _ in range(m)]


dq_trial = TrialFunction(V)
psi = TestFunction(V)
a = psi*dq_trial*dx
us = Function(W)
n = FacetNormal(mesh)
un = 0.5*(dot(us, n) + abs(dot(us, n)))
L1 = dtc*(inner(us, grad(psi))*q*dx          
          - (psi('+') - psi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)
q1 = Function(V); q2 = Function(V)
L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2})

# Placeholder for the total change in q
for i,q in enumerate(q_list):
    if i == 0:
        q_total = q
    else:
        q_total += q  
for i,q in enumerate(q1_list):
    if i == 0:
        q1_total = q
    else:
        q1_total += q  
for i,q in enumerate(q2_list):
    if i == 0:
        q2_total = q
    else:
        q2_total += q  
 
dq = Function(V)  # Placeholder for the total change in q


# create the solvers
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)



# 2. Solve Poisson for phi
phi_sol = TrialFunction(Vcg)
v = TestFunction(Vcg)
phi = Function(Vcg)
nullspace = VectorSpaceBasis(constant=True)

# The phi solver
a_phi = inner(grad(phi_sol), grad(v))*dx
Paphi = phi_sol*v*dx + inner(grad(phi_sol), grad(v))*dx
L_phi = q_total*v*dx
L1_phi = q1_total*v*dx
L2_phi = q2_total*v*dx
phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, aP=Paphi)
phi_solver = LinearVariationalSolver(phi_problem, nullspace=nullspace,
                                    solver_parameters={
                                        'ksp_type': 'gmres',
                                        #'ksp_monitor': None,
                                        'ksp_atol': 1.0e-11,
                                        #'ksp_converged_reason':None
                                    })
phi_problem_1 = LinearVariationalProblem(a_phi, L1_phi, phi, aP=Paphi)
phi_solver_1 = LinearVariationalSolver(phi_problem_1, nullspace=nullspace,
                                    solver_parameters={
                                        'ksp_type': 'gmres',
                                        #'ksp_monitor': None,
                                        'ksp_atol': 1.0e-11,
                                        #'ksp_converged_reason':None
                                    })
phi_problem_2 = LinearVariationalProblem(a_phi, L2_phi, phi, aP=Paphi)
phi_solver_2 = LinearVariationalSolver(phi_problem_2, nullspace=nullspace,
                                    solver_parameters={
                                        'ksp_type': 'gmres',
                                        #'ksp_monitor': None,
                                        'ksp_atol': 1.0e-11,
                                        #'ksp_converged_reason':None
                                    })

du = Function(W)

# The u solver
du_trial = TrialFunction(W)
u_test = TestFunction(W)
a = inner(u_test, du_trial)*dx

L1 = -dtc/mass*inner(u_test, grad(phi))*dx
du = Function(W) #constructor for a class object
du_prob = LinearVariationalProblem(a, L1, du)
du_solv = LinearVariationalSolver(du_prob)

t = 0.0
step = 0
output_freq = 20
outfile = VTKFile("advection.pvd")
outfile.write(*q_list, phi, *u_list)

while t < T - 0.5*dt:
    phi_solver.solve()
    for i in range(m):

        us.assign(u_list[i]) 
        q.assign(q_list[i])

        solv1.solve()
        du_solv.solve()
        q1_list[i].assign(q_list[i] + dq)
        u1_list[i].assign(u_list[i] + du)
        
    phi_solver_1.solve()

    for i in range(m):
        
        us.assign(u1_list[i])
        solv2.solve()
        du_solv.solve()
        q2_list[i].assign(0.75*q_list[i] + 0.25*(q1_list[i] + dq))
        u2_list[i].assign(0.75*u_list[i] + 0.25*(u1_list[i] + du))
    
    phi_solver_2.solve()
    for i in range(m):
        us.assign(u2_list[i])
        solv3.solve()
        du_solv.solve()
        q_list[i].assign((1.0/3.0)*q_list[i] + (2.0/3.0)*(q2_list[i] + dq))
        u_list[i].assign((1.0/3.0)*u_list[i] + (2.0/3.0)*(u2_list[i] + du))
    
    step += 1
    t += dt

    if step % output_freq == 0:
        outfile.write(*q_list, phi,*u_list)
        print("t=", t)


