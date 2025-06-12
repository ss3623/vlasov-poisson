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

for i, q in enumerate(q_list):
    # Example: Shift center for each stream
    q.interpolate(exp(-((x-0.5-0.1*i)**2)/(0.2**2/2)))

u_list = []
for i in range(m):
    u = Function(W)
    u_list.append(u)

for i, u in enumerate(u_list):
    # Example: Change frequency for each stream
    u.interpolate(0.5*(1 + sin(2*math.pi*(i+1)*x)))

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

q_total = Function(V)
q_total.assign(0)
for q in q_list:
    q_total += q  
dq_list = []
for i in range(m):
    dq = Function(V)
    dq_list.append(dq)

# create the solvers
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}


#DO STUFF FOR Q!!! and dq!!!!



# 2. Solve Poisson for phi
phi_sol = TrialFunction(Vcg)
v = TestFunction(Vcg)
phi = Function(Vcg)
nullspace = VectorSpaceBasis(constant=True)

# The phi solver
a_phi = inner(grad(phi_sol), grad(v))*dx
Paphi = phi_sol*v*dx + inner(grad(phi_sol), grad(v))*dx
L_phi = q_total*v*dx
phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, aP=Paphi)
phi_solver = LinearVariationalSolver(phi_problem, nullspace=nullspace,
                                    solver_parameters={
                                        'ksp_type': 'gmres',
                                        #'ksp_monitor': None,
                                        'ksp_atol': 1.0e-11,
                                        #'ksp_converged_reason':None
                                    })


du_list = [Function(W) for _ in range(m)]
u1 = Function(W); u2 = Function(W)

#construct u solver
du_trial = TrialFunction(W)
u_test = TestFunction(W)
a = inner(u_test, du_trial)*dx
dummy_L1 = -dtc/m*inner(u_test, Constant((0)))*dx
#Placeholder RHS
dummy_du = Function(W)
#Placeholder solution
du_prob = LinearVariationalProblem(a, dummy_L1, dummy_du)
du_solv = LinearVariationalSolver(du_prob)

#----velocity update------
for i in range(m):
    L1 = -dtc/m*inner(u_test, grad(phi)*dx)
    du_prob.L = L1
    du_prob.u = du_list[i]
    du_solv.solve()
    u_list[i].assign(u_list[i]+ du_list[i])



