from firedrake import *
import math
import matplotlib.pyplot as plt

#mesh setup
ncells = 80 
L = 1
mesh = PeriodicIntervalMesh(ncells,L)

#function spaces
V_dg = FunctionSpace(mesh,"DG",1) #for charge density q
V_cg = FunctionSpace(mesh,"CG",1) #for potential phi
W_cg = VectorFunctionSpace(mesh,"CG",1) #for velocity u

#Functions
q = Function(V_dg,name ='q')
u = Function(W_cg,name = 'u')
phi = Function(V_cg,name = 'phi')
m = Function(V_dg, name ='moment') #for moment


#Coordinate
x = SpatialCoordinate(mesh)[0]

#Initial Conditions
q0_expr= exp(-(x-0.5)**2/(0.2**2/2))
q.interpolate(q0_expr) #small initial gaussian bump
m_expr = u[0] * q
m.interpolate(m_expr)


#Time stepping
T = 3
dt = T/500 
t = 0.0

#Constants
dtc = Constant(dt)
mass = Constant(1.0)

#Poisson solver: -∇²φ = q
phi_trial = TrialFunction(V_cg)
phi_test = TestFunction(V_cg)
phi_sol = TrialFunction(V_cg)
a_phi = inner(grad(phi_trial), grad(phi_test)) * dx
L_phi = q * phi_test * dx
Pa_phi = phi_sol*phi_test*dx + inner(grad(phi_sol), grad(phi_test))*dx
#nullspace for periodic B.C.
nullspace = VectorSpaceBasis(constant=True)

poisson_problem = LinearVariationalProblem(a_phi,L_phi,phi,aP = Pa_phi)
poisson_solver = LinearVariationalSolver(poisson_problem, nullspace = nullspace,solver_parameters={
                                        'ksp_type': 'gmres',
                                        #'ksp_monitor': None,
                                        'ksp_atol': 1.0e-11,
                                        #'ksp_converged_reason':None
                                    })
"""
Summary so far:
- Set up function spaces: DG for q, CG for phi, Vector CG for u  
- Created functions: q (charge density), u (velocity), phi (potential)
- Initial conditions: small perturbation in q, zero velocity
- Time stepping parameters: T=3, dt=T/500
- Poisson solver: solves -∇²φ = q for electric potential
"""

#Velocity Solver: ∂u/∂t = -∇φ/m
u_trial = TrialFunction(W_cg)
u_test = TestFunction(W_cg)
du = Function(W_cg) #velocity update increment

a_u = inner(u_test,u_trial)*dx
L_u = -dtc/mass * inner(u_test, grad(phi)) * dx
u1 = Function(W_cg)
u2 = Function(W_cg)

velocity_problem = LinearVariationalProblem(a_u, L_u, du)
velocity_solver = LinearVariationalSolver(velocity_problem)

# Advection solver: ∂q/∂t + ∇·(uq) = 0
dq_trial = TrialFunction(V_dg)
psi = TestFunction(V_dg)
a = psi * dq_trial * dx

us = Function(W_cg)  # velocity for advection
n = FacetNormal(mesh)
un = 0.5 * (dot(us, n) + abs(dot(us, n)))  # upwind flux

L1 = dtc * (inner(us, grad(psi)) * q * dx
           - (psi('+') - psi('-')) * (un('+') * q('+') - un('-') * q('-')) * dS)

# Functions for RK3 stages
q1 = Function(V_dg)
q2 = Function(V_dg)
dq = Function(V_dg)
M0 = Function(V_dg)

# Create 3 problems for 3 RK stages
L2 = replace(L1, {q: q1})
L3 = replace(L1, {q: q2})

# Solvers
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
advection_prob1 = LinearVariationalProblem(a, L1, dq)
advection_solv1 = LinearVariationalSolver(advection_prob1, solver_parameters=params)
advection_prob2 = LinearVariationalProblem(a,L2,dq)
advection_solv2 = LinearVariationalSolver(advection_prob2, solver_parameters=params)
advection_prob3 = LinearVariationalProblem(a, L3, dq)
advection_solv3 = LinearVariationalSolver(advection_prob3, solver_parameters=params)

#output setup
step = 0
output_freq = 20

#solve poisson once before we start
poisson_solver.solve()
 #VTK files for visualisation
outfile = VTKFile("1d1v_single_stream.pvd")
outfile.write(q,phi,u)
# Store initial values for comparison
initial_charge = assemble(q * dx)
initial_moment = assemble(m * dx)

print(f"  Total charge: {initial_charge:.6f}")
print(m)
#print(f"  Total moment: {initial_moment:.6f}")
print("-" * 40)


#begin time loop

while t < T - 0.5*dt:
    poisson_solver.solve()
    us.assign(u) 
    advection_solv1.solve()
    velocity_solver.solve()
    q1.assign(q + dq)
    u1.assign(u + du)

    poisson_solver.solve()
    us.assign(u1)
    advection_solv2.solve()
    velocity_solver.solve()
    q2.assign(0.75*q + 0.25*(q1 + dq))
    u2.assign(0.75*u + 0.25*(u1 + du))

    poisson_solver.solve()
    us.assign(u2)
    advection_solv3.solve()
    velocity_solver.solve()
    q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
    u.assign((1.0/3.0)*u + (2.0/3.0)*(u2 + du))
    
    #update time and output
    step += 1
    t += dt

    if step % output_freq == 0:
        outfile.write(q, phi,u)
        # Calculate current values
        current_charge = assemble(q * dx) 
        M0.interpolate(u[0]**2 * q)
        total_M0 = assemble(M0 * dx)

        # Calculate relative errors
        charge_error = abs(current_charge - initial_charge) / abs(initial_charge)
        moment_error = (total_M0 - initial_moment)                                    

        print(f"Step: {step}, Time: {t:.3f}")
        print(f"  Total charge: {current_charge:.6f} (error: {charge_error:.2e})")
        print(f"  Total moment: {total_M0:.6f} (error: {moment_error:.2e})")
        print("-" * 40)



