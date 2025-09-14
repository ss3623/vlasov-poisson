from firedrake import *
import numpy as np
from scipy.special import roots_hermite
import os
from params import P

def w(v):
    """Weight function for moment calculation"""
    return v**2

def run_multistream(M=None, T=None):
    if M is None: 
        M = P.M
    if T is None: 
        T = P.T
    
    print(f"Running multistream simulation with M={M}, T={T}")
    
    # Create mesh and function spaces
    mesh = P.make_1d_mesh()
    x = SpatialCoordinate(mesh)[0]
    V_dg = FunctionSpace(mesh, "DG", 1)
    V_cg = FunctionSpace(mesh, "CG", 1)
    W_cg = VectorFunctionSpace(mesh, "CG", 1)

    # Initialize stream functions
    q_list = [Function(V_dg, name=f"q{i+1}") for i in range(M)]
    u_list = [Function(W_cg, name=f"u{i+1}") for i in range(M)]
    phi = Function(V_cg, name="phi")

    # Set up Hermite quadrature
    v_points, weights = roots_hermite(M)
    velocities = v_points * np.sqrt(2)
    adjusted_weights = weights * np.sqrt(2)

    # Initialize streams with proper initial conditions
    spatial_part = (1 + P.A*cos(P.k*x))/sqrt(2*pi)
    for i in range(M):
        u_list[i].interpolate(as_vector([velocities[i]]))
        q_list[i].interpolate(adjusted_weights[i] * spatial_part)
    
    # Compute initial total charge for verification
    q_total = Function(V_dg, name="q_total")
    q_total.interpolate(sum(q_list))
    print(f"Initial total charge: {assemble(q_total * dx)}")
        
    # Set up time stepping
    if T < 0.1:
        nsteps = max(100, int(T * 5000))
    else:
        nsteps = max(500, int(T * 200))

    dt = T / nsteps
    dtc = Constant(dt)
    print(f"Using {nsteps} time steps, dt = {dt}")
    
    # Stage functions for RK3
    q1_list = [Function(V_dg) for i in range(M)]
    q2_list = [Function(V_dg) for i in range(M)]
    u1_list = [Function(W_cg) for i in range(M)]
    u2_list = [Function(W_cg) for i in range(M)]
    
    # Temporary functions for solvers
    q_temp = Function(V_dg)
    u_temp = Function(W_cg)

    # Advection equation setup: ∂q/∂t + ∇·(uq) = 0
    dq_trial = TrialFunction(V_dg)
    psi = TestFunction(V_dg)
    a_adv = psi * dq_trial * dx
    n = FacetNormal(mesh)
    un = 0.5 * (dot(u_temp, n) + abs(dot(u_temp, n)))  # upwind flux
    L1 = dtc * (inner(u_temp, grad(psi)) * q_temp * dx
            - (psi('+') - psi('-')) * (un('+') * q_temp('+') - un('-') * q_temp('-')) * dS)
    
    # Stage 2 and 3 forms
    q1_temp = Function(V_dg)
    q2_temp = Function(V_dg)
    L2 = replace(L1, {q_temp: q1_temp})
    L3 = replace(L1, {q_temp: q2_temp})
    
    dq = Function(V_dg)
    
    # Advection solvers
    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    advection_prob1 = LinearVariationalProblem(a_adv, L1, dq)
    advection_solv1 = LinearVariationalSolver(advection_prob1, solver_parameters=params)
    advection_prob2 = LinearVariationalProblem(a_adv, L2, dq)
    advection_solv2 = LinearVariationalSolver(advection_prob2, solver_parameters=params)
    advection_prob3 = LinearVariationalProblem(a_adv, L3, dq)
    advection_solv3 = LinearVariationalSolver(advection_prob3, solver_parameters=params)
    
    # Poisson equation setup: -∇²φ = Σ q_i
    phi_trial = TrialFunction(V_cg)
    phi_test = TestFunction(V_cg)
    nullspace = VectorSpaceBasis(constant=True)
    
    a_phi = inner(grad(phi_trial), grad(phi_test)) * dx
    Paphi = phi_trial * phi_test * dx + a_phi
    
    # Total charges for each stage
    q_total_current = sum(q_list)
    q1_total = sum(q1_list)
    q2_total = sum(q2_list)

    # Poisson solvers for each RK3 stage
    poisson_problem = LinearVariationalProblem(a_phi, q_total_current*phi_test*dx, phi, aP=Paphi)
    poisson_solver = LinearVariationalSolver(poisson_problem, nullspace=nullspace,
                        solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1.0e-11})
    
    poisson_problem1 = LinearVariationalProblem(a_phi, q1_total * phi_test * dx, phi, aP=Paphi)
    poisson_solver1 = LinearVariationalSolver(poisson_problem1, nullspace=nullspace,
                        solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1.0e-11})
    
    poisson_problem2 = LinearVariationalProblem(a_phi, q2_total * phi_test * dx, phi, aP=Paphi)
    poisson_solver2 = LinearVariationalSolver(poisson_problem2, nullspace=nullspace,
                        solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1.0e-11})
    
    # Velocity equation setup: ∂u/∂t = -∇φ/m (assuming m=1)
    du_trial = TrialFunction(W_cg)
    u_test = TestFunction(W_cg)
    a_vel = inner(u_test, du_trial) * dx
    L_vel = -dtc * inner(u_test, grad(phi)) * dx
    du = Function(W_cg)
    velocity_problem = LinearVariationalProblem(a_vel, L_vel, du)
    velocity_solver = LinearVariationalSolver(velocity_problem)
    
    # Initial Poisson solve
    poisson_solver.solve()

    # Save initial conditions
    init_vtk_dir = 'vtk_ms_init'
    os.makedirs(init_vtk_dir, exist_ok=True)
    outfile_init = VTKFile(os.path.join(init_vtk_dir, f"M{M}_T_{T}_ms_init.pvd"))
    q_total.interpolate(sum(q_list))  # Update q_total
    outfile_init.write(*q_list, *u_list, q_total, phi)

    init_dir = 'ms_init'
    os.makedirs(init_dir, exist_ok=True)
    filename = f"M{M}_T_{T}_init_cp.h5"
    with CheckpointFile(os.path.join(init_dir, filename), 'w') as afile:
        afile.save_mesh(mesh, "mesh_1d")
        for i, (q, u) in enumerate(zip(q_list, u_list)):
            afile.save_function(q, name=f"q_{i+1}")
            afile.save_function(u, name=f"u_{i+1}")

    # SSP-RK3 time stepping loop
    t = 0.0
    step = 0
    while t < T - 0.5*dt:
        # Stage 1: q¹ = qⁿ + dt*L(qⁿ,uⁿ), u¹ = uⁿ + dt*(-∇φⁿ)
        poisson_solver.solve()
        for i in range(M):
            u_temp.assign(u_list[i])
            q_temp.assign(q_list[i])
            advection_solv1.solve()
            velocity_solver.solve()
            q1_list[i].assign(q_list[i] + dq)
            u1_list[i].assign(u_list[i] + du)
        
        # Stage 2: q² = (3/4)qⁿ + (1/4)(q¹ + dt*L(q¹,u¹))
        poisson_solver1.solve()
        for i in range(M):
            u_temp.assign(u1_list[i])
            q1_temp.assign(q1_list[i])
            advection_solv2.solve()
            velocity_solver.solve()
            q2_list[i].assign(0.75*q_list[i] + 0.25*(q1_list[i] + dq))
            u2_list[i].assign(0.75*u_list[i] + 0.25*(u1_list[i] + du))
        
        # Stage 3: qⁿ⁺¹ = (1/3)qⁿ + (2/3)(q² + dt*L(q²,u²))
        poisson_solver2.solve()
        for i in range(M):
            u_temp.assign(u2_list[i])
            q2_temp.assign(q2_list[i])
            advection_solv3.solve()
            velocity_solver.solve()
            q_list[i].assign((1.0/3.0)*q_list[i] + (2.0/3.0)*(q2_list[i] + dq))
            u_list[i].assign((1.0/3.0)*u_list[i] + (2.0/3.0)*(u2_list[i] + du))
        
        t += dt
        step += 1
        
        # Print progress occasionally
        if step % max(1, nsteps//10) == 0:
            print(f"Step {step}/{nsteps}, t = {t:.6f}")
    
    # Final output
    final_vtk_dir = 'vtk_ms_final'
    os.makedirs(final_vtk_dir, exist_ok=True)
    outfile_final = VTKFile(os.path.join(final_vtk_dir, f"M{M}_T_{T}_ms_final.pvd"))
    q_total.interpolate(sum(q_list))  # Update q_total
    outfile_final.write(*q_list, *u_list, q_total, phi)

    # Save final checkpoint
    final_dir = 'ms_final'
    os.makedirs(final_dir, exist_ok=True)
    with CheckpointFile(os.path.join(final_dir, f"M{M}_T_{T}_final_checkpoint.h5"), 'w') as afile:
        afile.save_mesh(mesh, "mesh_1d")
        for i, (q, u) in enumerate(zip(q_list, u_list)):
            afile.save_function(q, name=f"q_{i+1}")
            afile.save_function(u, name=f"u_{i+1}")
        # Also save the final moment
        moment = Function(V_dg, name="moment")
        moment.interpolate(sum([w(u[0]) * q for q, u in zip(q_list, u_list)]))
        afile.save_function(moment, name="moment")
    
    print(f"Simulation completed! Final time: {t:.6f}")
    
    moment = Function(V_dg, name="moment")
    moment.interpolate(sum([w(u[0]) * q for q, u in zip(q_list, u_list)]))
    
    final_charge = assemble(sum(q_list) * dx)
    moment_norm = norm(moment)
    print(f"Final total charge: {final_charge}")
    print(f"Final moment norm: {moment_norm}")
    
    return moment

if __name__ == "__main__":
    # Test with different parameters
    run_multistream(M=3, T=1.0)
    print("Simulation completed successfully!")