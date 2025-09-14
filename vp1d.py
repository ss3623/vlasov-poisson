from firedrake import *
from params import P
import os

def run_2d_vlasov(T=None):
    if T is None: 
        T = P.T
    
    print(f"Running 2D Vlasov simulation for T={T}")
    
    # Create mesh and coordinates
    mesh = P.make_2d_mesh()
    x, v = SpatialCoordinate(mesh)
    
    # Set up time stepping
    if T < 0.1:
        nsteps = max(100, int(T * 5000))
    else:
        nsteps = max(500, int(T * 200))
    
    dt = T/nsteps
    dtc = Constant(dt)
    print(f"Using {nsteps} time steps, dt = {dt}")
    
    # Function spaces
    V = FunctionSpace(mesh, 'DQ', 1)
    Wbar = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    
    # Initialize distribution function
    fn = Function(V, name="density")
    fn.interpolate(exp(-v**2/2) * (1 + P.A*cos(P.k*x)) / sqrt(2*pi))
    
    # Stage functions for RK3
    f1 = Function(V)
    f2 = Function(V)
    
    # Compute initial average
    One = Function(V).assign(1.0)
    fbar = assemble(fn*dx)/assemble(One*dx)
    print(f"Initial average density: {fbar}")
    
    # Potential function and temporary function
    phi = Function(Wbar, name="potential")
    fstar = Function(V)
    
    # Poisson equation setup: ∂²φ/∂x² = H*(f* - f̄)
    psi = TestFunction(Wbar)
    dphi = TrialFunction(Wbar)
    phi_eqn = dphi.dx(0)*psi.dx(0)*dx - P.H*(fstar-fbar)*psi*dx
    
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    shift_eqn = dphi.dx(0)*psi.dx(0)*dx + dphi*psi*dx
    phi_problem = LinearVariationalProblem(lhs(phi_eqn), rhs(phi_eqn), phi, aP=shift_eqn)
    
    params = {
        'ksp_type': 'gmres',
        'pc_type': 'lu',
        'ksp_rtol': 1.0e-8,
    }
    phi_solver = LinearVariationalSolver(phi_problem, nullspace=nullspace, solver_parameters=params)
    
    # Vlasov equation setup: ∂f/∂t + v*∂f/∂x - (∂φ/∂x)*∂f/∂v = 0
    df_out = Function(V)
    q = TestFunction(V)
    u = as_vector([v, -phi.dx(0)])  # velocity field [v_x, v_v]
    n = FacetNormal(mesh)
    un = 0.5*(dot(u, n) + abs(dot(u, n)))  # upwind flux
    df = TrialFunction(V)
    df_a = q*df*dx
    
    dS = dS_h + dS_v
    df_L = dtc*(div(u*q)*fstar*dx
        - (q('+') - q('-'))*(un('+')*fstar('+') - un('-')*fstar('-'))*dS
        - conditional(dot(u, n) > 0, q*dot(u, n)*fstar, 0.)*ds_tb
    )
    
    df_problem = LinearVariationalProblem(df_a, df_L, df_out)
    df_solver = LinearVariationalSolver(df_problem)
    
    # Initial solve
    fstar.assign(fn)
    phi_solver.solve()
    phi.assign(0.0)  # Reset phi after initial solve
    
    # Save initial conditions
    init_vtk_dir = 'vtk_vlasov_init'
    os.makedirs(init_vtk_dir, exist_ok=True)
    outfile_init = VTKFile(os.path.join(init_vtk_dir, f"T_{T}_vlasov_init.pvd"))
    outfile_init.write(fn, phi)
    
    init_dir = 'vlasov_init'
    os.makedirs(init_dir, exist_ok=True)
    with CheckpointFile(os.path.join(init_dir, f"T_{T}_init_checkpoint.h5"), 'w') as afile:
        afile.save_mesh(mesh, "mesh_2d")
        afile.save_function(fn, name="fn_init")
        afile.save_function(phi, name="phi_init")
    
    # SSP-RK3 time stepping loop
    t = 0.0
    for step in range(nsteps):
        # Stage 1: f¹ = fⁿ + dt*L(fⁿ)
        fstar.assign(fn)
        phi_solver.solve()
        df_solver.solve()
        f1.assign(fn + df_out)
        
        # Stage 2: f² = (3/4)fⁿ + (1/4)(f¹ + dt*L(f¹))
        fstar.assign(f1)
        phi_solver.solve()
        df_solver.solve()
        f2.assign(3*fn/4 + (f1 + df_out)/4)
        
        # Stage 3: fⁿ⁺¹ = (1/3)fⁿ + (2/3)(f² + dt*L(f²))
        fstar.assign(f2)
        phi_solver.solve()
        df_solver.solve()
        fn.assign(fn/3 + 2*(f2 + df_out)/3)
        
        t += dt
        
        # Print progress occasionally
        if step % max(1, nsteps//10) == 0:
            print(f"Step {step}/{nsteps}, t = {t:.6f}")
    
    # Save final results
    final_vtk_dir = 'vtk_vlasov_final'
    os.makedirs(final_vtk_dir, exist_ok=True)
    outfile_final = VTKFile(os.path.join(final_vtk_dir, f"T_{T}_vlasov_final.pvd"))
    outfile_final.write(fn, phi)
    
    final_dir = 'vlasov_final'
    os.makedirs(final_dir, exist_ok=True)
    with CheckpointFile(os.path.join(final_dir, f"T_{T}_final_checkpoint.h5"), 'w') as afile:
        afile.save_mesh(mesh,"mesh_2d")
        afile.save_function(fn, name="fn")
        afile.save_function(phi, name="phi")
    
    print(f"Simulation completed! Final time: {t:.6f}")
    
    # Print diagnostics
    final_mass = assemble(fn * dx)
    print(f"Final total mass: {final_mass}")
    
    return None

if __name__ == "__main__":
    run_2d_vlasov(T=1.0)
    print("2D Vlasov simulation completed successfully!")