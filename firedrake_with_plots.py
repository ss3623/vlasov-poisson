from firedrake import *
import math
import numpy as np
import matplotlib.pyplot as plt

# --- Mesh and function spaces ---
ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)
V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, = SpatialCoordinate(mesh)

# --- Initial conditions ---
u = Function(W)
q = Function(V).interpolate(1 + 0.01 * cos(2 * pi * x))
q_init = Function(V).assign(q)

# --- Time stepping parameters ---
T = 3
dt = T/500
dtc = Constant(dt)
mass = Constant(1.0)

# --- Snapshot settings ---
snapshot_times = [0.0, T/2, T]
tol = dt/2
snapshots = {}

def extract_field(field):
    # Extract 1D coordinates and values from a Firedrake Function
    coords = mesh.coordinates.dat.data_ro[:, 0].copy()
    vals = field.dat.data_ro.copy()
    # If vector-valued, take first component
    if vals.ndim > 1:
        vals = vals[:, 0]
    idx = np.argsort(coords)
    return coords[idx], vals[idx]

# --- Setup solvers for q-advection ---
dq_trial = TrialFunction(V)
psi = TestFunction(V)
a_q = psi * dq_trial * dx

us = Function(W)
n = FacetNormal(mesh)
un = 0.5 * (dot(us, n) + abs(dot(us, n)))

L1 = dtc * (
    inner(us, grad(psi)) * q * dx
    - (psi('+') - psi('-')) * (un('+') * q('+') - un('-') * q('-')) * dS
)

q1 = Function(V)
q2 = Function(V)
L2 = replace(L1, {q: q1})
L3 = replace(L1, {q: q2})

dq = Function(V)

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a_q, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a_q, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a_q, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

# --- Setup solver for phi (Poisson) ---
Vcg = FunctionSpace(mesh, "CG", 1)
phi_sol = TrialFunction(Vcg)
dphi = TestFunction(Vcg)
phi = Function(Vcg, name="phi")

nullspace = VectorSpaceBasis(constant=True)

aphi = inner(grad(phi_sol), grad(dphi)) * dx
Paphi = phi_sol * dphi * dx + inner(grad(phi_sol), grad(dphi)) * dx
F_phi = q * dphi * dx
phi_problem = LinearVariationalProblem(aphi, F_phi, phi, aP=Paphi)
phi_solver = LinearVariationalSolver(
    phi_problem,
    nullspace=nullspace,
    solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1e-11}
)

# --- Setup solver for u update ---
du_trial = TrialFunction(W)
u_test = TestFunction(W)
a_u = inner(u_test, du_trial) * dx
L_u = -dtc/mass * inner(u_test, grad(phi)) * dx
du = Function(W)
u1 = Function(W)
u2 = Function(W)
du_prob = LinearVariationalProblem(a_u, L_u, du)
du_solv = LinearVariationalSolver(du_prob)

# --- Initial solve and output ---
t = 0.0
step = 0
output_freq = 20

phi_solver.solve()
outfile = VTKFile("advection_1d.pvd")
outfile.write(q, phi, u)

# Record initial snapshot
coords_q, qv = extract_field(q)
_, phiv = extract_field(phi)
_, uv = extract_field(u)
snapshots[0.0] = (coords_q, qv, phiv, uv)

# --- Time-stepping loop ---
while t < T - 0.5 * dt:
    # Stage 1
    phi_solver.solve()
    us.assign(u)
    solv1.solve()
    du_solv.solve()
    q1.assign(q + dq)
    u1.assign(u + du)
    # Stage 2
    phi_solver.solve()
    us.assign(u1)
    solv2.solve()
    du_solv.solve()
    q2.assign(0.75 * q + 0.25 * (q1 + dq))
    u2.assign(0.75 * u + 0.25 * (u1 + du))
    # Stage 3
    phi_solver.solve()
    us.assign(u2)
    solv3.solve()
    du_solv.solve()
    q.assign((1.0/3.0) * q + (2.0/3.0) * (q2 + dq))
    u.assign((1.0/3.0) * u + (2.0/3.0) * (u2 + du))

    step += 1
    t += dt

    # Record mid and final snapshots
    for ts in snapshot_times:
        if ts not in snapshots and abs(t - ts) < tol:
            coords_q, qv = extract_field(q)
            _, phiv = extract_field(phi)
            _, uv = extract_field(u)
            snapshots[ts] = (coords_q, qv, phiv, uv)

    # Periodic VTK output
    if step % output_freq == 0:
        outfile.write(q, phi, u)
        print("t=", t)

# Ensure final snapshot is recorded
if T not in snapshots:
    coords_q, qv = extract_field(q)
    _, phiv = extract_field(phi)
    _, uv = extract_field(u)
    snapshots[T] = (coords_q, qv, phiv, uv)

# --- Plotting snapshots ---
for ts in sorted(snapshots):
    x_vals, q_vals, phi_vals, u_vals = snapshots[ts]
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(x_vals, q_vals)
    axs[0].set_ylabel(r'$q(x)$')
    axs[0].set_title(f'Snapshot at t={ts:.2f}')
    axs[1].plot(x_vals, phi_vals)
    axs[1].set_ylabel(r'$\phi(x)$')
    axs[2].plot(x_vals, u_vals)
    axs[2].set_ylabel(r'$u(x)$')
    axs[2].set_xlabel(r'$x$')
    fig.tight_layout()
    fname = f'vlasovpoisson_t{ts:.2f}.png'
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved plot {fname}")
