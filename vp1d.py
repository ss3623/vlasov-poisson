from firedrake import *

ncells = 50
L = 8*pi
base_mesh = PeriodicIntervalMesh(ncells, L)

H = 10.0
nlayers = 50
mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=H/nlayers)

x, v = SpatialCoordinate(mesh)
mesh.coordinates.interpolate(as_vector([x, v-H/2]))

V = FunctionSpace(mesh, 'DQ', 1)

Wbar = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

fn = Function(V, name="density")
A = Constant(0.05)
k = Constant(0.5)
fn.interpolate(v**2*exp(-v**2/2)*(1 + A*cos(k*x))/(2*pi)**0.5)

One = Function(V).assign(1.0)
fbar = assemble(fn*dx)/assemble(One*dx)

phi = Function(Wbar, name="potential")

fstar = Function(V)

psi = TestFunction(Wbar)
dphi = TrialFunction(Wbar)
phi_eqn = dphi.dx(0)*psi.dx(0)*dx - H*(fstar-fbar)*psi*dx

nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)

shift_eqn = dphi.dx(0)*psi.dx(0)*dx + dphi*psi*dx

phi_problem = LinearVariationalProblem(lhs(phi_eqn), rhs(phi_eqn),
                                      phi, aP=shift_eqn)

params = {
  'ksp_type': 'gmres',
  'pc_type': 'lu',
  'ksp_rtol': 1.0e-8,
  }
phi_solver = LinearVariationalSolver(phi_problem,
                                    nullspace=nullspace,
                                    solver_parameters=params)

dtc = Constant(0)

df_out = Function(V)

q = TestFunction(V)
u = as_vector([v, -phi.dx(0)])
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))
df = TrialFunction(V)
df_a = q*df*dx

dS = dS_h + dS_v

df_L = dtc*(div(u*q)*fstar*dx
  - (q('+') - q('-'))*(un('+')*fstar('+') - un('-')*fstar('-'))*dS
  - conditional(dot(u, n) > 0, q*dot(u, n)*fstar, 0.)*ds_tb
   )

df_problem = LinearVariationalProblem(df_a, df_L, df_out)
df_solver = LinearVariationalSolver(df_problem)

T = 50.0
t = 0.
ndump = 100
dumpn = 0
nsteps = 5000
dt = T/nsteps
dtc.assign(dt)

f1 = Function(V)
f2 = Function(V)

outfile = VTKFile("vlasov.pvd")
projected = VTKFile("proj_vp1d.pvd", project_output=True)

fstar.assign(fn)
phi_solver.solve()
outfile.write(fn, phi)
phi.assign(.0)


#----------the moments stuff------------------------------------

m_trial = TrialFunction(Wbar) #moment
r_test = TestFunction(Wbar)
a_moment = r_test * m_trial * dx
L_moment = H * r_test * v *fn *dx #w(v) = v, edit this later?
moment_problem = LinearVariationalProblem(a_moment, L_moment, m_trial)
moment_solver = LinearVariationalSolve(moment_problem)
moment_solver.solve()

for step in ProgressBar("Timestep").iter(range(nsteps)):

   fstar.assign(fn)
   phi_solver.solve()
   df_solver.solve()
   f1.assign(fn + df_out)

   fstar.assign(f1)
   phi_solver.solve()
   df_solver.solve()
   f2.assign(3*fn/4 + (f1 + df_out)/4)

   fstar.assign(f2)
   phi_solver.solve()
   df_solver.solve()
   fn.assign(fn/3 + 2*(f2 + df_out)/3)
   t += dt

   dumpn += 1
   if dumpn % ndump == 0:
       dumpn = 0
       outfile.write(fn, phi)
       projected.write(fn,phi)
       
with Checkpointfile("vlasov_checkpoint.h5",'w' as afile):
    afile.save_mesh(mesh, name = "2d_mesh")
    afile.save_function(fn,name = "fn")
    afile.save_function(phi, name = "phi")

