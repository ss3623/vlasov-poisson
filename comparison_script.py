from firedrake import *

# Load and compare both results
print("=== MOMENT COMPARISON ===")

# 1. Load Multistream data
with CheckpointFile("multistream_checkpoint.h5", 'r') as afile:
    mesh_1d = afile.load_mesh("1d_mesh")
    q_list = []
    u_list = []
    for i in range(4):
        q_list.append(afile.load_function(mesh_1d, f"q_{i+1}"))
        u_list.append(afile.load_function(mesh_1d, f"u_{i+1}"))

# 2. Compute multistream moment
V_dg = FunctionSpace(mesh_1d, "DG", 1)
moment_ms = Function(V_dg)
moment_expr = sum([ui[0] * qi for ui, qi in zip(u_list, q_list)])  # w(v) = v
moment_ms.interpolate(moment_expr)
multistream_final = assemble(moment_ms * dx)

print(f"Multistream final moment: {multistream_final:.6f}")

# 3. Load Vlasov data  
with CheckpointFile("vlasov_checkpoint.h5", 'r') as afile:
    mesh_2d = afile.load_mesh("2d_mesh")
    fn_vlasov = afile.load_function(mesh_2d, "fn")

# 4. Compute Vlasov moment (same as in your Vlasov code)
x, v = SpatialCoordinate(mesh_2d)
Wbar = FunctionSpace(mesh_2d, 'CG', 1, vfamily='R', vdegree=0)
m = Function(Wbar)
vlasov_moment_expr = 10.0 * v * fn_vlasov  # H * v * fn
vlasov_final = assemble(vlasov_moment_expr * dx)

print(f"Vlasov final moment: {vlasov_final:.6f}")
print(f"Difference: {abs(vlasov_final - multistream_final):.6f}")