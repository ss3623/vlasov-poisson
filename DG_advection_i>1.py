from firedrake import *
import math

ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)

V = FunctionSpace(mesh, "DG", 1)         
W = VectorFunctionSpace(mesh, "CG", 1) 

x, = SpatialCoordinate(mesh)
m = 2 #how many streams?
qm = [] #store it allll in a list
qm_init = []
for i in range(m):
    q = Function(V)
    qm.append(q)

for q in qm:
    q.interpolate(exp(-(x-0.5)**2/(0.2**2/2))) 
    q_init = Function(V).assign(q)
    qm_init.append(q_init)

#q1_init = Function(V).assign(q1)
#q2_init = Function(V).assign(q2)
#make a list for u's
um = []
for i in range(m):
    u = Function(W)
    um.append(u)
for u in um:
    u.interpolate(0.5*(1 + sin(2*math.pi*x))) #change something in f

# --- Time-stepping parameters ---
T = 3.0
dt = T/500.0
dtc = Constant(dt)
m = Constant(1.0)
