import math
from firedrake import *

class Params:
    k = 0.5          
    A = 0.05         
    H = 10.0         
    L = 8 * math.pi  
    T = 0.01              
    ncells = 50      
    nlayers = 200    
    M = 5            
    
    
    @property 
    def layer_height(self):
        return self.H / self.nlayers
    
    def make_1d_mesh(self):
        return PeriodicIntervalMesh(self.ncells,self.L,name = "1d_mesh")
    
    def make_2d_mesh(self):
        base = PeriodicIntervalMesh(self.ncells, self.L)
        mesh = ExtrudedMesh(base, layers=self.nlayers, 
                           layer_height=self.layer_height, name="2d_mesh")
        x, v = SpatialCoordinate(mesh)
        mesh.coordinates.interpolate(as_vector([x, v - self.H/2]))
        return mesh
    
    @staticmethod
    def w(v):
        return v**2

P = Params()
