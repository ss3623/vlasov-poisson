'''Load 2D Vlasov data from checkpoints
Load multistream data from checkpoints
Check if files exist before trying to load them
Handle errors
Provide a function to check what data is available'''

from firedrake import *
import numpy as np
import os

print("Current directory:", os.getcwd())
print("Parent directory contents:", os.listdir(".."))
#breakpoint()





def load_multistream_data(M, T, checkpoint_type="final"):
    """Load multistream data for given M, T, and checkpoint type"""
    
    if checkpoint_type == "initial":
        checkpoint_file = f"../ms_init/M{M}_T_{T}_init_cp.h5"
    elif checkpoint_type == "final":
        checkpoint_file = f"../ms_final/M{M}_T_{T}_final_checkpoint.h5"
    else:
        print(f"Invalid checkpoint_type: {checkpoint_type}")
        return None, None, None
    
    if os.path.exists(checkpoint_file):
        try:
            with CheckpointFile(checkpoint_file, 'r') as afile:
                mesh = afile.load_mesh("1d_mesh")
                
                # Load q and u functions for each stream
                q_list = []
                u_list = []
                for i in range(M):
                    q = afile.load_function(mesh, f"q_{i+1}")
                    u = afile.load_function(mesh, f"u_{i+1}")
                    q_list.append(q)
                    u_list.append(u)
                    
            return mesh, q_list, u_list
            
        except Exception as e:
            print(f"Error loading M={M}, T={T}, {checkpoint_type}: {e}")
            return None, None, None
    else:
        print(f"File not found: {checkpoint_file}")
        return None, None, None

def load_2d_vlasov_data(T, checkpoint_type="final"):
    """Load 2D Vlasov data for given T and checkpoint type"""
    
    if checkpoint_type == "initial":
        checkpoint_file = f"../vlasov_init/T_{T}_init_checkpoint.h5"
    elif checkpoint_type == "final":
        checkpoint_file = f"../vlasov_final/T_{T}_final_checkpoint.h5"
    else:
        print(f"Invalid checkpoint_type: {checkpoint_type}")
        return None, None, None
    
    if os.path.exists(checkpoint_file):
        try:
            with CheckpointFile(checkpoint_file, 'r') as afile:
                mesh_2d = afile.load_mesh("2d_mesh")  # Check what name you used!
                fn = afile.load_function(mesh_2d, "fn")
                phi = afile.load_function(mesh_2d, "phi")
                    
            return mesh_2d, fn, phi
            
        except Exception as e:
            print(f"Error loading 2D Vlasov T={T}, {checkpoint_type}: {e}")
            return None, None, None
    else:
        print(f"File not found: {checkpoint_file}")
        return None, None, None
mesh_2d, fn, phi = load_2d_vlasov_data(0.1, "final")