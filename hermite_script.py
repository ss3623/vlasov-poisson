from scipy.special import roots_hermite
import numpy as np

# Fixed velocities for streams
v_fixed = np.array([-3.0, -1.0, 1.0, 3.0])

# Get Hermite weights
v_hermite, weights = roots_hermite(M)

for i in range(M):
    # Fixed velocities
    u_list[i].interpolate(as_vector([v_fixed[i]]))
    
    # Hermite-weighted q values
    q_list[i].interpolate(weights[i] * exp(-(x-0.5)**2/(0.05**2)))