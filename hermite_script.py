from scipy.special import roots_hermite
import numpy as np

# Get Hermite quadrature points and weights for M points
v_hermite, weights = roots_hermite(M)
v_scaled = v_hermite / np.sqrt(2.0)
weights_scaled = weights / np.sqrt(2.0)

for i in range(M):
    u_list[i].interpolate(as_vector([v_scaled[i]]))
    center = 0.3 + i * 0.4  # your original centers
    v_i = v_scaled[i]
    q_list[i].interpolate(weights_scaled[i] * exp(-(x-center)**2/(0.05**2)) / exp(-v_i**2))

print("Hermite quadrature initialization:")
for i in range(M):
    print(f"Stream {i+1}: v = {v_scaled[i]:.3f}, weight = {weights_scaled[i]:.6f}")
