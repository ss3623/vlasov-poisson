import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from numpy.polynomial.hermite import hermgauss

def hermite_quadrature_demo():
    """
    Demonstrate Hermite quadrature with several examples
    """
    print("=" * 60)
    print("HERMITE QUADRATURE RULE DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Example 1: Simple polynomial
    print("EXAMPLE 1: Integrating f(x) = x²")
    print("-" * 40)
    
    def f1(x):
        return x**2
    
    # Analytical result: ∫_{-∞}^{∞} e^(-x²) x² dx = √π/2
    analytical_1 = np.sqrt(np.pi) / 2
    print(f"Analytical result: {analytical_1:.6f}")
    
    # Test different numbers of quadrature points
    for n in [1, 2, 3, 4]:
        x, w = hermgauss(n)
        numerical = np.sum(w * f1(x))
        error = abs(numerical - analytical_1)
        print(f"n={n}: {numerical:.6f}, error = {error:.2e}")
    
    print()
    
    # Example 2: Gaussian moment (higher order)
    print("EXAMPLE 2: Integrating f(x) = x⁴ (4th moment)")
    print("-" * 40)
    
    def f2(x):
        return x**4
    
    # Analytical result: ∫_{-∞}^{∞} e^(-x²) x⁴ dx = 3√π/4
    analytical_2 = 3 * np.sqrt(np.pi) / 4
    print(f"Analytical result: {analytical_2:.6f}")
    
    for n in [1, 2, 3, 4, 5]:
        x, w = hermgauss(n)
        numerical = np.sum(w * f2(x))
        error = abs(numerical - analytical_2)
        print(f"n={n}: {numerical:.6f}, error = {error:.2e}")
    
    print()
    
    # Example 3: Non-polynomial function
    print("EXAMPLE 3: Integrating f(x) = cos(x) (non-polynomial)")
    print("-" * 40)
    
    def f3(x):
        return np.cos(x)
    
    # Analytical result: ∫_{-∞}^{∞} e^(-x²) cos(x) dx = √π * e^(-1/4)
    analytical_3 = np.sqrt(np.pi) * np.exp(-0.25)
    print(f"Analytical result: {analytical_3:.6f}")
    
    for n in [2, 4, 6, 8, 10]:
        x, w = hermgauss(n)
        numerical = np.sum(w * f3(x))
        error = abs(numerical - analytical_3)
        print(f"n={n}: {numerical:.6f}, error = {error:.2e}")
    
    print()

def visualize_hermite_quadrature():
    """
    Visualize how Hermite quadrature points and weights work
    """
    print("VISUALIZATION: Hermite quadrature points and weights")
    print("-" * 50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Show quadrature points for different orders
    x_plot = np.linspace(-3, 3, 1000)
    weight_func = np.exp(-x_plot**2)
    
    ax1.plot(x_plot, weight_func, 'k-', linewidth=2, label='$e^{-x^2}$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Weight function')
    ax1.set_title('Hermite Quadrature Points')
    ax1.grid(True, alpha=0.3)
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, n in enumerate([2, 3, 4, 5]):
        x, w = hermgauss(n)
        # Scale weights for visualization
        w_scaled = w / np.max(w) * 0.8
        ax1.scatter(x, np.exp(-x**2), color=colors[i], s=100*w_scaled, 
                   alpha=0.7, label=f'n={n}')
    
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Right plot: Convergence example
    def test_function(x):
        return x**2 * np.exp(-0.5*x**2)
    
    # "True" value using high-order quadrature
    x_ref, w_ref = hermgauss(20)
    true_value = np.sum(w_ref * test_function(x_ref))
    
    n_values = range(1, 11)
    errors = []
    
    for n in n_values:
        x, w = hermgauss(n)
        approx = np.sum(w * test_function(x))
        error = abs(approx - true_value)
        errors.append(error)
    
    ax2.semilogy(n_values, errors, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of quadrature points (n)')
    ax2.set_ylabel('Absolute error')
    ax2.set_title('Convergence of Hermite Quadrature')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reference value: {true_value:.8f}")

def physics_application():
    """
    Physics application: Computing moments of Maxwell-Boltzmann distribution
    """
    print("\nPHYSICS APPLICATION: Maxwell-Boltzmann Distribution Moments")
    print("-" * 60)
    print("Computing <v²> and <v⁴> for Maxwell-Boltzmann distribution")
    print("f(v) ∝ exp(-mv²/2kT), normalized")
    print()
    
    # Parameters
    m = 1.0  # mass
    k = 1.0  # Boltzmann constant  
    T = 1.0  # temperature
    
    # Transform to standard Hermite form: exp(-x²)
    # Let x = v√(m/2kT), then v = x√(2kT/m)
    scale = np.sqrt(2*k*T/m)
    
    def velocity_squared_moment(x):
        v = x * scale
        return v**2
    
    def velocity_fourth_moment(x):
        v = x * scale  
        return v**4
    
    # Analytical results for Maxwell-Boltzmann
    analytical_v2 = 3*k*T/m  # <v²> = 3kT/m
    analytical_v4 = 15*(k*T/m)**2  # <v⁴> = 15(kT/m)²
    
    print(f"Analytical <v²>: {analytical_v2:.6f}")
    print(f"Analytical <v⁴>: {analytical_v4:.6f}")
    print()
    
    for n in [3, 5, 7, 10]:
        x, w = hermgauss(n)
        
        # The Hermite quadrature gives us the integral over exp(-x²)
        # We need to normalize by √π for the probability distribution
        normalization = 1.0 / np.sqrt(np.pi)
        
        v2_moment = normalization * np.sum(w * velocity_squared_moment(x))
        v4_moment = normalization * np.sum(w * velocity_fourth_moment(x))
        
        error_v2 = abs(v2_moment - analytical_v2)
        error_v4 = abs(v4_moment - analytical_v4)
        
        print(f"n={n:2d}: <v²> = {v2_moment:.6f} (error: {error_v2:.2e})")
        print(f"      <v⁴> = {v4_moment:.6f} (error: {error_v4:.2e})")
        print()

# Run all examples
if __name__ == "__main__":
    hermite_quadrature_demo()
    visualize_hermite_quadrature()
    physics_application()
    
    # Show the actual quadrature points and weights for small n
    print("\nHERMITE QUADRATURE POINTS AND WEIGHTS")
    print("-" * 40)
    for n in [2, 3, 4]:
        x, w = hermgauss(n)
        print(f"\nn = {n}:")
        for i in range(n):
            print(f"  x_{i+1} = {x[i]:8.5f}, w_{i+1} = {w[i]:8.5f}")