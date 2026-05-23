import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class GAV_ACO:

    def __init__(self,
                 n_ants: int = 50,
                 n_iterations: int = 100,
                 # Gaussian alpha parameters (Eq. ia*1)
                 alpha_0: float = 1.0,
                 alpha_max: float = 5.0,
                 alpha_min: float = 0.5,
                 alpha_mu: float = 50.0,  # mean iteration for alpha peak
                 alpha_sigma: float = 30.0,  # std for alpha
                 # Gaussian beta parameters (Eq. ia*1)
                 beta_0: float = 5.0,
                 beta_max: float = 5.0,
                 beta_min: float = 1.0,
                 beta_mu: float = 50.0,  # mean iteration for beta valley
                 beta_sigma: float = 30.0,  # std for beta
                 # Adaptive rho parameters (Eq. ia*2)
                 rho_init: float = 0.5,
                 zeta_threshold: float = 0.001,
                 # Pheromone bounds (Eq. ia*3)
                 z_const: float = 0.5,  # constant for min pheromone
                 Q: float = 100.0,
                 seed: int = None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.n_ants = n_ants
        self.n_iterations = n_iterations

        # Gaussian alpha parameters
        self.alpha_0 = alpha_0
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma

        # Gaussian beta parameters
        self.beta_0 = beta_0
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.beta_mu = beta_mu
        self.beta_sigma = beta_sigma

        # Adaptive rho parameters
        self.rho_init = rho_init
        self.rho_current = rho_init
        self.zeta_threshold = zeta_threshold

        # Pheromone bounds
        self.z_const = z_const
        self.Q = Q

        # Problem data
        self.coords = None
        self.n_nodes = 0
        self.base_idx = 0

        # Matrices
        self.dist_matrix = None
        self.eta_matrix = None
        self.tau_matrix = None
        self.tau_max = None
        self.tau_min = None

        # Best solution tracking
        self.best_tour = None
        self.best_length = float('inf')
        self.best_history = []
        self.iteration_best_history = []

        # External initial solution (from GM-CEO)
        self.initial_tour = None

    def setup(self, points: np.ndarray, base: np.ndarray, initial_tour: List[int] = None):
        """
        Setup problem with optional initial solution from GM-CEO
        """
        self.base_idx = 0
        self.coords = np.vstack([base.reshape(1, -1), points])
        self.n_nodes = len(self.coords)
        self.initial_tour = initial_tour

        # Distance matrix
        self.dist_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])

        # Heuristic matrix: eta = 1/d (Eq. 17)
        self.eta_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and self.dist_matrix[i, j] > 1e-10:
                    self.eta_matrix[i, j] = 1.0 / self.dist_matrix[i, j]

        # Initialize pheromone
        self.tau_matrix = np.ones((self.n_nodes, self.n_nodes)) * 0.1
        np.fill_diagonal(self.tau_matrix, 0)

        # If initial tour provided, boost its pheromone (GM-CEO integration)
        if initial_tour is not None:
            self._initialize_pheromone_from_tour(initial_tour)

    def _initialize_pheromone_from_tour(self, tour: List[int]):
        """Use GM-CEO solution to initialize pheromone matrix"""
        path = [self.base_idx] + tour + [self.base_idx]
        boost = 2.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.tau_matrix[u, v] += boost

        self.tau_matrix = np.clip(self.tau_matrix, 0.1, 10.0)
        np.fill_diagonal(self.tau_matrix, 0)

    def gaussian_alpha(self, t: int) -> float:
        """
        Gaussian-adjusted alpha(t) - Eq. (ia*1)

        alpha(t) = alpha(0) + (alpha_max - alpha_min) *
                   (1 - exp(-(t - alpha_mu)^2 / (2 * alpha_sigma^2)))
        """
        exponent = -((t - self.alpha_mu) ** 2) / (2.0 * self.alpha_sigma ** 2)
        gaussian_term = 1.0 - math.exp(exponent)
        alpha_t = self.alpha_0 + (self.alpha_max - self.alpha_min) * gaussian_term
        return max(self.alpha_min, min(self.alpha_max, alpha_t))

    def gaussian_beta(self, t: int) -> float:
        """
        Gaussian-adjusted beta(t) - Eq. (ia*1)

        beta(t) = beta(0) - (beta_max - beta_min) *
                  (1 - exp(-(t - beta_mu)^2 / (2 * beta_sigma^2)))
        """
        exponent = -((t - self.beta_mu) ** 2) / (2.0 * self.beta_sigma ** 2)
        gaussian_term = 1.0 - math.exp(exponent)
        beta_t = self.beta_0 - (self.beta_max - self.beta_min) * gaussian_term
        return max(self.beta_min, min(self.beta_max, beta_t))

    def adaptive_rho(self, t: int, L_star_t: float) -> float:
        """
        Adaptive pheromone evaporation rate - Eq. (ia*2)

        rho'(t) = min(L*(1), ..., L*(t-1)) / L*(t)

        rho(t+1) = 1 - 0.9 * (1 - sigmoid(-rho'(t)))  if zeta < 0.001
                   sigmoid(-rho'(t))                    otherwise
        """
        if t <= 1 or not self.iteration_best_history:
            return self.rho_init

        # Calculate rho'(t)
        min_prev = min(self.iteration_best_history)
        rho_prime = min_prev / L_star_t if L_star_t > 1e-10 else 1.0

        # Sigmoid function
        sigmoid_val = 1.0 / (1.0 + math.exp(-rho_prime))

        # Calculate zeta (relative error)
        L_prev = self.iteration_best_history[-1]
        zeta = abs(L_star_t - L_prev) / L_prev if L_prev > 1e-10 else 1.0

        # Adaptive rho update
        if zeta < self.zeta_threshold:
            rho_new = 1.0 - 0.9 * (1.0 - sigmoid_val)
        else:
            rho_new = sigmoid_val

        return max(0.01, min(0.99, rho_new))

    def compute_gamma(self, L_star_t: float) -> float:
        """
        Variable enhancement factor gamma(t) - Eq. (a4)

        gamma(t) = Q / L*(t)   if L*(t) < L_min (current best)
                   0           otherwise
        """
        if L_star_t < self.best_length:
            return self.Q / L_star_t
        return 0.0

    def update_pheromone_bounds(self, t: int, delta_tau: np.ndarray, n_customers: int):
        """
        Update pheromone bounds - Eq. (ia*3)

        tau_max(t) = tau_max(t) * (1 - rho(t)) + delta_tau^i(t)
        tau_min(t) = tau_max(t) * (1 - z) / (m/2 - 1) * z
        """
        if self.tau_max is None:
            self.tau_max = np.max(self.tau_matrix)

        self.tau_max = self.tau_max * (1.0 - self.rho_current) + np.max(delta_tau)

        m = max(n_customers, 2)
        denom = max(m / 2.0 - 1.0, 0.5)
        self.tau_min = self.tau_max * (1.0 - self.z_const) / denom * self.z_const
        self.tau_min = max(1e-6, min(self.tau_min, self.tau_max * 0.1))

    def tour_length(self, tour: List[int]) -> float:
        """Calculate tour length"""
        if not tour:
            return 0.0

        length = 0.0
        path = [self.base_idx] + tour + [self.base_idx]

        for i in range(len(path) - 1):
            length += self.dist_matrix[path[i], path[i + 1]]

        return length

    def build_tour(self, alpha_t: float, beta_t: float) -> List[int]:
        """Build tour with dynamic alpha and beta"""
        unvisited = set(range(1, self.n_nodes))
        tour = []
        current = self.base_idx

        while unvisited:
            probs = []
            candidates = []

            for j in unvisited:
                tau = self.tau_matrix[current, j]
                eta = self.eta_matrix[current, j]

                if eta > 0:
                    val = (tau ** alpha_t) * (eta ** beta_t)
                    probs.append(val)
                    candidates.append(j)

            if not candidates:
                j = unvisited.pop()
                tour.append(j)
                current = j
                continue

            total = sum(probs)
            probs = [p / total for p in probs]

            r = random.random()
            cumsum = 0.0
            selected_idx = 0

            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    selected_idx = i
                    break

            next_node = candidates[selected_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return tour

    def update_pheromone(self, all_tours: List[Tuple[List[int], float]],
                         gamma_t: float, t: int):
        """
        Update pheromone with bounds - Eq. (a4) + Eq. (ia*3)
        """
        # Step 1: Evaporation
        self.tau_matrix = (1.0 - self.rho_current) * self.tau_matrix

        # Step 2: Calculate deposits
        delta_tau = np.zeros((self.n_nodes, self.n_nodes))

        for tour, length in all_tours:
            if length > 1e-10:
                deposit = self.Q / length
                path = [self.base_idx] + tour + [self.base_idx]

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    delta_tau[u, v] += deposit

        # Step 3: Add gamma(t) to best path
        if gamma_t > 0 and self.best_tour is not None:
            path = [self.base_idx] + self.best_tour + [self.base_idx]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                delta_tau[u, v] += gamma_t

        # Step 4: Apply updates
        self.tau_matrix += self.rho_current * delta_tau

        # Step 5: Update bounds and clamp
        n_customers = len(all_tours[0][0]) if all_tours else self.n_nodes - 1
        self.update_pheromone_bounds(t, delta_tau, n_customers)

        if self.tau_min is not None and self.tau_max is not None:
            self.tau_matrix = np.clip(self.tau_matrix, self.tau_min, self.tau_max)

        np.fill_diagonal(self.tau_matrix, 0)

    def solve(self, points: np.ndarray, base: np.ndarray,
              initial_tour: List[int] = None, verbose: bool = True) -> Tuple:
        """
        Main GAV-ACO algorithm
        """
        self.setup(points, base, initial_tour)
        self.best_tour = None
        self.best_length = float('inf')
        self.best_history = []
        self.iteration_best_history = []
        self.rho_current = self.rho_init
        self.tau_max = None
        self.tau_min = None

        start_time = time.time()

        for t in range(1, self.n_iterations + 1):
            # Gaussian-adjusted parameters
            alpha_t = self.gaussian_alpha(t)
            beta_t = self.gaussian_beta(t)

            all_tours = []
            L_star_t = float('inf')
            best_tour_t = None

            # Build tours
            for _ in range(self.n_ants):
                tour = self.build_tour(alpha_t, beta_t)
                length = self.tour_length(tour)
                all_tours.append((tour, length))

                if length < L_star_t:
                    L_star_t = length
                    best_tour_t = tour.copy()

            # Update global best
            if L_star_t < self.best_length:
                self.best_length = L_star_t
                self.best_tour = best_tour_t.copy()

            # Compute gamma(t)
            gamma_t = self.compute_gamma(L_star_t)

            # Update adaptive rho
            self.rho_current = self.adaptive_rho(t, L_star_t)

            # Update pheromone
            self.update_pheromone(all_tours, gamma_t, t)

            # Record history
            self.iteration_best_history.append(L_star_t)
            self.best_history.append(self.best_length)

            if verbose and t % 20 == 0:
                print(f"  Iter {t:3d}: Best={self.best_length:.4f}, "
                      f"α={alpha_t:.3f}, β={beta_t:.3f}, "
                      f"ρ={self.rho_current:.4f}, γ={gamma_t:.4f}")

        runtime = time.time() - start_time

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Best tour length: {self.best_length:.4f}")
            print(f"  Runtime: {runtime:.3f} seconds")

        return self.best_tour, self.best_length, self.best_history, runtime


def generate_instance(n_points: int, area_size: float = 30.0, seed: int = 42):
    """Generate random instance"""
    np.random.seed(seed)
    random.seed(seed)
    base = np.array([area_size / 2, area_size / 2])
    points = np.random.uniform(0, area_size, (n_points, 2))
    return points, base


def generate_initial_solution(points: np.ndarray, base: np.ndarray) -> List[int]:
    """Simulate GM-CEO initial solution (nearest neighbor heuristic)"""
    coords = np.vstack([base.reshape(1, -1), points])
    n = len(coords)
    unvisited = set(range(1, n))
    tour = []
    current = 0

    while unvisited:
        nearest = min(unvisited, key=lambda j: np.linalg.norm(coords[current] - coords[j]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return tour


def main():
    """Main demo"""
    print("=" * 75)
    print("GAV-ACO Algorithm Implementation")
    print("=" * 75)

    test_cases = [20, 40, 60]

    for n in test_cases:
        print(f"\n{'=' * 75}")
        print(f"Test Case: {n} Target Points")
        print(f"{'=' * 75}")

        points, base = generate_instance(n, area_size=30.0, seed=42)
        print(f"Region: 30km × 30km, Base: ({base[0]:.1f}, {base[1]:.1f})")

        # Generate initial solution (simulating GM-CEO output)
        initial_tour = generate_initial_solution(points, base)
        initial_length = sum(
            np.linalg.norm(
                np.vstack([base, points])[initial_tour[i]] -
                np.vstack([base, points])[initial_tour[i + 1]]
            ) for i in range(len(initial_tour) - 1)
        )
        print(f"Initial solution (NN heuristic) length: {initial_length:.4f} km")

        # Run GAV-ACO
        print(f"\n--- GAV-ACO ---")
        gav_aco = GAV_ACO(
            n_ants=50,
            n_iterations=100,
            alpha_0=0.5, alpha_max=5.0, alpha_min=0.5,
            alpha_mu=30, alpha_sigma=25,
            beta_0=5.0, beta_max=5.0, beta_min=1.0,
            beta_mu=30, beta_sigma=25,
            rho_init=0.5,
            zeta_threshold=0.001,
            z_const=0.5,
            Q=100.0,
            seed=42
        )

        tour_gav, length_gav, history_gav, runtime_gav = gav_aco.solve(
            points, base, initial_tour=initial_tour, verbose=True
        )

        print(f"\nGAV-ACO Results:")
        print(f"  Tour: {tour_gav}")
        print(f"  Length: {length_gav:.4f} km")
        print(f"  Runtime: {runtime_gav:.3f} s")
        print(f"  Improvement: {(initial_length - length_gav) / initial_length * 100:.2f}%")

        # Visualization (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Plot 1: Initial solution
        ax1 = axes[0, 0]
        coords = np.vstack([base, points])
        ax1.scatter(points[:, 0], points[:, 1], c='lightblue', s=60,
                    edgecolors='navy', zorder=4)
        ax1.scatter(base[0], base[1], c='red', marker='^', s=200,
                    edgecolors='darkred', zorder=5, label='Base')
        if initial_tour:
            path_idx = [0] + initial_tour + [0]
            path_xy = coords[path_idx]
            ax1.plot(path_xy[:, 0], path_xy[:, 1], 'orange', linewidth=2,
                     alpha=0.7, label=f'Initial (L={initial_length:.2f})')
        ax1.set_title('Initial Solution (GM-CEO / NN)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: GAV-ACO solution
        ax2 = axes[0, 1]
        ax2.scatter(points[:, 0], points[:, 1], c='lightblue', s=60,
                    edgecolors='navy', zorder=4)
        ax2.scatter(base[0], base[1], c='red', marker='^', s=200,
                    edgecolors='darkred', zorder=5, label='Base')
        if tour_gav:
            path_idx = [0] + tour_gav + [0]
            path_xy = coords[path_idx]
            ax2.plot(path_xy[:, 0], path_xy[:, 1], 'g-', linewidth=2,
                     alpha=0.7, label=f'GAV-ACO (L={length_gav:.2f})')
        ax2.set_title('GAV-ACO Optimized Solution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Plot 3: Parameter evolution
        ax3 = axes[1, 0]
        iterations = range(1, gav_aco.n_iterations + 1)
        alphas = [gav_aco.gaussian_alpha(t) for t in iterations]
        betas = [gav_aco.gaussian_beta(t) for t in iterations]
        ax3.plot(iterations, alphas, 'b-', linewidth=2, label='α(t) - Pheromone')
        ax3.plot(iterations, betas, 'r-', linewidth=2, label='β(t) - Heuristic')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Gaussian-Adjusted Parameters')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Convergence
        ax4 = axes[1, 1]
        ax4.plot(range(1, len(history_gav) + 1), history_gav, 'g-',
                 linewidth=2, label='GAV-ACO Best')
        ax4.axhline(y=initial_length, color='orange', linestyle='--',
                    label=f'Initial ({initial_length:.2f})')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Best Tour Length (km)')
        ax4.set_title(f'Convergence (Runtime: {runtime_gav:.2f}s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    print(f"\n{'=' * 75}")
    print("All tests completed successfully!")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()