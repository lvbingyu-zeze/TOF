# VR-HHO Algorithm - Complete Python Implementation
# Based on the paper: VR-HHO for UAV Routing Optimization
# Includes VHCA (Vertical & Horizontal Crossover) and RLC (Reverse Learning Competition)

import numpy as np
import random
from typing import List, Tuple, Callable, Optional


class VRHHO:
    """
    VR-HHO Algorithm with VHCA and RLC for UAV Routing Optimization

    Paper Reference:
    - VHCA: Horizontal Crossover (HX) + Vertical Crossover (VX)
    - RLC: Reverse Learning Competition strategy
    """

    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 100,
                 dim: int = None,
                 penalty_weight: float = 1e6,
                 rlc_prob: float = 0.3,
                 hx_prob: float = 0.5,
                 vx_prob: float = 0.3,
                 bounds: Tuple[float, float] = (0, 1)):
        """
        Initialize VR-HHO parameters

        Args:
            population_size: Number of Harris hawks (N)
            max_iterations: Maximum number of iterations
            dim: Dimension of the problem (D = N^w_u * N_c^2)
            penalty_weight: Penalty W for infeasible solutions
            rlc_prob: Probability of applying RLC to underperforming individuals
            hx_prob: Probability of horizontal crossover
            vx_prob: Probability of vertical crossover
            bounds: Tuple of (lower_bound, upper_bound) for search space
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dim = dim
        self.penalty_weight = penalty_weight
        self.rlc_prob = rlc_prob
        self.hx_prob = hx_prob
        self.vx_prob = vx_prob
        self.bounds = bounds

        # Population: each hawk is a binary vector of length D
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def initialize_population(self, dim: int = None, seed: int = None):
        """
        Initialize population with binary encoding
        Each hawk X_i = (X_{i,1}, X_{i,2}, ..., X_{i,D}) where X_{i,j} in {0, 1}
        """
        if dim is not None:
            self.dim = dim
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize binary population
        self.population = np.random.randint(0, 2, size=(self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)

    def horizontal_crossover(self, xi: np.ndarray, xj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Horizontal Crossover (HX) - promotes information exchange among different hawks

        HX_i(t) = l1 * X_i(t) + (1-l1) * X_j(t) + o1 * (X_i(t) - X_j(t))
        HX_j(t) = l2 * X_j(t) + (1-l2) * X_j(t) + o2 * (X_j(t) - X_i(t))

        Note: For binary problems, we use sigmoid to convert to probability then threshold
        """
        l1 = np.random.uniform(0, 1, self.dim)
        l2 = np.random.uniform(0, 1, self.dim)
        o1 = np.random.uniform(-1, 1, self.dim)
        o2 = np.random.uniform(-1, 1, self.dim)

        # Compute offspring (continuous values)
        hx_i_cont = l1 * xi + (1 - l1) * xj + o1 * (xi - xj)
        hx_j_cont = l2 * xj + (1 - l2) * xj + o2 * (xj - xi)

        # Convert to binary using sigmoid threshold
        hx_i = (1 / (1 + np.exp(-hx_i_cont)) > np.random.uniform(0, 1, self.dim)).astype(int)
        hx_j = (1 / (1 + np.exp(-hx_j_cont)) > np.random.uniform(0, 1, self.dim)).astype(int)

        # Ensure binary bounds
        hx_i = np.clip(hx_i, self.bounds[0], self.bounds[1])
        hx_j = np.clip(hx_j, self.bounds[0], self.bounds[1])

        return hx_i, hx_j

    def vertical_crossover(self, xi: np.ndarray) -> np.ndarray:
        """
        Vertical Crossover (VX) - recombines solution components within an individual

        VX_i(t) = l * X_{i_p}(t) + (1-l) * X_{i_q}(t)

        Randomly selects two dimensions p and q to exchange
        """
        if self.dim < 2:
            return xi.copy()

        # Select two random distinct dimensions
        p, q = np.random.choice(self.dim, 2, replace=False)

        l = np.random.uniform(0, 1)

        vx = xi.copy()
        # Blend the two dimensions (for binary, use probabilistic combination)
        blend_val = l * xi[p] + (1 - l) * xi[q]
        vx[p] = 1 if random.random() < (1 / (1 + np.exp(-blend_val))) else 0
        vx[q] = 1 if random.random() < (1 / (1 + np.exp(-blend_val))) else 0

        vx = np.clip(vx, self.bounds[0], self.bounds[1])
        return vx

    def reverse_learning_competition(self, xi: np.ndarray, t: int) -> np.ndarray:
        """
        Reverse Learning Competition (RLC)

        fitness_R(t+1) = rand * (X_max + X_min) - X_i(t+1)

        Generates reverse candidate solution to escape local optima
        """
        rand_val = np.random.uniform(0, 1, self.dim)

        # Reverse solution: rand * (X_max + X_min) - X_i
        # For binary {0,1}: X_max=1, X_min=0
        x_max = np.ones(self.dim) * self.bounds[1]
        x_min = np.ones(self.dim) * self.bounds[0]

        reverse_cont = rand_val * (x_max + x_min) - xi

        # Convert to binary
        reverse_sol = (1 / (1 + np.exp(-reverse_cont)) > np.random.uniform(0, 1, self.dim)).astype(int)
        reverse_sol = np.clip(reverse_sol, self.bounds[0], self.bounds[1])

        return reverse_sol

    def evaluate_fitness(self, solution: np.ndarray, fitness_func: Callable,
                         is_feasible: Callable = None) -> float:
        """
        Evaluate fitness with penalty for infeasible solutions

        Args:
            solution: Candidate solution
            fitness_func: Objective function J_2
            is_feasible: Function to check feasibility (returns True/False)
        """
        if is_feasible is not None and not is_feasible(solution):
            # Apply penalty W and generate new solution
            return self.penalty_weight

        return fitness_func(solution)

    def update_position(self, xi: np.ndarray, best: np.ndarray,
                        t: int, max_t: int, randomization: float = 0.1) -> np.ndarray:
        """
        Update position of Harris hawk (simplified HHO position update)

        This is a simplified version focusing on the VR-HHO enhancements.
        In full HHO, this involves energy calculation and different attack strategies.
        """
        # Simplified exploration/exploitation balance
        E0 = 2 * random.random() - 1  # Initial energy
        E = 2 * E0 * (1 - t / max_t)  # Decaying energy

        if abs(E) >= 1:
            # Exploration: random position
            rand_hawk = self.population[random.randint(0, self.population_size - 1)]
            new_pos = rand_hawk - random.random() * abs(rand_hawk - 2 * random.random() * xi)
        else:
            # Exploitation: move toward best
            J = 2 * (1 - random.random())  # Random jump strength
            new_pos = best - E * abs(J * best - xi)

        # Convert to binary
        new_pos = (1 / (1 + np.exp(-new_pos)) > np.random.uniform(0, 1, self.dim)).astype(int)
        new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])

        return new_pos

    def optimize(self,
                 fitness_func: Callable,
                 is_feasible: Callable = None,
                 verbose: bool = True) -> Tuple[np.ndarray, float, List]:
        """
        Main optimization loop (Algorithm 2)

        Args:
            fitness_func: The middle-level objective function J_2
            is_feasible: Function to check solution feasibility
            verbose: Print progress

        Returns:
            best_solution, best_fitness, history
        """
        if self.population is None:
            raise ValueError("Population not initialized. Call initialize_population() first.")

        # Step 1: Evaluate initial fitness
        for i in range(self.population_size):
            self.fitness[i] = self.evaluate_fitness(self.population[i], fitness_func, is_feasible)

        # Find initial best
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        # Optimization loop
        for t in range(self.max_iterations):
            # Sort population by fitness (for RLC targeting underperforming individuals)
            sorted_indices = np.argsort(self.fitness)

            # RLC for underperforming individuals (bottom 30% by default)
            n_underperforming = int(self.population_size * self.rlc_prob)
            underperforming_indices = sorted_indices[-n_underperforming:]

            for idx in underperforming_indices:
                reverse_sol = self.reverse_learning_competition(self.population[idx], t)
                reverse_fitness = self.evaluate_fitness(reverse_sol, fitness_func, is_feasible)

                # Competitive replacement: keep better one
                if reverse_fitness < self.fitness[idx]:
                    self.population[idx] = reverse_sol
                    self.fitness[idx] = reverse_fitness

            # Update positions and apply VHCA
            new_population = np.zeros_like(self.population)
            new_fitness = np.zeros(self.population_size)

            for i in range(self.population_size):
                # Update position (standard HHO-like update)
                new_pos = self.update_position(self.population[i], self.best_solution, t, self.max_iterations)

                # Apply VHCA with certain probabilities
                if random.random() < self.hx_prob and self.population_size > 1:
                    # Horizontal crossover with random partner
                    partner_idx = random.randint(0, self.population_size - 1)
                    while partner_idx == i:
                        partner_idx = random.randint(0, self.population_size - 1)

                    hx_i, _ = self.horizontal_crossover(new_pos, self.population[partner_idx])

                    # Evaluate and select better
                    hx_fitness = self.evaluate_fitness(hx_i, fitness_func, is_feasible)
                    current_fitness = self.evaluate_fitness(new_pos, fitness_func, is_feasible)

                    if hx_fitness < current_fitness:
                        new_pos = hx_i

                if random.random() < self.vx_prob:
                    # Vertical crossover
                    vx = self.vertical_crossover(new_pos)
                    vx_fitness = self.evaluate_fitness(vx, fitness_func, is_feasible)
                    current_fitness = self.evaluate_fitness(new_pos, fitness_func, is_feasible)

                    if vx_fitness < current_fitness:
                        new_pos = vx

                new_population[i] = new_pos
                new_fitness[i] = self.evaluate_fitness(new_pos, fitness_func, is_feasible)

            # Update population
            self.population = new_population
            self.fitness = new_fitness

            # Update global best
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()

            self.history.append(self.best_fitness)

            if verbose and (t + 1) % 10 == 0:
                print(f"Iteration {t + 1}/{self.max_iterations}, Best Fitness: {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness, self.history


# ==================== DEMO / TEST ====================

def demo_uav_routing():
    """
    Demonstration of VR-HHO on a simplified UAV routing problem
    """
    print("=" * 60)
    print("VR-HHO Algorithm Demo: UAV Routing Optimization")
    print("=" * 60)

    # Problem parameters (simplified)
    N_u = 2  # Number of UAVs
    N_c = 3  # Number of customers
    N_w = 2  # Number of warehouses

    # Dimension: D = N^w_u * N_c^2 (simplified for demo)
    # In practice, this depends on the specific encoding
    D = N_u * N_c * N_w  # Simplified dimension for demo

    print(f"\nProblem Setup:")
    print(f"  UAVs: {N_u}, Customers: {N_c}, Warehouses: {N_w}")
    print(f"  Solution Dimension (D): {D}")

    # Define a sample fitness function (middle-level objective J_2)
    # This would typically involve path cost, energy, time, etc.
    def sample_fitness_j2(solution: np.ndarray) -> float:
        """
        Simplified J_2: Minimize total routing cost

        In practice, this would decode the binary solution into:
        - Warehouse assignments (a_wj from KMSE-IAO)
        - Service sequences (s_uj from GAV-ACO)
        - UAV paths (r_uij)
        """
        # Simplified: weighted sum with some structure
        # Lower values are better (minimization)
        cost = np.sum(solution) * 10  # Base cost

        # Add some structure to make it non-trivial
        for i in range(len(solution) - 1):
            if solution[i] == 1 and solution[i + 1] == 1:
                cost -= 2  # Reward consecutive 1s (efficient routing)

        # Add noise to simulate complex routing landscape
        cost += np.random.normal(0, 0.5)

        return float(cost)

    # Feasibility check (simplified)
    def is_feasible(solution: np.ndarray) -> bool:
        """Check if solution satisfies routing constraints"""
        # Simplified: ensure at least one assignment per UAV
        # In practice: check warehouse capacity, UAV range, etc.
        if np.sum(solution) == 0:
            return False
        if np.sum(solution) > len(solution) * 0.8:
            return False  # Too many assignments
        return True

    # Initialize and run VR-HHO
    print(f"\nInitializing VR-HHO...")

    vrhho = VRHHO(
        population_size=30,
        max_iterations=100,
        dim=D,
        penalty_weight=1e6,
        rlc_prob=0.3,
        hx_prob=0.5,
        vx_prob=0.3,
        bounds=(0, 1)
    )

    vrhho.initialize_population(dim=D, seed=42)

    print(f"Running optimization...\n")
    best_sol, best_fit, history = vrhho.optimize(
        fitness_func=sample_fitness_j2,
        is_feasible=is_feasible,
        verbose=True
    )

    print(f"\n{'=' * 60}")
    print(f"Optimization Results:")
    print(f"{'=' * 60}")
    print(f"Best Fitness (J_2): {best_fit:.6f}")
    print(f"Best Solution: {best_sol}")
    print(f"Solution Sum (active routes): {np.sum(best_sol)}")
    print(f"Total iterations: {len(history)}")
    print(f"Improvement: {history[0]:.6f} -> {history[-1]:.6f}")

    return vrhho, best_sol, best_fit, history


def demo_comparison():
    """
    Compare VR-HHO with and without VHCA/RLC enhancements
    """
    print("\n" + "=" * 60)
    print("Comparison: VR-HHO vs Basic HHO")
    print("=" * 60)

    D = 20

    def test_func(x):
        # Rastrigin-like function (multimodal, good for testing)
        return 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    # VR-HHO (with VHCA and RLC)
    vrhho = VRHHO(population_size=50, max_iterations=200, dim=D,
                  rlc_prob=0.3, hx_prob=0.5, vx_prob=0.3)
    vrhho.initialize_population(dim=D, seed=42)
    _, fit_vrhho, hist_vrhho = vrhho.optimize(test_func, verbose=False)

    # Basic HHO (without enhancements)
    basic_hho = VRHHO(population_size=50, max_iterations=200, dim=D,
                      rlc_prob=0.0, hx_prob=0.0, vx_prob=0.0)
    basic_hho.initialize_population(dim=D, seed=42)
    _, fit_basic, hist_basic = basic_hho.optimize(test_func, verbose=False)

    print(f"VR-HHO (with VHCA+RLC): {fit_vrhho:.6f}")
    print(f"Basic HHO:              {fit_basic:.6f}")
    print(f"Improvement:            {fit_basic - fit_vrhho:.6f}")

    return hist_vrhho, hist_basic


if __name__ == "__main__":
    # Run main demo
    vrhho, best_sol, best_fit, history = demo_uav_routing()

    # Run comparison
    hist_vrhho, hist_basic = demo_comparison()

    print("\n" + "=" * 60)
    print("VR-HHO implementation completed successfully!")
    print("=" * 60)
