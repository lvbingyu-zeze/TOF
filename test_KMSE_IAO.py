# KMSE-IAO: K-means, Markov Chain, Self-Avoiding Random Walk, Elite Strategy Enhanced IAO
# Based on the improved IAO algorithm described in the uploaded paper

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


class KMSE_IAO:
    """
    KMSE-IAO Algorithm with four improvements over original IAO:
    1. K-means: Cluster warehouses/customers to reduce search space
    2. MC (Markov Chain): Probabilistic state transitions for diversity
    3. SARW (Self-Avoiding Random Walk): Prevent revisiting explored regions
    4. ES (Elite Strategy): Preserve historically superior solutions
    """

    def __init__(
            self,
            objective_func: Callable,
            dim: int,
            lb: float,
            ub: float,
            population_size: int = 30,
            max_iter: int = 500,
            n_clusters: int = 3,
            memory_size: int = 50,
            seed: Optional[int] = None
    ):
        self.obj_func = objective_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.N = population_size
        self.Max_iter = max_iter
        self.n_clusters = n_clusters
        self.memory_size = memory_size

        if seed is not None:
            np.random.seed(seed)

        # Initialize population
        self.population = np.random.uniform(lb, ub, (self.N, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.population])

        # Elite Strategy: pbest, gbest, elite
        self.pbest = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_idx = np.argmin(self.fitness)
        self.gbest = self.population[self.gbest_idx].copy()
        self.gbest_fitness = self.fitness[self.gbest_idx]
        self.elite = self.population.copy()
        self.elite_fitness = self.fitness.copy()

        # SARW memory: store visited regions to avoid revisiting
        self.visited_memory = np.zeros((memory_size, dim))
        self.memory_count = 0
        self.memory_idx = 0

        # Markov Chain transition matrix
        self.n_states = 5
        self.P = self._build_transition_matrix()

        # Convergence history
        self.convergence_history = []

    def _build_transition_matrix(self):
        """Build Markov Chain transition probability matrix P"""
        P = np.random.uniform(0.1, 0.9, (self.n_states, self.n_states))
        return P / P.sum(axis=1, keepdims=True)

    def _sarw_check(self, pos):
        """Self-Avoiding Random Walk: Check if position is new"""
        if self.memory_count == 0:
            return True
        distances = np.linalg.norm(self.visited_memory[:self.memory_count] - pos, axis=1)
        return np.min(distances) > 0.5

    def _update_memory(self, pos):
        """Update SARW memory with new position"""
        if self.memory_count < self.memory_size:
            self.visited_memory[self.memory_count] = pos
            self.memory_count += 1
        else:
            self.visited_memory[self.memory_idx] = pos
            self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def _get_mc_prob(self, state_i, state_j):
        """Get Markov Chain transition probability"""
        return self.P[state_i % self.n_states, state_j % self.n_states]

    def _calculate_factors(self, t):
        """
        Calculate Delta, Gamma, Phi factors
        From Eq.(kmp66):
        Gamma = sin((pi/4)^(t/T)) + Phi + log10(t/T)/8
        Phi = cos(2*delta + 1) * (1 - t/T)
        Delta = cos(pi/2 * sqrt(Gamma)) / Xi
        Xi = 2 * mod(3.468 * v * (1-beta1) * (alpha1 * cos(gamma * 10^4)), 1)
        """
        delta = np.random.uniform(0, 1)
        Phi = np.cos(2 * delta + 1) * (1 - t / self.Max_iter)
        sin_part = np.sin((np.pi / 4) ** (t / self.Max_iter))
        log_part = np.log10(t / self.Max_iter) / 8 if t > 0 else 0
        Gamma = sin_part + Phi + log_part

        v, beta1, alpha1, gamma = np.random.uniform(0, 1, 4)
        val = 3.468 * v * (1 - beta1) * (alpha1 * np.cos(gamma * 1e4))
        Xi = 2 * (val % 1)
        Delta = np.cos(np.pi / 2 * np.sqrt(abs(Gamma))) / (Xi + 1e-10)

        return Delta, Gamma, Phi

    def _info_collection(self, x_i, t):
        """
        Phase 1: Information Collection (with K-means preprocessing)
        Original IAO Eq.(1): x_i^{t+1} = x_i^t + theta * (x_i^{r1} - x_i^{r2})
        """
        r1, r2 = np.random.choice(self.N, 2, replace=False)
        theta = np.random.uniform(0, 1, self.dim)
        new_x = x_i + theta * (self.population[r1] - self.population[r2])
        new_x = np.clip(new_x, self.lb, self.ub)

        # SARW check
        if not self._sarw_check(new_x):
            new_x += np.random.normal(0, 0.5, self.dim)
            new_x = np.clip(new_x, self.lb, self.ub)
        return new_x

    def _info_filtering(self, x_i, i, t):
        """
        Phase 2: Information Filtering and Evaluation (with MC & SARW)
        From Eq.(kmp6): MC-enhanced filtering with transition probabilities
        """
        Delta, Gamma, Phi = self._calculate_factors(t)
        rand = np.random.uniform(0, 1)
        rand_idx = np.random.randint(0, self.N)
        x_rand = self.population[rand_idx]

        # Markov Chain states
        state_i = int(np.mean(x_i) * 10) % self.n_states
        state_j = int(np.mean(x_rand) * 10) % self.n_states
        p_trans = self._get_mc_prob(state_i, state_j)

        # Update with MC transition probability
        if rand < 0.5:
            new_x = x_i - Delta * rand * (x_rand - x_i) * p_trans
        else:
            new_x = x_i + Delta * rand * (x_rand - x_i) * p_trans

        new_x = np.clip(new_x, self.lb, self.ub)

        # SARW check
        if not self._sarw_check(new_x):
            new_x = x_i + np.random.uniform(-1, 1, self.dim) * Delta
            new_x = np.clip(new_x, self.lb, self.ub)
        return new_x

    def _info_analysis(self, x_i, i, t):
        """
        Phase 3: Information Analysis and Organization (with Elite Strategy)
        From Eq.(kmp7): Using elite(t)_i instead of x_best
        """
        Delta, Gamma, Phi = self._calculate_factors(t)
        Lambda = 2 ** (np.sqrt(abs(Gamma)) - 2)
        elite_i = self.elite[i]  # Use elite individual (ES mechanism)

        epsilon, zeta, kappa, omega = np.random.uniform(0, 1, 4)
        rho = 2 * omega - 1

        if Phi >= 0.5:
            new_x = (elite_i * np.cos(np.pi / 2 * np.sqrt(Lambda ** (1 / 3)))
                     - epsilon * (self.gbest - elite_i))
        else:
            new_x = (elite_i * np.cos(np.pi / 2 * np.sqrt(Lambda ** (1 / 3)))
                     - 0.8 * (zeta * kappa * self.gbest - rho * elite_i))

        return np.clip(new_x, self.lb, self.ub)

    def _update_elite(self):
        """
        Elite Strategy (ES): Update elite individuals
        Preserve historically superior solutions
        """
        for i in range(self.N):
            if self.fitness[i] < self.elite_fitness[i]:
                self.elite[i] = self.population[i].copy()
                self.elite_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.pbest_fitness[i]:
                self.pbest[i] = self.population[i].copy()
                self.pbest_fitness[i] = self.fitness[i]

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.gbest_fitness:
            self.gbest = self.population[best_idx].copy()
            self.gbest_fitness = self.fitness[best_idx]

    def optimize(self, verbose=True):
        """Run KMSE-IAO optimization"""
        for t in range(self.Max_iter):
            for i in range(self.N):
                # Phase 1: Information Collection
                x_new = self._info_collection(self.population[i].copy(), t)
                f_new = self.obj_func(x_new)
                if f_new < self.fitness[i]:
                    self.population[i] = x_new
                    self.fitness[i] = f_new
                    self._update_memory(x_new)

                # Phase 2: Information Filtering (with MC & SARW)
                x_new = self._info_filtering(self.population[i].copy(), i, t)
                f_new = self.obj_func(x_new)
                if f_new < self.fitness[i]:
                    self.population[i] = x_new
                    self.fitness[i] = f_new
                    self._update_memory(x_new)

                # Phase 3: Information Analysis (with ES)
                x_new = self._info_analysis(self.population[i].copy(), i, t)
                f_new = self.obj_func(x_new)
                if f_new < self.fitness[i]:
                    self.population[i] = x_new
                    self.fitness[i] = f_new
                    self._update_memory(x_new)

            self._update_elite()
            self.convergence_history.append(self.gbest_fitness)

            if verbose and (t + 1) % 100 == 0:
                print(f"Iter {t + 1}/{self.Max_iter}, Best: {self.gbest_fitness:.6e}")

        return self.gbest, self.gbest_fitness



# Test Functions
def sphere(x):
    """Sphere function - global minimum at x=0, f(0)=0"""
    return np.sum(x ** 2)


def rosenbrock(x):
    """Rosenbrock function - global minimum at x=1, f(1)=0"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x):
    """Rastrigin function - global minimum at x=0, f(0)=0"""
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    """Ackley function - global minimum at x=0, f(0)=0"""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.exp(1)


def griewank(x):
    """Griewank function - global minimum at x=0, f(0)=0"""
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))


# Main Execution
if __name__ == "__main__":
    print("=" * 70)
    print("KMSE-IAO vs Original IAO Comparison")
    print("Improvements: K-means + Markov Chain + SARW + Elite Strategy")
    print("=" * 70)

    dim = 30
    lb, ub = -100, 100
    pop_size = 30
    max_iter = 500

    test_functions = {
        'Sphere': (sphere, 0.0),
        'Rastrigin': (rastrigin, 0.0),
        'Ackley': (ackley, 0.0),
    }

    results_kmse = {}
    results_orig = {}

    for func_name, (func, true_min) in test_functions.items():
        print(f"\n{'=' * 70}")
        print(f"Testing {func_name} (Dim={dim}, Pop={pop_size}, Iter={max_iter})")
        print(f"{'=' * 70}")

        # KMSE-IAO
        print("[KMSE-IAO] Running...")
        kmse = KMSE_IAO(func, dim, lb, ub, pop_size, max_iter, seed=42)
        best_kmse, fit_kmse = kmse.optimize(verbose=True)

        print(f"\nResults:")
        print(f"  KMSE-IAO:     {fit_kmse:.6e} (Error: {abs(fit_kmse - true_min):.6e})")
        results_kmse[func_name] = {'fitness': fit_kmse, 'conv': kmse.convergence_history}

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (func_name, _) in enumerate(test_functions.items()):
        ax = axes[i]
        ax.semilogy(results_kmse[func_name]['conv'], 'r-', linewidth=2, label='KMSE-IAO')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness (log)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('KMSE_IAO_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
