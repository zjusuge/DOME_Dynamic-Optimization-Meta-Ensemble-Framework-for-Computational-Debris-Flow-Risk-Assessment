import numpy as np
import random


class GCRA:
    """
    Greater Cane Rat Algorithm (GCRA)

    A lightweight and robust implementation for continuous optimization.

    Notes
    -----
    - Backward-compatible with the earlier interface used in this repository:
        GCRA(objective_function, population_size, max_iterations, dimensions, bounds)
    - Added support for:
        * random_state
        * verbose logging control
        * early stopping
        * elitism
        * mutation for diversity
    """

    def __init__(
        self,
        objective_function,
        population_size=30,
        max_iterations=100,
        dimensions=10,
        bounds=None,
        random_state=42,
        verbose=True,
        early_stopping_rounds=None,
        tolerance=1e-10,
        elite_fraction=0.10,
        mutation_probability=0.10
    ):
        self.objective_function = objective_function
        self.population_size = int(max(2, population_size))
        self.max_iterations = int(max(1, max_iterations))
        self.dimensions = int(max(1, dimensions))
        self.bounds = bounds if bounds is not None else [(0.0, 1.0)] * self.dimensions

        if len(self.bounds) != self.dimensions:
            raise ValueError(
                f"Length of bounds ({len(self.bounds)}) must equal dimensions ({self.dimensions})."
            )

        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.tolerance = float(max(tolerance, 0.0))
        self.elite_fraction = float(min(max(elite_fraction, 0.0), 0.5))
        self.mutation_probability = float(min(max(mutation_probability, 0.0), 1.0))

        # Reproducible random generators
        self.rng = np.random.default_rng(self.random_state)
        self.py_random = random.Random(self.random_state)

        # GCRA adaptive parameters
        self.alpha = 2.0   # exploration strength
        self.beta = 0.5    # exploitation strength
        self.p1 = 0.7      # foraging tendency
        self.p2 = 0.3      # hiding tendency

        # Tracking
        self.best_fitness_history = []
        self.convergence_data = []

        self._validate_bounds()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _validate_bounds(self):
        validated = []
        for idx, bound in enumerate(self.bounds):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise ValueError(f"Bound at index {idx} must be a (lower, upper) tuple/list.")
            lower, upper = float(bound[0]), float(bound[1])
            if lower > upper:
                lower, upper = upper, lower
            validated.append((lower, upper))
        self.bounds = validated

    def _clip_to_bounds(self, individual):
        clipped = np.asarray(individual, dtype=float).copy()
        for i in range(self.dimensions):
            lower, upper = self.bounds[i]
            clipped[i] = np.clip(clipped[i], lower, upper)
        return clipped

    def _safe_objective(self, individual):
        try:
            value = self.objective_function(np.asarray(individual, dtype=float))
            value = float(value)
            if not np.isfinite(value):
                return float("inf")
            return value
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------
    # Population initialization and evaluation
    # ------------------------------------------------------------------
    def initialize_population(self):
        """Initialize population uniformly within bounds."""
        population = []
        for _ in range(self.population_size):
            individual = np.zeros(self.dimensions, dtype=float)
            for i in range(self.dimensions):
                lower, upper = self.bounds[i]
                individual[i] = self.rng.uniform(lower, upper)
            population.append(individual)
        return population

    def evaluate_population(self, population):
        """Evaluate the full population safely."""
        fitness = np.zeros(len(population), dtype=float)
        for i, individual in enumerate(population):
            fitness[i] = self._safe_objective(individual)
        return fitness

    # ------------------------------------------------------------------
    # GCRA movement mechanisms
    # ------------------------------------------------------------------
    def adaptive_parameters(self, iteration):
        """
        Update GCRA parameters adaptively.

        Early iterations:
            stronger exploration
        Later iterations:
            stronger exploitation
        """
        progress = iteration / max(1, self.max_iterations - 1)

        self.alpha = max(0.1, 2.0 * (1.0 - progress))
        self.beta = min(1.0, 0.3 + 0.7 * progress)

        self.p1 = 0.7 - 0.3 * progress
        self.p2 = 0.3 + 0.4 * progress

    def foraging_behavior(self, individual, best_individual, iteration):
        """
        Exploration-oriented movement.
        """
        new_individual = individual.copy()
        progress = iteration / max(1, self.max_iterations - 1)
        exploration_factor = self.alpha * (1.0 - progress)

        for i in range(self.dimensions):
            lower, upper = self.bounds[i]
            span = upper - lower
            r1, r2, r3 = self.rng.random(3)

            if r1 < self.p1:
                # Move toward global best with exploration noise
                attraction = best_individual[i] - individual[i]
                noise = (r3 - 0.5) * span * 0.25
                new_individual[i] = individual[i] + exploration_factor * r2 * attraction + noise
            else:
                # Random local/global search
                new_individual[i] = individual[i] + exploration_factor * (2.0 * r3 - 1.0) * span * 0.35

        return self._clip_to_bounds(new_individual)

    def hiding_behavior(self, individual, best_individual, population):
        """
        Exploitation-oriented movement.
        """
        new_individual = individual.copy()
        safe_individual = population[self.py_random.randrange(len(population))]

        for i in range(self.dimensions):
            r1, r2 = self.rng.random(2)

            if r1 < self.p2:
                # Move toward a safe peer
                new_individual[i] = individual[i] + self.beta * r2 * (safe_individual[i] - individual[i])
            else:
                # Move toward the best known position
                new_individual[i] = individual[i] + self.beta * r2 * (best_individual[i] - individual[i])

        return self._clip_to_bounds(new_individual)

    def mutation(self, individual, iteration):
        """
        Small Gaussian mutation to preserve search diversity.
        """
        mutated = individual.copy()
        progress = iteration / max(1, self.max_iterations - 1)
        decay = max(0.05, 1.0 - progress)

        for i in range(self.dimensions):
            if self.rng.random() < self.mutation_probability:
                lower, upper = self.bounds[i]
                span = upper - lower
                sigma = max(span * 0.05 * decay, 1e-12)
                mutated[i] += self.rng.normal(0.0, sigma)

        return self._clip_to_bounds(mutated)

    def update_position(self, individual, best_individual, population, iteration):
        """
        Update a single individual using GCRA mechanisms.
        """
        if self.rng.random() < 0.5:
            candidate = self.foraging_behavior(individual, best_individual, iteration)
        else:
            candidate = self.hiding_behavior(individual, best_individual, population)

        candidate = self.mutation(candidate, iteration)
        return self._clip_to_bounds(candidate)

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------
    def optimize(self):
        """
        Run optimization and return:

        Returns
        -------
        best_individual : np.ndarray
        best_fitness : float
        """
        self.best_fitness_history = []
        self.convergence_data = []

        self._log("Initializing GCRA optimization...")

        population = self.initialize_population()
        fitness = self.evaluate_population(population)

        best_idx = int(np.argmin(fitness))
        best_individual = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        self._log(f"Initial best fitness: {best_fitness:.6f}")

        elite_count = max(1, int(np.ceil(self.elite_fraction * self.population_size)))
        no_improvement_rounds = 0

        for iteration in range(self.max_iterations):
            self.adaptive_parameters(iteration)

            # Store elites from the current population
            sorted_idx = np.argsort(fitness)
            elite_pairs = [
                (population[idx].copy(), float(fitness[idx]))
                for idx in sorted_idx[:elite_count]
            ]

            # Generate candidate population
            new_population = []
            for i in range(self.population_size):
                candidate = self.update_position(
                    population[i],
                    best_individual,
                    population,
                    iteration
                )
                new_population.append(candidate)

            new_fitness = self.evaluate_population(new_population)

            # Greedy replacement
            for i in range(self.population_size):
                if new_fitness[i] <= fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            # Elitism: preserve top individuals if needed
            worst_indices = np.argsort(fitness)[::-1][:elite_count]
            for (elite_individual, elite_fitness), worst_idx in zip(elite_pairs, worst_indices):
                if elite_fitness < fitness[worst_idx]:
                    population[worst_idx] = elite_individual.copy()
                    fitness[worst_idx] = elite_fitness

            # Update global best
            current_best_idx = int(np.argmin(fitness))
            current_best_fitness = float(fitness[current_best_idx])

            if current_best_fitness + self.tolerance < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1

            avg_fitness = float(np.mean(fitness)) if len(fitness) > 0 else float("inf")
            std_fitness = float(np.std(fitness)) if len(fitness) > 0 else float("inf")

            self.best_fitness_history.append(best_fitness)
            self.convergence_data.append({
                "iteration": int(iteration + 1),
                "best_fitness": float(best_fitness),
                "average_fitness": avg_fitness,
                "std_fitness": std_fitness,
                "alpha": float(self.alpha),
                "beta": float(self.beta),
                "p1": float(self.p1),
                "p2": float(self.p2)
            })

            report_every = max(1, self.max_iterations // 5)
            if iteration % report_every == 0 or iteration == self.max_iterations - 1:
                self._log(
                    f"Iteration {iteration + 1}/{self.max_iterations}: "
                    f"Best = {best_fitness:.6f}, Avg = {avg_fitness:.6f}, Std = {std_fitness:.6f}"
                )

            if (
                self.early_stopping_rounds is not None
                and no_improvement_rounds >= int(self.early_stopping_rounds)
            ):
                self._log(
                    f"Early stopping triggered after {iteration + 1} iterations "
                    f"(no improvement for {no_improvement_rounds} rounds)."
                )
                break

        self._log(f"GCRA optimization completed. Final best fitness: {best_fitness:.6f}")
        return best_individual, best_fitness

    def get_convergence_history(self):
        """Return convergence history."""
        return {
            "best_fitness_history": self.best_fitness_history,
            "convergence_data": self.convergence_data
        }
