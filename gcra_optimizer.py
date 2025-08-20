import numpy as np
import random


class GCRA:
    """
    Greater Cane Rat Algorithm (GCRA) Implementation
    Based on Agushaka et al. (2024) - cited in the manuscript
    """

    def __init__(self, objective_function, population_size=30, max_iterations=100,
                 dimensions=10, bounds=None):
        self.objective_function = objective_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dimensions = dimensions
        self.bounds = bounds if bounds else [(0, 1)] * dimensions

        # GCRA-specific parameters
        self.alpha = 2.0  # Exploration parameter
        self.beta = 0.5  # Exploitation parameter
        self.p1 = 0.7  # Probability for foraging behavior
        self.p2 = 0.3  # Probability for hiding behavior

        # Optimization tracking
        self.best_fitness_history = []
        self.convergence_data = []

    def initialize_population(self):
        """Initialize population of cane rats randomly within bounds"""
        population = []

        for _ in range(self.population_size):
            individual = []
            for i in range(self.dimensions):
                lower, upper = self.bounds[i]
                individual.append(random.uniform(lower, upper))
            population.append(np.array(individual))

        return population

    def evaluate_population(self, population):
        """Evaluate fitness for entire population"""
        fitness = []

        for individual in population:
            try:
                fit = self.objective_function(individual)
                if np.isnan(fit) or np.isinf(fit):
                    fit = float('inf')
                fitness.append(fit)
            except Exception as e:
                fitness.append(float('inf'))

        return np.array(fitness)

    def foraging_behavior(self, individual, best_individual, iteration):
        """
        Implement foraging behavior of cane rats
        Exploration phase focusing on food search
        """
        new_individual = individual.copy()

        for i in range(self.dimensions):
            r1, r2, r3 = random.random(), random.random(), random.random()

            # Adaptive foraging with decreasing exploration
            exploration_factor = self.alpha * (1 - iteration / self.max_iterations)

            if r1 < self.p1:  # Foraging towards best position
                new_individual[i] = (individual[i] +
                                     exploration_factor * r2 * (best_individual[i] - individual[i]))
            else:  # Random foraging
                new_individual[i] = individual[i] + exploration_factor * (2 * r3 - 1)

            # Ensure bounds
            lower, upper = self.bounds[i]
            new_individual[i] = np.clip(new_individual[i], lower, upper)

        return new_individual

    def hiding_behavior(self, individual, best_individual, population):
        """
        Implement hiding behavior of cane rats
        Exploitation phase focusing on safety and local search
        """
        new_individual = individual.copy()

        # Find a safe position (random individual from population)
        safe_individual = random.choice(population)

        for i in range(self.dimensions):
            r1, r2 = random.random(), random.random()

            if r1 < self.p2:  # Hide towards safe position
                new_individual[i] = (individual[i] +
                                     self.beta * r2 * (safe_individual[i] - individual[i]))
            else:  # Hide towards best position
                new_individual[i] = (individual[i] +
                                     self.beta * r2 * (best_individual[i] - individual[i]))

            # Ensure bounds
            lower, upper = self.bounds[i]
            new_individual[i] = np.clip(new_individual[i], lower, upper)

        return new_individual

    def update_position(self, individual, best_individual, population, iteration):
        """
        Update individual position using GCRA mechanisms
        Combines foraging and hiding behaviors
        """
        r = random.random()

        if r < 0.5:  # Foraging behavior
            return self.foraging_behavior(individual, best_individual, iteration)
        else:  # Hiding behavior
            return self.hiding_behavior(individual, best_individual, population)

    def adaptive_parameters(self, iteration):
        """Update GCRA parameters adaptively during optimization"""
        # Decrease exploration, increase exploitation over time
        progress = iteration / self.max_iterations

        self.alpha = 2.0 * (1 - progress)  # Decreasing exploration
        self.beta = 0.5 + 0.5 * progress  # Increasing exploitation

        # Adaptive probabilities
        self.p1 = 0.7 * (1 - progress) + 0.3 * progress
        self.p2 = 0.3 + 0.4 * progress

    def optimize(self):
        """
        Main GCRA optimization loop
        Returns best solution and fitness
        """
        print("Initializing GCRA optimization...")

        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)

        # Track best solution
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        print(f"Initial best fitness: {best_fitness:.6f}")

        # Optimization loop
        for iteration in range(self.max_iterations):
            # Update adaptive parameters
            self.adaptive_parameters(iteration)

            # Create new population
            new_population = []

            for i in range(self.population_size):
                # Update position using GCRA mechanisms
                new_individual = self.update_position(
                    population[i], best_individual, population, iteration
                )
                new_population.append(new_individual)

            # Evaluate new population
            new_fitness = self.evaluate_population(new_population)

            # Selection: keep better individuals
            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_individual = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            # Track convergence
            self.best_fitness_history.append(best_fitness)
            self.convergence_data.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'average_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })

            # Progress reporting
            if iteration % 20 == 0 or iteration == self.max_iterations - 1:
                print(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}, "
                      f"Avg fitness = {np.mean(fitness):.6f}")

        print(f"GCRA optimization completed. Final best fitness: {best_fitness:.6f}")

        return best_individual, best_fitness

    def get_convergence_history(self):
        """Return optimization convergence history"""
        return {
            'best_fitness_history': self.best_fitness_history,
            'convergence_data': self.convergence_data
        }