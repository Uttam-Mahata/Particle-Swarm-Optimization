import numpy as np

# Define the Rastrigin function
def rastrigin(position, A=10):
    D = len(position)
    return A * D + sum(x**2 - A * np.cos(2 * np.pi * x) for x in position)

# PSO Parameters
num_particles = 30    # Number of particles
num_dimensions = 2    # Dimensionality of the problem (2D)
max_iterations = 100  # Maximum number of iterations
inertia_weight = 0.7  # Inertia weight for velocity update
c1 = 1.5              # Cognitive coefficient
c2 = 1.5              # Social coefficient
bounds = (-5.12, 5.12)  # Bounds for the Rastrigin function

# Initialize particle positions and velocities
positions = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
personal_best_positions = np.copy(positions)
personal_best_scores = np.array([rastrigin(pos) for pos in positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# PSO algorithm
for iteration in range(max_iterations):
    for i in range(num_particles):
        # Update velocity
        r1, r2 = np.random.rand(2)
        velocities[i] = (
            inertia_weight * velocities[i]
            + c1 * r1 * (personal_best_positions[i] - positions[i])
            + c2 * r2 * (global_best_position - positions[i])
        )

        # Update position
        positions[i] += velocities[i]

        # Apply bounds
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        # Evaluate the new position
        score = rastrigin(positions[i])

        # Update personal best
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

        # Update global best
        if score < global_best_score:
            global_best_score = score
            global_best_position = positions[i]

    # Optional: Print progress
    if iteration % 10 == 0 or iteration == max_iterations - 1:
        print(f"Iteration {iteration + 1}/{max_iterations}, Global Best Score: {global_best_score}")

print("\nOptimization completed.")
print("Global Best Position:", global_best_position)
print("Global Best Score:", global_best_score)