# Particle-Swarm-Optimization

Let's solve a simple optimization problem using Particle Swarm Optimization (PSO) in Python. We'll use PSO to find the minimum of the Rastrigin function, a commonly used test function in optimization with multiple local minima. 

The **Rastrigin function** is defined as:
$$f(\mathbf{x}) = A D + \sum_{i=1}^{D} \left[ x_i^2 - A \cos(2 \pi x_i) \right]$$
where:
- $D$ is the dimension of the problem,
- $A$ is usually set to 10.

The function has a global minimum at $\mathbf{x} = \mathbf{0}$, with $f(\mathbf{0}) = 0$.

### PSO Implementation in Python

We'll implement the PSO algorithm as described, with particles updating their positions and velocities to minimize the Rastrigin function.

```python
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
```

### Explanation of the Code

1. **Initialization**:
   - We set up a swarm with randomly initialized positions and velocities within the bounds $[-5.12, 5.12]$ for each particle.
   - Each particle has a **personal best** position and the swarm has a **global best** position.
  
2. **PSO Algorithm**:
   - For each iteration, we:
     - Update each particle’s **velocity** using the inertia, cognitive, and social terms.
     - Update the **position** of each particle based on its new velocity.
     - **Clip** positions to stay within the search space bounds.
     - Evaluate the new position, and update personal and global bests if a better score is found.

3. **Termination**:
   - The loop runs until the maximum number of iterations is reached.
   - The global best position and score are printed, which represent the algorithm’s solution to the problem.

### Expected Output

The output should show the **global best score** converging towards the global minimum (close to 0) as the iterations progress. The **global best position** should be close to$[0, 0]$(the minimum of the Rastrigin function in 2D).

This approach can be modified for different optimization functions by adjusting the objective function definition and problem parameters.
