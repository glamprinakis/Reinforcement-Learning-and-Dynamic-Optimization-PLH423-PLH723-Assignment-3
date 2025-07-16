import numpy as np
import random
from scipy.optimize import linprog
import matplotlib.pyplot as plt
K = 100  # Number of content items
C = int(0.2 * K)  # Number of cached items
N = 2  # Number of recommended items
q = 0.1  # Probability of ending a viewing session
alpha = 0.9  # Probability of choosing a recommended item
umin = 0.5  # Relevance threshold
p_k = 1 / K  # Probability of picking any item from the catalogue

# Create content catalogue
U = np.random.uniform(low=0, high=1, size=(K, K))  # U_ij indicates how related content j is to content i
np.fill_diagonal(U, 0)  # No self-recommendations

# Create cost vector for content
cost = np.ones(K)
cost[random.sample(range(K), C)] = 0  # Randomly assign some items to cache (cost 0)

# Calculate relevance: 1 if relevance is above threshold, 0 otherwise
relevance = np.where(U > umin, 1, 0)


def policy_iteration():
    """
    Using policy iteration to optimize recommendations.
    Returns a matrix where the i-th row contains the indexes of the items recommended after watching item i.
    """
    # Initial policy: recommend the first N relevant items after each content
    policy = np.zeros((K, N), dtype=int)
    iterations = 0
    while True:
        # Policy evaluation
        iterations += 1
        V = np.zeros(K)  # Value function
        for i in range(K):
            relevant_items = np.where(relevance[i] == 1)[0]  # Find relevant items
            if len(relevant_items) > 0:
                for j in relevant_items:
                    if j in policy[i]:
                        # Bellman Equation
                        V[i] = cost[j] + q * V[j]

        # Policy improvement
        new_policy = np.zeros((K, N), dtype=int)
        for i in range(K):
            # Choose the N items that minimize the value function
            values = np.full(K, np.inf)  # Initialize to infinity
            relevant_items = np.where(relevance[i] == 1)[0]  # Find relevant items
            if len(relevant_items) > 0:
                values[relevant_items] = cost[relevant_items] + q * V[relevant_items]
            new_policy[i] = values.argsort()[:N]  # Get the indexes of the N items with smallest values

        # Check if the policy has changed
        if np.array_equal(policy, new_policy):
            return policy, iterations  # Add iterations to the return statement
        else:
            policy = new_policy


# Run policy iteration and print the policy
policy, iterations = policy_iteration()
print(f"Policy:\n{policy}")
print(f"Number of iterations: {iterations}")


def simulate_session(policy, max_steps=1000):
    """
    Simulate a viewing session.
    The session starts with a random item and ends when the user decides to quit or after max_steps.
    The function returns the total cost of the session.
    """
    current_item = np.random.randint(K)  # Start with a random item
    cost_total = cost[current_item]  # Initialize total cost with the cost of the first item
    for _ in range(max_steps):
        if np.random.uniform() < q:  # The user decides to quit
            break

        if policy is not None:  # If there is a policy
            recommended_items = policy[current_item]  # Get recommended items
            relevant_recommended_items = [item for item in recommended_items if
                                          relevance[current_item][item] == 1]  # Filter out non-relevant items

            if len(recommended_items) == len(relevant_recommended_items):  # If all recommended items are relevant
                current_item = np.random.choice(
                    relevant_recommended_items)  # Pick a random item from relevant recommended items
            else:  # If at least one recommended item is not relevant
                current_item = np.random.randint(K)  # Pick a random item
        else:  # If there is no policy
            current_item = np.random.randint(K)  # Pick a random item

        cost_total += cost[current_item]  # Add the cost of the picked item
    return cost_total


num_sessions = 1000  # Number of simulated sessions
# Simulate sessions with the policy obtained via policy iteration
costs = [simulate_session(policy) for _ in range(num_sessions)]
average_cost_policy = np.mean(costs)
print(f'Average total cost with policy iteration: {average_cost_policy}')

# Simulate sessions with a random recommendation policy
costs = [simulate_session(None) for _ in range(num_sessions)]
average_cost_random = np.mean(costs)
print(f'Average total cost with random recommendations: {average_cost_random}')


def epsilon_func(t, k):
    return t ** (-1 / 3) * (k * np.log(t)) ** (1 / 3)

def q_learning(epochs=10000, learning_rate=0.1, discount_factor=0.9, k=1):
    """
    Using Q-learning with epsilon-greedy strategy to optimize recommendations.
    Returns a Q-table where Q[i][j] gives the estimated reward of recommending item j after item i.
    """
    # Initialize Q-table with zeros
    Q = np.zeros((K, K))

    for epoch in range(epochs):
        item = np.random.randint(K)  # Start with a random item

        for t in range(K):  # For each step in the session
            if np.random.uniform() < q:  # The user decides to quit
                break

            # Choose the action
            epsilon = epsilon_func(t + 1, k)  # Compute epsilon for this step
            if np.random.uniform() < epsilon:  # Choose a random action with probability epsilon
                recommended_item = np.random.randint(K)
            else:  # Choose the item with the highest Q-value
                recommended_item = np.argmax(Q[item])

            # Take the action and get the reward
            reward = relevance[item, recommended_item] - cost[recommended_item]

            # Update the Q-value for the state-action pair
            Q[item][recommended_item] = (1 - learning_rate) * Q[item][recommended_item] + learning_rate * (
                        reward + discount_factor * np.max(Q[recommended_item]))

            # Move to the next state
            item = recommended_item

    return Q

def simulate_session_q_learning(Q, max_steps=1000):
    """
    Simulate a viewing session with Q-learning.
    The session starts with a random item and ends when the user decides to quit or after max_steps.
    The function returns the total cost of the session.
    """
    current_item = np.random.randint(K)  # Start with a random item
    cost_total = cost[current_item]  # Initialize total cost with the cost of the first item
    for _ in range(max_steps):
        if np.random.uniform() < q:  # The user decides to quit
            break
        if np.random.uniform() > alpha:  # The user picks a random item
            current_item = np.random.randint(K)
        else:  # The user picks one of the recommended items
            recommended_items = np.argsort(Q[current_item])[-N:]  # Get the indexes of the N items with highest Q-values
            relevant_recommended_items = [item for item in recommended_items if
                                          relevance[current_item][item] == 1]  # Filter out non-relevant items

            if len(recommended_items) == len(relevant_recommended_items):  # If all recommended items are relevant
                current_item = np.random.choice(
                    relevant_recommended_items)  # Pick a random item from relevant recommended items
            else:  # If at least one recommended item is not relevant
                current_item = np.random.randint(K)  # Pick a random item
        cost_total += cost[current_item]  # Add the cost of the picked item
    return cost_total

# Calculate the Q-table via Q-learning
Q = q_learning(epochs=10000, learning_rate=0.1, discount_factor=0.9, k=1)

# Simulate sessions with the policy obtained via Q-learning
costs_q_learning = [simulate_session_q_learning(Q) for _ in range(num_sessions)]
average_cost_q_learning = np.mean(costs_q_learning)
print(f'Average total cost with Q-learning: {average_cost_q_learning}')

def initialize_variables(K):
    C = int(0.2 * K)  # Number of cached items

    # Create content catalogue
    U = np.random.uniform(low=0, high=1, size=(K, K))  # U_ij indicates how related content j is to content i
    np.fill_diagonal(U, 0)  # No self-recommendations

    # Create cost vector for content
    cost = np.ones(K)
    cost[random.sample(range(K), C)] = 0  # Randomly assign some items to cache (cost 0)

    # Calculate relevance: 1 if relevance is above threshold, 0 otherwise
    relevance = np.where(U > umin, 1, 0)

    return cost, relevance


def q_learning_with_scores(K, cost, relevance, epochs=10000, learning_rate=0.1, discount_factor=0.9, k=1):
    Q = np.zeros((K, K))
    scores = []
    for epoch in range(epochs):
        item = np.random.randint(K)

        for t in range(K):
            if np.random.uniform() < q:
                break
            epsilon = epsilon_func(t + 1, k)
            if np.random.uniform() < epsilon:
                recommended_item = np.random.randint(K)
            else:
                recommended_item = np.argmax(Q[item])
            reward = relevance[item, recommended_item] - cost[recommended_item]
            Q[item][recommended_item] = (1 - learning_rate) * Q[item][recommended_item] + learning_rate * (
                    reward + discount_factor * np.max(Q[recommended_item]))
            item = recommended_item

        if epoch % 100 == 0:  # Compute average score every 100 epochs
            average_score = np.mean([simulate_session_q_learning(Q, max_steps=1000) for _ in range(100)])
            scores.append(average_score)

    return Q, scores

sim_cov=1
if(sim_cov):
    K_values = [10, 50, 100]
    num_runs = 10
    plt.figure(figsize=(10, 6))

    for K in K_values:
        all_scores = []
        for _ in range(num_runs):
            cost, relevance = initialize_variables(K)
            Q, scores = q_learning_with_scores(K, cost, relevance, epochs=10000, learning_rate=0.1, discount_factor=0.9,
                                               k=1)
            all_scores.append(scores)
        average_scores = np.mean(all_scores, axis=0)  # take the average across the runs

        # Plotting the convergence
        plt.plot(np.arange(0, 10000, 100), average_scores, label=f'K={K}')

    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.title('Convergence of Q-learning for different K')
    plt.legend()
    plt.grid(True)
    plt.show()




import numpy as np

K = 5  # Number of content items
C = 2  # Number of cached items
N = 1  # Number of recommended items
q = 0.2  # Probability of ending a viewing session
alpha = 0.9  # Probability of choosing a recommended item
umin = 0.5  # Relevance threshold

# Create content catalogue
U = np.array([[0.0, 0.9, 0.3, 0.2, 0.1],
              [0.9, 0.0, 0.2, 0.8, 0.3],
              [0.9, 0.2, 0.0, 0.8, 0.4],
              [0.9, 0.4, 0.1, 0.0, 0.2],
              [0.9, 0.8, 0.4, 0.2, 0.0]])

# Create cost vector for content
cost = np.ones(K)
cost[[1, 3]] = 0  # Assign some items to cache (cost 0)

# Calculate relevance: 1 if relevance is above threshold, 0 otherwise
relevance = np.where(U > umin, 1, 0)

# Run policy iteration
policy, iterations = policy_iteration()
print(f"Policy:\n{policy}")
print(f"Number of iterations: {iterations}")

# Run Q-learning
Q = q_learning(epochs=10000)
# Convert Q-table to policy
policy_q_learning = np.argmax(Q, axis=1)
print(f'Q-learning: {policy_q_learning}')

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
# Define the parameters to test
K_values = [10, 50, 100]
umin_values = [0.1, 0.4]
alpha_values = [0.6, 0.9]
q_values = [0.05,0.5]

# Initialize the result arrays
average_costs_policy = np.zeros((len(K_values), len(umin_values), len(alpha_values), len(q_values)))
average_costs_q_learning = np.zeros((len(K_values), len(umin_values), len(alpha_values), len(q_values)))

# Loop over all parameter combinations
for i, K in enumerate(K_values):
    for j, umin in enumerate(umin_values):
        for k, alpha in enumerate(alpha_values):
            for l, q in enumerate(q_values):
                # Initialize the variables
                cost, relevance = initialize_variables(K)

                # Run policy iteration
                policy, _ = policy_iteration()
                costs = [simulate_session(policy) for _ in range(num_sessions)]
                average_costs_policy[i, j, k, l] = np.mean(costs)

                # Calculate the Q-table via Q-learning
                Q = q_learning(epochs=10000, learning_rate=0.1, discount_factor=0.9, k=1)
                costs_q_learning = [simulate_session_q_learning(Q) for _ in range(num_sessions)]
                average_costs_q_learning[i, j, k, l] = np.mean(costs_q_learning)

# Now you can print the results in a readable format
for i, K in enumerate(K_values):
    for j, umin in enumerate(umin_values):
        for k, alpha in enumerate(alpha_values):
            for l, q in enumerate(q_values):
                print(f"For K={K}, umin={umin}, alpha={alpha}, q={q}:")
                print(f"Average total cost with policy iteration: {average_costs_policy[i, j, k, l]}")
                print(f"Average total cost with Q-learning: {average_costs_q_learning[i, j, k, l]}")
                print()
