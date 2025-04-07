import torch
import torch.optim as optim
import numpy as np
from Environment import *
from Soft_LQR import soft_LQR

# Critic Algorithm with the environment
def offline_critic_algorithm(env, fixed_policy, num_episodes=501, Δt=0.01, τ=0.5, γcritic=1):
    """
    Offline Critic Training Algorithm using Monte Carlo rollout under a fixed policy.

    Args:
        env (LQREnvironmentWithPolicy): Environment instance with dynamics and cost model.
        fixed_policy (Callable): Function implementing a fixed stochastic policy π(t, x).
                                 Must return a tuple (sampled_action, mean_action).
        num_episodes (int): Number of Monte Carlo rollouts to train the critic.
        Δt (float): Time discretization step.
        τ (float): Entropy regularization coefficient for log-probabilities.
        γcritic (float): (Unused) Discount factor placeholder.

    Returns:
        OnlyLinearValueNN: Trained neural network approximator for the value function.

    Training Procedure:
        - For each episode:
            * Sample initial state.
            * Roll out trajectory using fixed policy.
            * Compute cumulative (regularized) cost-to-go.
            * Predict value using critic at each state.
            * Minimize squared error between prediction and Monte Carlo return.
        - Loss used: 
            L^η(t, x) = (V_θ(t, x) - MC_return(t, x))^2
        
    """
    
    value_nn = OnlyLinearValueNN().to(env.device)
    optimizer = optim.Adam(value_nn.parameters(), lr=1e-3)
    Losses =[]
    for episode in range(num_episodes):
        # Sample initial state X0 ~ ρ
        X0 = env.sample_initial_state()
        X = X0.clone()
       
        costs = []
        states = []
        actions = []
       
        for n in range(env.N):
            t = n * Δt
            states.append((t, X))
           
            # Sample action according to the fixed policy
            α_t,_ = fixed_policy(t, X)
            actions.append(α_t)
           
            # Apply action and observe cost and new state
            X, f_t = env.step(α_t)
           
            costs.append(f_t)
           
            if n == env.N - 1:
                g_T = env.observe_terminal_cost()
       
        # Compute critic loss L^(η)
        loss = 0.0
        for n in range(env.N):
            t, Xn = states[n]
            predicted_matrix, predicted_offset = value_nn(torch.tensor([t], dtype=torch.float32).to(env.device))
            v_hat = Xn.T @ predicted_matrix @ Xn + predicted_offset.squeeze()
           
            running_cost = sum(costs[k] + τ * env.fixed_policy_log_prob(actions[k], t, Xn) * (Δt) for k in range(n, env.N))
            terminal_adjust = g_T if n == env.N - 1 else 0.0
            
            L_hat = (v_hat - running_cost - terminal_adjust) ** 2
            loss += L_hat
        Losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        if episode % 50 == 0:
            print(f"Episode {episode}, Critic Loss: {min(Losses)}")
            Losses=[]
            
        

    return value_nn

def evaluate_approximation(fixed_value,value_network):
    """
    Evaluates the trained value network against the exact value function from the soft LQR.

    Args:
        fixed_value (Callable): Ground truth value function V*(t, x) from the soft LQR solver.
        value_network (OnlyLinearValueNN): Trained value function approximator.

    Returns:
        None: Prints maximum absolute error on a test grid.

    Evaluation Details:
        - Evaluates V*(t, x) and V_θ(t, x) on a grid:
            * t ∈ {0, 1/6, 2/6, 1/2}
            * x ∈ [-3, 3]^2 (grid of 20×20)
        - Predicted value: V_θ(t, x) = xᵀ A(t) x + b(t)
        - Measures: max |V*(t, x) - V_θ(t, x)|

    """
    times = [0, 1/6, 2/6, 1/2]
    state_ranges = np.linspace(-3, 3, 20)

    max_error = 0
    for t in times:
        for x1 in state_ranges:
            for x2 in state_ranges:
                x = np.array([x1, x2])
                
                # Compute exact value
                exact_value = fixed_value(t, x)
                
                # Compute predicted value
                t_tensor = torch.tensor([t], dtype=torch.float32)
                x_tensor = torch.tensor(x, dtype=torch.float32)
                
                # Get matrix and offset from value network
                matrix, offset = value_network(t_tensor)
                predicted_value = x_tensor.T @ matrix.squeeze() @ x_tensor + offset.squeeze()
                
                # Compute error
                error = abs(exact_value - predicted_value.item())
                max_error = max(max_error, error)
    
    print(f"Max Error For the given case is : {max_error}")

def Exercise_3():
    """
    Executes the full critic training and evaluation pipeline for the soft LQR problem.

    Steps:
        1. Defines a 2D linear system with quadratic state-control cost and entropy-regularized objective.
        2. Constructs the stochastic environment for episodic rollouts.
        3. Initializes soft LQR controller to generate optimal (mean + noise) controls.
        4. Runs Monte Carlo-based critic training to learn V_θ(t, x).
        5. Evaluates learned value function against exact analytic solution.

    System Setup:
        - Dynamics: dx = (H·x + M·u) dt + σ dW
        - State cost: xᵀC·x, Control cost: uᵀD·u
        - Terminal cost: x_Tᵀ R·x_T
        - Regularization: τ · entropy
        - Time: T = 0.5, Δt = 0.005

    Returns:
        None: Trains and evaluates critic, prints max approximation error.
    """
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32) * 0.5
    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float32)
    D = torch.tensor([[1, 0.0], [0.0, 1]], dtype=torch.float32)
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float32) * 10.0
    sigma = torch.eye(2, dtype=torch.float32) * 0.5
    gamma = 1.0
    T = 0.5
    dt = 0.005
    tau = 0.5
    initial_distribution = torch.FloatTensor(2).uniform_(-2.0, 2.0)
    num_of_ts = int(T / dt)

    env_policy = LQREnvironmentWithPolicy(H, M, C, D, R, sigma, gamma, initial_distribution, T, dt)
    time_grid = np.linspace(0, T, num_of_ts)
    Lqr = soft_LQR(H.numpy(), M.numpy(), sigma.numpy(), C.numpy(), D.numpy(), R.numpy(), T, time_grid, tau, gamma)

    # Train the critic
    value_nn = offline_critic_algorithm(env_policy, Lqr.optimal_control)
    evaluate_approximation(Lqr.value_function,value_nn)