import torch
import torch.optim as optim
import numpy as np
from Environment import *
from Soft_LQR import soft_LQR

def offline_actor_critic_algorithm(env, fixed_value, num_episodes=501, Δt=0.01, τ=0.5, γcritic=1):
    """
    Offline Actor-Critic Training with Monte Carlo Rollouts and Oracle Advantage Estimation.

    This function jointly trains:
        - An actor (stochastic policy network) using policy gradients with a fixed value function.
        - A critic (value function network) to estimate the cost-to-go via quadratic regression.

    Args:
        env (LQREnvironmentWithPolicy): Linear-quadratic control environment with entropy-regularized dynamics.
        fixed_value (Callable): Oracle value function V*(t, x) for computing actor TD advantages.
        num_episodes (int): Number of Monte Carlo training episodes.
        Δt (float): Time discretization step size.
        τ (float): Entropy regularization parameter (balances exploration vs exploitation).
        γcritic (float): (Unused) Placeholder for future critic discounting.

    Returns:
        policy_nn (PolicyNeuralNetwork): Trained stochastic policy π_θ(t, x) ~ N(μ(t, x), Σ(t)).
        value_nn (OnlyLinearValueNN): Trained quadratic value function approximation:
            V̂(t, x) = xᵀ Q(t) x + b(t)

    Training Logic:
        - **Actor Update**: Policy gradient with entropy regularization:
            Loss = -log π_θ(α | x) · [ΔV + (f_t + τ·logπ) Δt]
            ΔV is computed using oracle value function (fixed_value).
        - **Critic Update**: Supervised regression of Monte Carlo cost-to-go using:
            Predicted V̂(t_n, x_n) ≈ ∑_{k=n}^N (f_k + τ·logπ_k)·Δt + g_T

    """

    policy_nn = PolicyNeuralNetwork(hidden_size=16, d=env.action_dim)
    optimizer_actor = optim.Adam(policy_nn.parameters(), lr=1e-3)
    value_nn = OnlyLinearValueNN().to(env.device)
    optimizer_critic = optim.Adam(value_nn.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        
        X0 = env.sample_initial_state()
        X = X0.clone()
       
        costs = []
        states = []
        actions = []
        values = []
        log_probs = []
        for n in range(env.N):
            t = n * Δt
            X = X.detach()
            states.append((t, X))
           
            # Sample action according to the fixed policy
            
            value = fixed_value(t, X)
            values.append(value)
            
            
            mean,var = policy_nn(torch.tensor([t], dtype=torch.float32).to(env.device))
            
            policy =  torch.distributions.MultivariateNormal((mean @ X).squeeze(), var)
            α_t = policy.sample()
            actions.append(α_t)
            log_prob = policy.log_prob(α_t)
            log_probs.append(log_prob)
            
           
            # Apply action and observe cost and new state
            X, f_t = env.step(α_t)
            
            costs.append(f_t)
            if n == env.N - 1:
                g_T = env.observe_terminal_cost()
            
       
        # Compute critic loss L^(η)
        loss_actor = 0.0
        loss_critic = 0.0
        for n in range(env.N - 1):
            t, Xn = states[n]
            
            delta_v = abs(values[n + 1] - values[n])
            running_cost = (costs[n] + τ * log_probs[n]) * (Δt)
            log_prob = log_probs[n]

            loss_actor += -log_prob * (delta_v + running_cost)

            predicted_matrix, predicted_offset = value_nn(torch.tensor([t], dtype=torch.float32).to(env.device))
            v_hat = Xn.T @ predicted_matrix @ Xn + predicted_offset.squeeze()
            
            running_cost = sum(costs[k] + τ * env.fixed_policy_log_prob(actions[k], t, Xn) * (Δt) for k in range(n, env.N))
            terminal_adjust = g_T if n == env.N - 1 else 0.0
            
            L_hat = (v_hat - running_cost - terminal_adjust) ** 2
            loss_critic += L_hat
            
            

        optimizer_actor.zero_grad()  
        loss_actor.backward()  
        optimizer_actor.step()  
       

        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()


        if episode % 50 == 0:
            print(f"Episode {episode}, Actor Loss: {loss_actor.item()}, Critic Loss: {loss_critic.item()}")
            
    return policy_nn,value_nn

def Exercise_5():
    """
    Runs a full offline actor-critic training loop for a Soft LQR environment.

    Objective:
        - Learn both the optimal stochastic policy and the value function
          for a 2D linear-quadratic control system using the offline actor-critic algorithm.

    Steps:
        1. Define system matrices H, M, C, D, R for dynamics and cost.
        2. Set up an entropy-regularized LQR environment with Gaussian noise.
        3. Compute the oracle value function V*(t, x) using a soft LQR solver.
        4. Train a stochastic actor and a quadratic critic using Monte Carlo episodes.

    Returns:
        None (but prints training loss for actor and critic every 50 episodes)
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
    policy_nn,value_nn = offline_actor_critic_algorithm(env_policy, Lqr.value_function)
