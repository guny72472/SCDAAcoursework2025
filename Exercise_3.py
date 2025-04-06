import torch
import torch.optim as optim
import numpy as np
from Environment import *
from Soft_LQR import soft_LQR

# Critic Algorithm with the environment
def offline_critic_algorithm(env, fixed_policy, num_episodes=501, Δt=0.01, τ=0.5, γcritic=1):
    # Initialize parameter η
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

def Exercise_3():
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