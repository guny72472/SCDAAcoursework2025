import torch
import torch.optim as optim
import numpy as np
from Environment import *
from Soft_LQR import soft_LQR

def offline_actor_algorithm(env, fixed_value, num_episodes=501, Δt=0.01, τ=0.5, γcritic=1):
    # Initialize parameter η
    policy_nn = PolicyNeuralNetwork(hidden_size=16, d=env.action_dim)
    optimizer = optim.Adam(policy_nn.parameters(), lr=1e-3)
   
    for episode in range(num_episodes):
        # Sample initial state X0 ~ ρ
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
            
       
        # Compute critic loss L^(η)
        loss = 0.0
        for n in range(env.N - 1):
            t, Xn = states[n]
            t_next, Xn_next = states[n + 1]
            delta_v = abs(values[n + 1] - values[n])
            running_cost = (costs[n] + τ * log_probs[n]) * (Δt)
            log_prob = log_probs[n]

            loss_actor = -log_prob * (delta_v + running_cost)
            loss += loss_actor
            

        optimizer.zero_grad()  # Reset gradients before next step
        loss.backward()  # Backpropagate the accumulated loss
        optimizer.step()  # Update policy parameters
       
        if episode % 50 == 0:
            print(f"Episode {episode}, Actor Loss: {loss.item()}")
            
    return policy_nn

def Exercise_4():
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
    policy_nn = offline_actor_algorithm(env_policy, Lqr.value_function)