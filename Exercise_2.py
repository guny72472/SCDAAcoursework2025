import numpy as np
import matplotlib.pyplot as plt
from Strict_LQR import strict_LQR
from Soft_LQR import soft_LQR

def simulate_trajectory(num_of_ts,x0,soft_Lqr,strict_Lqr,Terminal_time):
    X_soft = [x0]
    X_strict = [x0]
    dt = Terminal_time / num_of_ts
    for t in range(num_of_ts - 1):
        curr_ts = t*dt
        dw = np.sqrt(dt)*np.random.randn(len(X_soft[-1]))
        
        control_soft, mean_soft = np.array(soft_Lqr.optimal_control(curr_ts,X_soft[-1]))
        X_soft.append(X_soft[-1] + dt * (soft_Lqr.H @ X_soft[-1].T + soft_Lqr.M @ control_soft.T).T + dw @ soft_Lqr.sigma.T)
        
        control_strict = np.array(strict_Lqr.optimal_control(curr_ts,X_strict[-1]))
        X_strict.append(X_strict[-1] + dt * (strict_Lqr.H @ X_strict[-1].T + strict_Lqr.M @ control_strict.T).T + dw @ strict_Lqr.sigma.T)
        
    return np.array(X_strict),np.array(X_soft)

def plot_trajectories(num_of_ts,soft_Lqr,strict_Lqr,Terminal_time):
    time_grid = np.linspace(0, Terminal_time, num_of_ts)
    initial_conditions = [
            [2, 2],
            [2, -2],
            [-2, -2],
            [-2, 2]
        ]
        
    # Create subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
    
    axs = axs.ravel()
    axs2 = axs2.ravel()
    # Colors for different controllers
    strict_color = 'blue'
    soft_color = 'red'
        
    for i, x0 in enumerate(initial_conditions):
        # Simulate trajectories
        strict_traj,soft_traj = simulate_trajectory(num_of_ts,np.array(x0),soft_Lqr,strict_Lqr,Terminal_time)
        
        # Plot trajectories
        axs[i].plot(time_grid, strict_traj[:, 0], linewidth=2 ,
                        label='Strict LQR', color=strict_color)
        axs[i].plot(time_grid, strict_traj[:, 1],  linewidth=2,
                        color=strict_color)
            
        axs[i].plot(time_grid, soft_traj[:, 0], 
                        label='Soft LQR', color=soft_color,linestyle='--')
        axs[i].plot(time_grid, soft_traj[:, 1], 
                        color=soft_color, linestyle='--')
            
        axs[i].set_title(f'Initial Condition: {x0}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('State')
        axs[i].legend()
        axs[i].grid(True)

        
        axs2[i].plot(strict_traj[:, 0],strict_traj[:, 1],label='Strict LQR', color=strict_color,linewidth=2)
        axs2[i].plot(soft_traj[:, 0],soft_traj[:, 1],label='Soft LQR', color=soft_color,linestyle='--')        
        axs2[i].set_title(f'Initial Condition: {x0}')
        axs2[i].set_xlabel('State-x')
        axs2[i].set_ylabel('State-y')
        axs2[i].legend()
        
    plt.tight_layout()
    plt.show()


def Exercise_2():
    H = np.array([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = np.array([[1.0, 1.0], [0.0, 1.0]])
    sigma = np.eye(2) * 0.5
    C = np.array([[1.0, 0.1], [0.1, 1.0]]) * 1.0
    D = np.array([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = np.array([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    tau = 0.1
    gamma=10
    num_of_ts=1000
    # Create LQR Controller
    time_grid = np.linspace(0, T, num_of_ts)
    strict_Lqr = strict_LQR(H, M, sigma, C, D, R, T, time_grid)
    soft_Lqr = soft_LQR(H, M, sigma, C, D, R, T, time_grid,tau,gamma)
    plot_trajectories(num_of_ts,soft_Lqr,strict_Lqr,T)