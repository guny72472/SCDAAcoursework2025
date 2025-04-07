import numpy as np
import matplotlib.pyplot as plt
from Strict_LQR import strict_LQR
from Soft_LQR import soft_LQR

def simulate_trajectory(num_of_ts,x0,soft_Lqr,strict_Lqr,Terminal_time):
    """
    Simulates a single trajectory of the state under both strict and soft LQR controllers
    starting from the same initial state and under the same noise realization.

    Args:
        num_of_ts (int): Number of time steps for simulation.
        x0 (np.ndarray): Initial state vector (shape: [state_dim]).
        soft_Lqr (soft_LQR): Soft LQR controller instance.
        strict_Lqr (strict_LQR): Strict LQR controller instance.
        Terminal_time (float): Final time T for simulation horizon.

    Returns:
        tuple:
            - np.ndarray: Trajectory of state under strict LQR (shape: [num_of_ts, state_dim]).
            - np.ndarray: Trajectory of state under soft LQR (shape: [num_of_ts, state_dim]).

    """
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
    """
    Plots and compares the state trajectories of strict and soft LQR policies
    for four different initial conditions.

    For each initial condition, two plots are generated:
        1. Time evolution of both state variables.
        2. 2D phase trajectory (state-1 vs. state-2).

    Args:
        num_of_ts (int): Number of time steps for simulation.
        soft_Lqr (soft_LQR): Soft LQR controller instance.
        strict_Lqr (strict_LQR): Strict LQR controller instance.
        Terminal_time (float): Final time T for the simulations.

    Returns:
        None: Saves and shows the plots.
            - 'Ex2_1.png': Time vs. state plots (2x2 subplot grid).
            - 'Ex2_2.png': Phase plots of state-1 vs. state-2 (2x2 subplot grid).

    """
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

    fig.savefig("Ex2_1.png",dpi=300)    
    fig2.savefig("Ex2_2.png",dpi=300)    
    plt.tight_layout()
    plt.show()


def Exercise_2():
    """
    Executes the simulation and visualization of trajectories under strict and soft LQR control.

    Steps:
        1. Defines a 2D linear system with quadratic costs and additive Gaussian noise.
        2. Constructs both strict and soft LQR controllers.
        3. Simulates state trajectories for multiple initial conditions.
        4. Generates two sets of comparative plots:
            - Time vs. state trajectories.
            - Phase-space trajectories (x vs y).

    System parameters:
        - State/control dynamics: defined by matrices H, M.
        - Costs: state (C), control (D), terminal (R).
        - Noise: isotropic with covariance sigma @ sigmaᵀ.
        - Time horizon: T = 0.5 seconds.
        - Soft LQR parameters: entropy coefficient tau, discount gamma.

    """
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