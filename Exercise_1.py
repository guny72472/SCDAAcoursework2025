import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Strict_LQR import strict_LQR
def Monte_carlo(num_of_sim, num_of_ts, lqr, x0, Terminal_time):
    """
    Estimates the value function at initial state x0 for a given LQR system
    using Monte Carlo simulation.

    The system is evolved forward in time using the optimal control at each
    time step. The running and terminal costs are accumulated and averaged
    across simulations.

    Args:
        num_of_sim (int): Number of Monte Carlo simulations to average over.
        num_of_ts (int): Number of time steps in each simulation.
        lqr (strict_LQR): Instance of the strict LQR solver providing dynamics and optimal control.
        x0 (np.ndarray): Initial state vector (shape: [state_dim]).
        Terminal_time (float): Final time horizon T.

    Returns:
        float: The estimated expected cost (value function) from state x0.

    """
    dt = Terminal_time / num_of_ts
    cost_integrand = 0
    cost = 0
    for i in range(num_of_sim):
        X = x0
        for t in range(num_of_ts):
            curr_ts = t*dt
            control = np.array(lqr.optimal_control(curr_ts,X))
            dw = np.sqrt(dt)*np.random.randn(len(X))
            cost_integrand += np.dot(X, lqr.C @ X) + np.dot(control, lqr.D @ control)
            X = X + dt * (lqr.H @ X.T + lqr.M @ control.T).T + dw @ lqr.sigma.T
        cost += cost_integrand * dt + np.dot(X, lqr.R @ X)
        cost_integrand = 0
    
    return cost/num_of_sim

def Plot_Error_graph(H, M, sigma, C, D, R, T, time_grid, x0):
    """
    Plots error convergence of the Monte Carlo estimate of the value function
    with respect to:
      1. Number of Monte Carlo samples (fixed time discretization).
      2. Number of time steps (fixed number of simulations).

    The error is computed as the absolute difference between the analytical
    value function (from `strict_LQR.value_function()`) and the Monte Carlo estimate.

    Args:
        H (np.ndarray): Drift matrix for state dynamics.
        M (np.ndarray): Control matrix.
        sigma (np.ndarray): Diffusion matrix (noise coefficient).
        C (np.ndarray): State running cost matrix.
        D (np.ndarray): Control running cost matrix.
        R (np.ndarray): Terminal cost matrix.
        T (float): Time horizon.
        time_grid (np.ndarray): Array of time points for ODE solver in strict_LQR.
        x0 (np.ndarray): Initial state vector for simulation and comparison.

    Returns:
        None: Saves two plots as image files:
            - 'Ex1_{x0}.jpeg': Error vs. number of samples (log-log scale)
            - 'Ex2_{x0}.jpeg': Error vs. number of time steps (log-log scale)

    """

    lqr = strict_LQR(H, M, sigma, C, D, R, T, time_grid)
    num_of_sim = 100
    num_of_ts = 100

    arr_num_of_sim = [2**i for i in range(1, 10)]
    arr_num_of_ts = [2**i for i in range(4, 10)]

    errors=[]
    
    true_value = lqr.value_function(0, x0).item()
    
    for temp_num_of_sim in arr_num_of_sim:
        estimated_value = Monte_carlo(temp_num_of_sim,num_of_ts,lqr,x0,T)
        errors.append(abs(estimated_value - true_value))
        print(f"Done for {0}",temp_num_of_sim)
        
    slope_sim = -0.5
    ref_errors_sim = errors[0] * (np.array(arr_num_of_sim[0]) / np.array(arr_num_of_sim))**(-slope_sim)

    plt.figure()
    plt.loglog(arr_num_of_sim, errors, marker='o', label='Actual Error')
    plt.loglog(arr_num_of_sim, ref_errors_sim, '--', label='Reference slope -0.5')
    plt.xlabel('Number of Monte Carlo Samples')
    plt.ylabel('Error')
    plt.title('Error Convergence vs. Sample Size')
    plt.grid(True)
    plt.savefig(f"Ex1_{x0}.jpeg")
    
    errors=[]
    for temp_num_of_ts in arr_num_of_ts:
        estimated_value = Monte_carlo(num_of_sim,temp_num_of_ts,lqr,x0,T)
        errors.append(abs(estimated_value - true_value))
        print(f"Done for {0}",temp_num_of_ts)
    
    slope_ts = -1
    ref_errors_ts = errors[0] * (np.array(arr_num_of_ts[0]) / np.array(arr_num_of_ts))**(-slope_ts)

    plt.figure()
    plt.loglog(arr_num_of_ts, errors, marker='o', label='Actual Error')
    plt.loglog(arr_num_of_ts, ref_errors_ts, '--', label='Reference slope -1')
    plt.xlabel('Number of Time Steps')
    plt.ylabel('Error')
    plt.title('Error Convergence vs. Time Steps')
    plt.grid(True)
    plt.savefig(f"Ex2_{x0}.jpeg")

def Exercise_1():
    """
    Runs the full error benchmarking exercise for two different initial states.

    The function:
        - Sets up a sample 2D Linear-Quadratic Regulator (LQR) problem.
        - Computes the true value function using the strict solution (Riccati-based).
        - Estimates the value function using Monte Carlo simulations.
        - Plots the absolute error in log-log scale with respect to:
            - Number of Monte Carlo samples.
            - Number of time steps in simulation.

    Initial states tested:
        - x₀ = [1.0, 1.0]
        - x₀ = [2.0, 2.0]

    Returns:
        None: Two sets of log-log error plots are saved per initial state.
    """
    H = np.array([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = np.array([[1.0, 1.0], [0.0, 1.0]])
    sigma = np.eye(2) * 0.5
    C = np.array([[1.0, 0.1], [0.1, 1.0]]) * 1.0
    D = np.array([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = np.array([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5

    time_grid = np.linspace(0, T, 1000)
    x0 = np.array([1.0, 1.0])
    Plot_Error_graph(H, M, sigma, C, D, R, T, time_grid,x0)
            
    x0 = np.array([2.0, 2.0])
    Plot_Error_graph(H, M, sigma, C, D, R, T, time_grid,x0)


