import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Strict_LQR import strict_LQR
def Monte_carlo(num_of_sim, num_of_ts, lqr, x0, Terminal_time):
    
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
    lqr = strict_LQR(H, M, sigma, C, D, R, T, time_grid)
    num_of_sim = 10000
    num_of_ts = 10000

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
    plt.savefig(f"Temp1_{x0}.jpeg")
    
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
    plt.savefig(f"Temp2_{x0}.jpeg")

def Exercise_1():
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


