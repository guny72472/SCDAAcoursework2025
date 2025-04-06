import numpy as np
import torch
from scipy.integrate import solve_ivp

class strict_LQR:
    """
    A class to solve the strict Linear Quadratic Regulator (LQR) problem.
    
    This class computes:
    - The solution to the Riccati differential equation.
    - The optimal value function v(t, x).
    - The optimal control a(t, x).
    """
    def __init__(self, H, M, sigma, C, D, R, T, time_grid):
        """
        Initialize the strict LQR problem with system and cost matrices.
        
        Args:
            H (ndarray): Drift matrix of the state dynamics.
            M (ndarray): Input matrix for the control.
            sigma (ndarray): Noise matrix.
            C (ndarray): State cost matrix.
            D (ndarray): Control cost matrix (positive definite).
            R (ndarray): Terminal cost matrix.
            T (float): Terminal time.
            time_grid (ndarray): Time discretization grid over [0, T].
        """
        self.H = H
        self.M = M
        self.sigma = sigma
        self.C = C
        self.D = D
        self.R = R
        self.T = T
        self.time_grid = time_grid
        self.S_grid = self.solve_riccati_ode()
        self.dt = time_grid[1] - time_grid[0]

    def riccati_ode(self, t, S_flat):
        """
        Defines the matrix Riccati differential equation as a system of first-order ODEs.
        
        Riccati equation:
            dS/dt = S M D^{-1} M^T S - H^T S - S H - C
        
        Args:
            t (float): Current time (not used in this autonomous equation).
            S_flat (ndarray): Flattened Riccati matrix S(t).
        
        Returns:
            ndarray: Flattened time derivative of S(t).
        """
        S = S_flat.reshape(2, 2)
        dSdt = S @ self.M @ np.linalg.inv(self.D) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return dSdt.flatten()

    def solve_riccati_ode(self):
        """
        Solves the Riccati equation backward in time from T to 0.
        
        Uses SciPy's `solve_ivp` to numerically integrate the ODE.
        
        Returns:
            dict: A mapping from each time step to its corresponding S(t) matrix.
        """
        S_T = self.R.flatten()   # Terminal condition: S(T) = R
        sol = solve_ivp(self.riccati_ode, [self.T, 0], S_T, t_eval=np.flip(self.time_grid), atol=1e-8, rtol=1e-8)
        S_values = sol.y.T.reshape(-1, 2, 2)   # Reshape solution to (n_time, 2, 2)
        
        return {t: S for t, S in zip(np.flip(self.time_grid), S_values)}

    def value_function(self, t, x):
        """
        Evaluates the value function v(t, x) = x^T S(t) x + integral_{t}^{T} Tr[sigma sigma^T S(s)] ds.
        
        The integral is approximated using a Riemann sum over the time grid.
        
        Args:
            t (float): Current time.
            x (ndarray): State vector (2D column).
        
        Returns:
            torch.Tensor: Value function evaluated at (t, x).
        """

        # Find the closest grid time <= t
        closest_t = max([t_n for t_n in self.time_grid if t_n <= t])
        S_t = self.S_grid[closest_t]

        # Compute the integral term: sum of traces of sigma sigma^T S_n over [t, T]
        trace_term = 0
        for i in self.time_grid:
            if( i < closest_t):continue
            S_t_n = self.S_grid[i]
            trace_term += np.trace(self.sigma @ self.sigma.T @ S_t_n) * self.dt
        
        # Quadratic form + trace integral
        return torch.tensor(x.T @ S_t @ x + trace_term, dtype=torch.float32)

    def optimal_control(self, t, x):
        """
        Computes the optimal control: a(t, x) = -D^{-1} M^T S(t) x
        
        Args:
            t (float): Current time.
            x (ndarray): Current state (2D column).
        
        Returns:
            torch.Tensor: Optimal control vector at (t, x).
        """

        # Find the closest grid time <= t
        closest_t = max([t_n for t_n in self.time_grid if t_n <= t])
        S_t = self.S_grid[closest_t]

        # Compute optimal control law
        control = -np.linalg.inv(self.D) @ self.M.T @ S_t @ x
        return torch.tensor(control, dtype=torch.float32)
    

