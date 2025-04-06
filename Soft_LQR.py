import numpy as np
import torch
from scipy.integrate import solve_ivp
    
class soft_LQR:
    """
    A class to solve the soft (entropy-regularized) Linear Quadratic Regulator (LQR) problem.
    
    Compared to strict LQR, the soft LQR introduces a regularization term (entropy penalty),
    which results in a stochastic optimal control policy.
    """
    def __init__(self, H, M, sigma, C, D, R, T, time_grid, tau = 0, gamma = 1):
        """
        Initializes the soft LQR controller with system dynamics and cost structure.

        Args:
            H, M, sigma, C, D, R: System and cost matrices.
            T (float): Final time.
            time_grid (ndarray): Discretized time grid (numpy array or torch tensor).
            tau (float): Entropy regularization parameter (controls stochasticity).
            gamma (float): Scaling factor for entropy penalty.
        """
        self.m = m = len(R[0])
        self.H = H
        self.M = M
        self.sigma = sigma
        self.C = C
        self.D = D
        self.R = R
        self.T = T
        self.tau = tau
        self.gamma = gamma
        self.time_grid = time_grid
        self.dt = time_grid[1] - time_grid[0]

        # Regularized control cost matrix: D + τ / (2γ²) * I
        self.D_soft = D + (tau/(2*(gamma**2)))*np.eye(2)

        # Constant term from entropy regularization in the value function
        self.C_d_tau_gamma = - tau*np.log(((tau**(m/2))/(gamma**m))*((np.linalg.det(self.D_soft))**(0.5)))

        # Solve the Riccati equation backward in time
        self.S_grid = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """
        Defines the Riccati ODE (modified for soft LQR) as a first-order system.

        Riccati equation:
            dS/dt = S M D_soft^{-1} M^T S - H^T S - S H - C

        Args:
            t (float): Current time (not used; equation is autonomous).
            S_flat (ndarray): Flattened Riccati matrix S(t).

        Returns:
            ndarray: Flattened derivative of S(t).
        """
        S = S_flat.reshape(2, 2)
        dSdt = S.T @ self.M @ np.linalg.inv(self.D_soft) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return dSdt.flatten()

    def solve_riccati_ode(self):
        """
        Solves the Riccati ODE backward in time using `solve_ivp`.

        Returns:
            dict: Dictionary mapping each time t to the matrix S(t).
        """
        S_T = self.R.flatten()  
        sol = solve_ivp(self.riccati_ode, [self.T, 0], S_T, t_eval=np.flip(self.time_grid), atol=1e-8, rtol=1e-8)
        S_values = sol.y.T.reshape(-1, 2, 2)  
        return {t: S for t, S in zip(np.flip(self.time_grid), S_values)}

    def value_function(self, t, x):
        """
        Computes the value function for the soft LQR:
        
            v(t, x) = xᵀ S(t) x 
                      + ∫ₜᵀ Tr[σ σᵀ S(s)] ds 
                      + (T - t) C_d_tau_gamma ds

        Args:
            t (float): Current time.
            x (ndarray or torch.Tensor): Current state vector.

        Returns:
            torch.Tensor: Scalar value function v(t, x).
        """
        if isinstance(x, torch.Tensor):
            x=x.numpy()

        # Get the closest grid point to time t
        closest_t = max([t_n for t_n in self.time_grid if t_n <= t])
        S_t = self.S_grid[closest_t]

        # Approximate trace integral
        trace_term = 0
        for i in self.time_grid:
            if( i < closest_t):continue
            S_t_n = self.S_grid[i]
            trace_term += np.trace(self.sigma @ self.sigma.T @ S_t_n) * self.dt
        
        # Final value function includes trace and entropy terms
        final_value = x.T @ S_t @ x + trace_term + (self.T - closest_t)*self.C_d_tau_gamma
        return torch.tensor(final_value, dtype=torch.float32)

    def optimal_control(self, t, x):
        """
        Computes the stochastic optimal control for the soft LQR:
        
            a(t, x) ~ N(mean, covariance)
            where:
                mean = -D_soft^{-1} Mᵀ S(t) x
                covariance = τ D_soft
        
        Args:
            t (float): Current time.
            x (ndarray or torch.Tensor): Current state vector.

        Returns:
            Tuple[torch.Tensor, ndarray]: Sampled control vector and its mean.
        """
        if isinstance(x, torch.Tensor):
            x=x.numpy()
        # Get the closest time step to t
        closest_t = max([t_n for t_n in self.time_grid if t_n <= t])
        S_t = self.S_grid[closest_t]

        # Compute the mean of the control distribution
        mean = -np.linalg.inv(self.D_soft) @ self.M.T @ S_t @ x
        
        # Covariance of the control distribution
        covariance_matrix = self.tau*self.D_soft

        # Sample control from Gaussian policy
        control = np.random.multivariate_normal(mean, covariance_matrix)
        return torch.tensor(control, dtype=torch.float32), mean
    