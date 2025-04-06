import torch
import numpy as np
import torch.nn as nn

class LQREnvironmentWithPolicy:
    """
    A simulation environment for Linear Quadratic Regulator (LQR) systems,
    designed for use in Monte Carlo sampling and actor-critic reinforcement learning.

    Attributes:
        H, M (torch.Tensor): Dynamics matrices.
        C, D, R (torch.Tensor): Cost matrices for state, action, and terminal cost.
        sigma (torch.Tensor): Noise (diffusion) matrix.
        gamma (float): Scaling factor for entropy regularization.
        initial_distribution (torch.Tensor): Initial state distribution (assumed deterministic here).
        T (float): Total simulation time.
        dt (float): Discretization time step.
        N (int): Number of simulation steps.
        action_dim (int): Dimensionality of the control vector.
    """
    def __init__(self, H, M, C, D, R, sigma, gamma, initial_distribution, T, dt):
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.gamma = gamma
        self.initial_distribution = initial_distribution
        self.T = T
        self.dt = dt
        self.device = torch.device('cpu')
        self.current_state = None
        self.N = int(T / dt)
        self.action_dim = M.size(1)

    def sample_initial_state(self):
        """
        Initializes and returns the starting state from the initial distribution.
        (Assumes initial_distribution is a torch.Tensor.)

        Returns:
            torch.Tensor: The initial state.
        """
        self.current_state = self.initial_distribution.to(self.device)
        return self.current_state

    def observe_terminal_cost(self):
        """
        Computes the terminal cost: x(T)ᵀ R x(T)

        Returns:
            float: Scalar terminal cost.
        """
        return (self.current_state.T @ self.R @ self.current_state).item()

    def f(self, action, x):
        """
        Computes instantaneous cost:
            f(x, a) = xᵀ C x + aᵀ D a

        Args:
            action (torch.Tensor): Control action vector.
            x (torch.Tensor): State vector.

        Returns:
            float: Instantaneous cost at this (x, a).
        """
        xCx = (x.T @ self.C @ x).item()
        aDa = (action.T @ self.D @ action).item()
        return xCx + aDa

    def step(self, action):
        """
        Simulates one step of the LQR system using Euler-Maruyama discretization.

        Args:
            action (torch.Tensor): Action taken at current state.

        Returns:
            Tuple[torch.Tensor, float]: New state and cost incurred at current step.
        """

        action = action.view(-1, 1)

        # Generate Brownian noise: dW ~ N(0, dt)
        noise = torch.tensor(np.random.normal(0, np.sqrt(self.dt), size=(2,1)), dtype=torch.float32).to(self.device)
       
        # Ensure current_state is a column vector
        current_state_col = self.current_state.unsqueeze(1)

        # Euler-Maruyama state transition
        new_state_col = current_state_col + (self.H @ current_state_col + self.M @ action) * self.dt + self.sigma @ noise
        new_state = new_state_col.squeeze()

        # Compute costs
        cost = self.f(action,current_state_col)
        
        # Update current state to new state needed for next iteration
        self.current_state = new_state

        return self.current_state, cost


    def gaussian_quadratic_integral(self):
        """
        This is needed to compute the normalization constant of the Gaussian policy
        in the soft value function / policy log-prob.

        Returns:
            float: Normalization constant (or ∞ if matrix inversion fails).
        """
        
        try:
            epsilon = 1e-8  # Small value to ensure non-singularity
            adjusted_matrix = torch.eye(self.action_dim) / (2 * self.gamma**2) - self.D
            adjusted_matrix += torch.eye(self.action_dim) * epsilon  # Adding to diagonal

            precision_matrix = torch.inverse(adjusted_matrix)
            integral_value = torch.sqrt((2 * np.pi) ** self.action_dim * torch.linalg.det(precision_matrix)).item()
       
            return integral_value
        except torch.linalg.LinAlgError as e:
            print("Matrix inversion failed due to singular matrix:", e)
            return float('inf')

    def fixed_policy_log_prob(self, action, t, state):
        """
        Computes the (unnormalized) log-probability of an action under the current policy
        Useful in Monte Carlo estimation of gradients or log-likelihood ratios.

        Args:
            action (torch.Tensor): Action taken.
            t (float): Current time (not used here).
            state (torch.Tensor): Current state vector.

        Returns:
            float: Log-probability of action under current policy.
        """
        f_atx = self.f(action, state)
        integral_value = self.gaussian_quadratic_integral()
        log_denominator = np.log(integral_value)
        log_prob = f_atx - log_denominator
        return log_prob
    
# Define the Only Linear Value Neural Network
class OnlyLinearValueNN(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(OnlyLinearValueNN, self).__init__()
        self.device = device
        self.hidden_layer_width = 512  # Set hidden layer width

        # Define the hidden layer
        self.hidden_layer = nn.Linear(1, self.hidden_layer_width, device=device)

        # Define the matrix and offset layers following the hidden layer
        self.matrix = nn.Linear(self.hidden_layer_width, 2*2, device=device)
        self.offset = nn.Linear(self.hidden_layer_width, 1, device=device)

        # Non-linear activation function
        self.activation = nn.ReLU()

    def forward(self, t):
        # Pass input through the hidden layer and activation
        x = self.activation(self.hidden_layer(t))

        # Calculate the matrix elements and offset
        matrix_elements = self.matrix(x)
        matrix = matrix_elements.view(-1, 2, 2)
        matrix = torch.bmm(matrix, matrix.transpose(1, 2)) + torch.eye(2).to(matrix.device) * 1e-3

        offset = self.offset(x)
        return matrix, offset

class PolicyNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, d, device="cpu"):
        super(PolicyNeuralNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(1, hidden_size, device=device)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size, device=device)
        
        # Output for phi
        self.phi_output = nn.Linear(hidden_size, d * d).to(device)
        
        # Output for L matrix for Sigma
        self.sigma_output_L = nn.Linear(hidden_size, d * (d + 1) // 2).to(device)
        
        # remember dim
        self.d = d
        
        # precompute
        self.tri_indices = torch.tril_indices(self.d, self.d).to(device)
    
    def forward(self, t):
        # Forward pass
        t = t.view(-1, 1) # Ensure t is a column vector
        hidden = torch.relu(self.hidden_layer1(t))
        hidden = torch.sigmoid(self.hidden_layer2(hidden))
        
        # Compute phi
        phi_flat = self.phi_output(hidden)
        phi = phi_flat.view(-1, self.d, self.d)
        
        # Compute Sigma
        L_flat = self.sigma_output_L(hidden)
        
        # Create a lower triangular matrix L where L_flat fills the lower triangle
        L = torch.zeros(self.d, self.d, device=L_flat.device)
        L[self.tri_indices[0], self.tri_indices[1]] = L_flat
        
        # Compute Sigma = LL^T to ensure positive semi-definiteness
        Sigma = L @ L.T
        return phi, Sigma