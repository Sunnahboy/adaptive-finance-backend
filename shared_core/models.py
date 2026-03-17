
import numpy as np
import os
from typing import List, Union, Tuple, Dict, Optional


class LinUCB:
    """
    LinUCB with Sherman-Morrison Optimization.
    
    Features:
    - O(d^2) rank-1 updates (Sherman-Morrison formula)
    - O(d) diagonal approximation option
    - Drift handling via decay factor
    - Configurable regularization (lambda)
    - Native persistence (Save/Load)
    
    
    """
    
    def __init__(self, 
                 n_arms: int, 
                 n_features: int, 
                 alpha: float = 0.5, 
                 allow_diagonal: bool = False, 
                 decay_factor: float = 0.98, 
                 decay_strategy: str = 'interpolation',
                 epsilon: float = 1e-10, 
                 min_exploration: float = 0.01,
                 lambda_reg: float = 1.0):
        """
        Initialize LinUCB bandit.
        
        Args:
            n_arms: Number of actions/arms
            n_features: Feature dimension
            alpha: Exploration-exploitation trade-off parameter (typically 0.1-1.0)
            allow_diagonal: Use diagonal approximation (faster but less accurate)
            decay_factor: Non-stationarity handling (1.0 = no decay)
            decay_strategy: 'interpolation' or 'scale' for applying decay
            epsilon: Numerical stability threshold
            min_exploration: Minimum uncertainty to maintain exploration
            lambda_reg: Regularization constant (A_0 = lambda_reg * I)
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.allow_diagonal = allow_diagonal
        self.decay_factor = decay_factor
        self.decay_strategy = decay_strategy
        self.epsilon = epsilon
        self.min_exploration = min_exploration
        self.lambda_reg = lambda_reg
        
        # Initialize A (covariance matrix) and b (reward vector) for each arm
        # Start with A = lambda_reg * I, so A_inv = 1/lambda_reg * I
        if self.allow_diagonal:
            self.A_inv = [np.ones(n_features) / lambda_reg for _ in range(n_arms)]
        else:
            self.A_inv = [np.identity(n_features) / lambda_reg for _ in range(n_arms)]
        
        # Reward vectors (always 1D)
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        
        # Statistics for debugging/logging
        self.update_count = [0] * n_arms

    def select_arm(self, context: Union[List[float], np.ndarray]) -> Tuple[int, Dict]:
        """
        Select an arm using the UCB principle.
        
        For each arm: UCB = μ_a + α * σ_a
        where μ_a is the estimated reward and σ_a is the confidence bound.
        
        Args:
            context: Feature vector of shape (n_features,) or (n_features, 1)
            
        Returns:
            (selected_arm_index, debug_info_dict)
        """
        context = np.array(context)
        flat_context = context.flatten()
        
        col_context = context.reshape(-1, 1) if not self.allow_diagonal else None
        
        ucb_scores = np.zeros(self.n_arms)
        debug_info = {}
        
        for arm in range(self.n_arms):
            try:
                if self.allow_diagonal:
                    mean, uncertainty = self._compute_stats_diagonal(arm, flat_context)
                else:
                    mean, uncertainty = self._compute_stats_full(arm, flat_context, col_context)
                
                # Enforce minimum exploration
                uncertainty = max(uncertainty, self.min_exploration)
                ucb_scores[arm] = mean + uncertainty
                
                debug_info[arm] = {
                    "mean": float(mean),
                    "uncertainty": float(uncertainty),
                    "ucb_score": float(ucb_scores[arm]),
                    "updates": self.update_count[arm]
                }
            except Exception as e:
                # Numerical issue: fallback to random exploration
                ucb_scores[arm] = -999.0 
        
        # Select best arm (with tie-breaking)
        #  epsilon for floating-point comparison
        best_score = np.max(ucb_scores)
        candidates = np.where(np.abs(ucb_scores - best_score) < self.epsilon)[0]
        selected_arm = int(np.random.choice(candidates))
        
        return selected_arm, debug_info

    def update(self, arm: int, context: Union[List[float], np.ndarray], reward: float):
        """
        Update arm's A_inv and b with new observation (context, reward).
        
        Args:
            arm: Arm index
            context: Feature vector
            reward: Observed reward (typically 0.0 or 1.0 for classification)
        """
        context = np.array(context)
        
        # 1. Apply decay if non-stationary
        if self.decay_factor < 1.0:
            self._apply_decay(arm)
        
        # 2. Update using Sherman-Morrison formula
        if self.allow_diagonal:
            flat_context = context.flatten()
            self._update_diagonal(arm, flat_context, reward)
        else:
            col_context = context.reshape(-1, 1)
            flat_context = context.flatten()
            self._update_sherman_morrison(arm, col_context, flat_context, reward)
        
        self.update_count[arm] += 1

    # ========== PROTECTED METHODS (Internal Math) ==========

    def _compute_stats_diagonal(self, arm: int, flat_context: np.ndarray) -> Tuple[float, float]:
        """
        Compute mean and uncertainty for diagonal approximation (O(d)).
        
        For diagonal A_inv:
        - μ = θ^T x where θ = A_inv ⊙ b (element-wise product)
        - σ² = x^T (A_inv ⊙ I) x = Σ(x_i^2 * A_inv_ii)
        """
        theta = self.A_inv[arm] * self.b[arm]
        mean = np.dot(theta, flat_context)
        
        variance = np.dot(flat_context**2, self.A_inv[arm])
        uncertainty = self.alpha * np.sqrt(max(variance, 0.0))
        return mean, uncertainty

    def _compute_stats_full(self, arm: int, flat_context: np.ndarray, 
                           col_context: np.ndarray) -> Tuple[float, float]:
        """
        Compute mean and uncertainty for full matrix (O(d²)).
        
        For full A_inv:
        - μ = θ^T x where θ = A_inv @ b
        - σ² = x^T @ A_inv @ x
        """
        theta = self.A_inv[arm].dot(self.b[arm])
        mean = np.dot(theta, flat_context)
        
        variance = float(col_context.T.dot(self.A_inv[arm]).dot(col_context)[0, 0])
        uncertainty = self.alpha * np.sqrt(max(variance, 0.0))
        return mean, uncertainty

    def _update_sherman_morrison(self, arm: int, col_context: np.ndarray, 
                                flat_context: np.ndarray, reward: float):
        """
        Update A_inv using Sherman-Morrison rank-1 formula (O(d²)).
        
        Sherman-Morrison: 
        A_new^(-1) = A_old^(-1) - (A_old^(-1) x (A_old^(-1) x)^T) / (1 + x^T A_old^(-1) x)
        """
        inv = self.A_inv[arm]
        
        # Precompute A^(-1) * x
        inv_x = inv.dot(col_context)
        
        # Denominator: 1 + x^T A^(-1) x
        denom = float(1.0 + col_context.T.dot(inv_x)[0, 0])
        
        # Numerical stability check
        if np.abs(denom) < self.epsilon:
            self.b[arm] += reward * flat_context
            return 
        
        # Update A_inv
        numerator = inv_x.dot(inv_x.T)
        self.A_inv[arm] = inv - (numerator / denom)
        
        # Update b
        self.b[arm] += reward * flat_context

    def _update_diagonal(self, arm: int, flat_context: np.ndarray, reward: float):
        """
        Update A_inv using diagonal Sherman-Morrison formula (O(d)).
        
        diagonal application:
        A_inv_i = A_inv_i - (A_inv_i^2 * x_i^2) / (1 + Σ_j(A_inv_jj * x_j^2))
        """
        current_inv = self.A_inv[arm]
        
        # Denominator: 1 + Σ_j(A_inv_jj * x_j^2)
        denom = 1.0 + np.dot(current_inv * flat_context**2, np.ones_like(flat_context))
        
        # Numerical stability
        if np.abs(denom) < self.epsilon:
            self.b[arm] += reward * flat_context
            return
        
        # Subtract rank-1 term from each diagonal element
        # Include A_inv_i^2 term 
        self.A_inv[arm] = current_inv - (current_inv**2 * flat_context**2) / denom
        
        # Update b
        self.b[arm] += reward * flat_context

    def _apply_decay(self, arm: int):
        """
        Apply decay/forgetting factor for non-stationary environments.
        
        Two strategies:
        1. 'interpolation': A_inv_new = γ*A_inv_old + (1-γ)*I  (more conservative)
        2. 'scale': A_inv_new = A_inv_old/γ, b_new = γ*b_old    (exponential decay)
        """
        if self.decay_strategy == 'interpolation':
            if self.allow_diagonal:
                identity = np.ones(self.n_features)
            else:
                identity = np.eye(self.n_features)
            self.A_inv[arm] = (self.decay_factor * self.A_inv[arm]) + ((1 - self.decay_factor) * identity)
        else:  # 'scale'
            self.A_inv[arm] /= self.decay_factor
            self.b[arm] *= self.decay_factor

    # ========== PERSISTENCE (Save/Load) ==========

    def save_weights(self, filepath: str):
        """
        Save model state to compressed .npz file.
        
        Args:
            filepath: Output file path (e.g., 'models/bandit.npz')
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        np.savez_compressed(
            filepath,
            n_arms=self.n_arms,
            n_features=self.n_features,
            alpha=self.alpha,
            allow_diagonal=self.allow_diagonal,
            decay_factor=self.decay_factor,
            decay_strategy=self.decay_strategy,
            lambda_reg=self.lambda_reg,
            # Save lists as object arrays to handle ragged shapes
            A_inv=np.array(self.A_inv, dtype=object), 
            b=np.array(self.b, dtype=object),
            update_count=np.array(self.update_count)
        )

    def load_weights(self, filepath: str):
        """
        Load model state from .npz file.
        
        Args:
            filepath: Input file path
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If architecture doesn't match
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        data = np.load(filepath, allow_pickle=True)
        
        # Verify architecture compatibility
        if data['n_features'] != self.n_features or data['n_arms'] != self.n_arms:
            raise ValueError(
                f"Architecture mismatch! "
                f"Saved: {data['n_arms']} arms, {data['n_features']} features. "
                f"Current: {self.n_arms} arms, {self.n_features} features"
            )

        # Restore all parameters
        self.alpha = float(data['alpha'])
        self.allow_diagonal = bool(data['allow_diagonal'])
        self.decay_factor = float(data['decay_factor'])
        self.decay_strategy = str(data['decay_strategy'])
        self.lambda_reg = float(data.get('lambda_reg', 1.0))  # Backward compatible
        
        # Convert object arrays back to lists
        self.A_inv = list(data['A_inv'])
        self.b = list(data['b'])
        self.update_count = list(data['update_count'])