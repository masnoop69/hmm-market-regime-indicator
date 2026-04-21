import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist
from scipy.special import digamma
from scipy.optimize import brentq

class gaussianHMM:
    def __init__(self, n_states = 3, max_iter = 200, tol = 1e-6):
        # initial parameters for model fitting
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

    # ==================== Helper Functions =====================

    def _initialize(self, X, pi=None, A=None, mu=None, sigma=None):
        """
        Initialize the parameters of the model for subsequential fitting:
        pi - initial state distribution : (1 x N)
                equal weightage for all states
        A - transition matrix i->j : (N x N)
                equal weightage for all states, but with higher weightage for self-transition. this is due to
                regimes being sticky, and usually continueing to persist.
        mu and sigma - state parameters mu and sigma: (1 x N)
                we use equal quantiles of observations for regime specific mean and sd to initialize state 
                parameters.

        All parameters are optional. If provided, they are validated for shape and constraints.
        If not provided, sensible defaults are computed from the data.
        """

        n_states = self.n_states
        # ====================== Initialize pi and A ======================
        # pi: initial state distribution
        if pi is not None:
            pi = np.asarray(pi).flatten()
            if pi.shape != (n_states,):
                raise ValueError(f"pi must have shape ({n_states},), but got {pi.shape}")
            if not np.isclose(pi.sum(), 1.0):
                raise ValueError(f"pi must sum to 1, but got {pi.sum()}")
            self.pi = pi
        else:
            self.pi = np.ones(n_states) / n_states # assume equal probability

        # A: transition matrix
        if A is not None:
            A = np.asarray(A)
            if A.shape != (n_states, n_states):
                raise ValueError(f"A must have shape ({n_states}, {n_states}), but got {A.shape}")
            if not np.allclose(A.sum(axis=1), 1.0):
                raise ValueError(f"A must have rows that sum to 1, but got {A.sum(axis=1)}")
            self.A = A
        else:
            self.A = np.ones((n_states, n_states))
            np.fill_diagonal(self.A, n_states) # set diagonal to higher weightage
            self.A = self.A / self.A.sum(axis=1, keepdims=True) # normalize rows to sum to 1
        
        # ====================== Initialize mu and sigma ======================
        if mu is not None and sigma is not None:
            mu = np.asarray(mu).flatten()
            if mu.shape != (n_states,):
                raise ValueError(f"mu must have shape ({n_states},), but got {mu.shape}")
            sigma = np.asarray(sigma).flatten()
            if sigma.shape != (n_states,):
                raise ValueError(f"sigma must have shape ({n_states},), but got {sigma.shape}")
            self.mu, self.sigma = mu, sigma

        else: # initialize based on data
            # divide X into n_states equal quantiles: ie n_states = 5, percentiles = [0, 20, 40, 60, 80, 100]
            percentiles = np.percentile(X, np.linspace(0, 100, n_states + 1)) # values at borders of quantiles
            # classify X into each bin, slicing it at 20, 40, 60, 80 for 5 bins, 0-indexed from 0 to 4.
            bins = np.digitize(X, percentiles[1:-1])
            
            # vectorize calculation of components of mu, sigma
            counts = np.bincount(bins, minlength=n_states)
            sums = np.bincount(bins, weights=X, minlength=n_states)
            sums_sq = np.bincount(bins, weights=X**2, minlength=n_states)

            # calculate mu and sigma in (1 x N) vector:
            self.mu = np.where(counts > 0, sums / counts, np.mean(X)) # fallback to global mean if a bin is empty
            var = np.where(counts > 0, (sums_sq / counts) - (self.mu)**2, np.var(X)) # Var(X) = E[X^2] - (E[X])^2
            self.sigma = np.sqrt(np.maximum(var, 1e-12)) + 1e-6 # Sigma = sqrt(Var(X))

        self.X = X # Store X for internal method access
        self.T = len(X)
    
    def _forward(self):
        """
        Forward Algo:
        alpha_t(j) = sum_i(alpha_t-1(i) * a_ij) * b_j(o_t) = P(O_1,O_2,...O_t,q_t=j|lambda)

        Initial Alpha (t = 0):
        alpha_0(j) = pi_j * b_j(o_0)

        Vector operations per time slice:
        alpha_t = (alpha_t-1 @ A) * b(o_t)

        Initial Alpha (t = 0):
        alpha_0 = pi * b(o_0)

        c_t = p(O_t | O[0:t-1], lambda)
        
        total_probability_scale_t = c_t * c_t-1 * ... * c_0
                                  = P(O[0:t] | lambda)
        
        alpha_scaled_t(j) = alpha_t(j) / total_probability_scale_t
        """
        # initialize parameters
        T = self.T
        N = self.n_states
        alpha = np.zeros((T, N)) # alpha matrix across all time, for each state in (T x N) vector

        # emission matrix = b(o_t) for all t, j.
        self.emission = norm.pdf(self.X[:, np.newaxis], self.mu, self.sigma) 
        emission = self.emission
        c = np.zeros(T) # scaling factor

        # 1. initialization step
        alpha[0] = self.pi * emission[0] # alpha_0 = pi * b(o_0)
        c[0] = alpha[0].sum() + 1e-12 # scaling factor
        alpha[0] = alpha[0] / c[0] # scale alpha[0] to ensure alpha row sums to 1

        # 2. Recursion step: overwrite alpha[t] matrix with alpha_t
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * emission[t] # alpha_t = (alpha_t-1 @ A) * b(o_t)
            c[t] = alpha[t].sum() + 1e-12 # scaling factor
            alpha[t] = alpha[t] / c[t] # scale alpha[t]

        self.alpha = alpha
        self.c = c
    
    def _backward(self):
        """
        Backward Algo:
        beta_t(j) = sum_i(beta_t+1(i) * a_ij) * b_j(o_t) = P(O_t+1,O_t+2,...O_T|q_t=j,lambda)

        And Initial Beta (t = T-1):
        beta_T-1(j) = 1

        Vector operations per time slice:
        beta_t = (beta_t+1 * b(o_t+1)) @ A.T

        c_t+1 = p(O_t+1 | O[0:t], lambda)
        
        total_probability_scale_t+1 = c_t+1 * c_t+2 * ... * c_T-1
        
        beta_scaled_t(j) = beta_t(j) / total_probability_scale_t+1
        """
        T = self.T
        N = self.n_states
        beta = np.zeros((T, N)) # beta matrix across all time, for each state in (T x N) vector
        emission = self.emission
        c = self.c

        # 1. initialization step
        beta[T-1] = 1 # beta_T-1 = 1 (no scaling needed; recursion handles it via c[t+1])

        # 2. recursive step
        for t in range(T-2, -1, -1): # iterate backwards from T-2 to 0
            beta[t] = (beta[t+1] * emission[t+1]) @ self.A.T / c[t+1] # beta_t = (beta_t+1 * b(o_t+1)) @ A.T
        
        self.beta = beta

    def _compute_gamma(self):
        """
        gamma_t(j) = P(q_t = j | O, lambda) = (alpha_t(j) * beta_t(j)) / P(O | lambda)

        Since alpha and beta are already scaled, we can directly compute
        gamma_t(j) = alpha_scaled_t(j) * beta_scaled_t(j)
    
        """
        gamma = self.alpha * self.beta # hadamard product
        # Normalize each row to sum to 1 in case of underflow
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        self.gamma = gamma

    def _compute_xi(self):
        """
        xi_t(i, j) = P(q_t = i, q_t+1 = j | O, lambda) = (alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j)) / P(O | lambda)

        Since alpha and beta are already scaled, we can compute:

        xi_t(i, j) = alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j), 
        
        Note that c_t+1 (the scaling factor for the transition day itself) is missing, and the un-normalized sum of xi across t
        will be equal to c_t+1.

        Therefore, to normalize xi_t(i, j), we divide by c_t+1:
        xi_t(i, j) = [ alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j) ] / c_t+1

        Vectorized computation across one single time slide t:
        xi_t = (alpha_t * A * b(o_t+1) * beta_t+1) / c_t+1
        
        """
        alpha_b = self.alpha[:-1, :, np.newaxis]
        beta_b = (self.beta * self.emission)[1:, np.newaxis, :]
        A_b = self.A[np.newaxis, :, :]

        xi = alpha_b * A_b * beta_b
        self.xi = xi / self.c[1:, np.newaxis, np.newaxis] # divide by offset scaling factor to normalize entire row.

    def _baum_welch(self):
        """
        Gamma_t(j) = P(q_t = j | O, lambda)

        Thus, gamma at t = 0 would be the likely initial state probabilities, and we can update pi = gamma[0].

        Xi_t(i, j) = P(q_t = i, q_t+1 = j | O, lambda)

        To update A, we sum xi over all time steps t and divide by the sum of gamma over all time steps t.

        A = sum(xi_t(i, j) for t in 0 to T-2) / sum(gamma_t(i) for t in 0 to T-2)
        
        P(q_t+1 = j | q_t = i, O, lambda) = P(q_t = i, q_t+1 = j | O, lambda) / P(q_t = i | O, lambda)
                                          = xi_t(i, j) / gamma_t(i)
                                          = A*ij

        P(q_t+1 = j | q_t = i, lambda*) = A*ij
        Hence, pi, mean and variance are all calculated using the same logic.
        """
        # ========== E-Step ===========
        # calculate alpha, beta, gamma, xi for updating of pi, A, mu, sigma
        self._forward()
        self._backward()
        self._compute_gamma()
        self._compute_xi()

        # ========== M-Step ===========
        # update initial state probabilities
        self.pi = self.gamma[0]

        # update transition probabilities
        self.A = self.xi.sum(axis=0) / self.gamma[:-1].sum(axis=0)[:, np.newaxis] # xi already has only T-1 time steps as (T-1, N, N) Matrix

        # update mean and variance
        self.mu = np.sum(self.gamma * self.X[:, np.newaxis], axis = 0) / np.sum(self.gamma, axis = 0)
        var = np.sum(self.gamma * (self.X[:, np.newaxis] - self.mu) ** 2, axis = 0) / np.sum(self.gamma, axis = 0)
        self.sigma = np.sqrt(np.maximum(var, 1e-12))
        self.log_likelihood = np.sum(np.log(self.c))
    
    def _sort_states(self, criterion='mu'):
        """
        Sort states by mean (mu), volatility (sigma) or sharpe ratio (mu/sigma) to ensure consistent interpretation.
        Example: sort by sigma so State 0 is always the 'Quiet/Low Vol' state.
        """
        if criterion == 'mu':
            idx = np.argsort(self.mu)
        elif criterion == 'sigma':
            idx = np.argsort(self.sigma)
        elif criterion == 'sharpe':
            idx = np.argsort(self.mu / self.sigma)
        elif criterion == None:
            idx = np.arange(self.n_states)
        else:
            print("Criterion must be 'mu' or 'sigma'. Sorting by 'mu' by default...")
            idx = np.argsort(self.mu)

        # Reorder all state-dependent parameters
        self.mu = self.mu[idx]
        self.sigma = self.sigma[idx]
        self.pi = self.pi[idx]
        self.A = self.A[idx, :][:, idx]
        self.sort_idx = idx

        return self

    def _predict_proba(self, mode=None):
        """
        Returns the posterior probabilities of each state
        (Alpha for causal, or Gamma otherwise).
        """
        self._forward()
    
        if mode == 'infer':
            return self.alpha
    
        self._backward()
        self._compute_gamma()
        return self.gamma
    
    def _predict_posterior(self, mode=None):
        """
        Returns the most likely state at each time step (Posterior Decoding) through argmax of gamma, 
        the highest confidence state at time t.
    
        However, this may result in impossible transitions, like from bear to bull instantly.
        """
        gamma_or_alpha = self._predict_proba(mode)
        return np.argmax(gamma_or_alpha, axis=1)
    
    def _predict_viterbi(self, mode=None):
        """
        Returns the most likely sequence of states (Viterbi Decoding).
        Uses log-probabilities for numerical stability.
        """
        # Ensure emissions and scaling factors are updated for the current self.X
        self._forward()
        
        T = self.T
        N = self.n_states
        
        # log-space avoids underflow (pi, A, emission)
        log_pi = np.log(self.pi + 1e-12)
        log_A = np.log(self.A + 1e-12)
        log_emission = np.log(self.emission + 1e-12)
    
        viterbi = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
    
        # 1. initialization step
        viterbi[0] = log_pi + log_emission[0]

        if mode == "infer":
            path = np.zeros(T, dtype=int)
            path[0] = np.argmax(viterbi[0])
    
        # 2. recursion step
        for t in range(1, T):
            prob = viterbi[t-1][:, np.newaxis] + log_A
            # Max over the previous states (axis=0). Resulting shape: (N,)
            viterbi[t, :] = np.max(prob, axis=0) + log_emission[t, :]
            # Argmax over the previous states (axis=0) to find the best previous given current
            backpointer[t, :] = np.argmax(prob, axis=0)
            # In causal / online mode, choose the most likely state at time t given info up to t
            if mode == "infer":
                path[t] = np.argmax(viterbi[t])
        # 3. path reconstruction
        if mode == "infer":
            return path
        # Default mode: standard offline Viterbi with backtracking (non-causal)
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(viterbi[T-1])
    
        for t in range(T-2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]
    
        return path

    # ==================== Pubic API and Methods =====================

    def fit(self, X, sort="mu", verbose=True, **init_kwargs):
        """
        Fit the HMM to the data X.

        Parameters
        ----------
        X           : array-like, observation sequence.
        sort        : str, criterion for sorting states after fitting (default 'mu').
        verbose     : bool, if False suppresses all print output (default True).
        **init_kwargs : optional pi, A, mu, sigma arrays forwarded to _initialize().
        """
        # ========== Initialization stage ===========
        X = np.asarray(X).flatten()
        self._initialize(X, **init_kwargs)
        if verbose:
            print("="*50)
            print("Initial Parameters:")
            print("pi:", np.round(self.pi, 4))
            print("A:", np.round(self.A, 4))
            print("="*50)
            print("Initial State Specific Parameters:")
            for _ in range(self.n_states):
                print(f"State {_}: mu = {self.mu[_]:.4f}, sigma = {self.sigma[_]:.4f}")
        prev_log_likelihood = - np.inf

        # ========== Iterative Estimation stage ===========
        if verbose:
            print("="*50)
            print("Commencing iterative estimation...")
        for i in range(self.max_iter):
            self._baum_welch()
            if verbose:
                print(f"Iteration {(i + 1)}: Log Likelihood: {self.log_likelihood:.4f}")
            if np.abs(self.log_likelihood - prev_log_likelihood) < self.tol:
                if verbose:
                    print(f"Converged after {(i + 1)} iterations.")
                break
            prev_log_likelihood = self.log_likelihood
        
        if verbose:
            print("="*50)
        # Merge sorting into fit
        self._sort_states(criterion=sort)

        if verbose:
            print("Fitted State Specific Parameters:")
            for _ in range(self.n_states):
                print(f"State {_}: mu = {self.mu[_]:.4f}, sigma = {self.sigma[_]:.4f}")
            print("="*50)
            print("Final Parameters:")
            print("pi:", np.round(self.pi, 4))
            print("A:", np.round(self.A, 4))
            print("="*50)

        return self

    def predict(self, X, type = 'probability', mode = None):
        # Can take in a unseen X to predict the probabilities for, using the fitted parameters.
        X = np.asarray(X).flatten()

        self.X = X # Store X for internal method access
        self.T = len(X)

        if type == 'probability':
            return self._predict_proba(mode)
        elif type == 'posterior':
            return self._predict_posterior(mode)
        elif type =='viterbi':
            return self._predict_viterbi(mode)
        else:
            raise ValueError("Type must be 'probability', 'posterior' or 'viterbi'.")
    
    def predict_step(self, x_t, prev_alpha, A = None, mu = None, sigma = None):
        """
        Single-step causal-only. Since we already have regime specific parameters, we can just call on them to
        calculate state probabilities given observation.
        
        Parameters
        ----------
        x_t : float
            New observation at time t
        alpha_prev : (K,) array
            Scaled alpha from previous time step
        
        Returns
        -------
        alpha_t : (K,) array — updated scaled alpha
        c_t : float — scaling factor (conditional likelihood)
        emission_t : (K,) array — emission probabilities at this step

        """
        if A is not None:
            A = np.asarray(A)
            if A.shape != (self.n_states, self.n_states):
                raise ValueError(f"A must have shape ({self.n_states}, {self.n_states}), but got {A.shape}")
        else:
            A = self.A

        if mu is not None:
            mu = np.asarray(mu)
            if mu.shape != (self.n_states,):
                raise ValueError(f"mu must have shape ({self.n_states}, ), but got {mu.shape}")
        else:
            mu = self.mu

        if sigma is not None:
            sigma = np.asarray(sigma)
            if sigma.shape != (self.n_states,):
                raise ValueError(f"sigma must have shape ({self.n_states}, ), but got {sigma.shape}")
        else:
            sigma = self.sigma

        emission_t = norm.pdf(x_t, mu, sigma)
        alpha_t = prev_alpha @ A * emission_t
        c_t = alpha_t.sum() + 1e-12 # for computation of LLH
        alpha_t = alpha_t / c_t # for computation of next posterior

        return alpha_t, c_t, emission_t

    def get_regime_params(self):
        """
        Export fitted regime parameters for external access
        
        Returns
        -------
        dict with keys:
            'n_states': int
            'pi': (K,) initial state distribution
            'A': (K, K) transition matrix
            'mu': (K,) regime-specific means
            'sigma': (K,) regime-specific standard deviations
            'stationary': (K,) stationary distribution
        """
        
        eigvals, eigvecs = np.linalg.eig(self.A.T) # left eigenvector of A for eigenvalue 1
        idx = np.argmin(np.abs(eigvals - 1.0)) # find eigenvalue closest to 1, bypass floating point errors
        stationary = np.real(eigvecs[:, idx]) # in case of inmaginary solution
        stationary = stationary / stationary.sum() # normalize to 1 for probability
        
        return {
            'n_states': self.n_states,
            'pi': self.pi.copy(),
            'A': self.A.copy(),
            'mu': self.mu.copy(),
            'sigma': self.sigma.copy(),
            'stationary': stationary
        }

class studentHMM:
    def __init__(self, n_states = 3, max_iter = 200, tol = 1e-6):
        # initial parameters for model fitting
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

    # ==================== Helper Functions =====================

    def _initialize(self, X, pi = None, A = None, mu = None, sigma = None, nu = None):
        """
        Initialize the parameters of the model for subsequential fitting:
        pi - initial state distribution : (1 x N)
                equal weightage for all states
        A - transition matrix i->j : (N x N)
                equal weightage for all states, but with higher weightage for self-transition. this is due to
                regimes being sticky, and usually continueing to persist.
        mu and sigma - state parameters mu and sigma: (1 x N)
                we use equal quantiles of observations for regime specific mean and sd to initialize state 
                parameters.
        nu - degrees of freedom for student-t distribution: (1 x N)
                initialize DoF using kurtosis of the data
        """

        n_states = self.n_states
        # ====================== Initialize pi and A ======================
        # pi: initial state distribution
        if pi is not None:
            pi = np.asarray(pi).flatten()
            if pi.shape != (n_states,):
                raise ValueError(f"pi must have shape ({n_states},), but got {pi.shape}")
            if not np.isclose(pi.sum(), 1.0):
                raise ValueError(f"pi must sum to 1, but got {pi.sum()}")
            self.pi = pi
        else:
            self.pi = np.ones(n_states) / n_states # assume equal probability

        # A: transition matrix
        if A is not None:
            A = np.asarray(A)
            if A.shape != (n_states, n_states):
                raise ValueError(f"A must have shape ({n_states}, {n_states}), but got {A.shape}")
            if not np.isclose(A.sum(axis=1), 1.0).all():
                raise ValueError(f"A must have rows that sum to 1, but got {A.sum(axis=1)}")
            self.A = A
        else:
            self.A = np.ones((n_states, n_states))
            np.fill_diagonal(self.A, self.n_states) # set diagonal to higher weightage
            self.A = self.A / self.A.sum(axis=1, keepdims=True) # normalize rows to sum to 1
        
        # ====================== Initialize mu and sigma ======================
        if mu is not None and sigma is not None:
            mu = np.asarray(mu).flatten()
            if mu.shape != (n_states,):
                raise ValueError(f"mu must have shape ({n_states},), but got {mu.shape}")
            sigma = np.asarray(sigma).flatten()
            if sigma.shape != (n_states,):
                raise ValueError(f"sigma must have shape ({n_states},), but got {sigma.shape}")
            self.mu, self.sigma = mu, sigma

        else: # initialize based on data
            # divide X into n_states equal quantiles
            percentiles = np.percentile(X, np.linspace(0, 100, n_states + 1)) # values at borders of quantiles
            # classify X into each bin
            bins = np.digitize(X, percentiles[1:-1])
            
            counts = np.bincount(bins, minlength = n_states) 
            sums = np.bincount(bins, weights = X, minlength = n_states)
            sums_sq = np.bincount(bins, weights = X**2, minlength = n_states)

            # initialize mu and sigma
            self.mu = np.where(counts > 0, sums / counts, np.mean(X)) # fallback to global mean if a bin is empty
            var = np.where(counts > 0, (sums_sq / counts) - (self.mu)**2, np.var(X)) # Var(X) = E[X^2] - (E[X])^2
            self.sigma = np.sqrt(np.maximum(var, 1e-12)) + 1e-6 # Sigma = sqrt(Var(X))

        # initialize nu
        if nu is not None:
            nu = np.asarray(nu).flatten()
            if nu.shape != (n_states,):
                raise ValueError(f"nu must have shape ({n_states},), but got {nu.shape}")
            self.nu = nu
        else:
            from scipy.stats import kurtosis as scipy_kurtosis
            excess_kurt = scipy_kurtosis(X, fisher = True)
            excess_kurt = max(excess_kurt, 0.1)
            nu_init = np.clip(6.0 / excess_kurt + 4.0, 2.1, 30.0) # min DoF is 2.1 in case of cauchy
            self.nu = np.full(n_states, nu_init)

        self.X = X # Store X for internal method access
        self.T = len(X)
   
    def _forward(self):
        """
        Forward pass of the studentHMM.
        """
        # compute emission from t-dist first
        self.emission = np.maximum(
            t_dist.pdf(
                self.X[:, np.newaxis],
                df = self.nu,
                loc = self.mu,
                scale = self.sigma,
            ),
            1e-12,
        )
        
        # initialize alpha and c
        T, n_states = self.T, self.n_states
        alpha = np.zeros((T, n_states))
        c = np.zeros(T)

        alpha[0] = self.pi * self.emission[0]
        c[0] = alpha[0].sum() + 1e-12
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.emission[t]
            c[t] = alpha[t].sum() + 1e-12
            alpha[t] /= c[t]

        self.alpha = alpha
        self.c = c

    def _backward(self):
        T, n_states = self.T, self.n_states
        beta = np.zeros((T, n_states))
        beta[T-1] = 1.0

        for t in range(T-2, -1, -1):
            beta[t] = ((self.emission[t+1] * beta[t+1]) @ self.A.T) / self.c[t+1]

        self.beta = beta
    
    def _compute_gamma(self):
        """
        Same as gaussianHMM, posterior calculation for updating
        """
        gamma = self.alpha * self.beta
        self.gamma = gamma / (gamma.sum(axis = 1, keepdims = True) + 1e-12)

    def _compute_xi(self):
        """
        Same as gaussianHMM, posterior calculation for updating
        """
        alpha_b = self.alpha[:-1, :, np.newaxis] # (T-1, N, 1)
        beta_b = (self.beta * self.emission)[1:, np.newaxis, :] # (T-1, 1, N)
        A_b = self.A[np.newaxis, :, :] # (1, N, N)

        xi = alpha_b * A_b * beta_b
        self.xi = xi / self.c[1:, np.newaxis, np.newaxis]
    
    def _compute_u(self):
        """
        gaussian mixture model weight parameter for EM algo
        """
        delta_sq = np.maximum((self.X[:, np.newaxis] - self.mu)**2 / self.sigma**2, 1e-12) # in case of 0 values
        
        self.E_u = (self.nu + 1) / (self.nu + delta_sq)
        self.log_E_u = digamma((self.nu + 1.0) / 2.0) - np.log((self.nu + delta_sq) / 2.0)
        self.delta_sq = delta_sq
    
    def _update_nu_single(self, nu, u_bar, e_bar, nu_min=2.1, nu_max=100.0):
        def score(nu):
            return np.log(nu / 2.0) - digamma(nu / 2.0) + 1.0 + e_bar - u_bar
        
        # Check if root is bracketed
        g_min = score(nu_min)
        g_max = score(nu_max)
        
        if g_min * g_max > 0:
            # Root not bracketed — return boundary that is closer to zero
            return nu_min if abs(g_min) < abs(g_max) else nu_max
        
        return brentq(score, nu_min, nu_max, xtol=1e-6, maxiter=100)

    def _update_nu(self):
        """
        Update degrees of freedom for all states.
        """
        sum_gamma = self.gamma.sum(axis=0) + 1e-12  # (N,)
        
        # Per-state sufficient statistics (using the computed expected values)
        u_bar = (self.gamma * self.E_u).sum(axis=0) / sum_gamma  # (N,)
        e_bar = (self.gamma * self.log_E_u).sum(axis=0) / sum_gamma  # (N,)
        
        # Using default valid bounds for Student's T distribution
        nu_min = getattr(self, 'nu_min', 2.1)
        nu_max = getattr(self, 'nu_max', 100.0)

        for j in range(self.n_states):
            self.nu[j] = self._update_nu_single(
                self.nu[j], u_bar[j], e_bar[j],
                nu_min=nu_min, nu_max=nu_max
            )

    def _update_state_params(self):
        self.pi = self.gamma[0]
        self.A = (self.xi.sum(axis=0) / (self.gamma[:-1].sum(axis=0)[:, np.newaxis] + 1e-12))
        weighted_gamma = self.gamma * self.E_u # weighted confidence of state in j
        # average confidence in generating observation given weighted confidence that model is in state j
        self.mu = (weighted_gamma * self.X[:, np.newaxis]).sum(axis = 0) / (weighted_gamma.sum(axis = 0) + 1e-12)
        # average confidence in generating variance when in state j given confidence that model is in state j
        sigma2 = (weighted_gamma * (self.X[:, np.newaxis] - self.mu) ** 2).sum(axis = 0) / (self.gamma.sum(axis = 0) + 1e-12)
        self.sigma = np.sqrt(sigma2 + 1e-12)

    def _em_algo(self):
        # e step
        self._forward()
        self._backward()
        self._compute_gamma()
        self._compute_xi()
        self._compute_u()

        # m step
        self._update_state_params()
        self._update_nu()
        # llh computation
        self.llh = np.sum(np.log(self.c + 1e-12))

    def _sort_states(self, criterion='mu'):
        """
        Sort states by mean (mu), volatility (sigma) or sharpe ratio (mu/sigma) to ensure consistent interpretation.
        Example: sort by sigma so State 0 is always the 'Quiet/Low Vol' state.
        """
        if criterion == 'mu':
            idx = np.argsort(self.mu)
        elif criterion == 'sigma':
            idx = np.argsort(self.sigma)
        elif criterion == 'sharpe':
            idx = np.argsort(self.mu / self.sigma)
        elif criterion == None:
            idx = np.arange(self.n_states)
        else:
            print("Criterion must be 'mu' or 'sigma'. Sorting by 'mu' by default...")
            idx = np.argsort(self.mu)

        # Reorder all state-dependent parameters
        self.mu = self.mu[idx]
        self.sigma = self.sigma[idx]
        self.nu = self.nu[idx]
        self.pi = self.pi[idx]
        self.A = self.A[idx, :][:, idx]
        self.sort_idx = idx

        return self

    def _predict_proba(self, mode=None):
        """
        Returns the posterior probabilities of each state
        (Alpha for causal, or Gamma otherwise).
        """
        self._forward()
    
        if mode == 'infer':
            return self.alpha
    
        self._backward()
        self._compute_gamma()
        return self.gamma
    
    def _predict_posterior(self, mode=None):
        """
        Returns the most likely state at each time step (Posterior Decoding) through argmax of gamma, 
        the highest confidence state at time t.
    
        However, this may result in impossible transitions, like from bear to bull instantly.
        """
        gamma_or_alpha = self._predict_proba(mode)
        return np.argmax(gamma_or_alpha, axis=1)
    
    def _predict_viterbi(self, mode=None):
        self._forward()   # needed to compute emission matrix

        T, N = self.T, self.n_states

        log_pi       = np.log(self.pi + 1e-300)
        log_A        = np.log(self.A + 1e-300)
        # Use logpdf directly for numerical stability
        log_emission = t_dist.logpdf(
            self.X[:, np.newaxis],
            df=self.nu, loc=self.mu, scale=self.sigma,
        )

        viterbi     = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)

        viterbi[0] = log_pi + log_emission[0]

        if mode == 'infer':
            path = np.zeros(T, dtype=int)
            path[0] = np.argmax(viterbi[0])

        for t in range(1, T):
            prob = viterbi[t - 1][:, np.newaxis] + log_A
            viterbi[t]     = np.max(prob, axis=0) + log_emission[t]
            backpointer[t] = np.argmax(prob, axis=0)
            if mode == 'infer':
                path[t] = np.argmax(viterbi[t])

        if mode == 'infer':
            return path

        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(viterbi[T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        return path
    # ====================== Public API ===================

    def fit(self, X, sort="mu", verbose=True, **init_kwargs):
        """
        Fit the Student's T HMM to the data X.

        Parameters
        ----------
        X           : array-like, observation sequence.
        sort        : str, criterion for sorting states after fitting (default 'mu').
        verbose     : bool, if False suppresses all print output (default True).
        **init_kwargs : optional pi, A, mu, sigma, nu arrays forwarded to _initialize().
        """
        # ========== Initialization stage ===========
        X = np.asarray(X).flatten()
        self._initialize(X, **init_kwargs)
        if verbose:
            print("="*50)
            print("Initial Parameters:")
            print("pi:", np.round(self.pi, 4))
            print("A:", np.round(self.A, 4))
            print("="*50)
            print("Initial State Specific Parameters:")
            for _ in range(self.n_states):
                print(f"State {_}: mu = {self.mu[_]:.4f}, sigma = {self.sigma[_]:.4f}, nu = {self.nu[_]:.2f}")
        prev_log_likelihood = -np.inf

        # ========== Iterative Estimation stage ===========
        if verbose:
            print("="*50)
            print("Commencing iterative estimation...")
        for i in range(self.max_iter):
            self._em_algo()
            if verbose:
                print(f"Iteration {(i + 1)}: Log Likelihood: {self.llh:.4f}")
            if np.abs(self.llh - prev_log_likelihood) < self.tol:
                if verbose:
                    print(f"Converged after {(i + 1)} iterations.")
                break
            prev_log_likelihood = self.llh
        
        if verbose:
            print("="*50)
        # Merge sorting into fit
        self._sort_states(criterion=sort)

        if verbose:
            print("Fitted State Specific Parameters:")
            for _ in range(self.n_states):
                print(f"State {_}: mu = {self.mu[_]:.4f}, sigma = {self.sigma[_]:.4f}, nu = {self.nu[_]:.2f}")
            print("="*50)
            print("Final Parameters:")
            print("pi:", np.round(self.pi, 4))
            print("A:", np.round(self.A, 4))
            print("="*50)

        return self

    def predict(self, X, type='probability', mode=None):
        """
        Predict hidden states for observation sequence X.

        Parameters
        ----------
        X    : array-like, shape (T,)
        type : 'probability' | 'posterior' | 'viterbi'
        mode : None (offline) | 'infer' (causal/online)

        Returns
        -------
        Gamma matrix (T, N), posterior states (T,), or Viterbi path (T,)
        """
        X = np.asarray(X).flatten()
        self.X = X
        self.T = len(X)

        if type == 'viterbi':
            return self._predict_viterbi(mode=mode)

        self._forward()
        if mode == 'infer':
            if type == 'probability':
                return self.alpha
            return np.argmax(self.alpha, axis=1)

        self._backward()
        self._compute_gamma()

        if type == 'probability':
            return self.gamma
        if type == 'posterior':
            return np.argmax(self.gamma, axis=1)

        raise ValueError("type must be 'probability', 'posterior', or 'viterbi'.")

    def predict_step(self, x_t, prev_alpha):
        """
        Single-step causal forward update for online inference.
        """
        emission_t = np.maximum(
            t_dist.pdf(x_t, df=self.nu, loc=self.mu, scale=self.sigma),
            1e-12,
        )
        alpha_t = prev_alpha @ self.A * emission_t
        c_t = alpha_t.sum() + 1e-12
        alpha_t /= c_t
        return alpha_t, c_t, emission_t

    def get_regime_params(self):
        """
        Export all fitted regime parameters, including effective standard
        deviation (sigma_eff = sigma * sqrt(nu/(nu-2))) and kurtosis.
        """
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.real(eigvecs[:, idx])
        stationary /= stationary.sum()

        # Effective std dev: only defined for nu > 2
        sigma_eff = np.where(
            self.nu > 2,
            self.sigma * np.sqrt(self.nu / (self.nu - 2)),
            np.inf,
        )
        # Excess kurtosis: only defined for nu > 4
        kurtosis = np.where(self.nu > 4, 6.0 / (self.nu - 4), np.inf)

        return {
            'n_states':   self.n_states,
            'pi':         self.pi.copy(),
            'A':          self.A.copy(),
            'mu':         self.mu.copy(),
            'sigma':      self.sigma.copy(),     # scale parameter
            'sigma_eff':  sigma_eff,             # effective std dev
            'nu':         self.nu.copy(),
            'kurtosis':   kurtosis,
            'stationary': stationary,
        }
# =============== Visualization Tools =====================
class hmm_plot:
    def plot_regimes(price, regimes, hmm=None, returns=None, gamma=None, index=None, title=None):
        """
        Plots price (top) with regime shading, optionally state probabilities (middle),
        and optionally returns with regime-colored bars (bottom).

        Parameters
        ----------
        price    : array-like / pd.Series    price series to plot
        regimes  : array-like (int)          state label per time-step
        hmm      : fitted gaussianHMM        if provided, shows mu/sigma in legend
        returns  : array-like / pd.Series    if provided, adds a returns subplot at the bottom
        gamma    : (T, N) array              if provided, adds a state-probability subplot
        index    : x-axis labels (optional)  auto-detected from price.index if omitted
        title    : chart title (optional)
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        regimes_arr = np.asarray(regimes, dtype=int).flatten()
        T = len(regimes_arr)

        # ── Build the price series for the top panel ──
        raw = np.asarray(price).flatten()
        if len(raw) == T + 1:
            raw = raw[1:]
        T = min(T, len(raw))
        regimes_arr = regimes_arr[:T]
        plot_data = pd.Series(raw[:T])

        # ── Detect or assign date index ──
        if index is not None:
            plot_data.index = pd.Index(index)[:T]
        elif hasattr(price, 'index') and isinstance(price.index, pd.DatetimeIndex):
            plot_data.index = price.index[-T:]
        elif returns is not None and hasattr(returns, 'index') and isinstance(returns.index, pd.DatetimeIndex):
            plot_data.index = returns.index[:T]

        # ── Prepare returns series if provided ──
        has_returns = returns is not None
        if has_returns:
            returns_raw = np.asarray(returns).flatten()[:T]
            returns_series = pd.Series(returns_raw, index=plot_data.index)

        # ── Colour palette (red → yellow → green) ──
        unique_states = np.sort(np.unique(regimes_arr))
        n_states = len(unique_states)

        def _color(state_idx):
            pos = np.searchsorted(unique_states, state_idx)
            t = pos / max(n_states - 1, 1)
            r, g = int(255 * (1 - t)), int(255 * t)
            return f'rgb({r},{g},0)'

        def _label(state_idx):
            if hmm is not None:
                if hasattr(hmm, 'nu'):
                    nu = hmm.nu[state_idx]
                    sigma_eff = hmm.sigma[state_idx] * np.sqrt(nu / (nu - 2)) if nu > 2 else float('inf')
                    return f'Regime {state_idx}  (μ={hmm.mu[state_idx]*100:.3f}%, σ_eff={sigma_eff*100:.3f}%, ν={nu:.1f})'
                return f'Regime {state_idx}  (μ={hmm.mu[state_idx]*100:.3f}%, σ={hmm.sigma[state_idx]*100:.3f}%)'
            return f'Regime {state_idx}'

        # ── Determine subplot layout ──
        has_gamma = gamma is not None
        n_rows = 1 + int(has_gamma) + int(has_returns)

        if n_rows == 3:
            row_heights = [0.55, 0.25, 0.20]
            fig_height = 900
        elif n_rows == 2:
            row_heights = [0.65, 0.35]
            fig_height = 800
        else:
            row_heights = [1.0]
            fig_height = 600

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03
        )

        # ── Row assignments ──
        price_row = 1
        gamma_row = 2 if has_gamma else None
        returns_row = n_rows if has_returns else None

        # ── TOP: Price line ──
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=plot_data.values,
            mode='lines',
            name='Price',
            line=dict(color='black', width=0.7),
        ), row=price_row, col=1)

        # ── Regime shading on the price panel ──
        df = pd.DataFrame({'date': plot_data.index, 'regime': regimes_arr[:T]})
        df['group'] = (df['regime'] != df['regime'].shift()).cumsum()
        groups = [grp for _, grp in df.groupby('group')]

        for gi, grp in enumerate(groups):
            s   = int(grp['regime'].iloc[0])
            x0  = grp['date'].iloc[0]
            if gi + 1 < len(groups):
                x1 = groups[gi + 1]['date'].iloc[0]
            else:
                x1 = grp['date'].iloc[-1]

            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=_color(s), opacity=0.3,
                layer='below', line_width=0,
                row=price_row, col=1
            )

        # ── Legend entries ──
        for s in unique_states:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=_color(s), symbol='square'),
                name=_label(s),
            ), row=price_row, col=1)

        fig.update_yaxes(title_text='Price', row=price_row, col=1)

        # ── MIDDLE: State probabilities (stacked area) ──
        if has_gamma:
            gamma_arr = np.asarray(gamma)[:T]
            for s in unique_states:
                fig.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=gamma_arr[:, s],
                    mode='lines',
                    name=f'P(State {s})',
                    line=dict(color=_color(s), width=0.5),
                    fill='tonexty' if s > unique_states[0] else 'tozeroy',
                    stackgroup='gamma',
                    showlegend=False,
                ), row=gamma_row, col=1)
            fig.update_yaxes(title_text='State Prob.', range=[0, 1], row=gamma_row, col=1)

        # ── BOTTOM: Returns colored by regime ──
        if has_returns:
            bar_colors = [_color(regimes_arr[i]) for i in range(T)]
            fig.add_trace(go.Bar(
                x=returns_series.index,
                y=returns_series.values,
                marker_color=bar_colors,
                marker_line_width=0,
                name='Returns',
                showlegend=False,
            ), row=returns_row, col=1)
            fig.update_yaxes(title_text='Return', row=returns_row, col=1)

        # ── x-axis label on the bottom-most row ──
        fig.update_xaxes(title_text='Time', row=n_rows, col=1)

        # ── Layout ──
        fig.update_layout(
            title=dict(
                text=title or 'Market Regimes',
                font=dict(size=14, family='Arial', color='black'),
            ),
            height=fig_height,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                yanchor='top', y=0.99,
                xanchor='left', x=1.01,
                font=dict(family='monospace', size=11)
            ),
        )

        # ── Remove weekend / holiday gaps ──
        if isinstance(plot_data.index, pd.DatetimeIndex):
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        fig.show()

    def plot_regime_distributions(hmm, returns=None, regime_probs=None, threshold=0.8, x_range=None, n_points=500,
                                height=600, width=900, title=None):
        """
        Plot the PDF for each regime and the overall mixture distribution.

        The mixture weights are derived from the stationary distribution of the
        fitted transition matrix (i.e. the long-run fraction of time spent in each
        regime).  If observed returns are provided, empirical histograms for each
        regime are stacked underneath for visual comparison.

        Parameters
        ----------
        hmm : gaussianHMM or studentHMM
            A fitted HMM instance (must have .mu, .sigma, .A attributes).
        returns : array-like or pd.Series, optional
            Observed returns to overlay as a density histogram.
        regime_probs : array-like, optional
            1D array (hard labels) or 2D array (soft probs) to assign returns to regimes.
        threshold : float, default 0.8
            Confidence threshold for soft probabilities.
        x_range : tuple, default None
            Range of x-axis (daily return values) to plot over.
        n_points : int, default 500
            Number of evaluation points for the smooth PDF curves.
        height : int, default 600
        width : int, default 900
        title : str, optional

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from scipy.stats import norm, t as t_dist

        N = hmm.n_states
        is_t = hasattr(hmm, 'nu')

        # ── Stationary distribution from transition matrix (π A = π) ──
        # Solve by finding the left eigenvector with eigenvalue 1.
        eigvals, eigvecs = np.linalg.eig(hmm.A.T)
        idx = np.argmin(np.abs(eigvals - 1.0))            # closest eigenvalue to 1
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()         # normalise

        if is_t:
            sigma_eff = np.where(
                hmm.nu > 2,
                hmm.sigma * np.sqrt(hmm.nu / (hmm.nu - 2)),
                np.inf,
            )
        else:
            sigma_eff = hmm.sigma

        if x_range is None:
            if returns is not None:
                ret_flat = np.asarray(returns).flatten()
                lo, hi = np.percentile(ret_flat, 0.5), np.percentile(ret_flat, 99.5)
                pad = (hi - lo) * 0.1
                x_range = (lo - pad, hi + pad)
            else:
                overall_mu = np.sum(stationary * hmm.mu)
                overall_var = np.sum(stationary * (sigma_eff**2 + hmm.mu**2)) - overall_mu**2
                overall_sigma = np.sqrt(overall_var)
                x_range = (overall_mu - 4 * overall_sigma, overall_mu + 4 * overall_sigma)

        x = np.linspace(x_range[0], x_range[1], n_points)

        # ── Colour palette ──
        def _color(state_idx, alpha=1.0):
            t = state_idx / max(N - 1, 1)
            r, g = int(255 * (1 - t)), int(255 * t)
            return f'rgba({r},{g},0,{alpha})'

        fig = go.Figure()

        # ── Optional: empirical histogram per regime ──
        if returns is not None:
            import pandas as pd
            ret_vals = np.asarray(returns).flatten()
            if hasattr(returns, 'values'):
                ret_vals = returns.values.flatten()
            
            # Detect masking logic
            if regime_probs is not None:
                probs = np.asarray(regime_probs)
                if probs.ndim == 1:
                    hard_labels = probs.astype(int)
                    masks = [hard_labels == s for s in range(N)]
                else:
                    masks = [probs[:, s] >= threshold for s in range(N)]
            else:
                try:
                    hard_labels = hmm.predict(ret_vals, type='viterbi')
                except Exception:
                    hard_labels = np.zeros_like(ret_vals, dtype=int)
                masks = [hard_labels == s for s in range(N)]

            hist_bins = np.linspace(x_range[0], x_range[1], 100)
            centers = (hist_bins[:-1] + hist_bins[1:]) / 2
            bin_width = hist_bins[1] - hist_bins[0]
            
            for s in range(N):
                mask = masks[s]
                m_len = min(len(mask), len(ret_vals))
                obs_s = ret_vals[:m_len][mask[:m_len]]
                if len(obs_s) == 0: continue
                
                counts, _ = np.histogram(obs_s, bins=hist_bins)
                # Normalise to global density matching the mixture curve
                density_contrib = counts / (len(ret_vals) * bin_width)
                
                fig.add_trace(go.Bar(
                    x=centers, y=density_contrib,
                    name=f'Regime {s} Obs',
                    marker_color=_color(s, 0.45),
                    marker_line_width=0,
                    showlegend=True,
                ))
            
            fig.update_layout(barmode='stack')

        # ── Individual regime PDFs ──
        mixture_pdf = np.zeros_like(x)
        for s in range(N):
            if is_t:
                pdf = t_dist.pdf(x, df=hmm.nu[s], loc=hmm.mu[s], scale=hmm.sigma[s])
                label = f'Regime {s}  (μ={hmm.mu[s]*100:.3f}%, σ_eff={sigma_eff[s]*100:.3f}%, ν={hmm.nu[s]:.1f}, π={stationary[s]:.2%})'
            else:
                pdf = norm.pdf(x, hmm.mu[s], hmm.sigma[s])
                label = f'Regime {s}  (μ={hmm.mu[s]*100:.3f}%, σ={hmm.sigma[s]*100:.3f}%, π={stationary[s]:.2%})'

            weighted = stationary[s] * pdf
            mixture_pdf += weighted

            fig.add_trace(go.Scatter(
                x=x, y=pdf,
                mode='lines',
                line=dict(color=_color(s, 0.8), width=2, dash='dash'),
                name=label,
                fill='tozeroy',
                fillcolor=_color(s, 0.12),
            ))

        # ── Mixture (overall market distribution) ──
        fig.add_trace(go.Scatter(
            x=x, y=mixture_pdf,
            mode='lines',
            line=dict(color='white', width=3),
            name='Market Distribution (Mixture)',
        ))

        # ── Fitted overall Normal (dotted blue) ──
        overall_mu = np.sum(stationary * hmm.mu)
        overall_var = np.sum(stationary * (sigma_eff**2 + hmm.mu**2)) - overall_mu**2
        overall_sigma = np.sqrt(overall_var)
        fig.add_trace(go.Scatter(
            x=x, y=norm.pdf(x, overall_mu, overall_sigma),
            mode='lines',
            line=dict(color='rgba(80, 140, 255, 1)', width=2, dash='dot'),
            name=f'Fitted Normal  (μ={overall_mu*100:.3f}%, σ={overall_sigma*100:.3f}%)',
        ))

        title_suffix = "Student-t HMM" if is_t else "Gaussian HMM"
        fig.update_layout(
            title=dict(
                text=title or f'Regime Distributions ({N}-State {title_suffix})',
                font=dict(size=14, family='Arial', color='white'),
            ),
            xaxis_title='Daily Return',
            yaxis_title='Density',
            height=height,
            width=width,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            bargap=0,
            legend=dict(
                yanchor='top', y=0.99,
                xanchor='left', x=1.01,
                font=dict(family='monospace', size=11),
            ),
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                        zerolinecolor='rgba(128,128,128,0.5)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                        zerolinecolor='rgba(128,128,128,0.5)')

        return fig

    def plot_regime_diagnostics(observations, regime_probs, hmm=None, regime_names=None, threshold=0.8, lags=40, title=None, height=800, width=None):
        import numpy as np
        from scipy import stats
        from scipy.stats import t as t_dist
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import acf, pacf
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        obs = np.asarray(observations).flatten()
        probs = np.asarray(regime_probs)
        is_t = hmm is not None and hasattr(hmm, 'nu')

        if probs.ndim == 1:
            hard_labels = probs.astype(int)
            K = len(np.unique(hard_labels)) if hmm is None else hmm.n_states
            is_hard = True
        else:
            K = probs.shape[1]
            is_hard = False

        if regime_names is None:
            regime_names = [f"Regime {s}" for s in range(K)]

        if width is None:
            width = 450 * K

        dist_label = "Student-t" if is_t else "Normal"
        default_title = f"Regime Diagnostics: Q-Q, ACF, & PACF ({dist_label})"
        
        print(f"\\n{'='*70}")
        if is_hard:
            print(f"   REGIME DIAGNOSTICS — {dist_label.upper()} (Hard Classification)")
        else:
            print(f"   REGIME DIAGNOSTICS — {dist_label.upper()} (Confidence >= {threshold})")
        print(f"{'='*70}")

        fig = make_subplots(
            rows=3, cols=K, 
            subplot_titles=regime_names + [""]*2*K,
            vertical_spacing=0.1
        )

        def _color(state_idx, alpha=1.0):
            t = state_idx / max(K - 1, 1)
            r, g = int(255 * (1 - t)), int(255 * t)
            return f'rgba({r},{g},0,{alpha})'

        for s in range(K):
            if is_hard:
                mask = hard_labels == s
            else:
                mask = probs[:, s] >= threshold
            
            m_len = min(len(mask), len(obs))
            obs_s = obs[:m_len][mask[:m_len]]
            n_s = len(obs_s)

            if n_s < 3:
                print(f"\\n[Skipped] {regime_names[s]}: only {n_s} observations (need >= 3).")
                continue

            if hmm is not None:
                mu_s, sigma_s = hmm.mu[s], hmm.sigma[s]
            else:
                mu_s, sigma_s = np.mean(obs_s), np.std(obs_s)

            standardized = (obs_s - mu_s) / sigma_s

            if is_t:
                nu_s = hmm.nu[s]
                (osm, osr), (slope, intercept, _) = stats.probplot(
                    standardized, dist=t_dist, sparams=(nu_s,)
                )
                qq_x_label = f't(ν={nu_s:.1f}) Quantiles'
            else:
                (osm, osr), (slope, intercept, _) = stats.probplot(
                    standardized, dist="norm"
                )
                qq_x_label = 'Normal Quantiles'

            c = _color(s, 0.7)
            
            # ------------- ROW 1: Q-Q Plot ------------- 
            fig.add_trace(go.Scatter(
                x=osm, y=osr,
                mode='markers',
                marker=dict(color=c, size=6),
                name=f'{regime_names[s]} (n={n_s})',
                showlegend=False
            ), row=1, col=s+1)

            line_x = np.array([osm.min(), osm.max()])
            line_y = intercept + slope * line_x
            fig.add_trace(go.Scatter(
                x=line_x, y=line_y,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ), row=1, col=s+1)

            fig.update_xaxes(title_text=qq_x_label, row=1, col=s+1)
            if s == 0:
                fig.update_yaxes(title_text='Sample Quantiles', row=1, col=s+1)

            # ------------- ROW 2 & 3: ACF / PACF ------------- 
            current_lags = min(lags, (n_s // 2) - 1)
            if current_lags > 0:
                acf_vals = acf(standardized, nlags=current_lags, fft=True)[1:]
                pacf_vals = pacf(standardized, nlags=current_lags, method='ols')[1:]
                conf_int = 1.96 / np.sqrt(n_s)
                lag_idx = np.arange(1, len(acf_vals) + 1)
                
                # ACF bar/markers (row 2)
                fig.add_trace(go.Bar(
                    x=lag_idx, y=acf_vals,
                    marker_color=c, marker_line_width=0,
                    width=0.2, showlegend=False
                ), row=2, col=s+1)
                fig.add_trace(go.Scatter(
                    x=lag_idx, y=acf_vals,
                    mode='markers', marker=dict(color=c, size=5),
                    showlegend=False
                ), row=2, col=s+1)
                # Confidence intervals ACF
                fig.add_trace(go.Scatter(
                    x=[lag_idx[0], lag_idx[-1]], y=[conf_int, conf_int],
                    mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
                    showlegend=False
                ), row=2, col=s+1)
                fig.add_trace(go.Scatter(
                    x=[lag_idx[0], lag_idx[-1]], y=[-conf_int, -conf_int],
                    mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
                    showlegend=False
                ), row=2, col=s+1)
                if s == 0:
                    fig.update_yaxes(title_text='ACF', row=2, col=s+1)
                
                # PACF bar/markers (row 3)
                fig.add_trace(go.Bar(
                    x=lag_idx, y=pacf_vals,
                    marker_color=c, marker_line_width=0,
                    width=0.2, showlegend=False
                ), row=3, col=s+1)
                fig.add_trace(go.Scatter(
                    x=lag_idx, y=pacf_vals,
                    mode='markers', marker=dict(color=c, size=5),
                    showlegend=False
                ), row=3, col=s+1)
                # Confidence intervals PACF
                fig.add_trace(go.Scatter(
                    x=[lag_idx[0], lag_idx[-1]], y=[conf_int, conf_int],
                    mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
                    showlegend=False
                ), row=3, col=s+1)
                fig.add_trace(go.Scatter(
                    x=[lag_idx[0], lag_idx[-1]], y=[-conf_int, -conf_int],
                    mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
                    showlegend=False
                ), row=3, col=s+1)

                if s == 0:
                    fig.update_yaxes(title_text='PACF', row=3, col=s+1)
                fig.update_xaxes(title_text='Lags', row=3, col=s+1)

            # ------------- STATS PRINTOUT ------------- 
            skew = stats.skew(standardized)
            kurt = stats.kurtosis(standardized)
            jb_stat, jb_p = stats.jarque_bera(standardized)
            sw_data = standardized[:5000] if n_s > 5000 else standardized
            sw_stat, sw_p = stats.shapiro(sw_data)

            max_lag = max(1, n_s // 3)
            lb_lags = [l for l in [10, 20] if l <= max_lag]
            if lb_lags:
                lb_test = acorr_ljungbox(standardized, lags=lb_lags, return_df=True)

            if is_t:
                nu_s = hmm.nu[s]
                sigma_eff = sigma_s * np.sqrt(nu_s / (nu_s - 2)) if nu_s > 2 else float('inf')
                expected_kurt = 6.0 / (nu_s - 4) if nu_s > 4 else float('inf')
                print(f"\n── {regime_names[s]} (n={n_s}, μ={mu_s:.4f}, σ={sigma_s:.4f}, "
                      f"σ_eff={sigma_eff:.4f}, ν={nu_s:.1f}) ──")
            else:
                print(f"\n── {regime_names[s]} (n={n_s}, μ={mu_s:.4f}, σ={sigma_s:.4f}) ──")

            print(f"   Skewness:       {skew:>10.4f}")
            if is_t:
                print(f"   Excess Kurt:    {kurt:>10.4f}   (expected: {expected_kurt:.4f})")
            else:
                print(f"   Kurtosis:       {kurt:>10.4f}")
            print(f"   Jarque-Bera:    stat={jb_stat:.4f}, p={jb_p:.4e}  "
                  f"{'Normal' if jb_p >= 0.05 else 'Non-Normal'}")
            print(f"   Shapiro-Wilk:   stat={sw_stat:.4f}, p={sw_p:.4e}  "
                  f"{'Normal' if sw_p >= 0.05 else 'Non-Normal'}")

            if lb_lags:
                for lag_i in lb_lags:
                    lb_stat_i = lb_test.loc[lag_i, 'lb_stat']
                    lb_p_i    = lb_test.loc[lag_i, 'lb_pvalue']
                    print(f"   Ljung-Box({lag_i:>2d}): stat={lb_stat_i:.4f}, p={lb_p_i:.4e}  "
                          f"{'White noise' if lb_p_i >= 0.05 else 'Autocorrelated'}")
            else:
                print(f"   Ljung-Box:      [skipped — too few observations]")

            if is_t:
                ks_stat, ks_p = stats.kstest(standardized, lambda x: t_dist.cdf(x, df=nu_s))
                print(f"   KS Test (t):    stat={ks_stat:.4f}, p={ks_p:.4e}  "
                      f"{'Good fit' if ks_p >= 0.05 else 'Poor fit'}")

        fig.update_layout(
            title=dict(
                text=title or default_title,
                font=dict(size=14, family='Arial', color='white'),
            ),
            height=height,
            width=width,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.5)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.5)')
        
        return fig

    def plot_rolling_distribution(
    returns,
    window=252 * 4,
    step=21,
    start=None,
    end=None,
    bins=None,
    x_display_range=(-0.07, 0.07),
    y_range=(0, 80),
    frame_duration=50,
    height=600,
    width=1000,
    title_prefix="Rolling Returns Distribution",
    dark=True,
):
        """
        Animate the rolling return distribution over time with a draggable slider.

        Displays three overlaid layers per frame:
        1. Empirical histogram (bar chart, density-normalised)
        2. Kernel Density Estimate (KDE) smooth curve
        3. Fitted Normal distribution (dashed)

        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Daily return series (if DataFrame, first column is used).
        window : int, default 1008 (≈ 4 trading years)
            Number of observations in each rolling window.
        step : int, default 21 (≈ 1 trading month)
            Step size between successive frames.
        start : str or None
            Start date for subsetting (inclusive). Example: '2001-12-31'.
        end : str or None
            End date for subsetting (inclusive). Example: '2025-12-31'.
        bins : np.ndarray or None
            Custom bin edges for the histogram. Defaults to
            ``np.linspace(-0.12, 0.12, 100)``.
        x_display_range : tuple, default (-0.07, 0.07)
            x-axis display range for the KDE / normal curves.
        y_range : tuple, default (0, 80)
            y-axis display range (density).
        frame_duration : int, default 100
            Milliseconds per frame during playback.
        height : int, default 600
            Figure height in pixels.
        width : int, default 1000
            Figure width in pixels.
        title_prefix : str
            Prefix shown before the window-end date in the title.
        dark : bool, default True
            If True, uses a transparent / dark background with white text.
            If False, uses a light Plotly theme.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The animated figure (call ``fig.show()`` to render).
        """
        import pandas as pd
        import plotly.graph_objects as go
        from scipy.stats import norm, gaussian_kde

        # ── Coerce input ──
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        returns = returns.copy()

        # ── Subset by date ──
        if start or end:
            returns = returns.loc[start:end]

        # ── Histogram bins ──
        if bins is None:
            bins = np.linspace(-0.12, 0.12, 100)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # ── Build per-frame data ──
        frames_data = []
        dates = []

        for i in range(0, len(returns) - window, step):
            window_data = returns.iloc[i : i + window].values.flatten()
            end_date = returns.index[i + window]
            dates.append(end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date))

            mean = np.mean(window_data)
            std  = np.std(window_data)
            kde  = gaussian_kde(window_data)
            counts, _ = np.histogram(window_data, bins=bins, density=True)

            frames_data.append({
                'counts': counts,
                'mean':   mean,
                'std':    std,
                'kde':    kde,
            })

        if not frames_data:
            raise ValueError(
                f"Not enough data to form even one window "
                f"(need {window} observations, got {len(returns)})."
            )

        # ── Smooth x grid for KDE / normal curves ──
        x_range = np.linspace(x_display_range[0], x_display_range[1], 500)

        # ── Initial traces ──
        init = frames_data[0]
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=bin_centers, y=init['counts'],
            name='Empirical Data',
            marker_color='rgba(0, 255, 255, 0.6)',
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=init['kde'](x_range),
            mode='lines',
            line=dict(color='rgba(255, 0, 255, 1)', width=3),
            name='Optimal KDE Fit',
        ))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=norm.pdf(x_range, init['mean'], init['std']),
            mode='lines',
            line=dict(color='rgba(255, 255, 0, 1)', width=2, dash='dash'),
            name='Fitted Normal',
        ))

        # ── Animation frames ──
        frames = []
        for idx, fd in enumerate(frames_data):
            frames.append(go.Frame(
                data=[
                    go.Bar(
                        x=bin_centers, y=fd['counts'],
                        marker_color='rgba(0, 255, 255, 0.6)',
                    ),
                    go.Scatter(
                        x=x_range, y=fd['kde'](x_range),
                        mode='lines',
                        line=dict(color='rgba(255, 0, 255, 1)', width=3),
                    ),
                    go.Scatter(
                        x=x_range,
                        y=norm.pdf(x_range, fd['mean'], fd['std']),
                        mode='lines',
                        line=dict(color='rgba(255, 255, 0, 1)', width=2, dash='dash'),
                    ),
                ],
                name=str(idx),
                layout=go.Layout(
                    title_text=f"{title_prefix} ({window // 252}-Year Window ending: {dates[idx]})"
                ),
            ))
        fig.frames = frames

        # ── Slider steps ──
        slider_steps = []
        for idx, date in enumerate(dates):
            slider_steps.append(dict(
                method='animate',
                args=[
                    [str(idx)],                              # frame name to jump to
                    dict(
                        mode='immediate',
                        frame=dict(duration=0, redraw=True),  # instant jump when scrubbing
                        transition=dict(duration=0),
                    ),
                ],
                label=date,
            ))

        sliders = [dict(
            active=0,
            currentvalue=dict(
                prefix='Window ending: ',
                font=dict(size=13),
            ),
            pad=dict(t=50),
            steps=slider_steps,
        )]

        # ── Play / Pause buttons ──
        updatemenus = [dict(
            type='buttons',
            showactive=False,
            y=-0.12,
            x=0.08,
            xanchor='right',
            yanchor='top',
            buttons=[
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                        ),
                    ],
                ),
                dict(
                    label='⏸ Pause',
                    method='animate',
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                        ),
                    ],
                ),
            ],
        )]

        # ── Layout ──
        bg  = 'rgba(0,0,0,0)' if dark else 'white'
        fg  = 'white'          if dark else 'black'
        grid_color = 'rgba(128,128,128,0.2)' if dark else 'rgba(200,200,200,0.5)'
        zero_color = 'rgba(128,128,128,0.5)' if dark else 'rgba(150,150,150,0.7)'

        fig.update_layout(
            title=f"{title_prefix} ({window // 252}-Year Window ending: {dates[0]})",
            xaxis_title='Daily Return',
            yaxis_title='Density',
            height=height,
            width=width,
            plot_bgcolor=bg,
            paper_bgcolor=bg,
            font=dict(color=fg),
            bargap=0,
            sliders=sliders,
            updatemenus=updatemenus,
        )

        fig.update_xaxes(
            showgrid=True, gridcolor=grid_color,
            zerolinecolor=zero_color,
            range=list(x_display_range),
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=grid_color,
            zerolinecolor=zero_color,
            range=list(y_range),
        )

        return fig
