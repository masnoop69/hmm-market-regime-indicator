import numpy as np
from scipy.stats import norm

class gaussianHMM:
    def __init__(self, n_states = 5, max_iter = 100, tol = 1e-6):
        """
        Gaussian Hidden Markov Model: due to the continuous distribution of returns,
        we assume that the returns in each state follow a Gaussian distribution, where each
        Gaussian distribution, when layered on top of each other all at once, will fit the
        non-normality and non-stationarity of the market returns.

        Because each state (regime) can define its own mean, variance, therefore, the resulting
        distribution may be non-normal, and therefore be a good indication of overall market
        returns.
        """
        # initial parameters for model fitting
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

# ==================== Helper Functions =====================

    def _initialize(self, X):
        """
        Initialize the parameters of the model for subsequential fitting:
        pi - initial state distribution : 
                equal weightage for all states
        A - transition matrix i->j : 
                equal weightage for all states, but with higher weightage for self-transition. this is due to
                regimes being sticky, and usually continueing to persist.
        mu and sigma - state parameters mu and sigma:
                we use equal quantiles of observations for regime specific mean and sd to initialize state 
                parameters.

        """

        # ====================== Initialize pi and A ======================
        # pi: initial state distribution
        self.pi = np.ones(self.n_states) / self.n_states # assume equal probability

        # A: transition matrix
        self.A = np.ones((self.n_states, self.n_states))
        np.fill_diagonal(self.A, self.n_states) # set diagonal to higher weightage
        self.A = self.A / self.A.sum(axis=1, keepdims=True) # normalize rows to sum to 1
        
        # ====================== Initialize mu and sigma ======================
        # mu: state specific mean
        # sigma: state specific standard deviation
        self.mu = np.zeros(self.n_states)
        self.sigma = np.zeros(self.n_states)

        # divide X into n_states equal quantiles: ie n_states = 5, percentiles = [0, 20, 40, 60, 80, 100]
        percentiles = np.percentile(X, np.linspace(0, 100, self.n_states + 1)) # values at borders of quantiles
        # classify X into each bin, slicing it at 20, 40, 60, 80 for 5 bins, 0-indexed from 0 to 4.
        bins = np.digitize(X, percentiles[1:-1])
        
        # vectorize calculation of components of mu, sigma
        counts = np.bincount(bins, minlength=self.n_states) # [n1, n2,...n_states]
        sums = np.bincount(bins, weights=X, minlength=self.n_states) # [sum1, sum2,...sum_n_states]
        sums_sq = np.bincount(bins, weights=X**2, minlength=self.n_states) # [sum_sq1, sum_sq2,...sum_sq_n_states]

        # calculate mu and sigma in (1 x N) vector:
        self.mu = np.where(counts > 0, sums / counts, np.mean(X)) # fallback to global mean if a bin is empty
        var = np.where(counts > 0, (sums_sq / counts) - (self.mu)**2, np.var(X)) # Var(X) = E[X^2] - (E[X])^2
        self.sigma = np.sqrt(np.maximum(var, 1e-12)) + 1e-6 # Sigma = sqrt(Var(X))

        self.X = X # Store X for internal method access
        self.T = len(X)
    
    def _forward(self):
        """
        Forward algorithm to compute the probability of observing the sequence X given the model parameters.
        First part of the Forward-Backward Algo in Baum-Welch (EM) Algo.

        ALpha at t is defined as probability of observing sequence O = O_1,O_2,...O_t and hidden state q_t = j
        at time t.

        Forward Algo:
        alpha_t(j) = sum_i(alpha_t-1(i) * a_ij) * b_j(o_t) = P(O_1,O_2,...O_t,q_t=j|lambda)
            where:
                j is hidden state at t
                i is hidden state at t-1
                a_ij is transition probability from i to j
                b_j(o_t) is the probability of observing o_t given hidden state j

        And Initial Alpha (t = 0):
        alpha_0(j) = pi_j * b_j(o_0)

        We use a vectorization approach to compute this.

        Vector operations per time slice:
        alpha_t = (alpha_t-1 @ A) * b(o_t)

        Initial Alpha (t = 0):
        alpha_0 = pi * b(o_0)

        Here, b_j(o_t) is computed using Gaussian PDF as returns are continuous, hence probability of observing
        return X_t given state j is given by Normal PDF with mean mu_j and variance sigma_j^2, the state specific
        parameters.

        According to Bishop's Pattern Rocognition and Machine Learning, we must scale the Forward Alpha and the 
        Backwards Beta by a scaling factor c_t = 1 / alpha_t.sum() at each step for a filtered probability. This
        is because the alpha chain multiplies very small probabilities repeatedly, and alpha can quickly tend to
        zero, causing underflow.
        
        The scaling factors cancel out in the EM algo.

        To normalize:

        c_t = p(O_t | O[0:t-1], lambda)
        
        total_probability_scale_t = c_t * c_t-1 * ... * c_0
                                  = P(O[0:t] | lambda)
        
        alpha_scaled_t(j) = alpha_t(j) / total_probability_scale_t

        To compute this, we must scale alpha_t by c_t at each step.
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
        Backward algorithm to compute the probability of observing the sequence X given the model parameters.
        Second part of the Forward-Backward Algo in Baum-Welch (EM) Algo.

        Beta at t is defined as probability of observing sequence O = O_t+1,O_t+2,...O_T given hidden state q_t = j
        at time t.

        Backward Algo:
        beta_t(j) = sum_i(beta_t+1(i) * a_ij) * b_j(o_t) = P(O_t+1,O_t+2,...O_T|q_t=j,lambda)
            where:
                j is hidden state at t
                i is hidden state at t+1
                a_ij is transition probability from i to j
                b_j(o_t) is the probability of observing o_t given hidden state j

        And Initial Beta (t = T-1):
        beta_T-1(j) = 1

        We use a vectorization approach to compute this.

        Vector operations per time slice:
        beta_t = (beta_t+1 * b(o_t+1)) @ A.T

        Initial Beta (t = T-1):
        beta_T-1 = 1

        c_t+1 = p(O_t+1 | O[0:t], lambda)
        
        total_probability_scale_t+1 = c_t+1 * c_t+2 * ... * c_T-1
        
        beta_scaled_t(j) = beta_t(j) / total_probability_scale_t+1

        To compute this, we must scale beta_t by c_t+1 at each step backwards.
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
        Compute gamma, the probability of being in state j at time t given the sequence X.
        This means that it the confidence of the model that the hidden state at time t is j.
        gamma_t(j) = P(q_t = j | O, lambda) = (alpha_t(j) * beta_t(j)) / P(O | lambda)

        Since alpha and beta are already scaled, we can directly compute
        gamma_t(j) = alpha_scaled_t(j) * beta_scaled_t(j), 
        
        where alpha_t(j) has already been pre scaled by 1 / P(O[0:t] | lambda)
        and beta_t(j) has already been pre scaled by 1 / P(O[t+1:T-1] | O[0:t], lambda).

        Hence, multiplying the scaling factors:
        P(O[0:t] | lambda) * P(O[t+1:T-1] | O[0:t], lambda) = P(O[0:T-1] | lambda) = P(O | lambda),

        which is the denominator of the original (unscaled) Gamma_t.

        Therefore, vectorizing this approach by normalizing:
        gamma_t = alpha_t * beta_t / sum(alpha_t * beta_t)
        """
        gamma = self.alpha * self.beta # hadamard product
        # Normalize each row to sum to 1 in case of underflow
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        self.gamma = gamma

    def _compute_xi(self):
        """
        Compute xi, the probability of transitioning from state i to state j at time t given the sequence X.
        xi_t(i, j) = P(q_t = i, q_t+1 = j | O, lambda) = (alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j)) / P(O | lambda)

        Since alpha and beta are already scaled, we can compute:

        xi_t(i, j) = alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j), 
        
        where alpha_t(i) has already been pre scaled by 1 / P(O[0:t] | lambda) (historical data)
        and beta_t+1(j) has already been pre scaled by 1 / P(O[t+2:T-1] | O[0:t], lambda) (future data)
        
        Note that c_t+1 (the scaling factor for the transition day itself) is missing, and the un-normalized sum of xi across t
        will be equal to c_t+1.

        Therefore, to normalize xi_t(i, j), we divide by c_t+1:
        xi_t(i, j) = [ alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j) ] / c_t+1

        Vectorized computation across one single time slide t:
        xi_t = (alpha_t * A * b(o_t+1) * beta_t+1) / c_t+1

        However, in order to compute all xi_t(i, j) for all i, j at time t, we need to compute this in 3 dimensions simultaneously.
        We do this through extending alpha, [beta * b(O)] and A through their missing dimensions. We hence define these
        dimensions:
        1. alpha_dimension: a_d
        2. beta_dimension: b_d
        3. Time_dimension: t_d

        By multiplying each dimension together like a cube, we can compute all xi_t(i, j) for all i, j at time t simultaneously.

        We notice that there are missing dimensions from each of our matrices of alpha, beta and A in the format (t_d, a_d, b_d):
        1. alpha is (t_d, a_d) and is missing b_d. Alpha runs from t = 0 to T-2. 
        2. beta * b(O) is (t_d, b_d) and is missing a_d. Beta runs from t = 1 to T-1 
           (as we look at the next beta * b(O) at t+1 when we are computing xi at t).
        3. A is (a_d, b_d) and is missing t_d.

        Therefore, we need to extend each matrix in the direction of their missing dimensions by copying it over that dimension similar
        to making copies of a sheet of paper and stacking them on each other.
        1. alpha transforms from (t_d, a_d) -> (t_d, a_d, 1)
        2. beta * b(O) transforms from (t_d, b_d) -> (t_d, 1, b_d)
        3. A transforms from (a_d, b_d) -> (1, a_d, b_d)

        Then, we multiply the three broadcasted matrices together to get the final xi matrix.
        
        """
        alpha_b = self.alpha[:-1, :, np.newaxis]
        beta_b = (self.beta * self.emission)[1:, np.newaxis, :]
        A_b = self.A[np.newaxis, :, :]

        xi = alpha_b * A_b * beta_b
        self.xi = xi / self.c[1:, np.newaxis, np.newaxis] # divide by offset scaling factor to normalize entire row.

    def _baum_welch(self):
        """
        Baum-Welch algorithm to re-estimate the parameters of the HMM, using the concept of MLE and Bayesian
        statistics. Remember that Bayesian statistics is about updating our beliefs about a Hypothesis being true 
        given the Evidence. 
 
        Therefore:
        P(Hypothesis | Evidence) = [P(Evidence | Hypothesis) * P(Hypothesis)] / P(Evidence)

        In this case, the hypothesis is pi, A, mu and sigma, and the Evidence is the observation sequence O.

        Gamma is defined as confidence the model has of being in hidden state j  at time t given the 
        observation sequence O.

        Gamma_t(j) = P(q_t = j | O, lambda)

        Thus, gamma at t = 0 would be the likely initial state probabilities, and we can update pi = gamma[0].

        Xi is defined as the probability of transitioning from state i at time t to state j at time t+1 given the 
        observation sequence O.

        Xi_t(i, j) = P(q_t = i, q_t+1 = j | O, lambda)

        To update A, we sum xi over all time steps t and divide by the sum of gamma over all time steps t.

        A = sum(xi_t(i, j) for t in 0 to T-2) / sum(gamma_t(i) for t in 0 to T-2)

        This essentially is summing up the number of times the i to j jump happened over the observation 
        sequence divided by the total times we were in state i over the observation sequence, the number of 
        opportunities the model could have transitioned from state i to state j.

        Hence, A would thus be the expected rate of transition from state i to state j given state in i at any
        given time, given lambda (pi, A, state parameters[mu, sigma]).
        
        P(q_t+1 = j | q_t = i, O, lambda) = P(q_t = i, q_t+1 = j | O, lambda) / P(q_t = i | O, lambda)
                                          = xi_t(i, j) / gamma_t(i)
                                          = A*ij

        Where A*ij is the new transition probability from state i to state j given the observation sequence O,
        updated from the previous A_ij to maximize this.
        
        Summing them up over all t (and hence O) gives us the expected transition rate from state i to state j 
        given lambda.

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
    
        # viterbi[t, j]: max log-prob of state j at time t
        # backpointer[t, j]: state at t-1 that maximizes log-prob of state j at time t
        viterbi = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
    
        # 1. initialization step
        viterbi[0] = log_pi + log_emission[0]
    
        # If doing causal inference, we must decide the state at each time t
        # without looking ahead or backtracking from the future.
        if mode == "infer":
            path = np.zeros(T, dtype=int)
            path[0] = np.argmax(viterbi[0])
    
        # 2. recursion step
        for t in range(1, T):
            # prob shape: (N, N) where prob[i, j] is the log-prob of transitioning
            # from state i at t-1 to state j at t
            # viterbi[t-1][:, np.newaxis] broadcasts the previous max log-probs
            # to column shape (N, 1)
            # log_A is shape (N, N)
            prob = viterbi[t-1][:, np.newaxis] + log_A
            
            # Max over the previous states (axis=0). Resulting shape: (N,)
            viterbi[t, :] = np.max(prob, axis=0) + log_emission[t, :]
    
            # Argmax over the previous states (axis=0) to find the best previous
            # state for each current state.
            backpointer[t, :] = np.argmax(prob, axis=0)
    
            # In causal / online mode, choose the most likely state at time t
            # immediately using only information up to time t.
            if mode == "infer":
                path[t] = np.argmax(viterbi[t])
    
        # 3. path reconstruction
        if mode == "infer":
            # Causal mode: no backtracking, because backtracking uses future information.
            return path
    
        # Default mode: standard offline Viterbi with backtracking (non-causal)
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(viterbi[T-1])
    
        for t in range(T-2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]
    
        return path

# ==================== Pubic API and Methods =====================

    def fit(self, X, sort = "mu"):
        """
        Fit the HMM to the data X.
        """
        # ========== Initialization stage ===========
        X = np.asarray(X).flatten()
        self._initialize(X)
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
        print("="*50)
        print("Commencing iterative estimation...")
        for i in range(self.max_iter):
            self._baum_welch()
            print(f"Iteration {(i + 1)}: Log Likelihood: {self.log_likelihood:.4f}")
            if np.abs(self.log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged after {(i + 1)} iterations.")
                break
            prev_log_likelihood = self.log_likelihood
        
        print("="*50)
        # Merge sorting into fit
        self._sort_states(criterion=sort)

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
        """
        Predict the probabilities of each state (Gamma), the most likely state at each time step (Posterior Decoding), 
        or the most likely sequence of states (Viterbi Decoding).

        Where 'viterbi' and 'posterior' returns single output per time slice, 'probability' returns array of N
        for confidence of each state for each time slice.
        """
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


# =============== Visualization Tools =====================
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


def plot_regime_distributions(hmm, returns=None, x_range=(-0.08, 0.08), n_points=500,
                               height=600, width=900, title=None):
    """
    Plot the Gaussian PDF for each regime and the overall mixture distribution.

    The mixture weights are derived from the stationary distribution of the
    fitted transition matrix (i.e. the long-run fraction of time spent in each
    regime).  If observed returns are provided, an empirical histogram is shown
    underneath for visual comparison.

    Parameters
    ----------
    hmm : gaussianHMM
        A fitted gaussianHMM instance (must have .mu, .sigma, .A attributes).
    returns : array-like or pd.Series, optional
        Observed returns to overlay as a density histogram.
    x_range : tuple, default (-0.08, 0.08)
        Range of x-axis (daily return values) to plot over.
    n_points : int, default 500
        Number of evaluation points for the smooth PDF curves.
    height : int, default 600
        Figure height in pixels.
    width : int, default 900
        Figure width in pixels.
    title : str, optional
        Custom chart title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from scipy.stats import norm

    N = hmm.n_states
    x = np.linspace(x_range[0], x_range[1], n_points)

    # ── Stationary distribution from transition matrix (π A = π) ──
    # Solve by finding the left eigenvector with eigenvalue 1.
    eigvals, eigvecs = np.linalg.eig(hmm.A.T)
    idx = np.argmin(np.abs(eigvals - 1.0))            # closest eigenvalue to 1
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / stationary.sum()         # normalise

    # ── Colour palette (red → yellow → green, matching plot_regimes) ──
    def _color(state_idx, alpha=1.0):
        t = state_idx / max(N - 1, 1)
        r, g = int(255 * (1 - t)), int(255 * t)
        return f'rgba({r},{g},0,{alpha})'

    fig = go.Figure()

    # ── Optional: empirical histogram ──
    if returns is not None:
        import pandas as pd
        ret_vals = np.asarray(returns).flatten()
        if hasattr(returns, 'values'):
            ret_vals = returns.values.flatten()
        hist_bins = np.linspace(x_range[0], x_range[1], 100)
        counts, edges = np.histogram(ret_vals, bins=hist_bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=centers, y=counts,
            name='Observed Returns',
            marker_color='rgba(200, 200, 200, 0.35)',
            showlegend=True,
        ))

    # ── Individual regime PDFs ──
    mixture_pdf = np.zeros_like(x)
    for s in range(N):
        pdf = norm.pdf(x, hmm.mu[s], hmm.sigma[s])
        weighted = stationary[s] * pdf
        mixture_pdf += weighted

        fig.add_trace(go.Scatter(
            x=x, y=pdf,
            mode='lines',
            line=dict(color=_color(s, 0.8), width=2),
            name=f'Regime {s}  (μ={hmm.mu[s]*100:.3f}%, σ={hmm.sigma[s]*100:.3f}%, π={stationary[s]:.2%})',
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
    # Weighted mean and std from stationary distribution
    overall_mu = np.sum(stationary * hmm.mu)
    overall_var = np.sum(stationary * (hmm.sigma**2 + hmm.mu**2)) - overall_mu**2
    overall_sigma = np.sqrt(overall_var)
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, overall_mu, overall_sigma),
        mode='lines',
        line=dict(color='rgba(80, 140, 255, 1)', width=2, dash='dot'),
        name=f'Fitted Normal  (μ={overall_mu*100:.3f}%, σ={overall_sigma*100:.3f}%)',
    ))

    fig.update_layout(
        title=dict(
            text=title or f'Regime Distributions ({N}-State Gaussian HMM)',
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
