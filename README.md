# Gaussian Hidden Markov Model for Market Regime Detection

A from-scratch implementation of a Gaussian Hidden Markov Model (HMM) in Python, designed for identifying market regimes in financial time series. Built with NumPy and SciPy, this package uses the Baum-Welch (EM) algorithm for parameter estimation, and supports Viterbi and Posterior decoding for state prediction.

---

## Table of Contents

1. [Why a Gaussian HMM?](#1-why-a-gaussian-hmm)
2. [A Primer on Bayesian Statistics](#2-a-primer-on-bayesian-statistics)
3. [What is a Hidden Markov Model?](#3-what-is-a-hidden-markov-model)
4. [Model Definition](#4-model-definition)
5. [Parameter Initialization](#5-parameter-initialization)
6. [The Forward Algorithm](#6-the-forward-algorithm)
7. [The Backward Algorithm](#7-the-backward-algorithm)
8. [Computing Gamma (γ) — State Confidence](#8-computing-gamma-γ--state-confidence)
9. [Computing Xi (ξ) — Transition Confidence](#9-computing-xi-ξ--transition-confidence)
10. [The Baum-Welch Algorithm (EM)](#10-the-baum-welch-algorithm-em)
11. [Decoding: Viterbi vs Posterior](#11-decoding-viterbi-vs-posterior)
12. [Usage Guide](#12-usage-guide)
13. [API Reference](#13-api-reference)
14. [Future Improvements](#14-future-improvements)

---

## 1. Why a Gaussian HMM?

Market returns are **continuously distributed**, non-stationary and non-normal. Therefore, longer-term mean and variance do not hold, and modelling assumptions based on a single distribution fail to take these into consideration.

Because of this, we assume that the returns in each hidden state (regime) follow a **Gaussian distribution**, where each Gaussian has its own mean and variance, and layer multiple Gaussian distributions on top of each other (one per regime). The resulting "mixture distribution" can theoretically fit the non-normality and non-stationarity of real market returns. A single Gaussian cannot capture fat tails, volatility clustering, or regime switches — but a mixture of them might model something closer to true market conditions.

However, this is not to say that the distributions of market returns within each regime is stationary. In fact, the opposite is true. Because of this, Kalman Filters might be extended to account for this.

Because each state (regime) defines its own $\mu$ and $\sigma$, the overall distribution of returns becomes a weighted sum of Gaussians. 

**TLDR:**
- Each hidden state represents a **market regime** (e.g., bull, bear, high-vol, low-vol)
- Each regime has its own **mean return** ($\mu_j$) and **volatility** ($\sigma_j$)
- The model learns to switch between regimes based on observed data

Here is a quick infographic by Gemini that illustrates this point.
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/817aee79-ccc0-4c07-8d56-3c60524406f4" />


---

## 2. A Primer on Bayesian Statistics

Before diving into the HMM, it helps to understand the Bayesian framework that underpins the entire model. In fact, nearly everything the HMM does — from the Forward-Backward algorithm to the Baum-Welch parameter updates — is an application of Bayesian reasoning.

### The Core Idea

Classical (frequentist) statistics asks: "Given a fixed model, how likely is the data?" Bayesian statistics flips this around and asks: **"Given the data observed, how should the model's belief change?"**

This is formalized in **Bayes' Theorem**:

$$P(\text{Hypothesis} \mid \text{Evidence}) = \frac{P(\text{Evidence} \mid \text{Hypothesis}) \cdot P(\text{Hypothesis})}{P(\text{Evidence})}$$

Each term has a specific name and role:

| Term | Name | Meaning |
|------|------|---------|
| $P(H \mid E)$ | **Posterior** | Our updated belief about the hypothesis **after** seeing the evidence |
| $P(E \mid H)$ | **Likelihood** | How probable is the evidence if the hypothesis were true? |
| $P(H)$ | **Prior** | Our belief about the hypothesis **before** seeing any evidence |
| $P(E)$ | **Evidence** (marginal likelihood) | The total probability of the evidence across all possible hypotheses |

### An Intuitive Example

Suppose you want to know if the market is currently in a "crisis" regime ($H$), and you observe a daily return of -5% ($E$).

- **Prior** $P(H)$: Before looking at today's return, you might believe there's a 10% chance the market is in crisis mode (based on historical frequency).
- **Likelihood** $P(E \mid H)$: If the market *is* in crisis, what's the probability of seeing a -5% return? Under a crisis regime with $\mu = -0.02$, $\sigma = 0.03$, this might be quite plausible — say 5%.
- **Likelihood under alternative** $P(E \mid H^c)$: If the market is *not* in crisis, seeing -5% under a calm regime with $\mu = 0.001$, $\sigma = 0.008$ would be extremely unlikely — say 0.001%.
- **Evidence** $P(E)$: The total probability of seeing -5% across all regimes: $P(E) = P(E \mid H) \cdot P(H) + P(E \mid H^c) \cdot P(H^c)$.

Plugging in:

$$P(\text{Crisis} \mid -5\%) = \frac{0.05 \times 0.10}{0.05 \times 0.10 + 0.00001 \times 0.90} = \frac{0.005}{0.005009} \approx 99.8\%$$

Even though the prior probability of crisis was only 10%, a single -5% return is so unlikely under the calm regime that our posterior jumps to 99.8%. The evidence has overwhelmed the prior. This is the power of Bayesian updating — the data speaks for itself.

### Why This Matters for HMMs

The entire HMM framework is Bayesian at its core:

- **The hidden states are the hypothesis** — we cannot observe them directly, but we infer that they exist
- **The observations (returns) are the evidence** — they are what we actually see
- **The model parameters ($\pi$, $A$, $\mu$, $\sigma$) encode our prior beliefs** about regime behaviour
- **Gamma ($\gamma_t$) is the posterior** — our updated belief about which regime is current, given all the evidence

The Forward algorithm computes the likelihood of the data up to time $t$. The Backward algorithm extends this to include future data (only in training). When we multiply them together to get gamma, we are performing Bayesian inference: updating our belief about the hidden state at time $t$ using **all** available evidence — past, present, and future.

The Baum-Welch algorithm then goes one step further: it says "given these posteriors (our best beliefs about the hidden states), what model parameters would maximize the probability of the data?" This is the **Maximization** step — and the entire EM loop is fundamentally an iterative application of Bayes' theorem, alternating between:
1. **E-step:** "Given the current parameters, what do I believe about the hidden states?" (Bayesian inference)
2. **M-step:** "Given my beliefs about the hidden states, what parameters best explain the data?" (Maximum likelihood)

### Conditional Probability

One more concept that appears everywhere in HMMs: **conditional probability**. The notation $P(A \mid B)$ reads "the probability of $A$ given $B$" — it tells us how likely $A$ is when we already know that $B$ is true.

$$P(A \mid B) = \frac{P(A, B)}{P(B)} = \frac{P(A \cap B)}{P(B)}$$

Where $P(A, B)$ is the **joint probability** — the probability that both $A$ and $B$ happen simultaneously. This relationship is what lets us decompose complex joint probabilities into manageable conditional steps. For example, the transition probability $a_{ij} = P(q_{t+1} = j \mid q_t = i)$ is a conditional probability: the probability of moving to state $j$, given that we are currently in state $i$.

### Joint, Marginal, and Conditional — How They Connect

These three types of probability are the building blocks of everything in the HMM:

- **Joint probability** $P(A, B)$: The probability that both $A$ and $B$ occur together. In our model, $\alpha_t(j) = P(O_{0:t}, q_t = j)$ is a joint probability — the probability of seeing the observation sequence **and** being in state $j$.

- **Marginal probability** $P(A) = \sum_B P(A, B)$: The probability of $A$ regardless of $B$. We obtain the total likelihood $P(O) = \sum_j \alpha_T(j)$ by **marginalizing out** (summing over) all possible hidden states — we don't care which state produced the data, we just want the overall probability of the data.

- **Conditional probability** $P(A \mid B) = P(A, B) / P(B)$: The probability of $A$ given $B$ is true. Gamma $\gamma_t(j) = P(q_t = j \mid O)$ is exactly this — we take the joint probability $P(O, q_t = j)$ and divide by the marginal $P(O)$ to ask "given that we saw this specific data, what's the probability we were in state $j$?"

Understanding these relationships is crucial because the entire HMM pipeline is really just a sequence of decompositions: breaking apart joint probabilities, marginalizing over unknowns, and conditioning on observations.

---

## 3. What is a Hidden Markov Model?

### The Problem: We Can't See the Market's State

At any point in time, the market is arguably in some "state" or "regime" — maybe a bullish trend, a volatile correction, a quiet consolidation, or an outright crash. But we never observe these states directly. What we *do* observe are the returns: the noisy outputs that these hidden regimes generate.

This is exactly the setup of a **Hidden Markov Model**: there exists a hidden process (the sequence of market regimes) that evolves over time according to some transition rules, and at each time step, this hidden process produces an observable output (a return) through some emission distribution.

### The "Markov" Part

The "Markov" in Hidden Markov Model refers to the **Markov property** (or "memorylessness"): the hidden state at time $t+1$ depends **only** on the hidden state at time $t$, not on any earlier history. Therefore, the probability of the next state is independent of all previous states given the current state, and this means that the sequence of hidden states is conditionally independent.

$$P(q_{t+1} \mid q_t, q_{t-1}, \ldots, q_0) = P(q_{t+1} \mid q_t)$$

In market terms, this says: "The probability of what regime we transition to tomorrow depends only on what regime we are in today, not on what happened last week or last month." This is a simplifying assumption — real market dynamics likely have longer memory — but it makes the problem tractable and still captures the essential idea that regimes tend to persist and transitions follow patterns.

This can probably be explored further using a Hidden Semi-Markov Model (HSMM), where the hidden state is allowed to persist for a variable amount of time before transitioning to the next state, and the "mean-reversion" and "non-memoryless" properties of the market can be better captured.

### The "Hidden" Part

The key challenge is that we **never directly observe** $q_t$ (the regime). We only observe $O_t$ (the return). The relationship between the hidden state and the observation is governed by the **emission distribution**:

$$O_t \sim \mathcal{N}(\mu_{q_t}, \sigma_{q_t}^2)$$

Each hidden state $j$ has its own Gaussian distribution with mean $\mu_j$ and standard deviation $\sigma_j$. When the system is in state $j$ at time $t$, the observed return $O_t$ is drawn from that state's Gaussian. So a bear market might have $\mu = -0.002$ and $\sigma = 0.025$, while a bull market might have $\mu = 0.001$ and $\sigma = 0.006$.

The word "hidden" means we have to **infer** which state generated each observation — and that's where the entire apparatus of Forward-Backward, Gamma, Xi, Baum-Welch, and Viterbi comes in.

### How an HMM Generates Data

To understand how inference works, it helps to first think about how an HMM **generates** a sequence. The generative story is:

1. **Start** by sampling the initial state $q_0$ from the initial distribution $\pi$. For example, with 3 states and $\pi = [0.2, 0.5, 0.3]$, there's a 50% chance we start in state 1.

2. **Emit** an observation $O_0$ by sampling from the emission distribution of state $q_0$. If $q_0 = 1$ and state 1 has $\mu_1 = 0.001, \sigma_1 = 0.008$, we draw $O_0 \sim \mathcal{N}(0.001, 0.008^2)$.

3. **Transition** to the next state $q_1$ by sampling from the transition probabilities $A[q_0, :]$. If $q_0 = 1$ and the second row of $A$ is $[0.05, 0.90, 0.05]$, there's a 90% chance we stay in state 1 (regimes are sticky).

4. **Repeat** steps 2-3 for each subsequent time step.

The result is two sequences: a hidden state sequence $q_0, q_1, \ldots, q_{T-1}$ and an observation sequence $O_0, O_1, \ldots, O_{T-1}$. We only see the $O$ sequence. Our job is to recover the $q$ sequence — doing the "data generation story" in reverse.

### The Three Fundamental Problems of HMMs

Given a model $\lambda = (\pi, A, \theta)$, there are three core questions we need to answer:

**Problem 1: Evaluation** — "How well does the model explain the data?"

$$P(O \mid \lambda) = \text{?}$$

Given our model parameters and an observation sequence, what is the total probability that this model would have generated this data? This is answered by the **Forward algorithm** (Section 6). This probability is critical for comparing models (e.g., is a 3-state model better than a 5-state model?) and for computing the log-likelihood during training.

**Problem 2: Decoding** — "What hidden states generated the data?"

$$P(q_t = j \mid O, \lambda) = \text{?}$$

Given the observations, what is the most likely sequence of hidden states? This is answered by **Gamma** and the **Viterbi algorithm** (Sections 8, 11). This is the regime detection problem itself — the reason we built the model.

**Problem 3: Learning** — "What model best explains the data?"

$$\lambda^* = \arg\max_\lambda P(O \mid \lambda) = \text{?}$$

What parameters $\pi$, $A$, $\mu$, $\sigma$ maximize the probability of the observed returns? This is answered by the **Baum-Welch algorithm** (Section 10). Since we cannot observe the hidden states (latent property), we are unable to compute the optimal parameters — instead we iterate between estimating the states (E-step) and optimizing the parameters (M-step) to infer the hidden states from the observations.

### The Relationship Between Components

All the algorithms in this package work together in a pipeline. Here's how they connect:

```
Observed Returns (O)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                 BAUM-WELCH (EM)                     │
│                                                     │
│   ┌─────────── E-Step ──────────┐                   │
│   │                             │                   │
│   │   Forward  ──► α (scaled)   │                   │
│   │   Backward ──► β (scaled)   │                   │
│   │                             │                   │
│   │   α × β ──► γ (state conf.) │                   │
│   │   α × A × b × β ──► ξ       │                   │
│   │        (transition conf.)   │                   │
│   └─────────────────────────────┘                   │
│                  │                                  │
│                  ▼                                  │
│   ┌─────────── M-Step ──────────┐                   │
│   │                             │                   │
│   │   γ[0]        ──► π*        │                   │
│   │   Σξ / Σγ     ──► A*        │                   │
│   │   γ-weighted  ──► μ*, σ*    │                   │
│   └─────────────────────────────┘                   │
│                  │                                  │
│          Repeat until convergence                   │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Fitted Model (π*, A*, μ*, σ*)
        │
        ├──► Viterbi    ──► Most likely state SEQUENCE
        ├──► Posterior   ──► Most likely state per TIME STEP
        └──► Probability ──► Confidence per state per time step
```

The Forward-Backward algorithms are the computational engines. Gamma and Xi are the statistical summaries they produce. Baum-Welch orchestrates the whole training loop. And the decoding methods are how we extract actionable regime labels from the fitted model.

---

## 4. Model Definition

A Hidden Markov Model is defined by the parameter set $\lambda = (\pi, A, \theta)$, where:

| Symbol | Name | Description |
|--------|------|-------------|
| $N$ | Number of states | The number of hidden regimes the model assumes |
| $\pi$ | Initial state distribution | $\pi_j = P(q_0 = j)$ — the probability the system starts in state $j$ |
| $A$ | Transition matrix | $a_{ij} = P(q_{t+1} = j \mid q_t = i)$ — probability of transitioning from state $i$ to state $j$ |
| $\theta_j = (\mu_j, \sigma_j)$ | Emission parameters | State-specific mean and standard deviation of the Gaussian |
| $b_j(O_t)$ | Emission probability | The probability of observing return $O_t$ given hidden state $j$ |

### Transition Matrix ($A$)

The transition matrix $A$ is an $N \times N$ matrix where entry $a_{ij}$ represents the probability of moving from state $i$ to state $j$ in one time step. Each row sums to 1, because the system must go *somewhere* (including staying in the same state):

$$\sum_{j=1}^{N} a_{ij} = 1 \quad \text{for all } i$$

The transition probability $a_{ij} = P(q_{t+1} = j \mid q_t = i)$ is a **conditional probability**: given that we are currently in state $i$, what is the probability that we move to state $j$? This is the Markov property at work — the next state depends only on the current state.

In practice, the diagonal entries $a_{ii}$ (self-transitions) tend to be large, reflecting the "stickiness" of market regimes. A transition matrix like:

$$A = \begin{bmatrix} 0.95 & 0.03 & 0.02 \\ 0.04 & 0.92 & 0.04 \\ 0.02 & 0.05 & 0.93 \end{bmatrix}$$

says: "Once you're in a regime, you're very likely to stay there, with only small probabilities of switching."

### Emission Probability ($b_j$)

Since returns are continuous, the emission probability is computed using the **Gaussian PDF**:

$$b_j(O_t) = \frac{1}{\sigma_j \sqrt{2\pi}} \exp\left(-\frac{(O_t - \mu_j)^2}{2\sigma_j^2}\right)$$

This is the probability of observing return $O_t$ if we know we are in state $j$ — essentially asking "how likely is this return under regime $j$'s distribution?" It is the degree of confidence that we observe return $O_t$ given that we are in state $j$.

Note that $b_j(O_t)$ is a **conditional probability density**: $b_j(O_t) = P(O_t \mid q_t = j)$. It tells us how compatible the observed return is with each regime's statistical profile. A return of -4% would have a high emission probability under a crash regime (high $\sigma$, negative $\mu$) but near-zero probability under a calm bull regime (low $\sigma$, positive $\mu$). This is exactly how the model discriminates between regimes — returns that "fit" a regime's distribution are evidence for that regime.

### Initial State Distribution ($\pi$)

The initial state distribution $\pi$ is simply:

$$\pi_j = P(q_0 = j)$$

The probability that the hidden chain starts in state $j$. This is a **marginal probability** — it doesn't condition on any observations, it purely reflects our prior belief (or learned estimate) about which state the system is likely to begin in. After training, $\pi$ will usually concentrate on one or two states that best explain the first few observations.

---

## 5. Parameter Initialization

Before the EM algorithm can iteratively refine parameters, we need to estimate initial values for the parameters. Poor initialization can lead to convergence at a bad local optimum. Hence:

### Initial State Distribution ($\pi$)

We assume equal probability of starting in any state as it is not easily determined without actually observing the model:

$$\pi_j = \frac{1}{N} \quad \text{for all } j$$

### Transition Matrix ($A$)

We initialize with higher self-transition probability because in practice, regimes are **sticky** and they have clustering properties. They are more likely to persist than to transition to the next (bull market tends to stay bull market, bear market volatility tends to persist for a while). Following this property, the diagonal gets a weight of $N$, all off-diagonals get weight 1, and then we normalize each row to sum to 1:

$$A_{ij} = \begin{cases} \frac{N}{N + (N-1)} & \text{if } i = j \\ \frac{1}{N + (N-1)} & \text{if } i \neq j \end{cases}$$

For example, with $N = 3$ states:

$$A = \begin{bmatrix} 0.6 & 0.2 & 0.2 \\ 0.2 & 0.6 & 0.2 \\ 0.2 & 0.2 & 0.6 \end{bmatrix}$$

### Emission Parameters ($\mu_j$, $\sigma_j$)

We divide the observed returns $X$ into $N$ equal quantiles. For $N = 5$, we compute the 0th, 20th, 40th, 60th, 80th, and 100th percentile values and classify each observation into one of 5 bins. Then, for each bin $j$:

$$\mu_j = \frac{\sum_{x \in \text{bin}_j} x}{|\text{bin}_j|} \quad \quad \sigma_j = \sqrt{\frac{\sum_{x \in \text{bin}_j} x^2}{|\text{bin}_j|} - \mu_j^2}$$

This uses the identity $\text{Var}(X) = E[X^2] - (E[X])^2$, computed in a vectorized fashion using `np.bincount` with weights.

If a bin is empty, we fall back to the global mean and variance of $X$ as a safety net.

---

## 6. The Forward Algorithm

The Forward algorithm computes **alpha** ($\alpha$), the joint probability of observing the sequence up to time $t$ and being in state $j$ at time $t$.

### Definition

$$\alpha_t(j) = P(O_0, O_1, \ldots, O_t, \; q_t = j \mid \lambda)$$

What is the probability that we observed this specific sequence of returns so far, **and** we are currently in state $j$? It is a joint probability distribution of the observed sequence and the hidden state at time $t$.

This is a **joint probability** — not a conditional one. It tells us the probability of two things happening simultaneously: (1) the specific sequence $O_0, O_1, \ldots, O_t$ was observed, and (2) the hidden state at time $t$ happens to be $j$. The reason we compute this joint probability rather than just $P(q_t = j)$ is that it accumulates evidence from **all observations** seen so far, weighting each state by how compatible the data stream is with that state's parameters.

### Recursion

**Base case** ($t = 0$):

$$\alpha_0(j) = \pi_j \cdot b_j(O_0)$$

This is the **product of two probabilities**: the prior probability of starting in state $j$ (from $\pi$), and the likelihood of observing the first return $O_0$ under state $j$'s emission distribution. This product gives us the joint probability $P(q_0 = j, O_0)$ — the probability that we start in state $j$ **and** see observation $O_0$.

**Recursive step** ($t \geq 1$):

$$\alpha_t(j) = \left[\sum_{i=1}^{N} \alpha_{t-1}(i) \cdot a_{ij}\right] \cdot b_j(O_t)$$

For each state $j$ at time $t$, we sum over all possible previous states $i$: the probability of being in $i$ at $t-1$ with the correct observation history (that's $\alpha_{t-1}(i)$), times the probability of transitioning from $i$ to $j$ (that's $a_{ij}$). This sum is a **marginalization** over the previous state — we do not know for certain which state we came from, so we consider all possibilities weighted by their probabilities.

Then we multiply by the emission probability $b_j(O_t)$: how likely is the return $O_t$ under state $j$? The result is the joint probability $P(O_{0:t}, q_t = j)$ — the probability of seeing this exact sequence of observations **and** ending up in state $j$ at time $t$.

### Vectorized Form

Instead of looping over states, we express this as a single matrix operation per time step:

$$\boldsymbol{\alpha}_t = (\boldsymbol{\alpha}_{t-1} \; @ \; A) \odot \mathbf{b}(O_t)$$

Where $@$ is matrix multiplication and $\odot$ is element-wise (Hadamard) multiplication. The matrix multiplication $\boldsymbol{\alpha}_{t-1} @ A$ performs the summation-over-previous-states for all $N$ target states simultaneously, and the element-wise multiplication applies each state's emission probability.

### Scaling (Preventing Underflow)

Here's the problem: alpha multiplies very small probabilities at every time step. After a few hundred steps, $\alpha$ can quickly tend to zero, causing **numerical underflow**. Following Bishop's *Pattern Recognition and Machine Learning*, we scale alpha at each step by dividing by its sum:

$$c_t = \sum_j \alpha_t(j) \quad \quad \hat{\alpha}_t(j) = \frac{\alpha_t(j)}{c_t}$$

Where $c_t$ is the **scaling factor**. It has a probabilistic interpretation: $c_t = P(O_t \mid O_{0:t-1}, \lambda)$, the conditional probability of observing $O_t$ given everything that came before it. This is a **predictive probability** — how surprised the model is by the next observation given its history. The scaled $\hat{\alpha}_t$ always sums to 1 across states, which means it can be interpreted as a **filtered state distribution** — our best estimate of which state we're in at time $t$, using only observations up to and including $t$.

The total data likelihood can be recovered from the scaling factors:

$$P(O \mid \lambda) = \prod_{t=0}^{T-1} c_t \quad \implies \quad \log P(O \mid \lambda) = \sum_{t=0}^{T-1} \log(c_t)$$

This is how we compute the **log-likelihood** without ever computing the full (astronomically small) joint probability. Each $c_t$ is a manageable number, and summing their logs avoids all the underflow problems.

---

## 7. The Backward Algorithm

The Backward algorithm computes **beta** ($\beta$), the probability of observing the **future** sequence from time $t+1$ onwards, given that we are in state $j$ at time $t$.

### Definition

$$\beta_t(j) = P(O_{t+1}, O_{t+2}, \ldots, O_{T-1} \mid q_t = j, \lambda)$$

If we are in state $j$ right now at time $t$, what is the probability of seeing the rest of the sequence? This is a **conditional probability** — it conditions on the current state $j$ and asks about the probability of all future observations. Whereas alpha looks backwards ("what's the probability of everything up to now?"), beta looks forward ("what's the probability of everything yet to come?").

Together, alpha and beta give us a complete picture: alpha accounts for all evidence from the past, and beta accounts for all evidence from the future. Their product at time $t$ captures the full observation sequence, which is exactly what we need for computing gamma.

### Recursion

**Base case** ($t = T-1$):

$$\beta_{T-1}(j) = 1 \quad \text{for all } j$$

At the last time step, there is no future to observe, so the probability of "seeing no future observations" is trivially 1 — it's a certain event. This is the boundary condition that anchors the backwards recursion.

**Recursive step** (backwards, $t = T-2$ down to $0$):

$$\beta_t(j) = \sum_{i=1}^{N} a_{ji} \cdot b_i(O_{t+1}) \cdot \beta_{t+1}(i)$$

For each state $j$ at time $t$, we consider every possible next state $i$: the probability of transitioning from $j$ to $i$ (that's $a_{ji}$), times the probability of observing the next return $O_{t+1}$ in state $i$ (that's $b_i(O_{t+1})$), times the probability of all remaining observations from state $i$ onwards (that's $\beta_{t+1}(i)$). The sum over $i$ is again a **marginalization** — we don't know which state comes next, so we sum over all possibilities.

### Vectorized Form

$$\boldsymbol{\beta}_t = \left[(\boldsymbol{\beta}_{t+1} \odot \mathbf{b}(O_{t+1})) \; @ \; A^T\right]$$

We multiply beta and the emission element-wise first (combining future probability with the next observation's likelihood), then multiply by $A^T$ (transpose). The transpose is needed because we are looking at transitions **from** $j$ **to** $i$, which reverses the row-column convention of our transition matrix.

### Scaling

Just like alpha, beta must be scaled to prevent underflow. The reasoning mirrors the forward case, but runs in reverse.

Recall from the Forward algorithm that each scaling factor $c_t$ represents a conditional probability:

$$c_t = P(O_t \mid O_{0:t-1}, \lambda)$$

For **alpha**, the cumulative scaling product runs forward from the start:

$$C_t = c_0 \cdot c_1 \cdot \ldots \cdot c_t = P(O_{0:t} \mid \lambda)$$

$$\hat{\alpha}_t(j) = \frac{\alpha_t(j)}{C_t}$$

Where $C_t$ is `total_probability_scale_t` in the code. We achieve this by dividing alpha by $c_t$ at each forward step, since the cumulative product grows by one factor each time.

For **beta**, the cumulative scaling product runs backward from the end:

$$C_{t+1}^{\leftarrow} = c_{t+1} \cdot c_{t+2} \cdot \ldots \cdot c_{T-1} = P(O_{t+1:T-1} \mid O_{0:t}, \lambda)$$

$$\hat{\beta}_t(j) = \frac{\beta_t(j)}{C_{t+1}^{\leftarrow}}$$

Where $C_{t+1}^{\leftarrow}$ is `total_probability_scale_t+1` in the code. To achieve this cumulative product, we divide by $c_{t+1}$ at each backward step:

$$\hat{\beta}_t(j) = \frac{\beta_t(j)}{c_{t+1} \cdot c_{t+2} \cdot \ldots \cdot c_{T-1}}$$

Since the recursion works backwards from $T-2$ down to $0$, each step accumulates one more scaling factor. At $t = T-2$, we divide by $c_{T-1}$ (one factor). At $t = T-3$, the previous beta was already divided by $c_{T-1}$, and now we divide by $c_{T-2}$ as well (two factors). By the time we reach $t = 0$, beta has been divided by the full product $c_1 \cdot c_2 \cdot \ldots \cdot c_{T-1}$.

The key insight is that no new scaling factors are needed — the $c$ values from the forward pass serve double duty. They stabilize both alpha and beta, and they cancel out perfectly when we compute gamma and xi.

---

## 8. Computing Gamma (γ) — State Confidence

Gamma represents the **posterior probability** of being in state $j$ at time $t$, given the **entire** observation sequence (both past and future). This is the model's confidence that the hidden state at time $t$ is $j$.

### Definition

$$\gamma_t(j) = P(q_t = j \mid O, \lambda) = \frac{\alpha_t(j) \cdot \beta_t(j)}{P(O \mid \lambda)}$$

This is a **conditional probability** in the Bayesian sense. We are conditioning on the full observation sequence $O$ (the evidence) and asking about the hidden state $q_t$ (the hypothesis). In Bayesian terms:

- **Prior:** Our initial belief about being in state $j$ (captured in $\pi$ and $A$)
- **Likelihood:** How well the data fits state $j$ (captured in $b_j$)
- **Posterior:** Our updated belief after seeing all the data (that's $\gamma_t(j)$)

The numerator $\alpha_t(j) \cdot \beta_t(j)$ is the **joint probability** $P(O, q_t = j)$ — the probability of seeing the entire observation sequence **and** being in state $j$ at time $t$. The denominator $P(O)$ is the **marginal probability** of the observations — the total probability of the data regardless of which state we're in. Dividing joint by marginal gives us the conditional (posterior) probability.

### Why Scaling Makes This Simple

Since alpha and beta are already scaled, we can compute gamma directly as:

$$\gamma_t(j) = \hat{\alpha}_t(j) \cdot \hat{\beta}_t(j)$$

Here's why this works: $\hat{\alpha}_t(j)$ has been pre-divided by $P(O_{0:t} \mid \lambda)$, and $\hat{\beta}_t(j)$ has been pre-divided by $P(O_{t+1:T-1} \mid O_{0:t}, \lambda)$. Multiplying these denominators together:

$$P(O_{0:t} \mid \lambda) \cdot P(O_{t+1:T-1} \mid O_{0:t}, \lambda) = P(O_{0:T-1} \mid \lambda) = P(O \mid \lambda)$$

Which is exactly the denominator of the original (unscaled) gamma. Therefore, the product $\hat{\alpha} \cdot \hat{\beta}$ is already properly normalized — we just need to re-normalize each row to sum to 1 as a safety measure against floating-point drift:

$$\gamma_t(j) = \frac{\hat{\alpha}_t(j) \cdot \hat{\beta}_t(j)}{\sum_k \hat{\alpha}_t(k) \cdot \hat{\beta}_t(k)}$$

### Interpretation

$\gamma_t(j)$ tells you: "Given everything the model has seen — both before and after time $t$ — how confident is it that the market was in regime $j$ at time $t$?"

Note that $\gamma_t$ sums to 1 across states: $\sum_j \gamma_t(j) = 1$. This means at each time point, you get a full **probability distribution** over regimes. The model might say "70% bull, 25% neutral, 5% bear" — this is much richer than a hard assignment and captures the uncertainty inherent in regime classification.

---

## 9. Computing Xi (ξ) — Transition Confidence

Xi represents the **joint posterior probability** of being in state $i$ at time $t$ **and** state $j$ at time $t+1$, given the full observation sequence.

### Definition

$$\xi_t(i, j) = P(q_t = i, \; q_{t+1} = j \mid O, \lambda)$$

While gamma tells us the probability of being in a **single** state at a **single** time, xi tells us the probability of a specific **transition**: from state $i$ at time $t$ to state $j$ at time $t+1$. This is a **joint posterior probability** over two consecutive hidden states, conditioned on the full observation sequence.

The relationship between xi and gamma is:

$$\gamma_t(i) = \sum_{j=1}^{N} \xi_t(i, j)$$

Summing xi over all possible next states $j$ gives us gamma at time $t$ — if we marginalize out the future state, we're just left with the probability of the current state.

### Formula

$$\xi_t(i, j) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(O_{t+1}) \cdot \beta_{t+1}(j)}{P(O \mid \lambda)}$$

Each term carries a specific probabilistic meaning:
- $\alpha_t(i)$: probability of the observations up to time $t$ and being in state $i$ (evidence from the past)
- $a_{ij}$: probability of transitioning from $i$ to $j$ (the model's transition rule)
- $b_j(O_{t+1})$: probability of observing $O_{t+1}$ under state $j$ (how well the next observation fits)
- $\beta_{t+1}(j)$: probability of all future observations from state $j$ onwards (evidence from the future)
- $P(O)$: total probability of the data (normalizer to convert joint → conditional)

With scaled variables, we divide by $c_{t+1}$ (the scaling factor for the transition day itself, which is the only factor not yet accounted for in the product of scaled alpha and scaled beta):

$$\xi_t(i, j) = \frac{\hat{\alpha}_t(i) \cdot a_{ij} \cdot b_j(O_{t+1}) \cdot \hat{\beta}_{t+1}(j)}{c_{t+1}}$$

### 3D Vectorized Computation

To compute $\xi_t(i,j)$ for **all** $i$, $j$ at **all** time steps simultaneously, we need to work in 3 dimensions. We define the dimension format as $(t, i, j)$:

Each of our matrices is missing one dimension:
1. **Alpha** is $(T\text{-}1, N)$ → missing the $j$ dimension → extend to $(T\text{-}1, N, 1)$
2. **Beta $\cdot$ Emission** is $(T\text{-}1, N)$ → missing the $i$ dimension → extend to $(T\text{-}1, 1, N)$
3. **A** is $(N, N)$ → missing the time dimension → extend to $(1, N, N)$

Think of it like making copies of a sheet of paper and stacking them in the direction of the missing dimension. NumPy's broadcasting then multiplies the three "cubes" together element-wise:

```python
alpha_b = alpha[:-1, :, np.newaxis]           # (T-1, N, 1)
beta_b  = (beta * emission)[1:, np.newaxis, :]  # (T-1, 1, N)
A_b     = A[np.newaxis, :, :]                  # (1,   N, N)

xi = alpha_b * A_b * beta_b / c[1:, np.newaxis, np.newaxis]
```

The result is a $(T\text{-}1, N, N)$ tensor where $\xi[t, i, j]$ is the probability of regime $i$ at time $t$ transitioning to regime $j$ at time $t+1$.

**Note:** Alpha runs from $t = 0$ to $T\text{-}2$, and beta runs from $t = 1$ to $T\text{-}1$ (since we look at $\beta_{t+1}$ when computing $\xi_t$).

---

## 10. The Baum-Welch Algorithm (EM)

The Baum-Welch algorithm is an instance of the **Expectation-Maximization (EM)** algorithm, applied to HMMs. As discussed in the Bayesian primer, we are iteratively updating our beliefs about the model parameters given the observed data.

The fundamental challenge is a **chicken-and-egg problem**: to find the best parameters, we need to know the hidden states; but to know the hidden states, we need the parameters. EM breaks this circular dependency by alternating:
1. **Assume** the current parameters are correct, and estimate the hidden states (E-step)
2. **Assume** the hidden state estimates are correct, and optimize the parameters (M-step)

Each iteration is guaranteed to increase (or maintain) the log-likelihood, so the algorithm converges to a local maximum.

### E-Step (Expectation)

Compute the expected sufficient statistics using the current parameters:
1. Run the **Forward** algorithm → $\hat{\alpha}$, scaling factors $c$
2. Run the **Backward** algorithm → $\hat{\beta}$
3. Compute **Gamma** ($\gamma$) → state confidence at each time step
4. Compute **Xi** ($\xi$) → transition confidence at each time step

The E-step is where the Bayesian inference happens. Given our current model (our "prior" beliefs about how regimes behave), we compute the posterior distribution over hidden states at each time point. The gamma and xi values are **soft assignments** — they are not discretized to a single state but rather distribute probability mass across all states proportional to the evidence.

### M-Step (Maximization)

Re-estimate the parameters to maximize the expected log-likelihood:

**Update $\pi$** — Initial state distribution:

$$\pi_j^* = \gamma_0(j)$$

Gamma at $t = 0$ gives us the posterior probability of each state at the start of the sequence. This is the model's best estimate of which state it started in. If the model is confident it started in state 2 (perhaps because the first few observations were very consistent with state 2's distribution), then $\pi_2$ will be high.

**Update $A$** — Transition matrix:

$$a_{ij}^* = \frac{\sum_{t=0}^{T-2} \xi_t(i, j)}{\sum_{t=0}^{T-2} \gamma_t(i)}$$

The numerator is the **expected number of transitions from $i$ to $j$** across the entire sequence. The denominator is the **expected number of times we were in state $i$** — the total number of opportunities the model could have transitioned from $i$ to any other state $j$. The ratio is the expected rate of the $i \to j$ transition.

In plain English: "Out of all the times the model believed it was in state $i$, how often did it believe the next state was $j$?" If the model spent 100 expected time steps in state 1, and 5 of those involved a transition to state 2, then $a_{12}^* = 5/100 = 0.05$.

In code:
```python
A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, np.newaxis]
```

**Update $\mu_j$** — State-specific mean:

$$\mu_j^* = \frac{\sum_{t=0}^{T-1} \gamma_t(j) \cdot O_t}{\sum_{t=0}^{T-1} \gamma_t(j)}$$

This is a **gamma-weighted average** of all observations. Every observation in the sequence contributes to the mean of state $j$, but each contribution is weighted by $\gamma_t(j)$ — how confident the model is that $O_t$ was generated by state $j$. Observations that the model strongly attributes to state $j$ have a large influence; observations attributed to other states have almost none.

**Update $\sigma_j$** — State-specific standard deviation:

$$\hat{\sigma}_j^2 = \frac{\sum_{t=0}^{T-1} \gamma_t(j) \cdot (O_t - \hat{\mu}_j)^2}{\sum_{t=0}^{T-1} \gamma_t(j)}$$

Similarly, this is the gamma-weighted variance — the spread of observations around $\mu_j$, where each observation's contribution is weighted by the model's confidence that it belongs to state $j$.

### Convergence

The log-likelihood is computed from the scaling factors:

$$\mathcal{L} = \sum_{t=0}^{T-1} \log(c_t)$$

The algorithm iterates E-step → M-step until the change in log-likelihood between iterations falls below a tolerance threshold `tol`, or the maximum number of iterations `max_iter` is reached. In theory, the log-likelihood should monotonically increase with each iteration, since each M-step is guaranteed to find parameters at least as good as the current ones.

---

## 11. Decoding: Viterbi vs Posterior

After the model is fitted, we want to **decode** the hidden states — i.e., determine which regime the market was in at each time step. There are two approaches, each with different trade-offs.

### Posterior Decoding

Takes the **argmax of gamma** at each time step independently:

$$\hat{q}_t = \arg\max_j \; \gamma_t(j)$$

At each time $t$, we simply pick the state with the highest posterior probability. This maximizes the probability of **each individual** state assignment — for any given time point, this is the single best guess.

**Pros:** Maximizes the probability of each individual state assignment. Captures uncertainty well (via the full gamma distribution before taking the argmax).

**Cons:** May produce impossible transitions. For example, if $a_{02} = 0$ (state 0 cannot transition directly to state 2), posterior decoding might still assign state 0 at $t$ and state 2 at $t+1$, because it treats each time step independently and doesn't consider the sequence as a whole.

### Viterbi Decoding

Finds the **single most probable sequence** of states as a whole, respecting the transition structure:

$$\hat{Q} = \arg\max_{q_0, q_1, \ldots, q_{T-1}} \; P(q_0, q_1, \ldots, q_{T-1} \mid O, \lambda)$$

This optimizes the **joint probability** of the entire state sequence, not individual time points. The key insight is that the best state at time $t$ depends not just on the observations, but on what comes before **and** after in the sequence. A state that looks less likely individually might be chosen because it produces a much better overall path.

This uses **dynamic programming** in log-space to avoid underflow.

**Initialization:**

$$V_0(j) = \log \pi_j + \log b_j(O_0)$$

**Recursion:**

$$V_t(j) = \max_i \left[V_{t-1}(i) + \log a_{ij}\right] + \log b_j(O_t)$$

$$\psi_t(j) = \arg\max_i \left[V_{t-1}(i) + \log a_{ij}\right]$$

Where $V_t(j)$ stores the highest log-probability of any path ending in state $j$ at time $t$, and $\psi_t(j)$ (the **backpointer**) records which previous state achieved that maximum. This is a form of **dynamic programming** — instead of evaluating all $N^T$ possible paths (exponentially many), we only track the best path to each state at each time step, reducing the complexity to $O(T \cdot N^2)$.

**Backtracking:**

$$\hat{q}_{T-1} = \arg\max_j \; V_{T-1}(j)$$

$$\hat{q}_t = \psi_{t+1}(\hat{q}_{t+1}) \quad \text{for } t = T\text{-}2 \text{ down to } 0$$

We start from the best final state and trace backwards through the backpointers to recover the entire optimal path.

### Vectorized Implementation

The recursion step is vectorized by broadcasting `viterbi[t-1]` into a column vector and adding `log_A`:

```python
prob = viterbi[t-1][:, np.newaxis] + log_A   # (N, N) matrix
viterbi[t, :] = np.max(prob, axis=0) + log_emission[t, :]
backpointer[t, :] = np.argmax(prob, axis=0)
```

Where `prob[i, j]` is the log-probability of the best path arriving at state $i$ at $t\text{-}1$ and transitioning to state $j$ at $t$. We take the max over the source states (axis=0) for each target state.

**Pros:** Always produces a valid sequence consistent with the transition matrix.

**Cons:** Optimizes the full path, not individual states — a single state assignment might be suboptimal if it produces a better overall path.


<img width="1484" height="900" alt="newplot" src="https://github.com/user-attachments/assets/f907f72c-1868-4558-95b6-77e2c166b724" />

---

## 12. Usage Guide

### Installation

No installation required. Just place `HMM.py` in your working directory.

**Dependencies:**
```
numpy
scipy
plotly (for visualization)
pandas (for visualization)
yfinance (for data download)
```

### Quick Start

```python
from HMM import gaussianHMM, plot_regimes
import yfinance as yf
import numpy as np

# 1. Download data
spy = yf.download('^GSPC', start='2000-12-31', end='2020-12-31')

# 2. Compute log returns
returns = np.log(spy['Close'] / spy['Close'].shift(1)).dropna()
close = spy['Close'].dropna()

# 3. Initialize and fit the model
hmm = gaussianHMM(n_states=3, max_iter=100, tol=1e-6)
hmm.fit(returns, sort='sharpe')

# 4. Predict regimes
viterbi_states = hmm.predict(returns, type='viterbi')
probability_states = hmm.predict(returns, type='probability')

# 5. Visualize
plot_regimes(
    close,
    viterbi_states,
    hmm=hmm,
    returns=returns,
    gamma=probability_states,
    title="S&P 500 — HMM Market Regimes"
)
```

### State Sorting

After fitting, states are sorted by a criterion so that state labels are always interpretable:
- `sort='mu'` — Sort by mean return (State 0 = lowest mean)
- `sort='sigma'` — Sort by volatility (State 0 = lowest vol)
- `sort='sharpe'` — Sort by Sharpe ratio $\mu / \sigma$ (State 0 = worst risk-adjusted return)

### Visualization

`plot_regimes()` produces an interactive Plotly chart with up to 3 panels:

| Panel | Content | Appears when |
|-------|---------|--------------| 
| Top | Price with regime-colored background shading | Always |
| Middle | Stacked area chart of state probabilities ($\gamma$) | `gamma` is provided |
| Bottom | Daily returns as bars, colored by regime | `returns` is provided |

The color palette runs from **red** (worst regime) → **yellow** → **green** (best regime), based on the sorted state order.

---

## 13. API Reference

### `gaussianHMM(n_states=5, max_iter=100, tol=1e-6)`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | int | 5 | Number of hidden states (regimes) |
| `max_iter` | int | 100 | Maximum EM iterations |
| `tol` | float | 1e-6 | Convergence threshold for log-likelihood change |

### `.fit(X, sort='mu')`

Fit the HMM to observation sequence `X` using Baum-Welch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Observation sequence (e.g., log returns) |
| `sort` | str | Sort states after fitting: `'mu'`, `'sigma'`, `'sharpe'`, or `None` |

**Returns:** `self` (fitted model)

**Fitted attributes:**
- `hmm.pi` — Initial state distribution $(N,)$
- `hmm.A` — Transition matrix $(N, N)$
- `hmm.mu` — State means $(N,)$
- `hmm.sigma` — State standard deviations $(N,)$

### `.predict(X, type='probability')`

Predict hidden states for observation sequence `X` using the fitted parameters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Observation sequence (can be unseen data) |
| `type` | str | `'probability'`, `'posterior'`, or `'viterbi'` |

**Returns:**

| Type | Output shape | Description |
|------|--------------|-------------|
| `'probability'` | $(T, N)$ | Gamma matrix — posterior probability of each state at each time step |
| `'posterior'` | $(T,)$ | Argmax of gamma — most likely state per time step (independently) |
| `'viterbi'` | $(T,)$ | Most likely **sequence** of states (globally optimal path) |

### `plot_regimes(price, regimes, hmm=None, returns=None, gamma=None, index=None, title=None)`

Standalone visualization function for regime overlay plots.

| Parameter | Type | Description |
|-----------|------|-------------|
| `price` | array-like / pd.Series | Price series for the top panel |
| `regimes` | array-like (int) | State labels per time step |
| `hmm` | gaussianHMM (optional) | Fitted model — if provided, shows $\mu$/$\sigma$ in legend |
| `returns` | array-like (optional) | If provided, adds a returns panel at the bottom |
| `gamma` | $(T, N)$ array (optional) | If provided, adds a state probability panel in the middle |
| `index` | array-like (optional) | X-axis labels — auto-detected from price if omitted |
| `title` | str (optional) | Chart title |

---

## 14. Future Improvements

Several extensions could meaningfully improve its accuracy and practical utility for regime-based risk management or alpha-generation.

### 14.1 Kalman Filters for Non-Stationary Regime Distributions

The current model assumes that each regime's distribution is **stationary** — that is, the mean $\mu_j$ and volatility $\sigma_j$ of each regime are fixed across the entire observation period. In reality, this assumption does not hold. A bear market in 2008 has different statistical properties from a bear market in 2020, even though both are arguably the same "regime". The volatility of a crisis is not constant — it rises, peaks, and subsides, all while the model insists it belongs to a single fixed Gaussian.

A natural extension is to allow the regime-specific parameters to **evolve over time** using a **Kalman Filter**. Instead of treating $\mu_j$ and $\sigma_j$ as static values, we model them as latent variables with their own dynamics:

$$\mu_{j,t} = \mu_{j,t-1} + \eta_t^\mu \quad \quad \sigma_{j,t} = \sigma_{j,t-1} + \eta_t^\sigma$$

where $\eta$ are process noise terms. The Kalman Filter would then estimate these time-varying parameters online, allowing each regime to adapt its mean and volatility as market conditions within that regime shift.

This creates a **Switching State-Space Model**: the HMM handles the discrete regime transitions, while the Kalman Filter handles the continuous evolution of parameters within each regime. The result would be a model that recognises not just *which* regime the market is in, but also *how* that regime is changing in real time.

### 14.2 Hidden Semi-Markov Models (HSMM)

The standard HMM assumes the **Markov property** — the probability of the next state depends only on the current state, with no memory of how long it has persisted for. This means the **duration** spent in any state follows a geometric distribution: at every time step, there is a fixed probability of leaving the current state, regardless of how long it has persisted for.

Markets, however, exhibit **mean-reverting** behaviour. A regime that has persisted for a long time is arguably more likely to end soon, not less. Similarly, a regime that just started is likely to persist for at least some minimum duration. The memoryless property of standard HMMs cannot capture this.

**Hidden Semi-Markov Models (HSMMs)** address this by explicitly modelling the **duration distribution** of each state. Instead of a fixed self-transition probability, each state $j$ has a duration distribution $d_j(\tau)$ that specifies the probability of staying in state $j$ for exactly $\tau$ time steps:

$$d_j(\tau) = P(\text{duration} = \tau \mid q = j)$$

This could be parameterized as a Poisson, Negative Binomial, or log-normal distribution, allowing the model to learn that, say, bull markets typically last 200-400 trading days while crash regimes last 20-60 days.

### 14.3 State Probability Acceleration as a Predictive Signal

An interesting observation from the gamma (state probability) panel is that regime transitions are rarely instantaneous — the probability of the incoming regime **ramps up gradually** before the transition is confirmed by the Viterbi path. This "acceleration" of state probability could serve as an early warning signal.

<img width="1484" height="900" alt="acceleRATION" src="https://github.com/user-attachments/assets/059b75ad-b165-47d1-9381-dcc5860226ff" />

As observed, nearing to a regime change, the gradient of the probability of the incoming regime change might be a predictive indicator.
---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. — Scaling factor approach for Forward-Backward.
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE. — The foundational HMM tutorial.
