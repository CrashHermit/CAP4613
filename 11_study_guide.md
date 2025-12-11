# Study Guide 11: Dynamic Programming

## Overview
This study guide covers Dynamic Programming concepts, including memoization vs tabulation, applications in RL and NLP, and common implementation challenges.

---

## 1. What is Dynamic Programming?

### Definition
Dynamic Programming (DP) is an **optimization technique** that solves complex problems by:
1. Breaking them into **simpler, overlapping subproblems**
2. Solving each subproblem **once**
3. **Storing solutions** to avoid re-computation
4. **Combining solutions** to solve the original problem

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Optimal Substructure** | Optimal solution contains optimal solutions to subproblems |
| **Overlapping Subproblems** | Same subproblems solved multiple times |
| **Bellman's Principle** | Value of state = immediate reward + discounted future value |

### The Bellman Equation
```
V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]

Where:
  V(s) = value of state s
  R(s,a) = immediate reward
  γ = discount factor
  P(s'|s,a) = transition probability
```

> **Quiz Focus**: DP is an optimization technique that solves complex problems by breaking them into simpler, overlapping subproblems and storing solutions to avoid re-computation.

---

## 2. Top-Down vs Bottom-Up Approaches

### Comparison Table

| Aspect | Top-Down (Memoization) | Bottom-Up (Tabulation) |
|--------|------------------------|------------------------|
| **Direction** | Start with main problem, break down | Start with smallest, build up |
| **Implementation** | Recursive with cache | Iterative with table |
| **Computation Order** | On-demand (lazy) | Predetermined (eager) |
| **Stack Usage** | Uses call stack (risk overflow) | No recursion |
| **Memory** | Only stores needed subproblems | Stores all subproblems |
| **Speed** | Recursion overhead | Often faster |

### Top-Down (Memoization) Example: Fibonacci
```python
memo = {}
def fib(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
```

### Bottom-Up (Tabulation) Example: Fibonacci
```python
def fib(n):
    if n <= 1:
        return n
    table = [0] * (n + 1)
    table[1] = 1
    for i in range(2, n + 1):
        table[i] = table[i-1] + table[i-2]
    return table[n]
```

### When to Use Each

| Use Top-Down When | Use Bottom-Up When |
|-------------------|-------------------|
| Not all subproblems needed | All subproblems needed |
| Natural recursive structure | Clear iteration order |
| Easier to implement | Performance critical |
| Deep recursion is safe | Risk of stack overflow |

> **Quiz Focus**: Top-down starts with main problem using recursion with caching. Bottom-up solves smallest subproblems first and iteratively builds to the solution.

---

## 3. Dynamic Programming in Reinforcement Learning

### Model-Based RL with DP

DP is used when we have a **complete model** of the environment:
- Known transition probabilities P(s'|s,a)
- Known reward function R(s,a)

### Key DP Algorithms in RL

| Algorithm | Description | Process |
|-----------|-------------|---------|
| **Policy Evaluation** | Compute V(s) for given policy | Iterative Bellman updates |
| **Policy Improvement** | Make policy greedy w.r.t. V | Choose best action per state |
| **Policy Iteration** | Alternate eval & improvement | Converges to optimal policy |
| **Value Iteration** | Direct optimal value computation | Apply Bellman optimality |

### Policy Iteration Steps
```
1. Initialize policy π randomly
2. Policy Evaluation: Compute V^π for current policy
3. Policy Improvement: Update π to be greedy w.r.t. V^π
4. Repeat 2-3 until π converges
```

### Value Iteration
```
Repeat until convergence:
  For each state s:
    V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
```

> **Quiz Focus**: DP in model-based RL finds optimal policies by iteratively evaluating value functions using the Bellman equation, which requires a complete model of environment dynamics.

---

## 4. DP in Speech Recognition & Part-of-Speech Tagging

### Hidden Markov Models (HMMs)

**Components:**
- **Hidden States**: True underlying states (not observable)
- **Observations**: What we actually see/hear
- **Transition Probabilities**: P(next state | current state)
- **Emission Probabilities**: P(observation | state)

**Example: Doctor Diagnosis**
```
Hidden States: Healthy, Fever
Observations: "Normal", "Cold", "Dizzy"

Task: Infer health states from reported symptoms
```

### The Viterbi Algorithm

**Purpose**: Find most likely sequence of hidden states given observations

**Why DP?**
- Without DP: Exponential paths to check
- With DP: Polynomial time (store best path to each state)

**Process:**
1. **Initialize**: Probability of starting in each state
2. **Recurse**: For each time step, compute best path to each state
3. **Backtrack**: Reconstruct most probable state sequence

### Application Examples

| Application | Hidden States | Observations |
|-------------|---------------|--------------|
| **Speech Recognition** | Phonemes/words | Audio features |
| **POS Tagging** | Grammatical tags | Words |
| **Medical Diagnosis** | Health conditions | Symptoms |

> **Quiz Focus**: DP is used in algorithms like Viterbi to efficiently find the most likely sequence of hidden states (words/tags) given observed data (sounds/words).

---

## 5. Common DP Implementation Pitfalls

### 1. Incorrect State Definition
- **Problem**: State doesn't capture all necessary information
- **Result**: Wrong answers or missing optimal solutions
- **Fix**: Ensure state contains everything needed for decision

### 2. Wrong Recurrence Relation
- **Problem**: Formula doesn't correctly relate subproblems
- **Result**: Incorrect values propagate through computation
- **Fix**: Carefully derive and verify the recurrence

### 3. Infinite Recursion (Top-Down)
- **Problem**: No proper base case or cycle in subproblems
- **Result**: Stack overflow or infinite loop
- **Fix**: Define clear base cases, check for cycles

### 4. Incorrect Base Cases
- **Problem**: Wrong initialization values
- **Result**: All subsequent computations are wrong
- **Fix**: Verify base cases match problem definition

### 5. Wrong Computation Order (Bottom-Up)
- **Problem**: Dependencies not satisfied when needed
- **Result**: Using uncomputed values
- **Fix**: Ensure dependencies are computed first

> **Quiz Focus**: Common pitfalls include incorrectly defining state representation or recurrence relation, leading to wrong results, infinite recursion, or incorrect calculations.

---

## 6. DP Benefits and Limitations

### Benefits

| Benefit | Description |
|---------|-------------|
| **Optimal Solutions** | Guaranteed optimal when applicable |
| **Efficiency** | Avoids redundant computation |
| **Hierarchical** | Natural for nested decision problems |
| **Composable** | Subproblem solutions combine cleanly |
| **Interpretable** | Can trace through value functions |

### Limitations (Especially for Deep RL)

| Limitation | Description |
|------------|-------------|
| **State Space Explosion** | Infeasible for large/continuous spaces |
| **Model Dependency** | Requires known transition/reward functions |
| **Computational Cost** | Must sweep entire state space |
| **No Generalization** | Treats each state independently |

### When DP Works vs When to Use Alternatives

| Use DP | Use Model-Free RL |
|--------|-------------------|
| Small discrete state space | Large/continuous states |
| Known model | Unknown dynamics |
| Planning problems | Learning from interaction |
| Exact solution needed | Approximation acceptable |

---

## 7. Classic DP Problems

| Problem | State | Recurrence |
|---------|-------|------------|
| **Fibonacci** | n | F(n) = F(n-1) + F(n-2) |
| **Coin Change** | amount | min coins for amount |
| **Longest Common Subsequence** | (i, j) positions | LCS of prefixes |
| **Knapsack** | (item, capacity) | max value |
| **Edit Distance** | (i, j) positions | min operations |

---

## Quick Review Questions

1. **Q**: What is Dynamic Programming?
   **A**: An optimization technique that solves complex problems by breaking them into overlapping subproblems, solving each once, and storing solutions.

2. **Q**: What's the difference between top-down and bottom-up DP?
   **A**: Top-down uses recursion with memoization (solves on demand). Bottom-up uses iteration with tabulation (solves smallest first).

3. **Q**: How is DP used in reinforcement learning?
   **A**: In model-based RL to find optimal policies by iteratively computing value functions using the Bellman equation, requiring a complete environment model.

4. **Q**: How does DP apply to speech recognition/POS tagging?
   **A**: Through algorithms like Viterbi that efficiently find the most likely hidden state sequence (words/tags) given observations (sounds/words).

5. **Q**: What's a common DP implementation pitfall?
   **A**: Incorrectly defining the state representation or recurrence relation, leading to wrong results or infinite recursion.

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **Dynamic Programming** | Optimization via subproblem decomposition |
| **Memoization** | Top-down DP with caching |
| **Tabulation** | Bottom-up DP with table |
| **Bellman Equation** | Recursive value function definition |
| **Policy Evaluation** | Computing value function for a policy |
| **Policy Iteration** | Alternating evaluation and improvement |
| **Value Iteration** | Direct optimal value computation |
| **HMM** | Hidden Markov Model |
| **Viterbi Algorithm** | DP algorithm for most likely state sequence |
| **Optimal Substructure** | Optimal solution uses optimal subproblem solutions |
| **Overlapping Subproblems** | Same subproblems appear multiple times |


