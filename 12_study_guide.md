# Study Guide 12: Monte Carlo & Temporal Difference Methods

## Overview
This study guide covers Monte Carlo methods, Temporal Difference learning, bootstrapping, and when to use each approach in reinforcement learning.

---

## 1. Monte Carlo Methods

### Definition
Monte Carlo (MC) methods learn value functions by **averaging returns** from **complete episodes** of experience. They are **model-free** (don't need transition probabilities).

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Episode-based** | Must wait for episode to end |
| **Model-free** | No need for environment model |
| **Unbiased** | Uses actual returns (no estimation) |
| **High variance** | Returns can vary widely |
| **Sample-based** | Learns from actual experience |

### How MC Works
```
1. Generate complete episode: S₀, A₀, R₁, S₁, A₁, R₂, ... , Sₜ
2. Calculate return G from each state visited
3. Average returns to estimate V(s) or Q(s,a)
```

### Return Calculation
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T

Where:
  G_t = return from time t
  R = rewards
  γ = discount factor
  T = terminal time step
```

### MC Variants

| Variant | Description |
|---------|-------------|
| **First-Visit MC** | Average returns only from first visit to state |
| **Every-Visit MC** | Average returns from all visits to state |

> **Quiz Focus**: MC methods learn value functions by averaging returns from complete, finished episodes without needing an environment model.

---

## 2. Temporal Difference (TD) Learning

### Definition
TD learning updates value estimates **after each step** using observed rewards plus **estimates of future values** (bootstrapping). It combines ideas from MC and DP.

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Step-by-step** | Updates after each transition |
| **Model-free** | No need for environment model |
| **Bootstrapping** | Uses own estimates for updates |
| **Biased** | Estimates depend on other estimates |
| **Low variance** | More stable updates |
| **Online** | Can learn during episode |

### TD(0) Update Rule
```
V(S_t) ← V(S_t) + α [R_{t+1} + γV(S_{t+1}) - V(S_t)]
                    └────────────────────────────────┘
                              TD Target

Where:
  α = learning rate
  R_{t+1} = immediate reward
  γV(S_{t+1}) = discounted estimate of next state
  TD Error = [R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

> **Quiz Focus**: TD learning is model-free, updates at each step using immediate reward plus estimated value of next state (bootstrapping).

---

## 3. What is Bootstrapping?

### Definition
**Bootstrapping** means updating an estimate based on **other learned estimates** rather than waiting for the actual final outcome.

### Bootstrapping in TD
```
TD uses: V(S_t) ← V(S_t) + α[R + γV(S_{t+1}) - V(S_t)]
                                    ↑
                            This is an ESTIMATE!

MC uses: V(S_t) ← V(S_t) + α[G_t - V(S_t)]
                               ↑
                        This is the ACTUAL return
```

### Why Bootstrapping Matters

| Advantage | Disadvantage |
|-----------|--------------|
| Learn before episode ends | Introduces bias |
| Lower variance | Depends on estimate quality |
| Works in continuing tasks | Initial estimates affect learning |
| More sample efficient | Can propagate errors |

> **Quiz Focus**: Bootstrapping means updating estimates based on other learned estimates, rather than waiting for the complete actual outcome.

---

## 4. TD(0) vs Monte Carlo: Update Rules

### Side-by-Side Comparison

| Aspect | TD(0) | Monte Carlo |
|--------|-------|-------------|
| **When updated** | After EACH step | After COMPLETE episode |
| **Target** | R + γV(s') (estimated) | G_t (actual return) |
| **Bias** | Biased (uses estimates) | Unbiased |
| **Variance** | Lower | Higher |
| **Bootstrapping** | Yes | No |
| **Online learning** | Yes | No |

### Update Formulas
```
TD(0):
  V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
           └─────────────────────────────────────────┘
           Update after EACH step with estimated future

MC:
  V(S_t) ← V(S_t) + α[G_t - V(S_t)]
           └─────────────────────┘
           Update at END with actual return G_t
```

### Visual Timeline
```
Episode: S₀ → S₁ → S₂ → S₃ → Terminal

TD updates:
  S₀ updated after S₀→S₁
  S₁ updated after S₁→S₂
  S₂ updated after S₂→S₃
  (learns as it goes!)

MC updates:
  Wait until Terminal...
  Then update S₀, S₁, S₂ all at once
  (uses complete trajectory)
```

> **Quiz Focus**: TD(0) updates each step using immediate reward + estimated next state value. MC updates only at episode end using complete actual return.

---

## 5. When to Use MC vs TD

### Monte Carlo Preferred When:

| Scenario | Why MC? |
|----------|---------|
| **Episodic tasks with clear endings** | Can wait for complete return |
| **Easy to collect full trajectories** | Episodes are short |
| **Final outcome most informative** | Intermediate rewards less meaningful |
| **Bootstrapping bias problematic** | Need unbiased estimates |
| **Simple implementation needed** | MC is conceptually simpler |

### TD Preferred When:

| Scenario | Why TD? |
|----------|---------|
| **Continuing (non-episodic) tasks** | No episode end to wait for |
| **Very long episodes** | Can't wait for completion |
| **Online learning required** | Must update during episode |
| **Low variance critical** | Bootstrapping reduces variance |
| **Sample efficiency important** | TD uses data more efficiently |

### Summary Decision Guide
```
Is the task episodic with clear endings?
├── Yes → Are episodes short?
│         ├── Yes → Either can work; MC simpler
│         └── No → TD better (don't wait for long episodes)
└── No (continuing task) → Must use TD
```

> **Quiz Focus**: MC is preferred for strictly episodic tasks with clear start/end where full trajectories are easy to collect. Not suitable for continuous/very long tasks.

---

## 6. TD Algorithms: SARSA vs Q-Learning

### SARSA (On-Policy)
```
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                          ↑
                   Actual next action taken

On-policy: Learns value of policy being followed
```

### Q-Learning (Off-Policy)
```
Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                          ↑
                   Best possible action (greedy)

Off-policy: Learns optimal policy while following exploratory policy
```

### Comparison

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| **Policy type** | On-policy | Off-policy |
| **Next action** | Actual A' taken | max over all actions |
| **Learning target** | Current policy value | Optimal policy value |
| **Exploration impact** | Affects learned values | Doesn't affect Q-values |

---

## 7. Exploration-Exploitation in MC vs TD

### How Each Handles It

| Method | Exploration Strategy |
|--------|---------------------|
| **MC** | Often uses exploring starts or ε-soft policies |
| **TD (SARSA)** | On-policy ε-greedy |
| **TD (Q-learning)** | Off-policy (can explore freely) |

### Adaptation Speed

| Aspect | MC | TD |
|--------|-----|-----|
| **Feedback timing** | End of episode | Each step |
| **Adaptation** | Slower (wait for completion) | Faster (immediate updates) |
| **New information** | Averaged over trajectory | Incorporated immediately |

---

## 8. Bias-Variance Tradeoff

### MC: High Variance, No Bias
```
+ Uses actual returns (unbiased)
- Returns can vary wildly (high variance)
- Needs many samples to converge
```

### TD: Low Variance, Some Bias
```
+ Estimates are more stable (lower variance)
- Depends on accuracy of value estimates (biased)
+ Converges faster with fewer samples
```

### TD(λ): The Middle Ground
- Combines MC and TD
- λ = 0: Pure TD(0)
- λ = 1: Pure MC
- 0 < λ < 1: Blend of both

---

## Quick Review Questions

1. **Q**: What are Monte Carlo methods in RL?
   **A**: Algorithms that learn value functions by averaging returns from complete episodes, without needing an environment model.

2. **Q**: What is Temporal Difference learning?
   **A**: A model-free method that updates value estimates at each step using immediate reward plus estimated value of next state (bootstrapping).

3. **Q**: What is bootstrapping in TD methods?
   **A**: Updating value estimates based on other learned estimates rather than waiting for the complete actual outcome.

4. **Q**: What's the key difference between TD(0) and MC update rules?
   **A**: TD(0) updates each step using R + γV(s'). MC updates only at episode end using complete actual return G.

5. **Q**: When are MC methods preferred over TD?
   **A**: For strictly episodic tasks with clear endings where full trajectories are easy to collect. Not suitable for continuous tasks.

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **Monte Carlo (MC)** | Learning from complete episode returns |
| **Temporal Difference (TD)** | Learning from step-by-step transitions |
| **Bootstrapping** | Using estimates to update other estimates |
| **Return (G)** | Total discounted reward from a state |
| **TD Error** | Difference between TD target and current estimate |
| **TD Target** | R + γV(s') - what we update toward |
| **First-Visit MC** | Average returns from first visit only |
| **Every-Visit MC** | Average returns from all visits |
| **SARSA** | On-policy TD using actual next action |
| **Q-Learning** | Off-policy TD using max over actions |
| **On-Policy** | Learn value of policy being followed |
| **Off-Policy** | Learn optimal policy while following different policy |
| **Bias** | Systematic error in estimates |
| **Variance** | How much estimates fluctuate |
| **TD(λ)** | Blend of TD and MC controlled by λ |

---

## Summary Comparison Table

| Feature | Monte Carlo | TD Learning |
|---------|-------------|-------------|
| **Updates** | End of episode | Each step |
| **Target** | Actual return G | R + γV(s') |
| **Bootstrapping** | No | Yes |
| **Bias** | Unbiased | Biased |
| **Variance** | High | Low |
| **Model needed** | No | No |
| **Online learning** | No | Yes |
| **Works for continuing tasks** | No | Yes |
| **Sample efficiency** | Lower | Higher |

