# Study Guide 10: Reinforcement Learning Fundamentals

## Overview
This study guide covers the foundations of Reinforcement Learning, including heuristic search, comparison with supervised learning, and core RL concepts using examples like Tic-Tac-Toe.

---

## 1. Heuristic Search Algorithms

### Definition
Algorithms that use **domain-specific knowledge** (heuristics) to guide search, making it more efficient than brute-force methods by prioritizing promising paths.

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Efficiency** | Narrows search space significantly |
| **Approximation** | Finds "good enough" solutions, not necessarily optimal |
| **Domain Knowledge** | Uses problem-specific insights (educated guesses) |
| **Flexibility** | Adaptable to different problem domains |

### Common Heuristic Search Algorithms

| Algorithm | Description | Key Feature |
|-----------|-------------|-------------|
| **A*** | f(n) = g(n) + h(n) | Combines cost so far + estimated cost to goal |
| **Greedy Best-First** | Uses only h(n) | Fast but not optimal |
| **Simulated Annealing** | Probabilistic, accepts worse solutions with decreasing probability | Escapes local optima |
| **Genetic Algorithms** | Evolves population via crossover/mutation | Good for non-differentiable spaces |
| **Beam Search** | Keeps top-k candidates at each step | Memory efficient |

### When Heuristics Work Well
- Large problem spaces (combinatorial explosion)
- Near-optimal solutions are acceptable
- Imperfect/incomplete information
- Optimization problems
- Pathfinding/navigation
- Constraint satisfaction problems

> **Quiz Focus**: Heuristic search algorithms use domain-specific knowledge (an "educated guess") to guide search, making it more efficient by prioritizing promising paths.

---

## 2. Types of Machine Learning

### Comparison Table

| Type | Data | Goal | Example |
|------|------|------|---------|
| **Supervised** | Labeled input-output pairs | Learn mapping function | Image classification |
| **Unsupervised** | Unlabeled data | Discover patterns/clusters | Customer segmentation |
| **Transductive** | Labeled + specific unlabeled test set | Classify those specific instances | Semi-supervised label propagation |
| **Reinforcement** | Agent-environment interaction | Maximize cumulative reward | Game playing |

### Why Supervised Learning is Prominent
- Can achieve **high accuracy** on wide range of tasks
- Learns directly from **large labeled datasets**
- Reliable method for **prediction and classification**
- Well-established techniques and tools

> **Quiz Focus**: Supervised learning is prominent because it achieves high accuracy by learning directly from large amounts of labeled data.

---

## 3. Reinforcement Learning vs Supervised Learning

### Key Differences

| Aspect | Reinforcement Learning | Supervised Learning |
|--------|----------------------|-------------------|
| **Learning Process** | Trial-and-error interaction | Learning from labeled examples |
| **Feedback** | Scalar rewards (evaluative) | Correct answers (instructive) |
| **Timing** | Often delayed | Immediate |
| **Data** | Agent generates own data | Fixed dataset (i.i.d.) |
| **Objective** | Maximize cumulative reward | Minimize prediction error |
| **Exploration** | Must balance explore/exploit | Not applicable |

### RL Learning Process
```
Agent → Action → Environment
  ↑                    ↓
  └── State, Reward ←──┘
```

### Key RL Concepts
- **Agent**: The learner/decision maker
- **Environment**: What the agent interacts with
- **State**: Current situation
- **Action**: What agent can do
- **Reward**: Feedback signal (+ or -)
- **Policy**: Strategy mapping states to actions

> **Quiz Focus**: RL agents learn through trial-and-error interaction with environment guided by rewards. Supervised learning learns from static labeled datasets with known correct answers.

---

## 4. Tic-Tac-Toe: A Case Study

### Search Space Size

**Naive Calculation:**
- 9 cells, each can be X, O, or empty
- Upper bound: 3⁹ = **19,683** configurations

**Valid Game States:**
- Must follow turn-taking rules
- Game ends when someone wins
- Reachable valid states: ~**5,478**

**Game Tree:**
- Maximum 9 moves per game
- Number of terminal positions: **255,168**

> **Quiz Focus**: The 3×3 grid with each cell being X, O, or empty gives upper bound of 3⁹ = 19,683 configurations.

### RL Approach to Tic-Tac-Toe

**Environment Definition:**
- **State**: Current board configuration (9 cells)
- **Actions**: Place mark in empty cell
- **Rewards**: +1 win, -1 lose, 0 draw

**Learning Process:**
1. Agent plays many games
2. Initially moves are random
3. Associates states/actions with outcomes
4. Gradually prefers winning moves
5. Balances exploration vs exploitation
6. Eventually learns optimal strategy

---

## 5. Exploration vs Exploitation

### The Dilemma
- **Exploitation**: Use known best action (maximize immediate reward)
- **Exploration**: Try new actions (discover potentially better strategies)

### Why Both Matter

| Too Much Exploitation | Too Much Exploration |
|----------------------|---------------------|
| Miss better strategies | Waste time on bad actions |
| Stuck in local optima | Never leverage learning |
| Suboptimal long-term | Poor short-term performance |

### Balancing Strategies

| Strategy | Description |
|----------|-------------|
| **ε-greedy** | Exploit with probability 1-ε, explore with ε |
| **Decaying ε** | Start with high exploration, reduce over time |
| **UCB** | Upper Confidence Bound - explore uncertain actions |
| **Thompson Sampling** | Sample from posterior distribution |
| **Softmax** | Probability proportional to estimated value |

---

## 6. How RL Learns Optimal Strategies Against Any Opponent

### Key Mechanisms

1. **Trial and Error**
   - Experiment with different actions
   - Observe outcomes
   - Adjust strategy based on results

2. **Learning from Rewards**
   - Positive rewards reinforce good moves
   - Negative rewards discourage bad moves
   - Build value estimates over time

3. **Improving Over Time**
   - Policy evolves with experience
   - Value estimates become more accurate
   - Mistakes reduce, performance improves

4. **Self-Play**
   - Agent plays against itself
   - Discovers robust strategies
   - Learns to handle diverse opponent behaviors

5. **Generalization**
   - Learn patterns, not just specific positions
   - Adapt to unseen opponent strategies
   - Develop counter-strategies dynamically

> **Quiz Focus**: RL learns through self-play, exploring vast game variations and being rewarded for winning, discovering robust strategies that generalize to any opponent.

---

## 7. Deep Q-Networks (DQN)

### Core Concepts

**Q-Values**: Expected cumulative reward for taking action in state
```
Q(s, a) = Expected future rewards starting from state s, taking action a
```

### How DQN Works

1. **Experience Replay**
   - Store experiences (s, a, r, s') in replay buffer
   - Sample random mini-batches for training
   - Breaks correlation between sequential samples

2. **Neural Network Approximation**
   - Network takes state as input
   - Outputs Q-value for each action
   - Handles high-dimensional states (pixels)

3. **Target Network**
   - Separate network for computing targets
   - Updated periodically (not every step)
   - Provides stable learning targets

4. **Bellman Equation Update**
   ```
   Q(s,a) ← r + γ * max_a' Q(s', a')
   ```

---

## 8. Stationary vs Non-Stationary Environments

| Stationary | Non-Stationary |
|------------|----------------|
| Dynamics don't change | Dynamics change over time |
| Fixed transition probabilities | Shifting rewards/transitions |
| Can learn fixed optimal policy | Must continuously adapt |
| Example: Chess rules | Example: Stock market |

---

## Quick Review Questions

1. **Q**: What are Heuristic Search algorithms?
   **A**: Algorithms that use domain-specific knowledge (heuristics) to guide search efficiently by prioritizing promising paths.

2. **Q**: Why is Supervised Learning prominent today?
   **A**: It achieves high accuracy by learning directly from large labeled datasets, making it reliable for prediction and classification.

3. **Q**: What's the key difference between RL and Supervised Learning?
   **A**: RL learns through trial-and-error with reward signals; supervised learning learns from labeled examples with known correct answers.

4. **Q**: What's the search space size in Tic-Tac-Toe?
   **A**: Upper bound is 3⁹ = 19,683 (each of 9 cells can be X, O, or empty).

5. **Q**: How does RL learn to play against any opponent?
   **A**: Through self-play, exploring many game variations, being rewarded for winning, and discovering robust strategies that generalize.

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **Heuristic** | Rule of thumb or educated guess guiding search |
| **A*** | Search algorithm combining cost and estimated distance |
| **Agent** | The RL learner/decision maker |
| **Environment** | What the agent interacts with |
| **State** | Current situation representation |
| **Action** | What the agent can do |
| **Reward** | Feedback signal indicating action quality |
| **Policy** | Mapping from states to actions |
| **Q-Value** | Expected cumulative reward for state-action pair |
| **Exploration** | Trying new actions to discover better strategies |
| **Exploitation** | Using known best actions |
| **ε-greedy** | Explore with probability ε, exploit otherwise |
| **Self-play** | Agent learns by playing against itself |
| **DQN** | Deep Q-Network - neural network for Q-learning |


