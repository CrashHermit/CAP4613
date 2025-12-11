# Study Guide 08: Recurrent Neural Networks (RNNs)

## Overview
This study guide covers Recurrent Neural Networks, Time-Delay Neural Networks, and their applications in Deep Reinforcement Learning for processing sequential data.

---

## 1. What are Recurrent Neural Networks?

### Definition
RNNs are neural networks designed for **sequential data** (time-series, text, speech) that maintain an **internal hidden state** acting as memory of past inputs.

### Key Features
- **Feedback connections**: Output/hidden states fed back as input
- **Temporal memory**: Hidden state summarizes sequence history
- **Shared weights**: Same parameters used across all time steps

### How RNNs Work
```
At each time step t:
  h_t = f(W_xh * x_t + W_hh * h_{t-1} + b_h)
  y_t = g(W_hy * h_t + b_y)

Where:
  x_t = current input
  h_t = hidden state (memory)
  y_t = output
  W_hh = recurrent weights (hidden-to-hidden)
```

> **Quiz Focus**: RNNs maintain an internal hidden state that acts as memory, allowing them to capture information from previous time steps.

---

## 2. RNN Input-Output Architectures

| Architecture | Input | Output | Use Case |
|--------------|-------|--------|----------|
| **One-to-One** | Single | Single | Standard feedforward (no sequence) |
| **One-to-Many** | Single | Sequence | Image captioning, music generation |
| **Many-to-One** | Sequence | Single | Sentiment analysis, action recognition |
| **Many-to-Many (Sync)** | Sequence | Sequence (same length) | Video frame classification, RL actions |
| **Many-to-Many (Enc-Dec)** | Sequence | Sequence (different length) | Machine translation, instruction following |

### Examples in Deep RL
- **Many-to-One**: Process observation sequence → single action decision
- **Many-to-Many (Sync)**: Output action at each timestep based on observation history
- **Encoder-Decoder**: Receive instruction sequence → generate action sequence

> **Quiz Focus**: Main architectures are one-to-one, one-to-many, many-to-one, and many-to-many (synchronized or encoder-decoder).

---

## 3. Unfolding/Unrolling RNNs

### What is Unfolding?
Converting the RNN's cyclic graph into a **deep feedforward network** where each time step becomes a layer.

### Visualization
```
Folded (compact):     Unfolded (across time):
    ┌──┐              h₀ → h₁ → h₂ → h₃ → ...
    │  │                ↑     ↑     ↑     ↑
x → [RNN] → y         x₀    x₁    x₂    x₃
    └──┘                ↓     ↓     ↓     ↓
                       y₀    y₁    y₂    y₃
```

### Key Points
- Same weights (W_xh, W_hh, W_hy) **shared across all time steps**
- Enables standard backpropagation (called BPTT)
- Reveals why gradient problems occur (long chains of multiplications)

> **Quiz Focus**: Unrolling depicts the network as copies for each time step, with hidden state passed forward and same weights applied at every step.

---

## 4. Training: Backpropagation Through Time (BPTT)

### Process
1. **Forward Pass**: Process sequence step-by-step, storing hidden states
2. **Compute Loss**: Compare predictions to targets across all steps
3. **Backpropagate**: Compute gradients from final step backward
4. **Gradient Clipping**: Prevent exploding gradients
5. **Update Weights**: Apply accumulated gradients

### The Challenge with Recurrent Weights (W_hh)

**Why Hidden-to-Hidden Weights are Hardest to Train:**
- Applied **repeatedly** at each time step
- Gradients involve **long chains of multiplications**
- Same weight contributes to loss at EVERY time step
- Gradients must flow back through ENTIRE sequence

```
Gradient flow:
Loss → h_T → h_{T-1} → h_{T-2} → ... → h_1
       ↓         ↓          ↓              ↓
      W_hh     W_hh       W_hh           W_hh
```

> **Quiz Focus**: Hidden-to-hidden weights are difficult because they're applied repeatedly, causing gradients to either shrink (vanishing) or grow exponentially (exploding).

---

## 5. Gradient Problems in RNNs

### Vanishing Gradients
- Gradients **shrink exponentially** as they propagate back
- Network fails to learn **long-term dependencies**
- Signal from distant past becomes too weak

**Cause**: Repeated multiplication by values < 1

### Exploding Gradients
- Gradients **grow exponentially** during backpropagation
- Weight updates become **extremely large**
- Training becomes **unstable**, loss becomes NaN

**Cause**: Repeated multiplication by values > 1

> **Quiz Focus**: Exploding gradient problem occurs when gradients grow exponentially during backpropagation, causing extremely large weight updates and unstable training.

---

## 6. Solutions to Gradient Problems

### Gradient Clipping

| Method | Description |
|--------|-------------|
| **Clip by Value** | Constrain each gradient component to [-c, +c] |
| **Clip by Norm** | If ||g|| > threshold, rescale: g = g * (threshold/||g||) |

**Clip by Norm** is preferred because it preserves gradient direction.

### Advanced Architectures: LSTM & GRU

**Long Short-Term Memory (LSTM)**:
- **Cell state**: Acts as "conveyor belt" for information
- **Forget gate**: Decides what to discard from memory
- **Input gate**: Decides what new information to store
- **Output gate**: Decides what to output

**Gated Recurrent Unit (GRU)**:
- Simplified version with fewer gates
- Combines forget and input into single "update gate"

### Why LSTMs Solve the Problem
1. **Gradient Stability**: Cell state allows gradients to flow unchanged
2. **Memory Control**: Gates learn what to remember/forget
3. **Long-term Dependencies**: Can capture patterns across long sequences

---

## 7. Time-Delay Neural Networks (TDNNs)

### Definition
Feedforward networks that process a **fixed-size window** of sequential input using "delay taps."

### Comparison: TDNN vs RNN

| Feature | TDNN | RNN |
|---------|------|-----|
| **Structure** | Feedforward | Recurrent |
| **Memory** | Fixed window | Theoretically infinite |
| **Training** | Standard backprop | BPTT (more complex) |
| **Gradient Issues** | Minimal | Vanishing/exploding |
| **Flexibility** | Fixed sequence length | Variable length |

### When to Use TDNNs
- Short-horizon temporal patterns
- Stability is critical
- Simpler implementation needed

---

## 8. RNN Applications in Deep RL

### Key Applications

1. **Memory-based Policies**
   - POMDP environments (partial observability)
   - Hidden state acts as belief state

2. **State Estimation/Embedding**
   - Compress observation sequences into informative vectors
   - Feed to policy/value networks

3. **Temporal Credit Assignment**
   - Propagate reward signals backward through time
   - Learn which past actions caused future rewards

4. **Prediction/Model Learning**
   - Learn environment dynamics from sequences
   - Used for planning and simulation

5. **Multi-step Decision Sequencing**
   - Execute coordinated action sequences
   - Long-term planning

---

## Quick Review Questions

1. **Q**: What are RNNs and what problems do they solve?
   **A**: RNNs process sequential data by maintaining a hidden state that acts as memory, allowing them to capture temporal dependencies.

2. **Q**: What are the main RNN input-output architectures?
   **A**: One-to-one, one-to-many, many-to-one, and many-to-many (synchronized or encoder-decoder).

3. **Q**: How is an RNN "unrolled"?
   **A**: By depicting it as copies for each time step, with hidden state passed between steps and same weights applied throughout.

4. **Q**: Why are hidden-to-hidden weights difficult to train?
   **A**: They're applied repeatedly at each step, causing gradients to vanish or explode during backpropagation.

5. **Q**: What is the Exploding Gradient Problem?
   **A**: Gradients grow exponentially during backpropagation, causing extremely large weight updates and training instability.

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **Hidden State** | Internal memory vector updated at each time step |
| **BPTT** | Backpropagation Through Time - training algorithm for RNNs |
| **Vanishing Gradient** | Gradients shrink to zero over long sequences |
| **Exploding Gradient** | Gradients grow exponentially, causing instability |
| **Gradient Clipping** | Limiting gradient magnitude to prevent explosion |
| **LSTM** | Long Short-Term Memory - gated RNN architecture |
| **GRU** | Gated Recurrent Unit - simplified gated RNN |
| **Cell State** | LSTM's memory that can pass information unchanged |
| **TDNN** | Time-Delay Neural Network - feedforward with fixed window |
| **POMDP** | Partially Observable MDP - requires memory for decisions |


