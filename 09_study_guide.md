# Study Guide 09: Recursive Neural Networks (RvNNs)

## Overview
This study guide covers Recursive Neural Networks, their distinction from Recurrent Neural Networks, and their application to hierarchical/tree-structured data.

---

## 1. What is a Recursive Neural Network (RvNN)?

### Definition
A Recursive Neural Network applies the **same set of weights recursively** over a **hierarchical, tree-like data structure** to learn representations of structured data.

### Key Characteristics
- Processes **tree-structured** data (not sequential)
- Applies **shared composition function** at each node
- Builds representations **bottom-up** from leaves to root
- Ideal for data with natural hierarchy (parse trees, scene graphs)

### Example: Sentence Processing
```
          [Sentence]          ← Root representation
           /      \
       [NP]       [VP]        ← Phrase representations
       /  \       /   \
    [The] [cat] [sat] [down]  ← Word embeddings (leaves)
```

> **Quiz Focus**: An RvNN applies the same weights recursively over hierarchical tree structures to learn representations of structured data like parse trees.

---

## 2. RvNN vs RNN: Key Differences

| Feature | Recursive NN (RvNN) | Recurrent NN (RNN) |
|---------|---------------------|-------------------|
| **Data Structure** | Tree/hierarchical | Sequential/linear |
| **Processing Direction** | Bottom-up (leaves → root) | Left-to-right (time) |
| **Dependencies** | Structural (parent-child) | Temporal (past → present) |
| **Information Flow** | Along tree branches | Along time axis |
| **Applications** | NLP parsing, scene graphs | Time series, speech |
| **Training** | Backprop Through Structure | Backprop Through Time |

### Visual Comparison
```
RNN (Sequential):
x₁ → h₁ → h₂ → h₃ → h₄
      ↑     ↑     ↑     ↑
     x₁    x₂    x₃    x₄

RvNN (Tree):
         [root]
        /      \
    [node]    [node]
    /    \        \
 [leaf] [leaf]  [leaf]
```

> **Quiz Focus**: RvNNs operate on hierarchical tree structures based on syntactic relationships, while RNNs process data in linear sequences based on time/position.

---

## 3. How RvNNs Process Tree-Structured Input

### Bottom-Up Composition

**Process:**
1. **Start at leaves**: Convert each leaf (word/token) to vector embedding
2. **Combine children**: Apply shared composition function to child vectors
3. **Create parent**: Produce new representation for parent node
4. **Repeat recursively**: Continue until reaching root
5. **Root = final representation**: Use for downstream tasks

### Mathematical Formulation
```
For binary tree at each parent node:
  p = f(W · [c₁; c₂] + b)

Where:
  c₁, c₂ = child representations
  W = shared weight matrix
  b = bias
  f = activation function (tanh, ReLU)
  p = parent representation
```

### Example: Sentiment Analysis
```
Input: "not very good"

Step 1: Word embeddings
  "not" → v₁, "very" → v₂, "good" → v₃

Step 2: Combine "very" + "good"
  p₁ = f(W · [v₂; v₃] + b)  → "very good" (positive)

Step 3: Combine "not" + p₁
  p₂ = f(W · [v₁; p₁] + b)  → "not very good" (negative)

Final: p₂ captures negation effect!
```

> **Quiz Focus**: RvNNs start at leaves and recursively apply shared weights at each parent node, combining children representations until reaching the root.

---

## 4. Training: Backpropagation Through Structure (BPTS)

### Definition
**BPTS** is the training algorithm for RvNNs, propagating error signals from root down through the tree structure.

### Process
1. **Forward pass**: Compute representations bottom-up
2. **Compute loss**: At root (or any labeled node)
3. **Backpropagate**: From root down to leaves
4. **Follow tree paths**: Calculate gradients at each node
5. **Accumulate gradients**: Same weights used everywhere, so sum gradients

### Comparison to BPTT

| BPTS (RvNN) | BPTT (RNN) |
|-------------|------------|
| Traverse tree structure | Traverse time steps |
| Variable branch depths | Linear chain |
| Children → parent paths | Past → present paths |
| Structural dependencies | Temporal dependencies |

### Challenges
- **Variable structure**: Each input has different tree shape
- **Gradient issues**: Deep trees cause vanishing/exploding gradients
- **Batching difficulty**: Hard to batch different tree structures

> **Quiz Focus**: BPTS propagates error signals from the root down to the leaves, recursively calculating gradients at each node in the structure.

---

## 5. Limitations of Recursive Neural Networks

### Primary Challenge: Pre-parsed Structure Required

**The Problem:**
- RvNNs require input data **already organized as trees**
- This requires **external parsing** (e.g., syntactic parser for NLP)
- Parsing can be **error-prone** and introduce noise
- Many data types **don't naturally fit** tree structures

### Other Limitations

| Limitation | Description |
|------------|-------------|
| **Parsing Dependency** | Need pre-computed tree structure |
| **Parallelism Constraints** | Sequential dependency limits GPU efficiency |
| **High Latency** | Deep trees = slow inference |
| **Memory Overhead** | Store intermediate states for each node |
| **Structural Rigidity** | Can't handle unstructured/noisy data |
| **Training Complexity** | Variable shapes complicate batching |

### Comparison: When to Use Each

| Use RvNN When | Use RNN When |
|---------------|--------------|
| Data has natural hierarchy | Data is sequential |
| Parse trees available | No structural parsing |
| Compositionality matters | Order/time matters |
| Structure aids interpretation | Simple sequence processing |

> **Quiz Focus**: A primary limitation is that RvNNs require input data to be pre-parsed into tree structure, which is complex, error-prone, and many data types don't naturally fit hierarchical formats.

---

## 6. Applications Where RvNNs Excel

### 1. Sentiment Analysis
- Captures how modifiers affect sentiment
- "not good" vs "very good" vs "not very good"
- Structure reveals scope of negation/intensification

### 2. Syntax Parsing
- Natural fit for grammatical structure
- Learn representations for phrases/clauses
- Score parse tree quality

### 3. Scene Understanding
- Process scene graphs hierarchically
- Objects → object groups → scenes
- Capture part-whole relationships

### Additional Applications
- Mathematical expression analysis
- Code/AST understanding
- Semantic role labeling
- Question answering with structure

---

## 7. Key Architectural Concepts

### Shared Weights
- **Same parameters** at every composition step
- Reduces parameter count
- Enables **compositional consistency**
- Mirrors how humans apply same rules everywhere

### Recursive Representation
- Each node's vector summarizes **entire subtree below**
- Root vector represents **entire input**
- Intermediate nodes = meaningful subcomponents
- Enables **interpretability** (can inspect any node)

---

## Quick Review Questions

1. **Q**: What is a Recursive Neural Network?
   **A**: A neural network that applies the same weights recursively over hierarchical tree-like data structures to learn structured representations.

2. **Q**: How does RvNN differ from RNN?
   **A**: RvNNs operate on tree structures based on structural relationships; RNNs process linear sequences based on time/position.

3. **Q**: How does an RvNN process tree-structured input?
   **A**: Bottom-up: starts at leaves, recursively applies shared composition function at parent nodes, combining children until reaching root.

4. **Q**: How is backpropagation performed in RvNNs?
   **A**: Backpropagation Through Structure (BPTS) - error signals propagate from root down to leaves through the tree structure.

5. **Q**: What is the main limitation of RvNNs?
   **A**: They require pre-parsed tree structures, which is complex, error-prone, and many data types don't naturally fit hierarchical formats.

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **RvNN** | Recursive Neural Network - processes tree structures |
| **RNN** | Recurrent Neural Network - processes sequences |
| **Bottom-Up Composition** | Building representations from leaves to root |
| **BPTS** | Backpropagation Through Structure - training for RvNNs |
| **Compositionality** | Meaning of whole derived from parts + combination rules |
| **Parse Tree** | Tree showing grammatical structure of sentence |
| **Constituency Tree** | Parse tree with phrase types (NP, VP, etc.) |
| **Dependency Tree** | Parse tree with grammatical relations (subject, object) |
| **Scene Graph** | Hierarchical representation of image objects/relations |
| **Shared Weights** | Same parameters used at all composition steps |

