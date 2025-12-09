# Study Guide 07: Convolutional Neural Networks (CNNs)

## Overview
This study guide covers the key concepts for Quiz 07 on Convolutional Neural Networks, focusing on their architecture, advantages over fully connected networks, and applications in Deep Reinforcement Learning.

---

## 1. CNNs vs Fully Connected Networks

### Key Differences

| Feature | CNN | Fully Connected Network |
|---------|-----|------------------------|
| **Connectivity** | Local (small regions) | Every neuron to every neuron |
| **Parameters** | Shared weights (few parameters) | Unique weights (many parameters) |
| **Spatial Awareness** | Preserves spatial structure | Flattens input, loses spatial info |
| **Scalability** | Efficient for large inputs | Parameter explosion with large inputs |

### Five Key Advantages of CNNs

1. **Local Connectivity**
   - Each neuron connects only to a small region called the **receptive field**
   - Inspired by human visual cortex
   - Dramatically reduces parameters and connections

2. **Parameter Sharing**
   - Same filter (kernel) scans entire input
   - A 5×5 filter reuses just 25 weights across entire image
   - Enables position-independent feature detection

3. **Spatial Hierarchies**
   - Early layers detect basic features (edges, gradients)
   - Deeper layers learn complex patterns (shapes, objects)
   - FCNs flatten input → lose all spatial information

4. **Reduced Structure**
   - A single filter may have only dozens of parameters
   - Same input in FCN would require millions of parameters

5. **Translation Invariance**
   - Features recognized regardless of position in image
   - Same filter applied everywhere → same feature detected anywhere

> **Quiz Focus**: CNNs use convolutional layers to capture spatial features, while fully connected networks treat all inputs equally. CNNs use local receptive fields and shared weights.

---

## 2. Pooling and Activation Functions

### Pooling Layers

**Purpose**: Reduce spatial dimensions of feature maps

| Pooling Type | Description |
|--------------|-------------|
| **Max Pooling** | Takes maximum value from local region |
| **Average Pooling** | Takes average value from local region |

**Key Benefits**:
- **Dimensionality Reduction**: 2×2 pooling with stride 2 reduces data by 75%
- **Feature Invariance**: Detects features regardless of exact position
- **Translation Invariance**: If feature moves slightly, output may stay same
- **Computational Efficiency**: Fewer parameters in subsequent layers
- **Robustness**: Less sensitive to noise and small distortions

### Activation Functions

**Purpose**: Introduce non-linearity for learning complex patterns

| Function | Description |
|----------|-------------|
| **ReLU** | Outputs 0 for negative, input value for positive |
| **Sigmoid** | Squashes values to (0,1) |
| **Tanh** | Squashes values to (-1,1) |

**Why Non-linearity Matters**:
- Without it, network is just a linear system
- Cannot model complex real-world relationships
- ReLU helps mitigate vanishing gradient problem
- Creates sparse representations (many neurons inactive)

> **Quiz Focus**: Pooling simplifies feature maps by reducing spatial size; activation functions introduce non-linearity to learn complex patterns.

---

## 3. CNNs for Visual Input in Reinforcement Learning

### Why CNNs Excel in RL

1. **Spatial Hierarchy**
   - Early layers: edges, corners
   - Deeper layers: shapes, objects, scenes
   - Mimics human visual perception
   - Agent builds compositional understanding of visual state

2. **Parameter Efficiency**
   - Sparse connections via convolutional layers
   - Same filter applied everywhere (parameter sharing)
   - Less overfitting, faster training
   - Feasible for resource-constrained environments

3. **Translation Invariance**
   - Objects recognized regardless of position
   - Critical in dynamic environments where objects move
   - Stable policies without retraining for every position

4. **Automatic Feature Extraction**
   - Learns relevant features from raw pixels
   - No manual feature engineering needed
   - Captures color, shape, texture automatically
   - Focuses on what matters (paths, obstacles) ignores background

5. **Scalability**
   - Handles high-resolution inputs efficiently
   - No parameter explosion with larger inputs
   - Compatible with Q-learning, policy gradient methods

> **Quiz Focus**: CNNs extract spatial features efficiently, allowing agents to recognize patterns and objects in complex visual observations.

---

## 4. Popular CNN Architectures

### Evolution of CNN Architectures

| Architecture | Year | Key Innovation |
|--------------|------|----------------|
| **LeNet-5** | 1990s | First successful CNN (digit recognition) |
| **AlexNet** | 2012 | Deep CNNs, ReLU, Dropout |
| **VGG** | 2014 | Simple 3×3 stacked convolutions |
| **GoogLeNet** | 2014 | Inception module (multi-scale) |
| **ResNet** | 2015 | Residual connections (skip connections) |

### Architecture Details

**LeNet-5**
- 7 layers: 2 conv + 2 pooling + 3 FC
- Handwritten digit recognition
- Foundation for modern CNNs

**AlexNet**
- 5 conv layers + 3 FC layers
- First to use ReLU activation
- Introduced dropout regularization
- Won ImageNet 2012

**VGG (VGG-16, VGG-19)**
- Stacked 3×3 filters
- Deeper but uniform structure
- Good for transfer learning
- Computationally expensive

**GoogLeNet (Inception)**
- Inception module: parallel 1×1, 3×3, 5×5 convolutions
- Multi-scale feature learning
- 1×1 convolutions reduce computation
- Auxiliary classifiers combat vanishing gradients

**ResNet (Residual Networks)**
- **Skip connections**: gradients flow directly through network
- Learns residual mapping (easier to optimize)
- Enabled 152+ layer networks
- Solved vanishing gradient problem

> **Quiz Focus**: AlexNet, VGG, ResNet, and Inception are landmark architectures for image classification and feature learning.

---

## 5. Problems CNNs Solved

### Limitations of Fully Connected Networks

| Problem | FCN Issue | CNN Solution |
|---------|-----------|--------------|
| **Computational Complexity** | Every neuron connected to all | Local connectivity + shared weights |
| **Overfitting** | Too many parameters | Pooling, dropout, fewer params |
| **No Spatial Features** | Flattens input | Convolutional layers preserve structure |
| **No Translation Invariance** | Must relearn at every position | Shared filters detect anywhere |
| **Scalability** | Parameter explosion | Efficient for any input size |

### Example: 100×100 Image
- **FCN**: 10,000 input neurons × 500 hidden = **5 million weights**
- **CNN**: 5×5 filter = only **25 weights** (reused across entire image)

> **Quiz Focus**: CNNs reduce parameter count through weight sharing and preserve spatial relationships, making them ideal for high-dimensional image inputs.

---

## 6. CNN Architecture Components

### Core Building Blocks

1. **Convolutional Layer**
   - Learnable filters slide over input
   - Produces feature maps
   - Detects patterns (edges → shapes → objects)

2. **Activation Layer**
   - Non-linear transformation
   - ReLU most common: max(0, x)
   - Enables complex pattern learning

3. **Pooling Layer**
   - Downsamples feature maps
   - Max pooling retains strongest signal
   - Reduces computation, adds invariance

4. **Fully Connected Layer**
   - At end of network
   - Maps features to outputs (actions, Q-values)
   - Makes final predictions

5. **Normalization Layer (Batch Norm)**
   - Normalizes to mean 0, variance 1
   - Stabilizes training
   - Addresses internal covariate shift

---

## 7. Cross-Correlation Operation

### How Convolution Works

1. **Input Feature Map**: Image or previous layer output
2. **Kernel (Filter)**: Small learnable weight matrix (e.g., 3×3)
3. **Operation**: Slide kernel, compute element-wise products, sum
4. **Output Feature Map**: Shows where features are detected

**Mathematical Formula**:
```
g(i,j) = Σ Σ f(i+m, j+n) × k(m,n)
```
Where f = input, k = kernel

**Parameters**:
- **Stride**: How much kernel moves each step
- **Padding**: Zeros added to preserve edge information

---

## 8. Convolution as Sparsity

### Key Concept
Each neuron connects to only a small local region, not entire input.

**Benefits of Sparse Connectivity**:
- **Local Receptive Fields**: Focus on spatially localized features
- **Reduced Parameters**: 3×3 kernel = 9 connections vs 10,000 for FCN
- **Efficient Computation**: Fewer multiply-adds
- **Preserved Spatial Hierarchy**: Maintains 2D structure throughout

---

## Quick Review Questions

1. **Q**: How does a CNN differ from a fully connected network?
   **A**: CNNs use local connectivity and shared weights to capture spatial features; FCNs connect every neuron to every neuron and lose spatial information.

2. **Q**: What do pooling layers do?
   **A**: Reduce spatial dimensions, decrease parameters, and add translation invariance.

3. **Q**: What do activation functions do?
   **A**: Introduce non-linearity so networks can learn complex patterns.

4. **Q**: Why are CNNs good for RL?
   **A**: They efficiently extract spatial features from raw pixels, enabling agents to learn visual patterns for decision-making.

5. **Q**: Name four popular CNN architectures.
   **A**: AlexNet, VGG, ResNet, Inception (GoogLeNet)

6. **Q**: What problem do skip connections solve?
   **A**: Vanishing gradients in deep networks (used in ResNet)

---

## Key Terms Glossary

| Term | Definition |
|------|------------|
| **Receptive Field** | The region of input a neuron "sees" |
| **Kernel/Filter** | Small learnable weight matrix |
| **Feature Map** | Output of convolution showing detected features |
| **Stride** | How much the filter moves each step |
| **Padding** | Zeros added around input edges |
| **Translation Invariance** | Detecting features regardless of position |
| **ReLU** | Rectified Linear Unit: max(0, x) |
| **Max Pooling** | Takes maximum value in local region |
| **Skip Connection** | Direct path for gradients (ResNet) |
| **Batch Normalization** | Normalizes layer inputs for stable training |

