# Flashcards for Study Review

## Quiz 07: CNNs

### Card 1: CNN vs Fully Connected Network
**Q**: How does a CNN differ from a fully connected network?

**A**: CNNs use local connectivity (each neuron connects only to a small receptive field), parameter sharing (same filters slide across the entire image), and preserve spatial structure. Fully connected networks connect every neuron to every neuron, lose spatial information by flattening input, and require far more parameters.

**Status**: ✅ Mastered

---

### Card 2: Pooling and Activation Functions
**Q**: What roles do pooling and activation functions play in CNNs?

**A**: 
- **Pooling**: Reduces spatial dimensions (e.g., 2×2 pooling), decreasing parameters and computational cost, filtering irrelevant details, and providing translation invariance (recognizes features regardless of exact position).
- **Activation functions** (like ReLU): Introduce non-linearity, which is critical because without it, multiple layers would be equivalent to a single linear layer. This enables the network to learn complex, non-linear patterns.

**Status**: ✅ Mastered

---

### Card 3: CNNs for Visual Input in RL
**Q**: Why are CNNs suitable for visual input in reinforcement learning environments?

**A**: 
1. **Spatial Hierarchy**: Builds complexity layer by layer (edges → shapes → objects), allowing the agent to understand its environment compositionally.
2. **Automatic Feature Extraction**: Learns relevant features from raw pixels without manual feature engineering, enabling the agent to understand what it's seeing.
3. **Translation Invariance**: Recognizes features regardless of their position in the image, which is critical in dynamic RL environments where objects move around.

**Status**: ✅ Mastered

---

### Card 4: Popular CNN Architectures
**Q**: What are examples of popular CNN architectures used in computer vision?

**A**: 
1. **AlexNet** (2012): First to use ReLU activation function; won ImageNet 2012; 5 convolutional + 3 fully connected layers
2. **ResNet** (2015): Uses skip/residual connections that allow information to bypass layers; enables training of very deep networks (152+ layers); solves vanishing gradient problem
3. **VGG** (2014): Stacks many 3×3 convolutional filters in a simple, uniform structure; good for transfer learning

**Status**: ✅ Mastered

---

_More flashcards will be added as you complete questions..._

