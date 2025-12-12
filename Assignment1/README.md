# Assignment 1: Multi-layer Perceptron Implementation

## 📖 Overview
This assignment focuses on implementing a neural network classifier from scratch using only NumPy. The goal is to build a deep understanding of forward propagation, backpropagation, and gradient descent by implementing these algorithms without relying on deep learning frameworks.

## 🎯 Learning Objectives
- Implement forward propagation for multi-layer networks
- Derive and implement backpropagation algorithm
- Understand gradient descent optimization
- Perform gradient checking for numerical verification
- Apply the network to CIFAR-10 classification

## 📊 Dataset
- **Dataset:** CIFAR-10
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Images:** 50,000
- **Test Images:** 10,000
- **Image Size:** 32×32×3 (RGB)

## 🧠 Network Architecture
### Basic Implementation (Assignment1.ipynb)
- **Input Layer:** 3072 neurons (32×32×3 flattened)
- **Hidden Layer:** 50 neurons with ReLU activation
- **Output Layer:** 10 neurons with softmax activation
- **Loss Function:** Cross-entropy loss
- **Optimizer:** Mini-batch gradient descent

### Bonus Implementation (Assignment1_bonus.ipynb)
- Extended architecture with multiple hidden layers
- Different activation functions comparison
- Advanced initialization techniques
- Learning rate scheduling experiments

## 📈 Key Results
| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~XX% |
| **Test Accuracy** | ~XX% |
| **Training Time** | ~XX minutes |
| **Parameters** | ~XX,XXX |

## 🔍 Implementation Details

### Forward Propagation
```python
# Pseudo-code structure
def forward_pass(X, W, b):
    # Layer 1: Linear transformation + ReLU
    s1 = X @ W1 + b1
    h1 = relu(s1)
    
    # Layer 2: Linear transformation + Softmax
    s2 = h1 @ W2 + b2
    p = softmax(s2)
    
    return p, h1, s1
```

### Backpropagation
```python
# Gradient computation through chain rule
def backward_pass(X, Y, p, h1, s1, W2):
    # Output layer gradients
    G_batch = -(Y - p)  # dL/ds2
    
    # Hidden layer gradients
    G_W2 = h1.T @ G_batch
    G_b2 = np.mean(G_batch, axis=0)
    
    # Propagate to hidden layer
    G_h1 = G_batch @ W2.T
    G_s1 = G_h1 * relu_derivative(s1)
    
    # Input layer gradients
    G_W1 = X.T @ G_s1
    G_b1 = np.mean(G_s1, axis=0)
    
    return G_W1, G_b1, G_W2, G_b2
```

### Gradient Checking
The implementation includes numerical gradient verification:
```python
def check_gradients(X, Y, W, b, h=1e-6):
    # Compare analytical vs numerical gradients
    grad_analytical = compute_gradients_analytical(X, Y, W, b)
    grad_numerical = compute_gradients_numerical(X, Y, W, b, h)
    
    relative_error = np.abs(grad_analytical - grad_numerical) / (
        np.abs(grad_analytical) + np.abs(grad_numerical)
    )
    
    return relative_error < 1e-6  # Should be True
```

## 📁 Files Description

### Assignment1.ipynb
**Main assignment implementation containing:**
1. **Data Loading & Preprocessing**
   - CIFAR-10 dataset loading
   - Data normalization and preprocessing
   - Train/validation split

2. **Network Implementation**
   - Forward propagation function
   - Backpropagation algorithm
   - Parameter initialization
   - Mini-batch gradient descent

3. **Training Loop**
   - Training with different learning rates
   - Loss and accuracy tracking
   - Validation monitoring

4. **Evaluation & Analysis**
   - Test set evaluation
   - Learning curves visualization
   - Weight visualization
   - Error analysis

### Assignment1_bonus.ipynb
**Extended experiments including:**
1. **Architecture Variations**
   - Different hidden layer sizes
   - Multiple hidden layers
   - Activation function comparisons

2. **Optimization Techniques**
   - Learning rate scheduling
   - Momentum implementation
   - Adaptive learning rates

3. **Regularization Methods**
   - Weight decay (L2 regularization)
   - Dropout implementation
   - Early stopping

### Assignment1.pdf
**Comprehensive report containing:**
- Mathematical derivations of backpropagation
- Experimental results and analysis
- Comparison of different configurations
- Discussion of findings and insights

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy matplotlib jupyter
```

### Execution
```bash
# Navigate to Assignment1 directory
cd Assignment1/

# Launch Jupyter notebook
jupyter notebook Assignment1.ipynb
```

### Expected Runtime
- Basic implementation: ~10-15 minutes
- Bonus experiments: ~30-45 minutes

## 📊 Hyperparameters Explored

| Parameter | Values Tested | Best Value |
|-----------|---------------|------------|
| Learning Rate | [0.01, 0.001, 0.0001] | 0.001 |
| Batch Size | [50, 100, 200] | 100 |
| Hidden Units | [20, 50, 100] | 50 |
| Epochs | [20, 40, 60] | 40 |

## 🎓 Key Insights Learned

### Mathematical Understanding
- **Chain Rule Mastery:** Deep understanding of how gradients flow backward through the network
- **Matrix Operations:** Efficient vectorized implementations for batch processing
- **Numerical Stability:** Importance of proper initialization and normalization

### Implementation Challenges
- **Gradient Vanishing:** Understanding why deeper networks can be difficult to train
- **Overfitting:** Observing the gap between training and validation performance
- **Debugging:** Using gradient checking to verify implementation correctness

### Performance Optimization
- **Vectorization:** NumPy operations for efficient computation
- **Batch Processing:** Balancing batch size with convergence speed
- **Memory Management:** Handling large datasets efficiently

## 🐛 Common Issues & Solutions

### Issue 1: Exploding Gradients
**Problem:** Gradients become very large, causing training instability
**Solution:** 
- Use smaller learning rates
- Implement gradient clipping
- Better weight initialization

### Issue 2: Poor Convergence
**Problem:** Loss doesn't decrease or decreases very slowly
**Solution:**
- Check gradient computation with numerical gradients
- Adjust learning rate
- Verify data preprocessing

### Issue 3: Overfitting
**Problem:** Large gap between training and validation accuracy
**Solution:**
- Implement regularization techniques
- Reduce model complexity
- Use early stopping

## 📚 Theory References

### Essential Concepts
1. **Backpropagation Algorithm**
   - Chain rule application in neural networks
   - Efficient gradient computation

2. **Optimization Theory**
   - Gradient descent variants
   - Learning rate selection strategies

3. **Regularization Techniques**
   - L1/L2 regularization theory
   - Dropout as ensemble method

### Recommended Reading
- Deep Learning Book, Chapter 6 (Deep Feedforward Networks)
- CS231n Lecture Notes on Backpropagation
- Neural Networks and Deep Learning (Michael Nielsen)

## 🏆 Assessment Criteria

### Implementation Quality (40%)
- Correct forward propagation
- Proper backpropagation implementation
- Code clarity and documentation

### Experimental Analysis (30%)
- Thorough hyperparameter exploration
- Meaningful results interpretation
- Comparison of different approaches

### Report Quality (30%)
- Clear mathematical derivations
- Insightful discussion of results
- Professional presentation

---

**Note:** This assignment forms the foundation for understanding all subsequent deep learning concepts. Mastering these fundamentals is crucial for success in the remaining course assignments. 