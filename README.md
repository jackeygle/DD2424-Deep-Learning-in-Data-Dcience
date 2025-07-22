# DD2424 - Deep Learning in Data Science
*KTH Royal Institute of Technology*

[![Course](https://img.shields.io/badge/Course-DD2424-blue)](https://www.kth.se/student/kurser/kurs/DD2424)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)

## 🎯 Course Overview

This repository contains all assignments and implementations for the **DD2424 Deep Learning in Data Science** course at KTH Royal Institute of Technology. The course focuses on understanding neural network fundamentals, implementing deep learning algorithms from scratch, and applying them to real-world problems.

### 🧠 Learning Objectives
- Implement neural networks from first principles
- Understand backpropagation and gradient descent algorithms
- Master different network architectures (MLP, CNN, RNN)
- Apply deep learning to computer vision and NLP tasks
- Gain hands-on experience with optimization techniques

## 📁 Repository Structure

```
DD2424-Deep-Learning-in-Data-Science/
├── Assignment1/           # Multi-layer Perceptron Implementation
│   ├── Assignment1.ipynb         # Main assignment notebook
│   ├── Assignment1_bonus.ipynb   # Bonus exercises
│   └── Assignment1.pdf           # Assignment report
├── Assignment2/           # Convolutional Neural Networks
├── Assignment3/           # Recurrent Neural Networks  
│   └── assignment3bonus.ipynb    # Bonus implementation
├── Assignment4/           # Advanced Deep Learning
│   ├── assignment4.ipynb         # Main assignment
│   └── assignment4bonus.ipynb    # Bonus exercises
├── Reports/              # Written reports and analysis
├── Utils/                # Helper functions and utilities
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🚀 Assignments Overview

### 📊 Assignment 1: Multi-layer Perceptron & Backpropagation
**Focus:** Implementing neural networks from scratch using only NumPy

**Key Components:**
- Forward propagation implementation
- Backpropagation algorithm
- Gradient descent optimization
- CIFAR-10 classification task
- Gradient checking and numerical verification

**Files:**
- `Assignment1.ipynb` - Main implementation with 2-layer neural network
- `Assignment1_bonus.ipynb` - Extended exercises and improvements
- `Assignment1.pdf` - Detailed analysis and results report

**Learning Outcomes:**
- Deep understanding of neural network mathematics
- Hands-on implementation of backpropagation
- Experience with hyperparameter tuning

### 🔄 Assignment 3: Recurrent Neural Networks
**Focus:** Sequence modeling and text generation

**Key Components:**
- Vanilla RNN implementation
- LSTM/GRU architectures
- Character-level text generation
- Sequence-to-sequence modeling

**Files:**
- `assignment3bonus.ipynb` - Advanced RNN implementations

### 🎯 Assignment 4: Advanced Deep Learning Topics
**Focus:** State-of-the-art techniques and applications

**Key Components:**
- Transfer learning strategies
- Advanced optimization techniques
- Regularization methods
- Final project implementation

**Files:**
- `assignment4.ipynb` - Main assignment implementation
- `assignment4bonus.ipynb` - Extended experiments and analysis

## 🛠️ Technical Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook/Lab
- NumPy, Matplotlib for basic implementations
- TensorFlow/PyTorch for advanced assignments

### Installation
```bash
# Clone the repository
git clone https://github.com/jackeygle/DD2424-Deep-Learning-in-Data-Dcience.git
cd DD2424-Deep-Learning-in-Data-Dcience

# Create virtual environment
python -m venv dd2424_env
source dd2424_env/bin/activate  # On Windows: dd2424_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## 📈 Key Achievements

### Technical Implementations
- ✅ **From-scratch neural networks** using only NumPy
- ✅ **Backpropagation algorithm** with mathematical derivations
- ✅ **CNN architectures** for image classification
- ✅ **RNN models** for sequence prediction
- ✅ **Advanced optimizers** and regularization techniques

### Dataset Experience
- **CIFAR-10**: Image classification with CNNs
- **Text Corpora**: Character-level language modeling
- **Custom Datasets**: Real-world problem applications

### Performance Results
| Assignment | Dataset | Best Accuracy | Architecture |
|------------|---------|---------------|--------------|
| Assignment 1 | CIFAR-10 | ~45% | 2-layer MLP |
| Assignment 3 | Text Data | High Perplexity | LSTM |
| Assignment 4 | Various | Project-specific | Advanced Models |

## 🎓 Course Insights

### What I Learned
1. **Mathematical Foundations**: Deep understanding of gradient computation and chain rule
2. **Implementation Skills**: Building neural networks from basic linear algebra operations
3. **Debugging Techniques**: Identifying and fixing gradient vanishing/exploding problems
4. **Optimization Strategies**: Comparing SGD, Adam, and other optimization algorithms
5. **Architecture Design**: Understanding when to use different network types

### Key Challenges Overcome
- **Gradient Checking**: Ensuring numerical and analytical gradients match
- **Hyperparameter Tuning**: Finding optimal learning rates and network sizes
- **Overfitting Prevention**: Implementing dropout and regularization techniques
- **Performance Optimization**: Vectorizing operations for computational efficiency

### Practical Applications
This knowledge directly applies to:
- Computer vision projects (object detection, image classification)
- Natural language processing (sentiment analysis, machine translation)
- Time series forecasting and prediction
- Generative modeling and creative applications

## 📚 Resources Used

### Primary Textbooks
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- Course lectures and materials from KTH professors

### Online Resources
- [CS231n Stanford Course](http://cs231n.stanford.edu/) - Convolutional Networks
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528)
- [Deep Learning Book](https://www.deeplearningbook.org/) - Mathematical foundations

### Implementation References
- NumPy documentation for efficient array operations
- Matplotlib for visualization and result plotting
- Scientific Python ecosystem best practices

## 🏆 Course Grade & Reflection

**Overall Performance:** [Your Grade Here]

**Most Valuable Learning:**
The course provided an invaluable foundation in deep learning by requiring implementation from first principles. Unlike using high-level frameworks, building neural networks from scratch gave me genuine understanding of:
- How gradients flow through complex networks
- Why certain architectural choices matter
- How to debug and optimize learning algorithms
- The mathematical beauty underlying modern AI systems

**Future Applications:**
This deep understanding serves as a solid foundation for:
- Research in machine learning and AI
- Building custom neural architectures for specific problems
- Contributing to open-source deep learning frameworks
- Teaching and mentoring others in AI/ML

## 📧 Contact & Links

- **Author:** Jackeygle
- **Institution:** KTH Royal Institute of Technology
- **Course Period:** 2024 fall
- **Portfolio:** https://jackeysproject.web.app
- **GitHub:** https://github.com/jackeygle

## 🙏 Acknowledgments

- **KTH Faculty:** Excellent instruction and challenging assignments
- **Course TAs:** Patient support during office hours and debugging sessions
- **Study Group:** Collaborative learning and problem-solving discussions
- **Open Source Community:** NumPy, Matplotlib, and Jupyter ecosystems

---

*"The best way to understand neural networks is to implement them yourself. Every line of code teaches you something frameworks cannot."*

## 📄 License

This project is for educational purposes. Please respect academic integrity when using this code for your own coursework.

---

**Note:** This repository showcases academic work completed for DD2424 at KTH. The implementations demonstrate understanding of fundamental deep learning concepts through hands-on coding rather than using pre-built frameworks. 
