# Assignment 4: Advanced Deep Learning Topics

## 📖 Overview
The final assignment explores advanced deep learning concepts including transfer learning, attention mechanisms, generative models, and optimization techniques. This assignment integrates knowledge from previous assignments to tackle complex real-world problems using state-of-the-art approaches.

## 🎯 Learning Objectives
- Master transfer learning and fine-tuning strategies
- Understand attention mechanisms and their applications
- Implement advanced optimization techniques
- Explore generative adversarial networks (GANs)
- Apply deep learning to novel problem domains

## 📊 Potential Applications
- **Computer Vision:** Image classification with transfer learning
- **Natural Language Processing:** Attention-based sequence models
- **Generative Modeling:** GANs for image generation
- **Custom Projects:** Novel applications of deep learning

## 🧠 Advanced Architectures

### Transfer Learning Pipeline
```python
# Pre-trained model fine-tuning
def create_transfer_model(base_model, num_classes):
    """
    Create transfer learning model
    """
    # Freeze base layers
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(base_model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    base_model.fc = classifier
    return base_model
```

### Attention Mechanism
```python
def attention_forward(query, key, value, mask=None):
    """
    Scaled dot-product attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### GAN Architecture
```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

## 📈 Key Results
| Approach | Dataset | Metric | Performance |
|----------|---------|--------|-------------|
| **Transfer Learning** | Custom Dataset | Accuracy | ~XX% |
| **Attention Model** | Sequence Data | BLEU/Perplexity | ~XX |
| **GAN** | Generated Images | FID Score | ~XX |

## 📁 Files Description

### assignment4.ipynb
**Main assignment implementation featuring:**

1. **Transfer Learning Experiments**
   - Pre-trained model selection (ResNet, VGG, etc.)
   - Feature extraction vs. fine-tuning comparison
   - Custom dataset application
   - Performance evaluation

2. **Attention Mechanisms**
   - Self-attention implementation
   - Multi-head attention
   - Application to sequence tasks
   - Visualization of attention weights

3. **Advanced Optimization**
   - Learning rate scheduling
   - Batch normalization effects
   - Optimizer comparisons (Adam, RMSprop, etc.)
   - Regularization techniques

### assignment4bonus.ipynb
**Extended experiments including:**

1. **Generative Adversarial Networks**
   - Basic GAN implementation
   - Training stability techniques
   - Mode collapse analysis
   - Generated sample evaluation

2. **Custom Architecture Design**
   - Novel network architectures
   - Architecture search techniques
   - Performance benchmarking

3. **Real-world Applications**
   - End-to-end project implementation
   - Production considerations
   - Model deployment strategies

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision transformers pillow matplotlib jupyter
```

### Execution
```bash
# Navigate to Assignment4 directory
cd Assignment4/

# Launch main assignment
jupyter notebook assignment4.ipynb

# Launch bonus experiments
jupyter notebook assignment4bonus.ipynb
```

### Expected Runtime
- Transfer learning: ~30-45 minutes
- Attention experiments: ~20-30 minutes
- GAN training: ~60-90 minutes

## 📊 Experimental Design

### Transfer Learning Study
| Base Model | Dataset | Frozen Layers | Fine-tuned Layers | Results |
|------------|---------|---------------|-------------------|---------|
| ResNet-50 | ImageNet → Custom | All conv layers | Final classifier | XX% accuracy |
| VGG-16 | ImageNet → Custom | First 3 blocks | Last 2 blocks + classifier | XX% accuracy |
| EfficientNet | ImageNet → Custom | Feature extractor | Classifier only | XX% accuracy |

### Attention Analysis
```python
# Visualize attention patterns
def plot_attention_weights(attention_weights, input_tokens, output_tokens):
    """
    Create heatmap of attention weights
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=input_tokens,
                yticklabels=output_tokens,
                cmap='Blues')
    plt.title('Attention Weights Visualization')
    plt.show()
```

## 🎓 Key Insights Learned

### Transfer Learning Mastery
- **Feature Reusability:** Understanding which layers transfer well across domains
- **Fine-tuning Strategies:** When to freeze vs. fine-tune different layers
- **Domain Adaptation:** Bridging the gap between source and target domains

### Attention Mechanisms
- **Interpretability:** Attention weights provide insights into model decisions
- **Long-range Dependencies:** Better handling of sequential information
- **Computational Efficiency:** Parallelization advantages over RNNs

### Generative Modeling
- **GAN Training Dynamics:** Understanding generator vs. discriminator balance
- **Mode Collapse:** Identifying and mitigating training instabilities
- **Evaluation Metrics:** Beyond visual inspection (FID, IS scores)

## 🔬 Advanced Experiments

### Learning Rate Scheduling
```python
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# OneCycleLR for super-convergence
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=len(dataloader)
)
```

### Data Augmentation Strategies
```python
# Advanced augmentation pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Model Ensemble Techniques
```python
def ensemble_predict(models, x):
    """
    Ensemble prediction using multiple models
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```

## 🐛 Common Issues & Solutions

### Issue 1: Transfer Learning Overfitting
**Problem:** Model performs well on training but poorly on validation
**Solutions:**
- Reduce learning rate for pre-trained layers
- Increase dropout rate
- Use more aggressive data augmentation

### Issue 2: GAN Training Instability
**Problem:** Generator or discriminator dominates training
**Solutions:**
- Balance learning rates between G and D
- Use gradient penalty (WGAN-GP)
- Implement progressive growing

### Issue 3: Attention Mechanism Complexity
**Problem:** High computational cost for long sequences
**Solutions:**
- Use sparse attention patterns
- Implement efficient attention variants
- Sequence chunking strategies

## 📚 Theory References

### Core Concepts
1. **Transfer Learning Theory**
   - Domain adaptation principles
   - Feature transferability analysis

2. **Attention Mechanisms**
   - Transformer architecture
   - Self-attention vs. cross-attention

3. **Generative Models**
   - GAN theory and variants
   - Evaluation metrics for generative models

### Recommended Reading
- "Attention Is All You Need" (Vaswani et al.)
- "Deep Residual Learning for Image Recognition" (He et al.)
- "Generative Adversarial Networks" (Goodfellow et al.)

## 🏆 Assessment Criteria

### Technical Implementation (40%)
- Correct implementation of advanced techniques
- Code quality and documentation
- Experimental reproducibility

### Innovation & Creativity (30%)
- Novel applications or improvements
- Creative problem-solving approaches
- Original experimental design

### Analysis & Insights (30%)
- Deep understanding of advanced concepts
- Thoughtful interpretation of results
- Connection to theoretical foundations

## 🎨 Project Examples

### Example 1: Medical Image Classification
- **Dataset:** X-ray images for disease detection
- **Approach:** Transfer learning from ImageNet
- **Results:** 95% accuracy, outperforming radiologists

### Example 2: Creative Text Generation
- **Dataset:** Poetry corpus
- **Approach:** Transformer with attention
- **Results:** Coherent, creative verse generation

### Example 3: Art Style Transfer
- **Dataset:** Artistic paintings + photographs
- **Approach:** Cycle-GAN implementation
- **Results:** High-quality style transformation

## 🚀 Future Directions

### Emerging Techniques
- **Vision Transformers:** Applying attention to computer vision
- **Self-Supervised Learning:** Learning without labels
- **Neural Architecture Search:** Automated model design

### Production Considerations
- **Model Compression:** Quantization and pruning
- **Edge Deployment:** Mobile and embedded systems
- **MLOps:** Continuous integration and deployment

---

**Note:** This assignment represents the culmination of the DD2424 course, demonstrating mastery of advanced deep learning concepts and their practical applications. The skills developed here are directly applicable to cutting-edge research and industry applications. 