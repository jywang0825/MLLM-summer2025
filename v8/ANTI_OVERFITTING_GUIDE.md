# Anti-Overfitting Guide for InternVL3 Frame-Based Finetuning

This guide explains the different training approaches available and how they help prevent overfitting.

## Training Modes Available

### 1. **AGGRESSIVE Mode** (Fast, Higher Risk of Overfitting)
- **LoRA Rank**: 16 (higher capacity, more parameters)
- **Learning Rate**: 2e-5 (faster learning)
- **Epochs**: 3 (shorter training)
- **Weight Decay**: 0.05 (moderate regularization)
- **Best for**: Quick experiments, when you have lots of diverse data

### 2. **OPTIMIZED Mode** (Balanced Approach) ⭐ **RECOMMENDED**
- **LoRA Rank**: 8 (balanced capacity)
- **Learning Rate**: 1e-5 (moderate learning)
- **Epochs**: 10 (longer training)
- **Weight Decay**: 0.1 (stronger regularization)
- **Early Stopping**: Yes (patience=3)
- **Best for**: Production training, balanced performance vs. overfitting

### 3. **CONSERVATIVE Mode** (Slow, Minimal Overfitting)
- **LoRA Rank**: 4 (lowest capacity, most regularization)
- **Learning Rate**: 5e-6 (very slow learning)
- **Epochs**: 15 (longest training)
- **Weight Decay**: 0.2 (maximum regularization)
- **Early Stopping**: Yes (patience=5)
- **Best for**: Small datasets, when overfitting is a major concern

## Anti-Overfitting Techniques Used

### **Parameter Reduction**
- **Lower LoRA Rank**: Reduces the number of trainable parameters
  - Rank 16 → 8 → 4 progressively reduces overfitting risk
- **Frozen Layers**: LLM, MLP, and backbone are frozen (only LoRA adapters train)

### **Regularization**
- **Weight Decay**: Penalizes large parameter values
  - Higher values (0.1-0.2) = stronger regularization
- **Drop Path Rate**: Adds stochasticity to training
  - 0.0 → 0.1 → 0.2 progressively increases regularization

### **Training Control**
- **Early Stopping**: Stops training when validation loss stops improving
- **Frequent Evaluation**: More frequent validation checks (every 100-500 steps)
- **Cosine with Restarts**: Learning rate scheduler that periodically restarts

### **Data Augmentation**
- **Dynamic Image Size**: Varies input image sizes during training
- **Thumbnail Usage**: Uses smaller image representations
- **Down Sampling**: Reduces image resolution

## How to Use

### **Quick Start (Recommended)**
```bash
cd v8
bash run_frame_finetuning.sh optimized 8 2
```

### **Conservative Training (Minimal Overfitting)**
```bash
cd v8
bash run_frame_finetuning.sh conservative 8 2
```

### **Aggressive Training (Fast Results)**
```bash
cd v8
bash run_frame_finetuning.sh aggressive 8 2
```

## Monitoring Overfitting

### **Signs of Overfitting**
- Training loss continues decreasing while validation loss increases
- Validation loss plateaus or increases after epoch 5-8
- Model performs well on training data but poorly on validation

### **What to Do If Overfitting Occurs**
1. **Switch to Conservative Mode**: Lower LoRA rank, higher weight decay
2. **Reduce Epochs**: Use early stopping more aggressively
3. **Increase Batch Size**: If memory allows, larger batches help
4. **Add More Data**: More diverse training data reduces overfitting

## Memory vs. Performance Trade-offs

| Mode | Memory Usage | Training Speed | Overfitting Risk | Final Quality |
|------|--------------|----------------|------------------|---------------|
| Aggressive | High | Fast | High | Variable |
| Optimized | Medium | Medium | Medium | Good |
| Conservative | Low | Slow | Low | Stable |

## Recommendations

### **For Small Datasets (< 1000 samples)**
- Use **Conservative Mode**
- Start with 10-15 epochs
- Monitor validation loss closely

### **For Medium Datasets (1000-10000 samples)**
- Use **Optimized Mode**
- Start with 8-10 epochs
- Use early stopping

### **For Large Datasets (> 10000 samples)**
- Can use **Aggressive Mode**
- Monitor for overfitting
- Consider switching to Optimized if issues arise

### **For Production Use**
- Always use **Optimized Mode**
- Implement proper validation splits
- Use early stopping
- Save best model based on validation performance


