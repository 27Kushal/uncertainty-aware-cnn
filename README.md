# The Model That Knows When It Doesn't Know: Open-Set Image Classification

## Overview
Standard image classification models operate under the **Closed-World Assumption**: they assume every image encountered in production belongs to one of the categories seen during training. When presented with completely unfamiliar data (Out-of-Distribution or OOD inputs), standard Convolutional Neural Networks (CNNs) confidently hallucinate incorrect answers due to the exponentiation in the Softmax activation function.

This project implements an **Open-Set Image Classifier** capable of not only recognizing familiar categories with high accuracy but also accurately flagging unfamiliar, never-before-seen inputs as `"Uncertain / Unknown"`. 

### Key Features
- **Strict Data Pipeline:** Simulates an open-world environment by partitioning CIFAR-10 into known vs. completely excluded categories.
- **Custom Model Architecture:** Adapted ResNet-18 pipeline optimized for high-resolution processing of small inputs, with integrated Monte Carlo Dropout.
- **Advanced OOD Detection:** Implements Energy-based detection to intercept bad predictions before they are made.
- **Methodology Comparison:** Evaluates and compares Maximum Softmax Probability (MSP), Predictive Entropy, Energy Scores, and MC-Dropout.
- **Failure Mode Analysis:** Deep visual and statistical breakdown of where standard uncertainty metrics fail (e.g., Near-OOD vs. Far-OOD semantic overlap).

---

## 1. Tech Stack
- **Language:** Python 3
- **Deep Learning Framework:** PyTorch, Torchvision
- **Data Processing:** NumPy, Pandas
- **Evaluation & Visuals:** Scikit-learn, Matplotlib, Seaborn

---

## 2. Dataset & Preprocessing Pipeline
The model utilizes the **CIFAR-10 dataset**, strictly partitioned to simulate real-world open-set conditions:
* **Known Categories (In-Distribution, Labels 0-4):** `airplane`, `automobile`, `bird`, `cat`, `deer`. 
  * *Usage:* Training (25,000 images), Validation (2,500 images), ID Testing (2,500 images).
* **Unknown Categories (Out-of-Distribution, Excluded):** `dog`, `frog`, `horse`, `ship`, `truck`. 
  * *Usage:* Strictly hidden during training. Used exclusively for testing (5,000 images) to evaluate the robustness of the uncertainty detector.

**Preprocessing Pipeline:**
* **Training:** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, and `Normalize(mean, std)`.
* **Testing:** `ToTensor()`, `Normalize(mean, std)`.

---

## 3. Model Architecture
The core model is a modified **ResNet-18**:
1. **Initial Convolution Tweak:** The `conv1` layer was adjusted (`kernel_size=3`, `stride=1`) and `maxpool` was removed to preserve high-resolution feature maps for small $32 \times 32$ images.
2. **Monte Carlo (MC) Dropout Integration:** A `Dropout(p=0.3)` layer was injected before the final fully connected layer. This acts as a regularizer during training and enables epistemic uncertainty sampling (MC-Dropout) during inference.

---

## 4. Uncertainty Estimation Methods
To identify OOD inputs, four distinct mathematical approaches were implemented and compared:

1. **Maximum Softmax Probability (MSP - Baseline):** Uses the highest Softmax probability. Highly vulnerable to overconfidence due to Softmax shift-invariance.
2. **Predictive Shannon Entropy:** Measures probability dispersion across all 5 classes ($\mathcal{H} = -\sum P_i \log P_i$).
3. **Energy Score (Optimal Method):** Calculates the negative Helmholtz Free Energy directly from raw logits ($E(\mathbf{x}) = \log \sum e^{z_i}$). It avoids the Softmax normalization trap, relying on the magnitude of feature activations. Unfamiliar objects trigger lower global neuron activations, yielding lower energy scores.
4. **Monte Carlo Dropout:** Runs the model 10 times with random neuron suppression active during testing, measuring prediction variance (Epistemic Uncertainty).

### Method Comparison (OOD Detection AUROC)
| Uncertainty Method | AUROC (%) | Separation Ability |
| :--- | :--- | :--- |
| **Energy Score** | **~88.5%** | **Best** |
| Predictive Entropy | ~86.2% | Good |
| MC-Dropout (10x) | ~86.0% | Good |
| MSP (Baseline) | ~85.1% | Weakest |

---

## 5. Classification Behavior: Before vs. After Uncertainty Handling
The rejection threshold was determined dynamically using the **90th percentile of the Known Validation Set's Energy Scores** (ensuring zero data leakage from the OOD test set). 

When testing the model on an "Open-World" mix of 2,500 Known and 5,000 Unknown images:

| Metric | Without Uncertainty Handling | With Energy Uncertainty Detector |
| :--- | :--- | :--- |
| **Open-World Accuracy** | ~31.0% | **~82.0%** |
| **OOD Images Caught** | 0.0% *(Blindly guesses)* | **~78.0%** *(Correctly outputs "Unknown")* |
| **Selective Accuracy** | ~31.0% | **~94.0%** *(Accuracy when model chooses to answer)* |

*Conclusion:* Adding the Energy-based uncertainty detector transforms a dangerous, overconfident classifier into a reliable, cautious system.

---

## 6. Failure Analysis & Observations
Despite the Energy Score successfully filtering out the majority of OOD inputs, ~20-22% of unknown images still slip past the detector with high confidence. 

Through visual inspection and class-wise breakdown, two primary failure modes emerge:
1. **Near-OOD vs. Far-OOD Semantic Overlap:** 
   * The highest false-accept rates occur with `dog` (often predicted as `cat` with high energy) and `horse` (predicted as `deer`). 
   * *Why:* The convolutional feature extractor learned to identify fur, four legs, and pointed ears. Because dogs and horses share these low-level visual textures, they trigger massive raw activations in the network, tricking the Energy score. Conversely, a `frog` shares very few features with the known classes and is easily rejected.
2. **Background Shortcut Learning:**
   * The model occasionally accepts `ship` inputs as `airplane`. Visual inspection reveals this happens primarily when both objects are surrounded by flat blue textures (sky vs. ocean). 

**Future Work / Mitigation:** Implementing Contrastive Learning (e.g., SupCon) rather than Cross-Entropy would force the model to cluster embeddings by holistic object shape rather than relying heavily on localized textures and backgrounds.

---

## How to Run
1. Clone the repository and navigate to the project directory.
2. Ensure you have the required dependencies installed (`torch`, `torchvision`, `scikit-learn`, `pandas`, `seaborn`, `matplotlib`).
3. Run the primary notebook or script. The code will automatically download CIFAR-10, apply the custom open-world splits, train the ResNet model, compute uncertainty metrics, and generate matplotlib visualizations of the failure modes.
