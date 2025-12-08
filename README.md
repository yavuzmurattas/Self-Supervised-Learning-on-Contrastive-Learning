# SimCLR Reproducibility: Self-Supervised vs. Supervised Baseline on STL-10

## **Goal**
The goal of this project is to evaluate the effectiveness of **Self-Supervised Learning (SimCLR)** on the **STL-10** dataset using a **ResNet-50** backbone. 

The project compares two distinct training paradigms to demonstrate the power of learning from unlabeled data:
1.  **SimCLR Pretraining + Linear Evaluation:** The model is pretrained on 100k unlabeled images using contrastive loss, followed by training a linear classifier on frozen features.
2.  **Supervised Training from Scratch (Strong Baseline):** An identical ResNet-50 is trained from random initialization using only labeled data, but with the **same strong augmentations** (Blur, Jitter, etc.) used in SimCLR to ensure a fair and rigorous comparison.

## **Dataset**
- **STL-10** (`torchvision.datasets.STL10`):
  - **Unlabeled Split:** 100,000 unlabeled images (Used for SSL Pretraining).
  - **Train Split:** 5,000 labeled images (Used for Linear Evaluation and Scratch Training).
  - **Test Split:** 8,000 labeled images (Used for final evaluation).
  - **Image Size:** 96×96 RGB.

## **Methodology (SimCLR Implementation)**
This project follows the original SimCLR framework (*Chen et al.*) closely:
- **Backbone:** ResNet-50.
- **Projection Head:** 2-layer MLP (Hidden: 2048, Output: 128).
- **Loss Function:** NT-Xent Loss (Temperature $\tau=0.5$).
- **Optimizer:** SGD + Cosine Annealing Learning Rate Scheduler.
- **Augmentations:** Random Resized Crop, Horizontal Flip, **Color Jitter (Strong)**, **Gaussian Blur (p=0.5)**, Random Grayscale.

## **Folder Hierarchy**
```text
project_root/
│
├── Codes/
│   ├── ssl_pretraining_stl10.py    # SimCLR Pretraining (ResNet-50, Unlabeled Data)
│   ├── finetuning_stl10.py         # Linear Evaluation (Frozen Encoder + Linear Layer)
│   └── training_scratch_stl10.py   # Supervised Training from Scratch (Strong Baseline)
│
├── Weights/
│   ├── resnet50_ssl_stl10.pth            # Pretrained Encoder Weights (Epoch 50)
│   ├── resnet50_ssl_finetuned_stl10.pth  # Best Linear Classifier Weights
│   └── resnet50_scratch_stl10.pth        # Best Scratch Model Weights
│
└── Evaluation_Metrics/             # Loss and Accuracy Plots
    ├── PreTrained_Evaluation_Metric/
    ├── FineTuning_Metrics/
    └── Train_Scratch_Metrics/
```

## **Evaluation Metrics (STL-10)**
To ensure a fair and rigorous comparison, all experimental setups utilize an identical ResNet-50 backbone architecture. Performance metrics are reported on the official STL-10 test set (8,000 images).
```text

| Setting                           | SSL Pretrain Epochs | Supervised Epochs | Test Loss |  Test Acc  | Test Macro-F1 |
|-----------------------------------| :-----------------: | :---------------: | :-------: | :--------: | :-----------: |
| ResNet-50 (Scratch, Strong Aug.)  |         0           |         30        |  1.2360   |   53.73%   |    0.5271     |
| ResNet-50 (SimCLR + Linear Eval)  |         50          |         30        |  0.7641   |   72.25%   |    0.7219     |
```

## **Conclusion**

This project demonstrates the substantial impact of Self-Supervised Learning (SimCLR) in scenarios with limited labeled data. By pretraining a **ResNet-50** encoder on 100,000 unlabeled STL-10 images, we achieved a Test Accuracy of **72.25%**, significantly outperforming the supervised baseline (trained from scratch with strong augmentations), which only reached **53.73%**.

**Key Findings:**
1.  **Label Efficiency:** The SimCLR approach yielded a **+18.52% absolute improvement** in accuracy compared to the supervised baseline, proving that the model successfully leveraged unlabeled data to learn robust visual representations.
2.  **Rapid Convergence:** A remarkable observation is that the SimCLR model surpassed the final performance of the scratch model (Epoch 30) within the very **first epoch** of linear evaluation (~59% vs ~47%). This indicates that the pretrained features were already semantically separable before any supervised fine-tuning.
3.  **Generalization:** The close gap between Training and Validation metrics in the Linear Evaluation phase suggests that SSL learned features that generalize well, significantly minimizing the overfitting gap often observed in deep neural networks trained on limited data.
