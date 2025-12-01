# Self-Supervised vs From-Scratch on STL-10 (ResNet-18, PyTorch)

**Goal**  
The goal of this project is to compare **contrastive self-supervised pretraining + fine-tuning** against **pure from-scratch supervised training** on the **STL-10** dataset. The same backbone (ResNet-18) is first pretrained in a self-supervised way on unlabeled data and then fine-tuned on labeled data, and this is compared with an identical ResNet-18 trained from random initialization using only labeled data.

**Dataset**  
- **STL-10** (from `torchvision.datasets.STL10`): 10 classes, 96×96 RGB images  
- Usage in this project:
  - `unlabeled` split → self-supervised pretraining (no labels used)
  - `train` split → labeled train/validation for supervised fine-tuning and scratch training
  - `test` split → held-out evaluation only

**Folder Hierarchy**
```text
project_root/
│
├── Codes/
│   ├── ssl_pretrain_stl10.py      # Self-supervised (SimCLR-style contrastive) pretraining
│   ├── finetune_stl10.py          # Fine-tuning with SSL weights + evaluation + plots
│   └── train_scratch_stl10.py     # Supervised training from scratch + evaluation + plots
│
├── Weights/
│   ├── resnet18_ssl_stl10.pth              # SSL-pretrained encoder weights
│   ├── resnet18_ssl_finetuned_stl10.pth    # Best fine-tuned weights
│   └── resnet18_scratch_stl10.pth          # Best trained from scratch weights
└── (plots regarding evaluation metrics can be saved in the Evaluation_Metrics/ folder)
```

**Evaluation Metrics (STL-10)**
```text
All models use the same ResNet-18 backbone and are evaluated on the STL-10 test split.

| Setting                               | SSL Pretrain Epochs | Supervised Epochs | Test Loss | Test Acc | Test Macro-F1 |
|---------------------------------------|--------------------:|------------------:|----------:|---------:|--------------:|
| ResNet-18 (scratch, supervised)       | 0                   | 30                | 1.2054    | 0.6418   | 0.6386        |
| ResNet-18 (SimCLR-style SSL + FT)     | 50                  | 30                | 0.9708    | 0.7416   | 0.7376        |
```

**Conclusion** 
These results show that contrastive self-supervised pretraining on the unlabeled split of STL-10, followed by supervised fine-tuning, consistently improves test accuracy (≈0.64 → ≈0.74) and macro-averaged F1-score compared to training the same ResNet-18 architecture from scratch.

