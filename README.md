# Bone Fracture Segmentation – Research Update

## Data Processing

- **Dataset:** FracAtlas
- Processed the original dataset into a format suitable for binary fracture segmentation.
- Removed one image due to an image dimension mismatch between the image and its annotation.
- Generated binary segmentation masks from the provided COCO annotations.
- Created training, validation, and test splits:
  - **Training:** 2,856 images (502 fracture-positive)
  - **Validation:** 613 images (108 fracture-positive)
  - **Testing:** 613 images (108 fracture-positive)
- The resulting dataset contains 4,082 images, including 718 fracture-positive images and 3,364 fracture-negative images.

## Baselines

### Model Configuration

- **Architecture:** U-Net with a ResNet-18 encoder
- **Encoder initialization:** ImageNet-pretrained weights
- **Input:** Single-channel grayscale X-rays
- **Output:** Binary fracture segmentation mask
- **Loss function:** Binary Cross-Entropy + Dice Loss
- **Optimizer:** AdamW
- **Learning rate:** 1e-4
- **Weight decay:** 1e-4
- **Training:** Up to 30 epochs, with early stopping based on validation performance.

### Evaluation Metrics

The baseline is evaluated using the following metrics:
- **Overall Dice:** Dice score averaged across all validation images.
- **Fracture-Positive Dice:** Dice score averaged only across images containing fractures.
- **IoU:** Intersection over Union.
- **Fracture-Positive IoU:** Intersection over Union averaged only across images containing fractures.
- **Precision:** Proportion of predicted fracture pixels that overlap with ground-truth fracture pixels.
- **Recall:** Proportion of ground-truth fracture pixels recovered by the model.

Fracture-positive Dice and IoU are reported separately because the dataset is imbalanced, with far more fracture-negative than fracture-positive images.

### Limited Annotation Experiments

To investigate the effectiveness of semi-supervised learning when pixel-level
annotations are limited, experiments will simulate scenarios in which only a
small proportion of the training dataset has segmentation annotations.

The following labeled-data ratios will be evaluated:

- 10% labeled data
- 20% labeled data
- 30% labeled data
- 100% labeled data

For semi-supervised experiments, the remaining training images will be used
as unlabeled data. Supervised and semi-supervised approaches will use identical
labeled subsets to ensure fair comparisons.

The primary objective is to determine whether unlabeled images can improve
fracture segmentation performance when only a limited number of annotated
examples are available.

### Baseline Results

| Labeled Data | Overall Dice | Fracture-Positive Dice |  Overall IoU  | Fracture-Positive IoU | Precision | Recall |
| ------------ | ------------ | ---------------------- | ------------- | --------------------- | --------- | ------ |
| 100%         | 0.2869       | 0.3508                 | 0.2699        | 0.2544                | 0.1536    | 0.3997 |
| 30%          | 0.5577       | 0.2768                 | 0.5447        | 0.2027                | 0.2641    | 0.2645 |
| 20%          | 0.1975       | 0.0006                 | 0.1974        | 0.0003                | 0.0031    | 0.0004 |
| 10%          | 7.7 x 10<sup>-7</sup> | 4.4 x 10<sup>-6</sup> | 3.9 x 10<sup>-7</sup> | 2.2 x 10<sup>-6</sup> | 1.9 x 10<sup>-6</sup> | 7.6 x 10<sup>-6</sup> |

Preliminary results show inconsistent performance across the different annotation ratios. While the 100% and 30% experiments achieved measurable segmentation performance, the 10% and 20% experiments produced almost no overlap between predicted and ground-truth fracture masks. The cause of this variation is not yet clear and will require further investigation.

## Next Steps

- **Semi-Supervised Experiments:** Implement and evaluate semi-supervised learning approaches using the remaining unlabeled training images. Compare performance against fully supervised baselines across different labeled-data ratios (10%, 20%, and 30%).

- **Cross-Dataset Generalization:** Investigate whether incorporating additional fracture X-ray datasets as unlabeled training data can improve segmentation performance and generalization. Evaluate models on a separate dataset to assess their ability to generalize across differences in imaging sources, patient populations, and annotation conventions.

- **Cross-Anatomical Generalization:** Investigate whether unlabeled images from previously unseen anatomical regions can improve fracture segmentation through unsupervised domain adaptation. For example, train using labeled hand X-rays alongside unlabeled hip X-rays, then evaluate performance on a separate labeled hip test set. This will assess whether the model can adapt to new anatomical regions without requiring additional pixel-level annotations.