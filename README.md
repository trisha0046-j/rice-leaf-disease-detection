# Rice Leaf Disease Detection (PRCP-1001)

A deep learning project that classifies rice leaf images into three common diseases using CNNs and transfer learning (MobileNetV2).

## Problem Statement

Identify the disease affecting a rice leaf from its image, so that farmers and agronomists can detect problems early.

**Classes**

| Label | Disease |
|---|---|
| 0 | Bacterial Leaf Blight |
| 1 | Brown Spot |
| 2 | Leaf Smut |

## Dataset

- 119 images in total: 40 Bacterial Leaf Blight, 40 Brown Spot, 39 Leaf Smut (the project brief mentions 120; one image was missing in the supplied data and no images were duplicated)
- Images vary widely in size (from about 250×200 up to 3081×897 pixels), so all images are resized to 128×128
- No corrupted images were found
- The dataset is **not included** in this repository. Place the project zip (`PRCP-1001-RiceLeaf.zip`) in the project folder before running the notebook.

## Approach

1. **EDA** — class distribution, image dimensions, sample images, corrupted-image check, pixel-value analysis
2. **Preprocessing** — RGB conversion, resize to 128×128, normalization to 0–1, stratified 80/20 train-test split (95 train / 24 test)
3. **Baseline CNN** — 3 convolution blocks + dense layer with dropout
4. **Augmented CNN** — same CNN with random flip, rotation, zoom and translation
5. **Transfer learning** — MobileNetV2 (ImageNet weights, frozen base) + custom classifier head
6. **Evaluation** — accuracy, classification report, confusion matrix on the held-out test set
7. **Prediction demo** — predicting the disease and confidence for an unseen image

## Results

| Model | Test Accuracy |
|---|---:|
| Baseline CNN | 66.67% |
| Augmented CNN | 66.67% |
| **MobileNetV2 Transfer Learning** | **83.33%** |

MobileNetV2 per-class results on the test set:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bacterial Leaf Blight | 0.80 | 1.00 | 0.89 |
| Brown Spot | 0.86 | 0.75 | 0.80 |
| Leaf Smut | 0.86 | 0.75 | 0.80 |

**Key takeaways**

- Transfer learning improved test accuracy by about 16.7 percentage points over the custom CNNs, which makes sense for such a small dataset.
- Data augmentation did not raise test accuracy in this experiment, although it reduced the test loss of the custom CNN (about 1.17 to 0.58).
- The test set has only 24 images, so each image is worth roughly 4 percentage points. The results should be treated as indicative, not production-ready.

## Challenges

- Very small dataset, which increases the risk of overfitting
- Slight mismatch between the expected (120) and available (119) images
- Inconsistent image sizes and backgrounds
- Limited gains from augmentation on a small test set

## Future Improvements

- Collect a larger, more diverse dataset (different fields, lighting, devices)
- Use MobileNetV2's own `preprocess_input` and fine-tune the top layers
- Use k-fold cross-validation for a more reliable estimate
- Try other pretrained models (EfficientNet, ResNet) and compare
- Deploy the best model as a Streamlit or FastAPI app

## Tech Stack

Python · TensorFlow / Keras · NumPy · Pandas · Pillow · Scikit-learn · Matplotlib · Seaborn · Jupyter

## How to Run

```bash
git clone https://github.com/<your-username>/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# place PRCP-1001-RiceLeaf.zip in this folder, then:
jupyter notebook Rice_Leaf_Disease_Detection.ipynb
```

## Project Structure

```
rice-leaf-disease-detection/
├── Rice_Leaf_Disease_Detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Trisha** — AI Engineer Trainee, DataMites
B.Tech CSE · Python · Machine Learning · Deep Learning · Computer Vision

[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
