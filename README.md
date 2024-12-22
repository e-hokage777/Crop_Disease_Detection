# Summary
In Sub-Saharan African, agricture contributes to about 23% of the continent’s GDP. However, pests and diseases are capable of reducing annual yields by about 40%. This project aims to combat yield reduction by providing means to quickly detect diseases in crops in real-time by utilizing object detection technology. A model was developed to detect diseases in `pepper`, `tomato` and `corn` crops. The model achieved a Mean Average Precision of 
40%.

# Data Overview
The data is made up of images of pepper, tomato and corn crops. Within each image is at least one of the 23 different target diseases.
Diseases include:
- Corn_Cercospora Leaf Spot
- Corn Common Rust
- Corn Healthy
- Corn Northern Leaf Blight
- Corn Streak
- Pepper Bacterial Spot
- Pepper Cercospora
- Pepper Early Blight
- Pepper Fusarium
- Pepper Healthy
- Pepper Late Blight
- Pepper Leaf Blight
- Pepper Leaf Curl
- Pepper Leaf Mosaic
- Pepper Septoria
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Fusarium
- Tomato Healthy
- Tomato Late Blight
- Tomato Leaf Curl
- Tomato Mosaic
- Tomato Septoria

# EDA Insights
- Duplicate boxes for different diseases (predominant between `Tomato Septoria` and `Pepper Septoria` (includes mislabeling)
- Highly imbalanced

# Training Results
- MAP@50: 0.40
- MAP Per Class:


# Training Instructions


# Inference Instructions