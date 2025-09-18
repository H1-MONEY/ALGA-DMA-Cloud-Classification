## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/
cd ALGA-Cloud-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

1. Prepare your dataset in the following structure:
```
data/
├── 1_cumulus/
├── 2_altocumulus/
├── 3_cirrus/
├── 4_clearsky/
├── 5_stratocumulus/
├── 6_cumulonimbus/
└── 7_mixed/
```

2. Train the model:
```bash
python src/train.py --dataset_path data/ --num_epochs 50 --batch_size 32
```

### Inference

```python
from src.model import create_model
from examples.inference_example import CloudClassifier

# Initialize classifier
classifier = CloudClassifier("models/best_model.pth")

# Predict single image
result = classifier.predict_single_image("path/to/image.jpg")
print(f"Predicted: {result['predicted_label']} (confidence: {result['confidence']:.3f})")

# Visualize prediction
classifier.visualize_prediction("path/to/image.jpg")
```
