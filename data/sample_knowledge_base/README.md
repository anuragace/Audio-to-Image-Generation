# Sample Knowledge Base Dataset

This folder contains a small starter dataset for the audio-to-image retrieval project.

It has 15 educational diagram images and metadata that can be used to build a Pinecone vector index.

## Files

```text
data/sample_knowledge_base/
  train/
    images/
      *.png
    metadata.csv
  README.md
```

The images are generated educational diagrams for machine learning topics.

## What Each Image Record Needs

Each image should have these fields:

| Field | Required | Why it matters |
| --- | --- | --- |
| `id` | Yes | Unique identifier for the image/vector. Use this as the Pinecone vector id. |
| `file_name` | Yes | Used by Hugging Face `imagefolder` datasets to locate the image file. |
| `image_path` | Yes | Stored in Pinecone metadata so the app can find the matching dataset image. |
| `topic` | Recommended | Broad concept, such as Linear Regression or Neural Networks. |
| `title` | Recommended | Human-readable title for the diagram. |
| `description` | Yes | Main text used to explain the image and help retrieval. |
| `search_queries` | Recommended | Extra keywords and phrases users may say. |
| `embedding_text` | Yes | Final text that should be converted into a CLIP embedding and inserted into Pinecone. |

## Why `image_path` Matters

The app retrieves metadata from Pinecone, then uses `image_path` to find the actual image in the dataset.

Example Pinecone metadata:

```json
{
  "image_path": "images/linear_regression_best_fit.png",
  "description": "A scatter plot showing data points and a straight best-fit line used in linear regression."
}
```

The Hugging Face dataset row should have the same `image_path`.

## Recommended Pinecone Vector Design

For each row:

```text
vector id: id
vector values: OpenCLIP embedding of embedding_text
namespace: text_embeddings
metadata:
  image_path
  description
  topic
  title
```

If using OpenCLIP `ViT-B-32`, your Pinecone index should use:

```text
dimension: 512
metric: cosine
```

## Loading Locally With Hugging Face Datasets

You can test this dataset locally with:

```python
from datasets import load_dataset

dataset = load_dataset(
    "imagefolder",
    data_dir="data/sample_knowledge_base",
    split="train",
)

print(dataset[0])
```

The dataset should include an `image` column plus the metadata fields from `metadata.csv`.

## Dataset Topics

The 15 included images are:

1. Linear Regression - Line of Best Fit
2. Support Vector Machines - SVM Margin
3. Principal Component Analysis - PCA Components
4. Decision Trees - Decision Tree Split
5. Neural Networks - Neural Network Layers
6. Gradient Descent - Loss Minimization
7. Model Evaluation - Confusion Matrix
8. K-Nearest Neighbors - KNN Classification
9. K-Means Clustering - Cluster Centroids
10. Machine Learning Workflow - Train Test Split
11. Model Generalization - Overfitting vs Underfitting
12. Model Evaluation - ROC Curve
13. Naive Bayes - Prior, Likelihood, Posterior
14. Convolutional Neural Networks - CNN Convolution
15. Transformers - Attention Mechanism
