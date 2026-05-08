import csv
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "sample_knowledge_base"
TRAIN_DIR = DATASET_DIR / "train"
IMAGE_DIR = TRAIN_DIR / "images"

WIDTH = 1024
HEIGHT = 768


TOPICS = [
    {
        "id": "linear_regression_best_fit",
        "topic": "Linear Regression",
        "title": "Line of Best Fit",
        "description": "A scatter plot showing data points and a straight best-fit line used in linear regression.",
        "search_queries": "linear regression, line of best fit, scatter plot, prediction, supervised learning",
    },
    {
        "id": "support_vector_machine_margin",
        "topic": "Support Vector Machines",
        "title": "SVM Margin",
        "description": "A classification diagram showing two classes, a decision boundary, and the maximum margin in a support vector machine.",
        "search_queries": "support vector machine, SVM, margin, decision boundary, classification",
    },
    {
        "id": "pca_components",
        "topic": "Principal Component Analysis",
        "title": "PCA Components",
        "description": "A two-dimensional point cloud with first and second principal component axes showing dimensionality reduction.",
        "search_queries": "PCA, principal component analysis, dimensionality reduction, eigenvectors, components",
    },
    {
        "id": "decision_tree_split",
        "topic": "Decision Trees",
        "title": "Decision Tree Split",
        "description": "A simple decision tree showing how features split data into branches and leaf predictions.",
        "search_queries": "decision tree, feature split, branches, leaves, classification tree",
    },
    {
        "id": "neural_network_layers",
        "topic": "Neural Networks",
        "title": "Neural Network Layers",
        "description": "A neural network diagram with input neurons, hidden layers, weighted connections, and output neurons.",
        "search_queries": "neural network, input layer, hidden layer, output layer, deep learning",
    },
    {
        "id": "gradient_descent_steps",
        "topic": "Gradient Descent",
        "title": "Gradient Descent",
        "description": "A loss curve showing gradient descent steps moving downhill toward the minimum loss.",
        "search_queries": "gradient descent, loss function, optimization, learning rate, minimum",
    },
    {
        "id": "confusion_matrix",
        "topic": "Model Evaluation",
        "title": "Confusion Matrix",
        "description": "A confusion matrix explaining true positives, false positives, false negatives, and true negatives.",
        "search_queries": "confusion matrix, true positive, false positive, recall, precision",
    },
    {
        "id": "knn_neighbors",
        "topic": "K-Nearest Neighbors",
        "title": "KNN Classification",
        "description": "A K-nearest neighbors diagram showing a query point classified by nearby labeled points.",
        "search_queries": "KNN, k nearest neighbors, query point, classification, distance",
    },
    {
        "id": "clustering_kmeans",
        "topic": "K-Means Clustering",
        "title": "K-Means Clusters",
        "description": "A clustering diagram showing three groups of points and their centroids in k-means clustering.",
        "search_queries": "k-means, clustering, centroid, unsupervised learning, clusters",
    },
    {
        "id": "dataset_split_workflow",
        "topic": "Machine Learning Workflow",
        "title": "Train Test Split",
        "description": "A dataset split diagram showing training data used to fit a model and test data used for evaluation.",
        "search_queries": "train test split, model validation, training data, testing data, machine learning workflow",
    },
    {
        "id": "overfitting_underfitting",
        "topic": "Model Generalization",
        "title": "Overfitting vs Underfitting",
        "description": "A comparison of underfitting, good fit, and overfitting curves on the same kind of data.",
        "search_queries": "overfitting, underfitting, good fit, bias variance, model generalization",
    },
    {
        "id": "roc_curve",
        "topic": "Model Evaluation",
        "title": "ROC Curve",
        "description": "An ROC curve showing true positive rate versus false positive rate with an AUC label.",
        "search_queries": "ROC curve, AUC, true positive rate, false positive rate, classifier evaluation",
    },
    {
        "id": "naive_bayes",
        "topic": "Naive Bayes",
        "title": "Naive Bayes",
        "description": "A probability diagram showing how Naive Bayes combines prior and likelihood evidence to classify a sample.",
        "search_queries": "naive bayes, probability, prior, likelihood, posterior, classification",
    },
    {
        "id": "cnn_convolution",
        "topic": "Convolutional Neural Networks",
        "title": "CNN Convolution",
        "description": "A convolutional neural network diagram showing an image, a sliding filter, a feature map, and classification.",
        "search_queries": "CNN, convolution, filter, feature map, image classification",
    },
    {
        "id": "attention_mechanism",
        "topic": "Transformers",
        "title": "Attention Mechanism",
        "description": "A transformer attention diagram showing query, key, value vectors and weighted attention scores.",
        "search_queries": "attention mechanism, transformer, query key value, self attention, weights",
    },
]


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = font(42, bold=True)
SUBTITLE_FONT = font(24)
LABEL_FONT = font(22)
SMALL_FONT = font(18)


def wrap_text(text, width=44):
    return "\n".join(textwrap.wrap(text, width=width))


def base_canvas(title, topic):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 96), fill="#111827")
    draw.text((40, 24), title, fill="white", font=TITLE_FONT)
    draw.text((40, 76), topic, fill="#cbd5e1", font=SMALL_FONT)
    return image, draw


def footer(draw, description):
    draw.rounded_rectangle((36, 650, 988, 730), radius=14, fill="#e5e7eb")
    draw.text((56, 668), wrap_text(description, 92), fill="#111827", font=SMALL_FONT)


def axes(draw, origin=(120, 570), x_end=(900, 570), y_end=(120, 150)):
    draw.line((origin, x_end), fill="#111827", width=4)
    draw.line((origin, y_end), fill="#111827", width=4)
    draw.polygon([(900, 570), (880, 560), (880, 580)], fill="#111827")
    draw.polygon([(120, 150), (110, 170), (130, 170)], fill="#111827")


def draw_linear_regression(draw):
    axes(draw)
    points = [(180, 510), (230, 485), (290, 450), (350, 420), (420, 390), (500, 335), (570, 310), (660, 260), (730, 240), (810, 195)]
    for x, y in points:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#2563eb")
    draw.line((160, 520, 850, 180), fill="#ef4444", width=5)
    draw.text((620, 150), "best-fit line", fill="#ef4444", font=LABEL_FONT)


def draw_svm(draw):
    axes(draw)
    left = [(250, 450), (300, 500), (320, 420), (380, 470), (410, 390), (460, 430)]
    right = [(600, 260), (650, 300), (690, 230), (730, 330), (780, 250), (820, 310)]
    for x, y in left:
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#2563eb")
    for x, y in right:
        draw.rectangle((x - 12, y - 12, x + 12, y + 12), fill="#f97316")
    draw.line((500, 570, 500, 140), fill="#111827", width=5)
    draw.line((430, 570, 430, 140), fill="#94a3b8", width=3)
    draw.line((570, 570, 570, 140), fill="#94a3b8", width=3)
    draw.text((590, 155), "margin", fill="#475569", font=LABEL_FONT)


def draw_pca(draw):
    axes(draw)
    center = (500, 365)
    for i in range(36):
        x = 300 + i * 12
        y = int(500 - i * 7 + math.sin(i) * 35)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2563eb")
    draw.line((260, 520, 760, 220), fill="#ef4444", width=5)
    draw.line((430, 210, 590, 520), fill="#16a34a", width=5)
    draw.text((720, 210), "PC1", fill="#ef4444", font=LABEL_FONT)
    draw.text((600, 500), "PC2", fill="#16a34a", font=LABEL_FONT)
    draw.ellipse((center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8), fill="#111827")


def draw_decision_tree(draw):
    boxes = [
        (430, 140, 600, 200, "Feature A?"),
        (230, 300, 410, 360, "Feature B?"),
        (650, 300, 830, 360, "Class 1"),
        (120, 500, 300, 560, "Class 0"),
        (400, 500, 580, 560, "Class 1"),
    ]
    for x1, y1, x2, y2, text in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill="#dbeafe", outline="#1d4ed8", width=3)
        draw.text((x1 + 22, y1 + 18), text, fill="#111827", font=LABEL_FONT)
    draw.line((515, 200, 320, 300), fill="#111827", width=4)
    draw.line((515, 200, 740, 300), fill="#111827", width=4)
    draw.line((320, 360, 210, 500), fill="#111827", width=4)
    draw.line((320, 360, 490, 500), fill="#111827", width=4)


def draw_neural_network(draw):
    layers = [[(180, y) for y in (230, 330, 430)], [(400, y) for y in (190, 280, 370, 460)], [(620, y) for y in (230, 330, 430)], [(840, y) for y in (280, 380)]]
    for layer_a, layer_b in zip(layers, layers[1:]):
        for a in layer_a:
            for b in layer_b:
                draw.line((a, b), fill="#cbd5e1", width=2)
    colors = ["#2563eb", "#16a34a", "#16a34a", "#f97316"]
    labels = ["input", "hidden", "hidden", "output"]
    for layer, color, label in zip(layers, colors, labels):
        for x, y in layer:
            draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=color)
        draw.text((layer[0][0] - 40, 530), label, fill="#111827", font=LABEL_FONT)


def draw_gradient_descent(draw):
    axes(draw)
    curve = []
    for x in range(180, 850, 12):
        y = 180 + int(((x - 540) ** 2) / 1800)
        curve.append((x, y))
    draw.line(curve, fill="#2563eb", width=5)
    steps = [(250, 430), (350, 320), (450, 245), (540, 180)]
    for a, b in zip(steps, steps[1:]):
        draw.line((a, b), fill="#ef4444", width=4)
    for x, y in steps:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#ef4444")
    draw.text((570, 175), "minimum loss", fill="#111827", font=LABEL_FONT)


def draw_confusion_matrix(draw):
    x0, y0, cell = 260, 180, 180
    labels = [("TP", "#bbf7d0"), ("FP", "#fecaca"), ("FN", "#fecaca"), ("TN", "#bbf7d0")]
    positions = [(x0, y0), (x0 + cell, y0), (x0, y0 + cell), (x0 + cell, y0 + cell)]
    for (label, color), (x, y) in zip(labels, positions):
        draw.rectangle((x, y, x + cell, y + cell), fill=color, outline="#111827", width=3)
        draw.text((x + 65, y + 65), label, fill="#111827", font=TITLE_FONT)
    draw.text((x0 + 40, y0 - 45), "Predicted +", fill="#111827", font=LABEL_FONT)
    draw.text((x0 + 220, y0 - 45), "Predicted -", fill="#111827", font=LABEL_FONT)
    draw.text((70, y0 + 70), "Actual +", fill="#111827", font=LABEL_FONT)
    draw.text((70, y0 + 250), "Actual -", fill="#111827", font=LABEL_FONT)


def draw_knn(draw):
    axes(draw)
    blue = [(260, 350), (310, 420), (360, 375), (390, 440)]
    orange = [(650, 260), (710, 300), (760, 240), (800, 330)]
    query = (470, 360)
    for x, y in blue:
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#2563eb")
    for x, y in orange:
        draw.rectangle((x - 12, y - 12, x + 12, y + 12), fill="#f97316")
    draw.ellipse((query[0] - 90, query[1] - 90, query[0] + 90, query[1] + 90), outline="#111827", width=3)
    draw.polygon([(470, 338), (492, 382), (448, 382)], fill="#111827")
    draw.text((500, 350), "query point", fill="#111827", font=LABEL_FONT)


def draw_kmeans(draw):
    axes(draw)
    clusters = [((300, 390), "#2563eb"), ((560, 270), "#16a34a"), ((760, 450), "#f97316")]
    for center, color in clusters:
        cx, cy = center
        for i in range(12):
            x = cx + int(math.cos(i) * 45 + math.sin(i * 2) * 18)
            y = cy + int(math.sin(i) * 38 + math.cos(i * 3) * 16)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw.line((cx - 16, cy, cx + 16, cy), fill="#111827", width=4)
        draw.line((cx, cy - 16, cx, cy + 16), fill="#111827", width=4)
    draw.text((400, 150), "centroids mark cluster centers", fill="#111827", font=LABEL_FONT)


def draw_train_test(draw):
    draw.rounded_rectangle((100, 220, 380, 420), radius=14, fill="#dbeafe", outline="#1d4ed8", width=3)
    draw.text((170, 300), "Dataset", fill="#111827", font=TITLE_FONT)
    draw.line((380, 320, 520, 250), fill="#111827", width=5)
    draw.line((380, 320, 520, 420), fill="#111827", width=5)
    draw.rounded_rectangle((520, 170, 880, 310), radius=14, fill="#bbf7d0", outline="#15803d", width=3)
    draw.rounded_rectangle((520, 370, 880, 510), radius=14, fill="#fed7aa", outline="#c2410c", width=3)
    draw.text((610, 220), "Train set", fill="#111827", font=TITLE_FONT)
    draw.text((625, 420), "Test set", fill="#111827", font=TITLE_FONT)


def draw_overfitting(draw):
    labels = ["Underfit", "Good fit", "Overfit"]
    x_offsets = [80, 370, 660]
    for label, x0 in zip(labels, x_offsets):
        draw.rectangle((x0, 180, x0 + 250, 540), outline="#94a3b8", width=3)
        draw.text((x0 + 55, 555), label, fill="#111827", font=LABEL_FONT)
        pts = []
        for i in range(8):
            x = x0 + 35 + i * 28
            y = 430 - int(math.sin(i / 1.2) * 70) - i * 8
            pts.append((x, y))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#2563eb")
        if label == "Underfit":
            draw.line((x0 + 35, 410, x0 + 220, 310), fill="#ef4444", width=4)
        elif label == "Good fit":
            draw.line(pts, fill="#ef4444", width=4)
        else:
            jagged = []
            for p in pts:
                jagged.append((p[0], p[1] - 35))
                jagged.append((p[0] + 12, p[1] + 35))
            draw.line(jagged, fill="#ef4444", width=4)


def draw_roc(draw):
    axes(draw, origin=(160, 570), x_end=(860, 570), y_end=(160, 150))
    draw.line((160, 570, 860, 150), fill="#94a3b8", width=3)
    curve = [(160, 570), (230, 380), (360, 260), (540, 200), (860, 150)]
    draw.line(curve, fill="#2563eb", width=6)
    draw.text((500, 260), "AUC", fill="#111827", font=TITLE_FONT)
    draw.text((400, 605), "False Positive Rate", fill="#111827", font=LABEL_FONT)
    draw.text((40, 280), "True Positive Rate", fill="#111827", font=LABEL_FONT)


def draw_naive_bayes(draw):
    boxes = [
        (80, 250, 270, 350, "Prior\nP(class)"),
        (330, 250, 540, 350, "Likelihood\nP(data|class)"),
        (610, 250, 850, 350, "Posterior\nP(class|data)"),
    ]
    for x1, y1, x2, y2, text in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill="#dbeafe", outline="#1d4ed8", width=3)
        draw.multiline_text((x1 + 25, y1 + 22), text, fill="#111827", font=LABEL_FONT, spacing=4)
    draw.text((285, 285), "+", fill="#111827", font=TITLE_FONT)
    draw.text((560, 285), "=", fill="#111827", font=TITLE_FONT)
    draw.text((330, 430), "Choose class with highest probability", fill="#111827", font=LABEL_FONT)


def draw_cnn(draw):
    draw.rectangle((90, 230, 250, 390), fill="#dbeafe", outline="#1d4ed8", width=3)
    for i in range(4):
        draw.line((90 + i * 40, 230, 90 + i * 40, 390), fill="#93c5fd", width=2)
        draw.line((90, 230 + i * 40, 250, 230 + i * 40), fill="#93c5fd", width=2)
    draw.rectangle((130, 270, 210, 350), outline="#ef4444", width=5)
    draw.line((250, 310, 380, 310), fill="#111827", width=4)
    draw.rectangle((380, 230, 540, 390), fill="#bbf7d0", outline="#15803d", width=3)
    draw.text((405, 300), "Feature\nMap", fill="#111827", font=LABEL_FONT)
    draw.line((540, 310, 680, 310), fill="#111827", width=4)
    draw.rounded_rectangle((680, 250, 890, 370), radius=14, fill="#fed7aa", outline="#c2410c", width=3)
    draw.text((725, 295), "Classify", fill="#111827", font=TITLE_FONT)
    draw.text((112, 420), "image", fill="#111827", font=LABEL_FONT)
    draw.text((128, 190), "filter", fill="#ef4444", font=LABEL_FONT)


def draw_attention(draw):
    labels = [("Query", 160, 240, "#dbeafe"), ("Key", 420, 240, "#bbf7d0"), ("Value", 680, 240, "#fed7aa")]
    for label, x, y, color in labels:
        draw.rounded_rectangle((x, y, x + 180, y + 90), radius=14, fill=color, outline="#111827", width=3)
        draw.text((x + 45, y + 28), label, fill="#111827", font=LABEL_FONT)
    draw.line((250, 330, 510, 440), fill="#111827", width=4)
    draw.line((510, 330, 510, 440), fill="#111827", width=4)
    draw.line((770, 330, 510, 440), fill="#111827", width=4)
    draw.rounded_rectangle((370, 440, 650, 540), radius=14, fill="#e0e7ff", outline="#4338ca", width=3)
    draw.text((405, 470), "Weighted attention", fill="#111827", font=LABEL_FONT)


DRAWERS = [
    draw_linear_regression,
    draw_svm,
    draw_pca,
    draw_decision_tree,
    draw_neural_network,
    draw_gradient_descent,
    draw_confusion_matrix,
    draw_knn,
    draw_kmeans,
    draw_train_test,
    draw_overfitting,
    draw_roc,
    draw_naive_bayes,
    draw_cnn,
    draw_attention,
]


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for item, drawer in zip(TOPICS, DRAWERS):
        image, draw = base_canvas(item["title"], item["topic"])
        drawer(draw)
        footer(draw, item["description"])

        file_name = f"{item['id']}.png"
        image_path = f"images/{file_name}"
        image.save(IMAGE_DIR / file_name)

        embedding_text = f"{item['topic']}. {item['title']}. {item['description']} Keywords: {item['search_queries']}."
        rows.append(
            {
                "id": item["id"],
                "file_name": image_path,
                "image_path": image_path,
                "topic": item["topic"],
                "title": item["title"],
                "description": item["description"],
                "search_queries": item["search_queries"],
                "embedding_text": embedding_text,
            }
        )

    csv_path = TRAIN_DIR / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} images in {IMAGE_DIR}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
