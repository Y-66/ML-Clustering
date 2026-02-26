############################################################
# ERROR ANALYSIS FUNCTION
############################################################

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def error_analysis(model_name, data, true_encoded, pred_clusters, texts, encoder):

    print(f"\n===== ERROR ANALYSIS: {model_name} =====")

    # Confusion Matrix
    cm = confusion_matrix(true_encoded, pred_clusters)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"Confusion_Matrix_{model_name}.png", dpi=300)
    plt.close()

    # Category agreement
    label_names = encoder.classes_
    category_agreement = {}

    for i, label in enumerate(label_names):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0
        category_agreement[label] = acc

    print("\nCategory-wise agreement:")
    for k, v in category_agreement.items():
        print(k, round(v, 3))

    # Worst category
    worst_category = min(category_agreement,
                         key=category_agreement.get)

    print("Worst clustered category:", worst_category)

    worst_index = list(label_names).index(worst_category)

    misclassified = data[
        (true_encoded == worst_index) &
        (pred_clusters != worst_index)
    ]

    print("Misclassified samples:",
          len(misclassified))

    # Top confusing words
    if len(misclassified) > 0:

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=15
        )

        X_err = vectorizer.fit_transform(
            misclassified["Cleaned_Content"]
        )

        top_words = vectorizer.get_feature_names_out()

        print("Top confusing words:")
        print(top_words)

        # Save to txt file
        with open(f"Error_{model_name}_TopWords.txt", "w") as f:
            f.write("Worst Category: " + worst_category + "\n")
            f.write("Misclassified Samples: " +
                    str(len(misclassified)) + "\n\n")
            f.write("Top Confusing Words:\n")
            for w in top_words:
                f.write(w + "\n")