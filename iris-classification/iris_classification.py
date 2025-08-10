import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_and_preprocess_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, iris.target_names, iris.feature_names

def perform_eda(X, y, feature_names, target_names):
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = y
    df["species_name"] = df["species"].map(lambda x: target_names[x])

    print("Dataset head:\n", df.head())
    print("\nDataset info:")
    df.info()
    print("\nDataset description:\n", df.describe())
    print("\nSpecies distribution:\n", df["species_name"].value_counts())

    sns.pairplot(df, hue="species_name", palette="viridis")
    plt.suptitle("Pair Plot of Iris Features by Species", y=1.02)
    plt.savefig("Beginner Projects/Iris Flower Classification/reports/iris_pairplot.png")
    plt.close()
    print("Saved pairplot to reports/iris_pairplot.png")

    for feature in feature_names:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="species_name", y=feature, data=df, hue="species_name", palette="viridis", legend=False)
        plt.title(f"Box Plot of {feature} by Species")
        plt.savefig(f"Beginner Projects/Iris Flower Classification/reports/iris_boxplot_{feature.replace(' ', '_')}.png")
        plt.close()
        print(f"Saved boxplot for {feature} to reports/iris_boxplot_{feature.replace(' ', '_')}.png")

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target_names):
    print(f"\n--- Training and Evaluating {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    #confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"Beginner Projects/Iris Flower Classification/reports/confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()
    print(f"Saved confusion matrix for {model_name} to reports/confusion_matrix_{model_name.replace(' ', '_')}.png")

    return accuracy

def main():
    os.makedirs("Beginner Projects/Iris Flower Classification/reports", exist_ok=True)

    X_train, X_test, y_train, y_test, target_names, feature_names = load_and_preprocess_data()
    iris_full = load_iris()
    perform_eda(iris_full.data, iris_full.target, iris_full.feature_names, iris_full.target_names)

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Support Vector Machine": SVC(kernel="linear", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42)
    }

    accuracies = {}
    for name, model in models.items():
        accuracies[name] = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name, target_names)

    print("\n--- Model Comparison ---")
    for name, acc in accuracies.items():
        print(f"{name}: Accuracy = {acc:.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=list(accuracies.keys()), palette="viridis", legend=False)
    plt.title("Model Accuracies on Iris Dataset")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("Beginner Projects/Iris Flower Classification/reports/model_accuracies.png")
    plt.close()
    print("Saved model accuracies plot to reports/model_accuracies.png")

if __name__ == "__main__":
    main()


