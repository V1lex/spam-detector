from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

DATA_URL = "hf://datasets/AbdulHadi806/mail_spam_ham_dataset/mail_data.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df = df[["Category", "Message"]].dropna().drop_duplicates()
    return df


def get_models() -> dict:
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC())
        ]),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision_spam": precision_score(y_test, preds, pos_label="spam"),
        "recall_spam": recall_score(y_test, preds, pos_label="spam"),
        "f1_spam": f1_score(y_test, preds, pos_label="spam"),
        "predictions": preds,
    }


def save_model(model, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def print_model_report(model_name: str, y_test, preds) -> None:
    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")
    print(classification_report(y_test, preds))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    df = load_data()

    X = df["Message"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y,
    )

    models = get_models()

    results = []
    best_model = None
    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision (spam)": metrics["precision_spam"],
            "Recall (spam)": metrics["recall_spam"],
            "F1-score (spam)": metrics["f1_spam"],
        })

        print_model_report(name, y_test, metrics["predictions"])

        if metrics["f1_spam"] > best_f1:
            best_f1 = metrics["f1_spam"]
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).sort_values(
        by="F1-score (spam)",
        ascending=False
    )

    print("\nFinal comparison:")
    print(results_df.to_string(index=False))

    safe_model_name = best_model_name.lower().replace(" ", "_")
    model_path = project_root / "models" / f"spam_classifier_{safe_model_name}.joblib"

    save_model(best_model, model_path)

    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Model size: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    main()