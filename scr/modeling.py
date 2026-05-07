import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


def create_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn label based on customer purchase frequency.
    """

    customer_orders = df.groupby(
        'customer_unique_id'
    )['order_id'].nunique()

    df['churn'] = df['customer_unique_id'].map(
        lambda x: 1 if customer_orders[x] == 1 else 0
    )

    return df


def prepare_model_data(
    df: pd.DataFrame,
    features: list,
    target: str
):
    """
    Prepare train/test datasets.
    """

    model_df = df[features + [target]].dropna()

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train,
    y_train
):
    """
    Train logistic regression model.
    """

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(
    X_train,
    y_train
):
    """
    Train random forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate classification model.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    return predictions


def get_feature_importance(
    model,
    features: list
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based model.
    """

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='importance',
        ascending=False
    )

    return importance_df