# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # Updated metrics
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set MLflow tracking URI from environment variable, fallback to localhost if not set
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

api = HfApi(token=os.getenv("HF_TOKEN"))


Xtrain_path = "hf://datasets/Shaggys86/tourism/Xtrain.csv"
Xtest_path = "hf://datasets/Shaggys86/tourism/Xtest.csv"
ytrain_path = "hf://datasets/Shaggys86/tourism/ytrain.csv"
ytest_path = "hf://datasets/Shaggys86/tourism/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
original_numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
    ]


# List of categorical features in the dataset and their nature based upon data definition
categorical_features_ordinal = [
    'CityTier',
    'PreferredPropertyStar',
    'PitchSatisfactionScore'
]

categorical_features_nominal = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation'
]

# Combine original numerical features with ordinal features for scaling
features_to_scale = original_numeric_features + categorical_features_ordinal

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), features_to_scale),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features_nominal)
)

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss') # Changed to XGBClassifier

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__subsample': [0.7, 0.8],
    'xgbclassifier__colsample_bytree': [0.7, 0.8],
    'xgbclassifier__reg_lambda': [0.1, 1]
} # Updated param_grid keys

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc') # Changed scoring
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score) # Updated metric name

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics - Updated to classification metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train)
    test_precision = precision_score(ytest, y_pred_test)

    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    train_roc_auc = roc_auc_score(ytrain, best_model.predict_proba(Xtrain)[:, 1])
    test_roc_auc = roc_auc_score(ytest, best_model.predict_proba(Xtest)[:, 1])

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })


    # Save the model locally
    model_path = "best_tourism_pkg_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Shaggys86/tourism_pkg_prediction_model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_pkg_prediction_model_v1.joblib",
        path_in_repo="best_tourism_pkg_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
