# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Shaggys86/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier columns (not useful for modeling)
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)

# replacing the Fe Male value in gender column with Female as this looks to be a mistake in the data
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Dropping any rows with duplicate data.
df.drop_duplicates(inplace=True)

# dropping any rows with null data from the dataframe
df = df.dropna()

# Define the target variable for the classification task
target = 'ProdTaken'

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

# Split into X (features) and y (target)
X = df.drop(columns=[target])
y = df[target]

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Shaggys86/tourism-package-prediction",
        repo_type="dataset",
    )
