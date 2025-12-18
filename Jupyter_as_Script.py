# %% [markdown]
# # Appendix A

# %%
import pandas as pd
import pycaret
import numpy as np
import copy

# %% [markdown]
# ## Read Raw Data

# %%
df_raw = pd.read_csv('training_data_ht2025.csv')

# %%
df_raw

# %% [markdown]
# ### Check Data Contents

# %%
df_raw['increase_stock'].value_counts()

# %% [markdown]
# Based on above code that shows the values on the target variable *increase_stock*, there's a problem of Imbalance Dataset, where the records with low bike demand are much more compared to records with high bike demands.

# %%
df_raw['holiday'].value_counts()

# %%
df_raw['weekday'].value_counts()

# %%
df_raw['summertime'].value_counts()

# %%
df_raw['increase_stock'].value_counts()

# %%
df_raw['hour_of_day'].value_counts()

# %%
df_raw['day_of_week'].value_counts()

# %%
df_raw['month'].value_counts()

# %%
df_raw['snow'].value_counts()

# %% [markdown]
#  Note: The Snow Features only have one value: 0, and therefore isn't really useful for this dataset. And In Feature selection we can drop them.

# %%
df_raw.isnull().sum()

# %% [markdown]
# Based on the checking results, there're no NaN values, so we can proceed with feature processing.

# %% [markdown]
# #### Cyclical Encoding

# %% [markdown]
# Function to encode Ordinal Variables into Cyclical encoding using Sine and Cosine

# %%
def cyclical_encode(x, max_val, start_val=0):
    x_arr = np.asarray(x, dtype=float)

    # If values are 1..period (e.g. months 1..12), shift to 0..period-1
    if not start_val:
        x_arr = x_arr - 1

    angle = 2 * np.pi * x_arr / max_val
    sin_x = np.sin(angle)
    cos_x = np.cos(angle)
    
    if np.isscalar(x):
        return float(sin_x), float(cos_x)

    return sin_x, cos_x

# %%


# %%
# Transform Cyclical Values for hour_of_day

# Transform Cyclical Values for hour_of_day
# Hour of day ranges from 0 to 23
df_hour_feat = pd.DataFrame(df_raw['hour_of_day'].apply(lambda x: cyclical_encode(x, max(df_raw['hour_of_day']), min(df_raw['hour_of_day']))).to_list())
df_hour_feat.columns = ['hour_of_day_sin', 'hour_of_day_cos']
df_hour_feat


# Transform Cyclical Values for day_of_week
# Day of week ranges from 0 (Monday) to 6 (Sunday)
df_day_feat = pd.DataFrame(df_raw['day_of_week'].apply(lambda x: cyclical_encode(x, 6, 0)).to_list())
df_day_feat.columns = ['day_of_week_sin', 'day_of_week_cos']
df_day_feat

# Transform Cyclical Values for month
# Month ranges from 1 to 12
df_month_feat = pd.DataFrame(df_raw['month'].apply(lambda x: cyclical_encode(x, 12, 1)).to_list())
df_month_feat.columns = ['month_sin', 'month_cos']
df_month_feat

# Concat Cyclical Features
df_cyclical = pd.concat([df_hour_feat,df_day_feat, df_month_feat], axis=1)
df_cyclical

# %%
# Creating Final Feature and Target DataFrames
df_features = copy.deepcopy(df_raw)

# Creating Target Variable and Mapping text to binary
df_target = df_features['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# Dropping original cyclical columns and target from features
df_features = df_features.drop(['hour_of_day', 'day_of_week', 'month', 'increase_stock'], axis=1)

# Concatenating Cyclical Features to Features DataFrame
df_features = pd.concat([df_cyclical, df_features], axis=1)

df_features

# %%
df_target

# %%
# Correlation Analysis between Features and Target
df_corr = pd.concat([df_features, df_target], axis=1).corr()
df_corr['increase_stock'].sort_values(key=abs, ascending=False)

# %% [markdown]
# Based on above result, The variables hour_of_day_cos, temp, humidity, hour_of_day_sin, and summertime held the biggest correlation to the target variable. Because of this, we could check the trends more based on those attributes

# %%
from imblearn.over_sampling import SMOTE

# Applying SMOTE to Balance the Dataset
smote = SMOTE(sampling_strategy='minority')

print("Before SMOTE:\n", df_target.value_counts())

print()

X_resampled, y_resampled = smote.fit_resample(df_features, df_target)

print("After SMOTE:\n", y_resampled.value_counts())

# Note: Don't use SMOTE outside training data to avoid data leakage, this block is just for experimentation purposes. In practice, SMOTE should only be applied to the training set within each cross-validation folds.


# %% [markdown]
# ## Model Experiment

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# #### K-Fold Classification Report

# %%
## Function to do K-Fold Cross Validation and return the classification report
def kfold_classification_reports(clf, X, y, n_splits=10, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    reports = []
    
    for train_index, test_index in kf.split(X):
        # Applying SMOTE to Balance the Dataset
        smote = SMOTE(sampling_strategy='minority')
        
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        reports += [report]

    detailed = []
    for i, rep in enumerate(reports):
        fold_result = {
            "fold": i + 1,
            "accuracy": rep["accuracy"],
            "precision": rep["weighted avg"]["precision"],
            "recall": rep["weighted avg"]["recall"],
            "f1": rep["weighted avg"]["f1-score"],
        }
        detailed.append(fold_result)

    # compute averaged (generalized) metrics
    accuracies = [d["accuracy"] for d in detailed]
    precisions = [d["precision"] for d in detailed]
    recalls = [d["recall"] for d in detailed]
    f1s = [d["f1"] for d in detailed]

    report_dict = {
        "general": {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        },
        "detailed": detailed
    }
    
    return report_dict

# %% [markdown]
# ## Benchmark Model

# %% [markdown]
# For the benchmark model, we use a naive model that predict each instance as the majority class in the training dataset. This will provide a baseline accuracy to compare the performance of more sophisticated models.

# %%
from sklearn.dummy import DummyClassifier

clf_dummy = DummyClassifier(strategy="stratified")

# %%
reports = kfold_classification_reports(clf_dummy, df_features, df_target)
reports['general']

# %% [markdown]
# ## Logistic Regression

# %% [markdown]
# For Logistic Regression,  we will do hyperparameter tuning on Regularization Strength (*C*), Penalty type (*penalty*), and Solver type (*solver*)

# %%


# %%

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

#Grid Search to find the best hyperparameters for Logistic Regression
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg']
    },
    scoring='f1',
    cv=10,
    n_jobs=-1,
    verbose=1
)
clf_grid_lg = grid_search.fit(df_features, df_target)

# %%
clf_grid_lg.best_params_

# %%
clf_lg = LogisticRegression(**clf_grid_lg.best_params_)

# %%
reports = kfold_classification_reports(clf_lg, df_features, df_target)
reports['general']

# %% [markdown]
# ## Linear Discriminant Analysis (LDA)

# %% [markdown]
# For LDA, we us Hyperparameter tuning for Solver Type (*solver*)

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()

# Grid Search to find the best hyperparameters for LDA
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={
        'solver': ['svd', 'lsqr', 'eigen']
    },
    scoring='f1',
    cv=10,
    n_jobs=-1,
    verbose=1
)
clf_grid_lda = grid_search.fit(df_features, df_target)


# %%
clf_grid_lda.best_params_

# %%
clf_lda = LinearDiscriminantAnalysis(**clf_grid_lda.best_params_)
clf_lda

# %%
reports = kfold_classification_reports(clf_lda, df_features, df_target)
reports['general']

# %% [markdown]
# ## K Nearest Neighbor (KNN)

# %% [markdown]
# For K Nearest Neighbor, we use Hyperparameter tuning for selecting the numer of neighbors (*n_neighbors*), weight function used in prediction (*weights*), and the metric used for distance computation (*metric*)

# %%
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

#Grid Search to find the best hyperparameters for KNN
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={
        'n_neighbors': range(3, 21, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    scoring='f1',
    cv=10,
    n_jobs=-1,
    verbose=1
)
clf_grid_knn = grid_search.fit(df_features, df_target)

# %%
clf_grid_knn.best_params_

# %%
clf_knn = KNeighborsClassifier(**clf_grid_knn.best_params_)
clf_knn

# %%
reports = kfold_classification_reports(clf_knn, df_features, df_target)
reports['general']

# %% [markdown]
# ## Random Forest Classifier

# %% [markdown]
# For Random Forest Classifier, we use Hyperparameter tuning for selecting number of trees (*n_estimator*), maximum depth of each trees (*max_depth*), minimum number of samples required to split (*min_samples_split*), and minimum samples required to be a leaf node (*min_samples_leaf*)

# %%
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={
        'n_estimators': [20, 50, 100, 200, 250],
        'max_depth': [3, 5, 8, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    scoring='f1',
    cv=10,
    n_jobs=-1,
    verbose=1
)
clf_grid_rf = grid_search.fit(df_features, df_target)

# %%
clf_grid_rf.best_params_

# %%
clf_rf = RandomForestClassifier(**clf_grid_rf.best_params_)
clf_rf

# %%
reports = kfold_classification_reports(clf_rf, df_features, df_target)
reports['general']

# %% [markdown]
# ## Gradient Boosting

# %% [markdown]
# For Gradient Boosting, we use Hyperparameter tuning to determine, the number of sequential estimator (*n_estimators*), the learning rate (*learning_rate*), Maximum depth of the individual regression estimators (*max_depth*), and the fraction of samples to be used for fitting the individual estimators (*subsample*).

# %%
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
grid_search = GridSearchCV(
    estimator=clf,
    param_grid={
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
        'max_depth': [3, 5, 8, 10],
        'subsample': [0.6, 0.8, 1.0]
    },
    scoring='f1',
    cv=10,
    n_jobs=-1,
    verbose=1
)
clf_grid_gb = grid_search.fit(df_features, df_target)

# %%
clf_grid_gb.best_params_

# %%
clf_gb = GradientBoostingClassifier(**clf_grid_gb.best_params_)
clf_gb

# %%
reports = kfold_classification_reports(clf_gb, df_features, df_target)
reports['general']

# %% [markdown]
# ### Saving Model

# %% [markdown]
# Saving all model so we can use it later

# %%
# Applying SMOTE to Balance the Full Training Dataset
smote = SMOTE(sampling_strategy='minority')

print("Before SMOTE:\n", df_target.value_counts())

print()

X_resampled, y_resampled = smote.fit_resample(df_features, df_target)

print("After SMOTE:\n", y_resampled.value_counts())

# %%
import pickle

# Save the Dummy Classifier model used as benchmark
with open('clf_dummy.pkl', 'wb') as f:
    pickle.dump(clf_dummy, f)

# %%

# Train the Logistic Regression model on the full training dataset
clf_lg = LogisticRegression(**clf_grid_lg.best_params_)
clf_lg.fit(X_resampled, y_resampled)

# Save the trained Logistic Regression model
with open('clf_lg.pkl', 'wb') as f:
    pickle.dump(clf_lg, f)

# %%
# Train the LDA model on the full training dataset
clf_lda = LinearDiscriminantAnalysis(**clf_grid_lda.best_params_)
clf_lda.fit(X_resampled, y_resampled)


# Save the trained LDA model
with open('clf_lda.pkl', 'wb') as f:
    pickle.dump(clf_lda, f)

# %%
# Train the KNN model on the full training dataset
clf_knn = KNeighborsClassifier(**clf_grid_knn.best_params_)
clf_knn.fit(X_resampled, y_resampled)

# Save the trained KNN model
with open('clf_knn.pkl', 'wb') as f:
    pickle.dump(clf_knn, f)

# %%
# Train the Random Forest Classifier model on the full training dataset
clf_rf = RandomForestClassifier(**clf_grid_rf.best_params_)
clf_rf.fit(X_resampled, y_resampled)

# Save the trained Random Forest Classifier model
with open('clf_rf.pkl', 'wb') as f:
    pickle.dump(clf_rf, f)

# %%
# Train the Gradient Boosting Classifier model on the full training dataset
clf_gb = GradientBoostingClassifier(**clf_grid_gb.best_params_)
clf_gb.fit(X_resampled, y_resampled)

# Save the trained Gradient Boosting Classifier model
with open('clf_gb.pkl', 'wb') as f:
    pickle.dump(clf_gb, f)

# %%



