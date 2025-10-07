# -*- coding: utf-8 -*-
"""

Intro to Machine Learning
AER 850
Alina Saleem
501129840
10/01/2025

Project 1
"""
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
#Step 1: Data Processing

#reading the data

data=pd.read_csv("C:/Users/alina/Documents/GitHub/AER850_Project1/Project 1 Data.csv")

#Step 2: Data Visualization
#data splitting using stratified sampling

X = data[['X', 'Y', 'Z']]
y = data['Step']

#stratified train/test split
#plitting the data into training 
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)

for train_idx, test_idx in splitter.split(X,y):
    strat_train=data.iloc[train_idx].reset_index(drop=True)
    strat_test=data.iloc[test_idx].reset_index(drop=True)
 
    # test will be done with 20% of data 
X_train = strat_train.drop("Step", axis=1)
y_train = strat_train["Step"]
X_test = strat_test.drop("Step", axis=1)
y_test = strat_test["Step"]

per_class_stats = strat_train.groupby("Step")[["X","Y","Z"]] \
    .agg(["mean","std","min","max","median","count"])
    

train_counts = y_train.value_counts().sort_index()
test_counts  = y_test.value_counts().sort_index()

print("Train class counts:\n", train_counts)
print("\nTest class counts:\n", test_counts)


#histogram
#bc this is a multi class classification problem, histograms can
#help give a sense of range/outliers
X_train.hist(bins=50, figsize=(12,4))
plt.suptitle("Histogram (Train)")
plt.show()


#boxplots
steps_sorted=np.sort(strat_train["Step"].unique())
#helps to extract certain values for each class
def feature_by_step(data,feature):
    return [data.loc[data["Step"]==s,feature].values for s in steps_sorted]

#boxplot for X
plt.figure()
plt.boxplot(feature_by_step(strat_train, "X"), tick_labels=steps_sorted)
plt.title("X distribution per step — train")
plt.xlabel("Step"); plt.ylabel("X")
plt.tight_layout(); plt.show()

#boxplot for Y
plt.figure()
plt.boxplot(feature_by_step(strat_train, "Y"), tick_labels=steps_sorted)
plt.title("Y distribution per step — train")
plt.xlabel("Step"); plt.ylabel("Y")
plt.tight_layout(); plt.show()

#boxplot for Z
plt.figure()
plt.boxplot(feature_by_step(strat_train, "Z"), tick_labels=steps_sorted)
plt.title("Z distribution per step — train")
plt.xlabel("Step"); plt.ylabel("Z")
plt.tight_layout(); plt.show()

#2d scatter plot
#each plot compares classes as projections 
#XY Projection 
plt.figure()
sc=plt.scatter(strat_train["X"],strat_train["Y"],strat_train["Step"])
plt.title("XY projection colored by step for Train set")
plt.xlabel("X"); plt.ylabel("Y")
cb=plt.colorbar(sc)
cb.set_label("Step")
plt.tight_layout(); plt.show()

#XZ Projection
plt.figure()
sc=plt.scatter(strat_train["X"],strat_train["Z"],strat_train["Step"])
plt.title("XZ projection colored by step for Train set")
plt.xlabel("X"); plt.ylabel("Z")
cb=plt.colorbar(sc)
cb.set_label("Step")
plt.tight_layout(); plt.show()

#YZ projection
plt.figure()
sc=plt.scatter(strat_train["Y"],strat_train["Z"],strat_train["Step"])
plt.title("YZ projection colored by step for Train set")
plt.xlabel("Y"); plt.ylabel("Z")
cb=plt.colorbar(sc)
cb.set_label("Step")
plt.tight_layout(); plt.show()


#step 3 correlation analysis 
#correlation between features
#features are the XYZ coords where target is step

#feature to feature correlation is corr_features
corr_features = X_train.corr(method="pearson")
print("feature-feature Pearson (train): \n", corr_features, "\n")

plt.figure(figsize=(4,3))
sns.heatmap(np.abs(corr_features), annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
plt.title("absolute pearson correlation: feature-feature (train)")
plt.tight_layout()
plt.show()

#feature-target correlation
corr_data=strat_train[["X", "Y", "Z", "Step"]].copy()
#could also do corr_matrix=strat_train.corr(method="pearson")

corr_all=corr_data.corr(method="pearson")
print("feature-target pearson (train):")
print(corr_all["Step"].loc[["X", "Y", "Z"]], "\n")
plt.figure(figsize=(5,4))
sns.heatmap(np.abs(corr_all), annot=True, cmap="coolwarm", center=0, fmt=".2f", square="True")
plt.title("pearson correlation: features-step (train)")
plt.tight_layout()
plt.show()

# step 4: classification model development/engineering


#model 1 svm with scaling and gridsearchCV
#cross validation
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#svc with scaling 
svc_pipe=Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=False))
                   ])

svc_param_grid={
    "clf__kernel": ["rbf", "linear", "poly"],
    #regularization strength
    "clf__C": [0.1,1,10,100],
    #kernel coeff
    "clf__gamma": ["scale", "auto"]}

#testing every combo, using macro-f1
svc_grid = GridSearchCV(
    estimator=svc_pipe,
    param_grid=svc_param_grid,
    scoring = "f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=0)

#refitting the best svc config on all training data
svc_grid.fit(X_train, y_train)

#model 2 KNN with scaling and gridsearchCV

#knn with scaling 
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

knn_param_grid = {
    "clf__n_neighbors": [3,5,7,9,11],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1,2]
    }

#knn with grid search cv 
knn_grid = GridSearchCV(
    estimator=knn_pipe,
    param_grid=knn_param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=0)
knn_grid.fit(X_train,y_train)

#model 3 random forest classifier with grid search cv 
#various descision trees averaged together 
rf=RandomForestClassifier(random_state=42)


rf_param_grid ={
    "n_estimators": [100,200,400],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4],
    "max_features": ["sqrt", "log2"]
    }

#rf with grid search cv
rf_grid= GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=0
    )

rf_grid.fit(X_train,y_train)

#model 4 random forest with randomnized search cv

rf = RandomForestClassifier(random_state=42)

rf_param_dist={
    "n_estimators": np.arange(100,1000,100),
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4],
    "max_features":["sqrt", "log2", None],
    "bootstrap": [True, False]
    }

#rf with randomized search cv 
rf_rand_search= RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_dist,
    n_iter=30,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    refit=True,
    verbose=1
    )

rf_rand_search.fit(X_train,y_train)
print("Best SVC params:", svc_grid.best_params_)
print("Best KNN params:", knn_grid.best_params_)
print("Best RF params:", rf_grid.best_params_)
print("Best RF (w/ randomnized search cv) params:", rf_rand_search.best_params_)

#displying the best models 
best_svc = svc_grid.best_estimator_
best_knn = knn_grid.best_estimator_
best_rf  = rf_grid.best_estimator_
best_rf_rand  = rf_rand_search.best_estimator_

#step 5 model performance 

#used to compute and print metrics
#macro avergaring used to evaluate every class 

def eval_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)                         
    acc  = accuracy_score(y_te, y_pred)                   
    prec = precision_score(y_te, y_pred, average="macro", 
                           zero_division=0)
    rec  = recall_score(y_te, y_pred, average="macro",   
                        zero_division=0)
    f1   = f1_score(y_te, y_pred, average="macro")  
    
    print(f"{name:12s} | acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    return y_pred, acc, prec, rec, f1                 

#runs each best-rated model on unseen test data 

models = {
    "SVC": best_svc,
    "KNN": best_knn,
    "RF":  best_rf,
    "RF (random search)":  best_rf_rand
}


results = {}
for name, mdl in models.items():
    y_pred, acc, prec, rec, f1 = eval_model(name, mdl, X_test, y_test)
    results[name] = {
        "model": mdl, "y_pred": y_pred,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1
    }
    

#sort with macro-f1 scores 

ranked = sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True)
print("ranking by macro-F1:")

for name, res in ranked:
    print(f"{name:12s} | F1={res['f1']:.3f}  Acc={res['acc']:.3f}  "
          f"Prec={res['prec']:.3f}  Rec={res['rec']:.3f}")

best_name, best_res = ranked[0]
best_model = best_res["model"]



print(f"\nSelected best model (by macro-F1): {best_name}")

#per-class f1 result
print("classification report:")
print(classification_report(y_test, best_res["y_pred"], digits=3))

#confusion matrix
#shows where the model confuses one class for another
steps_sorted = np.sort(y_train.unique())                 
cm = confusion_matrix(y_test, best_res["y_pred"],        
                      labels=steps_sorted)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=steps_sorted, yticklabels=steps_sorted)


plt.xlabel("predicted"); plt.ylabel("true")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout(); plt.show()

#step 6  stacked model performance 
#picking svc and rf 
#combining predictions of svc and rf 
base_estimators = [
    ("svc", best_svc),
    ("rf_rand", best_rf_rand)
    ]

final_meta=LogisticRegression(max_iter=1000, random_state=42)

stack= StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_meta,
    cv=cv,
    stack_method="auto",
    passthrough=False,
    n_jobs=-1
    )

#training stack then training
stack.fit(X_train, y_train)
 
y_pred_stack=stack.predict(X_test)

#model performance metrics for stacked model 
acc_s = accuracy_score(y_test, y_pred_stack)

prec_s=precision_score(y_test, y_pred_stack, average="macro", zero_division=0)

rec_s=precision_score(y_test, y_pred_stack, average="macro", zero_division=0)

f1_s=f1_score(y_test, y_pred_stack, average="macro")
 
print("stacked model")
print(f"Accuracy={acc_s:.3f}  Precision(macro)={prec_s:.3f}  Recall(macro)={rec_s:.3f}  F1(macro)={f1_s:.3f}")

#confusion matrix for stack
steps_sorted=np.sort(y_train.unique())
cm_s=confusion_matrix(y_test, y_pred_stack, labels=steps_sorted) 

plt.figure(figsize=(6,5))
sns.heatmap(cm_s, annot=True, fmt="d", cmap="Purples", xticklabels=steps_sorted, yticklabels=steps_sorted)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("confusion matrix-stacking svc and rf")
plt.tight_layout()
plt.show()


# step 7 model packaging

#reuse training model
final_model=stack

joblib.dump(final_model, "final_trained_model.joblib")
print("model saves as final_trained_model.joblib")

loaded_model=joblib.load("final_trained_model.joblib")
print("model loaded successfully")

#table of new measurements
#setting up the model to predict class(step) for each row

new_points = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
],
    columns=["X", "Y", "Z"])

#printing input row and respecive predicted class(step)

predicted_steps=loaded_model.predict(new_points)

#iterating column nanmes

for i, coords in enumerate(new_points):
    print(f"Coordinates {coords} → Predicted Maintenance Step: {predicted_steps[i]}")

#saving the last confusion matrix as png 
plt.savefig("confusion_matrix_RF.png", dpi=200, bbox_inches="tight")
