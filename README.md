# AER850 Project 1: Step Classification

Alina Saleem
501129840
AER 850
Project 1

## Overview
The project is based on the augmented reality based instruction modules, actively growing in the manufacturing and maintenance sectors within the aerospace industry. A micro-level coordinate system is experimented with using multi-class classiication-based ML algorithms, to predict the maintenance step given a specific part and its coordinates. The device being ued in this project is an inverted of the 'FlightMax Fill Motion Simulator'.

<img width="630" height="490" alt="image" src="https://github.com/user-attachments/assets/19eadc3a-5034-4435-a2f1-ac2215068e42" />

There are 13 unique steps which, in this project is commonly referred to as a 'feature', these steps are defined within the process of disassembling the inverter. Each step has X, Y, Z axis points which are referred to as 'classes'. 

--- 
## Models implemented

**1** SVC, with GridSearchCV
**2** KNN, with GridSearchCV
**3** Random Forest, with with GridSearchCV
**4** Random Forest, with RandomizedSearchCV

---
## Packages imported
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

---
## How to Run

Using Python 3.11+ installed 
IDE

Clone this repo

Download file 'Project 1 Data.csv'

Run the python script:
Python script labeled 'Project_1.py'

---
## output file

final_trained_model.joblib

