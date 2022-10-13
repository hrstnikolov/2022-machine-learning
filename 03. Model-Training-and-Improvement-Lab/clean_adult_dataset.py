import pandas as pd
import numpy as np


income_data = pd.read_csv("../datasets/adult/adult.data", header=None)
income_data.columns = [
    "age",
    "workclass",
    "final_weight",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_class",
]

income_target = income_data.income_class
income_target = income_target.str.strip()

income_attributes = income_data.drop(columns="income_class")
income_attributes = pd.get_dummies(income_attributes, drop_first=True)
scaler = MinMaxScaler()
income_attributes = scaler.fit_transform(income_attributes)

income_attributes_train, income_attributes_test, \
income_target_train, income_target_test, \
= train_test_split(income_attributes, income_target, train_size=0.8, test_size=0.2)