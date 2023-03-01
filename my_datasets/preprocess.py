import numpy as np
import pandas as pd

import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path().home() / 'Desktop/machine-learning-2022/my_datasets'


################################################################################
# BANK DATASET
################################################################################


def clean_bank(df):
    """Cleans the bank dataset. Returns a pd.DataFrame."""
    
    relevant_cols = [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        # 'day',
        # 'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
        'y'
    ]

    return (df
            [relevant_cols]
            .assign(
                default=lambda df_: df_.default.replace({'yes': True, 'no': False, np.nan: False}),
                housing=lambda df_: df_.housing.replace({'yes': True, 'no': False, np.nan: False}),
                loan=lambda df_: df_.loan.replace({'yes': True, 'no': False, np.nan: False}),
                y=lambda df_: df_.y.replace({'yes': True, 'no': False, np.nan: False}),
            )
            .astype({'age':'uint8', 'campaign': 'uint8', 'pdays': 'int16' ,'previous': 'uint8'})
            .astype({col: 'category' for col in ['job', 'marital', 'education', 'contact', 'poutcome']})
           )

def divide_bank(df):
    """Create dummy vars and divide into attributes and labels."""
    
    attributes = (df
                  .drop(columns='y')
                  .pipe(pd.get_dummies)
                  .pipe(StandardScaler().fit_transform)
                 )
    labels = df.y
    
    return attributes, labels


def load_bank(return_X_y=False):
    """Summary function to load the bank dataset."""
    
    bank = pd.read_csv(str(CURRENT_DIR / 'bank-full.csv'), sep=';')
    bank = clean_bank(bank)
    if return_X_y:
        return divide_bank(bank)
    return bank


################################################################################
# OTHER DATASET
################################################################################