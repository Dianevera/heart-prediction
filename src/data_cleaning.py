import pandas as pd
import os

from sklearn.impute import KNNImputer

df = pd.read_csv("data/frmgham2.csv")

df['SEX'].where(df.SEX != 2, 0, inplace = True)

# KNN to fill in data

"""
We use the K-nearest-neighbours that looks at the nearest neighbors to an empty variable and then gives the empty data an average value of the 5 nearest values.
"""

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

df.loc[:, df.columns != 'BMI'] = round(df.loc[:, df.columns != 'BMI'])

# Save the data

out_path = "data/clean_data.csv"
if os.path.isfile(out_path):
    os.remove(out_path)
df.to_csv(out_path, index=False)
