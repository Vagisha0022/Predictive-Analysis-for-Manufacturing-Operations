
import pandas as pd
import numpy as np

df = pd.read_csv("predictive_maintenance.csv")
df = df.drop(['UDI', 'Type', 'Air temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Failure Type'], axis=1)
df.to_csv('final.csv', index=False)
print(df.head())
