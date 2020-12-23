import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

vis_data = pd.read_csv("train.csv", encoding='ISO-8859-1', low_memory=False)

# iterating the columns (balance_due)
# for col in vis_data.columns:
#    print(col)

balance_due = vis_data["balance_due"]

scaler = StandardScaler()
result = scaler.fit_transform(balance_due.values.reshape(-1, 1))
print(np.min(result))
