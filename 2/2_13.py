import pandas as pd
import numpy as np

vis_data = pd.read_csv("train.csv", encoding='ISO-8859-1', low_memory=False)

# iterating the columns (balance_due)
# for col in vis_data.columns:
#     print(col)

print(vis_data.balance_due.count())
balance = vis_data.balance_due.dropna()
print(balance.count())

max_b = np.max(balance)
min_b = np.min(balance)
print(max_b - min_b)
