import pandas as pd
import numpy as np

vis_data = pd.read_csv("train.csv", encoding='ISO-8859-1', low_memory=False)

# iterating the columns (balance_due)
# for col in vis_data.columns:
#    print(col)

balance_due = vis_data["balance_due"]
res = np.sqrt(vis_data.balance_due[vis_data.balance_due > 0])
print(np.median(res) - np.mean(res))
