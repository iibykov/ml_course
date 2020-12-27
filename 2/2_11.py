from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

vis_data = pd.read_csv("train.csv", encoding='ISO-8859-1', low_memory=False)

# iterating the columns (balance_due, discount_amount)
# for col in vis_data.columns:
#     print(col)

# 2.11.6
pf = PolynomialFeatures(3)
poly_features = pf.fit_transform(vis_data[['balance_due', 'discount_amount']])
print(np.mean(poly_features, axis=0))

# 2.11.7
# ticket_issued_date
ticket_issued_date = pd.to_datetime(vis_data.ticket_issued_date.dropna())
is_weekend = (ticket_issued_date.dt.weekday > 4)
print(is_weekend.value_counts())
