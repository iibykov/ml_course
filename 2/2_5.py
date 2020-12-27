import pandas as pd

vis_data = pd.read_csv("train.csv", encoding='ISO-8859-1', low_memory=False)

# iterating the columns (state)
# for col in vis_data.columns:
#    print(col)

items_counts = vis_data['state'].value_counts(sort=False)
top = items_counts.loc[[items_counts.idxmax()]]
value, count = top.index[0], top.iat[0]
# print(value, count)

result = vis_data['state'].fillna(value)
