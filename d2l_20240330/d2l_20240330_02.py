import os
import pandas as pd
import torch

data_file = os.path.join(".", 'data', 'house_tiny.csv')

data = pd.read_csv(data_file)

print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(outputs)
# 不设置numeric_only会报错
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

# dtype不设置则为True/False，设置后为1/0
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print(inputs)

X, y= torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y)