import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from d2l_20240331_func import download, DATA_HUB, DATA_URL, load_array

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train', './data/kaggle_house'))
test_data = pd.read_csv(download('kaggle_house_test', './data/kaggle_house'))

# print(train_data.shape)
# print(train_test.shape)

# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))
# 哪些特征是数值特征 - 非object为数值
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 均值为0 方差为1
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# na置为0，即均值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 字符串使用独热编码 one-hot
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Linear(128, 64),
        nn.Dropout(),
        nn.Linear(64, 32),
        nn.Dropout(),
        nn.Linear(32, 1),
    )


def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, lr, wd, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, lr, wd, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}\n'
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, wd, batch_size = 5, 100, 1, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, wd, batch_size)
print(f'{k}-折验证: \n'
      f'平均训练log rmse: {float(train_l):f},\n'
      f'平均验证log rmse: {float(valid_l):f}')


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, wd, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, wd, batch_size)
    print(f'train log rmse {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('./data/kaggle_house/submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, wd, batch_size)
