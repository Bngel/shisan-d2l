from torchvision import datasets
from torch.utils import data
from torchvision import transforms
import time

trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

batch_size = 256


def get_dataloader_workers():
    # 使用4个worker（进程）
    return 4


# windows 必须在 main 方法下使用 num_workers 参数
if __name__ == '__main__':
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers())

    start_time = time.time()

    for X, y in train_iter:
        continue

    end_time = time.time()

    print(f'{end_time - start_time} sec')
