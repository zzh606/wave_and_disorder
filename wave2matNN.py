from scipy import io as scio
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

L = 16
nr = 50
t0 = 1  # hopping系数，由wannier函数决定
alpha = 1  # hopping系数，由wannier函数决定
rho_max = 0.5
eps_max = 0.5
scale = 1
wavefunc_num = np.linspace(0, L*L-1, L*L)


class Flatten(nn.Module):
    # 构造函数，没有什么要做的
    def __init__(self):
        # 调用父类构造函数
        super(Flatten, self).__init__()

    # 实现forward函数
    def forward(self, input):
        # 保存batch维度，后面的维度全部压平，例如输入是28*28的特征图，压平后为784的向量
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # 如果使用view(3,2)或reshape(3,2)，得到的tensor并不是转置的效果，
        # 而是相当于将原tensor的元素按行取出，然后按行放入到新形状的tensor中
        return x.unsqueeze(1)  # return x.view(self.shape)


# 参考 https://www.jianshu.com/p/2d9927a70594
class D2L_Dataset(Dataset):
    def __init__(self, path, dataset_file):
        self.rho_arr = []
        self.eps_arr = []
        self.files = self.get_filelist(path)

        if os.path.exists(dataset_file):
            print('加载' + dataset_file)
            data = scio.loadmat(dataset_file)
            self.image_arr = data['image_arr']
            self.IIPR_arr = data['IIPR_arr']
            self.image_arr = torch.from_numpy(self.image_arr).type(torch.FloatTensor)
            self.IIPR_arr = torch.from_numpy(self.IIPR_arr).type(torch.FloatTensor)
            
        else:
            print('生成' + dataset_file)
            flag = 0
            for ff in self.files:
                f = ff.replace('.mat', '')
                f = f.split('-')
                rho = float(f[3].split('=')[1])
                eps = float(f[4].split('=')[1])
                self.rho_arr.append(rho)
                self.eps_arr.append(eps)
    
                data = scio.loadmat(ff)
                if flag == 0:
                    self.x_image_arr = data['x_image_arr']
                    self.y_image_arr = data['y_image_arr']
                    self.IIPR_arr = data['IIPR_arr']
                    flag = 1
                else:
                    self.x_image_arr = np.concatenate((self.x_image_arr, data['x_image_arr']), axis=0)
                    self.y_image_arr = np.concatenate((self.y_image_arr, data['y_image_arr']), axis=0)
                    self.IIPR_arr = np.concatenate((self.IIPR_arr, data['IIPR_arr']), axis=0)
            self.x_image_arr = np.expand_dims(self.x_image_arr, 1)
            self.y_image_arr = np.expand_dims(self.y_image_arr, 1)
            self.image_arr = np.concatenate((self.x_image_arr, self.y_image_arr), axis=1)
            
            scio.savemat(dataset_file,
                         {'image_arr': self.image_arr, 'IIPR_arr': self.IIPR_arr})

            self.image_arr = torch.from_numpy(self.image_arr).type(torch.FloatTensor)
            self.IIPR_arr = torch.from_numpy(self.IIPR_arr).type(torch.FloatTensor)

    def __len__(self):
        return np.size(self.image_arr, 0)

    def __getitem__(self, item):
        return self.image_arr[item, :, :, :], self.IIPR_arr[item]

    def get_filelist(self, path):
        Filelist = []
        for home, dirs, files in os.walk(path):
            for filename in files:
                # 文件名列表，包含完整路径
                Filelist.append(os.path.join(home, filename))
                # # 文件名列表，只包含文件名
                # Filelist.append(filename)
        return Filelist


class L2D_Dataset(Dataset):
    def __init__(self, path, dataset_file):
        self.rho_arr = []
        self.eps_arr = []
        self.files = self.get_filelist(path)

        if os.path.exists(dataset_file):
            print('加载' + dataset_file)
            data = scio.loadmat(dataset_file)
            self.image_arr = data['image_arr']
            self.IIPR_arr = data['IIPR_arr']
            self.image_arr = torch.from_numpy(self.image_arr).type(torch.FloatTensor)
            self.IIPR_arr = torch.from_numpy(self.IIPR_arr).type(torch.FloatTensor)

        else:
            print('生成' + dataset_file)
            flag = 0
            for ff in self.files:
                f = ff.replace('.mat', '')
                f = f.split('-')
                rho = float(f[3].split('=')[1])
                eps = float(f[4].split('=')[1])
                self.rho_arr.append(rho)
                self.eps_arr.append(eps)

                data = scio.loadmat(ff)
                if flag == 0:
                    self.x_image_arr = data['x_image_arr']
                    self.y_image_arr = data['y_image_arr']
                    self.IIPR_arr = data['IIPR_arr']
                    flag = 1
                else:
                    self.x_image_arr = np.concatenate((self.x_image_arr, data['x_image_arr']), axis=0)
                    self.y_image_arr = np.concatenate((self.y_image_arr, data['y_image_arr']), axis=0)
                    self.IIPR_arr = np.concatenate((self.IIPR_arr, data['IIPR_arr']), axis=0)
            self.x_image_arr = np.expand_dims(self.x_image_arr, 1)
            self.y_image_arr = np.expand_dims(self.y_image_arr, 1)
            self.image_arr = np.concatenate((self.x_image_arr, self.y_image_arr), axis=1)

            scio.savemat(dataset_file,
                         {'image_arr': self.image_arr, 'IIPR_arr': self.IIPR_arr})

            self.image_arr = torch.from_numpy(self.image_arr).type(torch.FloatTensor)
            self.IIPR_arr = torch.from_numpy(self.IIPR_arr).type(torch.FloatTensor)

    def __len__(self):
        return np.size(self.image_arr, 0)

    def __getitem__(self, item):
        return self.IIPR_arr[item].reshape((L, L)), self.image_arr[item, :, :, :]

    def get_filelist(self, path):
        Filelist = []
        for home, dirs, files in os.walk(path):
            for filename in files:
                # 文件名列表，包含完整路径
                Filelist.append(os.path.join(home, filename))
                # # 文件名列表，只包含文件名
                # Filelist.append(filename)
        return Filelist

class D2LCNN(nn.Module):
    def __init__(self):
        super(D2LCNN, self).__init__()
        # CNN
        # self.reshape = Reshape(-1, 2, 16, 16)
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1)  # group??
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fla1 = Flatten()
        self.fc1 = nn.Linear(2048, 2048)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, L * L)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.fla1(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(F.relu(x))
        return x


class L2DCNN(nn.Module):
    def __init__(self):
        super(L2DCNN, self).__init__()
        # CNN
        self.reshape1 = Reshape(-1, 1, L, L)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)  # group??
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fla1 = Flatten()
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, L * L * 2)

    def forward(self, x):
        # x = self.reshape1(x)
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.fla1(x)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        return x.reshape((-1, 2, L, L))


## 参考 https://blog.csdn.net/liangdaojun/article/details/105330007
## 参考 https://blog.csdn.net/dss_dssssd/article/details/84103834
# 直接定义函数 ， 不需要维护参数，梯度等信息
# 注意所有的数学操作需要使用tensor完成
def mape_loss_func(preds, labels):
    return torch.mean(torch.abs(labels-preds) / labels)


def d2l_train(model, train_loader, device, optimizer, epoch):
    # tensor可以直接取代Variable
    d2lcnn = model
    d2lcnn.train()

    losses = []
    for i, (image, IIPR) in enumerate(train_loader):  # for each training step
        image, IIPR = image.to(device), IIPR.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = d2lcnn(image)

        loss = mape_loss_func(output, IIPR)
        # loss = F.mse_loss(output, IIPR)

        optimizer.zero_grad()
        losses.append(loss)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('Epoch : %d, Step: %d, Loss: %f' % (epoch, i, loss))
            # plt.figure(1)
            # plt.scatter(wavefunc_num, IIPR.cpu()[0])
            # plt.scatter(wavefunc_num, output.data.cpu().numpy()[0])
            # plt.show()
    return losses


def l2d_train(model, train_loader, device, optimizer, epoch, model_pretrained):
    # tensor可以直接取代Variable
    l2dcnn = model
    l2dcnn.train()

    losses = []
    for i, (IIPR_image, dis_image) in enumerate(train_loader):  # for each training step
        IIPR_image = IIPR_image.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = l2dcnn(IIPR_image)
        output = model_pretrained(output)
        # loss = mape_loss_func(output.reshape(-1, L, L), IIPR_image)
        loss = mape_loss_func(output, IIPR_image.reshape(-1, L*L))
        optimizer.zero_grad()
        losses.append(loss)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('Epoch : %d, Step: %d, Loss: %f' % (epoch, i, loss))
            # plt.figure(1)
            # plt.scatter(wavefunc_num, IIPR_image.cpu()[10])
            # plt.scatter(wavefunc_num, output.data.cpu().numpy()[10])
            # plt.show()
    return losses


def d2l_validation(model, device, optimizer, test_loader, epoch):
    d2lcnn = model
    d2lcnn.eval()

    test_loss = 0
    all_y = []
    cnt = 0

    flag = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device)
            output = d2lcnn(X)
            cnt += 1
            if flag == 0:
                w_aver = np.sum(output.data.cpu().numpy(), 1) / (L ** 2)
                output_arr = output.data.cpu().numpy()
                loss = mape_loss_func(output, y)
                test_loss += loss
                flag = 1
            else:
                w_aver = np.concatenate((w_aver, np.sum(output.data.cpu().numpy(), 1) / (L ** 2)), axis=0)
                output_arr = np.concatenate((output_arr, output.data.cpu().numpy()), axis=0)
                loss = mape_loss_func(output, y)
                test_loss += loss                 # sum up batch loss
                # collect all y and y_pred in all batches
                all_y.extend(y)

    test_loss /= cnt
    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    # show information
    print('Test set ({:d} samples): Average loss: {:f} Average accuracy: {:f}%\n'.format(len(all_y), test_loss, (1-test_loss)*100))

    sample = len(output_arr) // 400

    w_scale = (w_aver - np.min(w_aver)) / (np.max(w_aver) - np.min(w_aver))
    for i in range(0, len(w_aver), sample):
        plt.scatter(wavefunc_num, output_arr[i], color=(w_scale[i], 0, 1-w_scale[i]))
    plt.show()

    return test_loss


def l2d_validation(model, device, optimizer, test_loader, epoch, model_pretrained):
    l2dcnn = model
    l2dcnn.eval()

    test_loss = 0
    all_y = []
    cnt = 0

    flag = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            # distribute data to device
            X = X.to(device)
            output = l2dcnn(X)
            output = model_pretrained(output)

            cnt += 1
            if flag == 0:
                w_aver = np.sum(output.data.cpu().numpy(), 1) / (L ** 2)
                output_arr = output.data.cpu().numpy()
                X_arr = X.data.cpu().numpy()
                loss = mape_loss_func(output.reshape((-1, L, L)), X)
                test_loss += loss
                flag = 1
            else:
                w_aver = np.concatenate((w_aver, np.sum(output.data.cpu().numpy(), 1) / (L ** 2)), axis=0)
                output_arr = np.concatenate((output_arr, output.data.cpu().numpy()), axis=0)
                X_arr = np.concatenate((X_arr, X.data.cpu().numpy()), axis=0)
                loss = mape_loss_func(output.reshape((-1, L, L)), X)
                test_loss += loss                 # sum up batch loss
                # collect all y and y_pred in all batches
                all_y.extend(X)

    test_loss /= cnt
    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    # show information
    print('Test set ({:d} samples): Average loss: {:f} Average accuracy: {:f}%\n'.format(len(all_y), test_loss, (1-test_loss)*100))

    sample = len(output_arr) // 400
    plt.subplot(1, 2, 1)
    w_scale = (w_aver - np.min(w_aver)) / (np.max(w_aver) - np.min(w_aver))
    for i in range(0, len(w_aver), sample):
        plt.scatter(wavefunc_num, output_arr[i], color=(w_scale[i], 0, 1-w_scale[i]))
    plt.subplot(1, 2, 2)
    w_X = np.sum(X_arr, 1).sum(1) / (L**2)
    w_X_scale = (w_X - np.min(w_X)) / (np.max(w_X) - np.min(w_X))
    for i in range(0, len(w_X_scale), sample):
        plt.scatter(wavefunc_num, X_arr[i], color=(w_X_scale[i], 0, 1-w_X_scale[i]))
    plt.show()

    return test_loss


def l2d2l_validation(model1, model2, device, test_loader, epoch):
    l2dcnn = model1
    d2lcnn = model2
    l2dcnn.eval()
    d2lcnn.eval()

    test_loss = 0
    all_y = []
    cnt = 0

    flag = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            # distribute data to device
            X = X.to(device)
            output = l2dcnn(X)
            output = d2lcnn(output)

            cnt += 1
            if flag == 0:
                w_aver = np.sum(output.data.cpu().numpy(), 1) / (L ** 2)
                output_arr = output.data.cpu().numpy()
                X_arr = X.data.cpu().numpy()
                loss = mape_loss_func(output.reshape((-1, L, L)), X)
                test_loss += loss
                flag = 1
            else:
                w_aver = np.concatenate((w_aver, np.sum(output.data.cpu().numpy(), 1) / (L ** 2)), axis=0)
                output_arr = np.concatenate((output_arr, output.data.cpu().numpy()), axis=0)
                X_arr = np.concatenate((X_arr, X.data.cpu().numpy()), axis=0)
                loss = mape_loss_func(output.reshape((-1, L, L)), X)
                test_loss += loss                 # sum up batch loss
                # collect all y and y_pred in all batches
                all_y.extend(X)

    test_loss /= cnt
    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    # show information
    print('Test set ({:d} samples): Average loss: {:f} Average accuracy: {:f}%\n'.format(len(all_y), test_loss, (1-test_loss)*100))

    sample = len(output_arr) // 500
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.axis([0, 255, 0, 0.8])
    axes2.axis([0, 255, 0, 0.8])
    w_scale = (w_aver - np.min(w_aver)) / (np.max(w_aver) - np.min(w_aver))
    for i in range(0, len(w_aver), sample):
        axes1.scatter(wavefunc_num, output_arr[i], color=(w_scale[i], 0, 1-w_scale[i]))
    w_X = np.sum(X_arr, 1).sum(1) / (L**2)
    w_X_scale = (w_X - np.min(w_X)) / (np.max(w_X) - np.min(w_X))
    for i in range(0, len(w_X_scale), sample):
        axes2.scatter(wavefunc_num, X_arr[i], color=(w_X_scale[i], 0, 1-w_X_scale[i]))
    fig.show()

    return test_loss


def d2lcnn_test(path, dataset):
    print('===================D2LNN===========================')
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    torch.cuda.empty_cache()

    full_dataset = D2L_Dataset(path, dataset)  # n*2*16*16

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4)

    d2lcnn = D2LCNN()
    if not os.path.exists('d2lcnn.pth'):
        print('创建新模型')
        d2lcnn = d2lcnn.to(device)
    else:
        print('加载模型')
        d2lcnn.load_state_dict(torch.load('d2lcnn.pth', map_location='cuda:0'))
        d2lcnn.eval()
        d2lcnn.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        d2lcnn = nn.DataParallel(d2lcnn)

    d2lcnn_params = list(d2lcnn.parameters())

    optimizer = optim.Adam(d2lcnn_params, lr=1e-4)  # 初始化优化器

    # record training process
    epoch_train_losses = []
    epoch_test_losses = []

    # start training
    epoch_num = 30
    for epoch in range(epoch_num):  # train entire dataset 5 times
        train_losses = d2l_train(d2lcnn, train_loader, device, optimizer, epoch)
        epoch_test_loss = d2l_validation(d2lcnn, device, optimizer, valid_loader, epoch)
        # save results
        epoch_train_losses.append(train_losses)
        epoch_test_losses.append(epoch_test_loss)

        torch.save(d2lcnn.state_dict(), 'd2lcnn.pth')

    torch.save(d2lcnn, 'D2LCNN_model.pth')


def l2dcnn_test(path, dataset):
    print('===================L2DCNN===========================')
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    full_dataset = L2D_Dataset(path, dataset)  # n*2*16*16

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4)

    l2dcnn = L2DCNN()
    if not os.path.exists('l2dcnn.pth'):
        print('创建新模型')
        l2dcnn = l2dcnn.to(device)
    else:
        print('加载模型')
        l2dcnn.load_state_dict(torch.load('l2dcnn.pth', map_location='cuda:0'))
        l2dcnn.eval()
        l2dcnn.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        l2dcnn = nn.DataParallel(l2dcnn)

    l2dcnn_params = list(l2dcnn.parameters())

    # optimizer = optim.Adam(l2dcnn_params, lr=1e-4, weight_decay=0.01)  # 初始化优化器
    optimizer = optim.Adam(l2dcnn_params, lr=1e-4)

    # record training process
    epoch_train_losses = []
    epoch_test_losses = []

    # start training
    d2lcnn_pretrained = torch.load('D2LCNN_model.pth', map_location='cuda:0')
    epoch_num = 50
    for epoch in range(epoch_num):  # train entire dataset 5 times
        train_losses = l2d_train(l2dcnn, train_loader, device, optimizer, epoch, d2lcnn_pretrained)
        epoch_test_loss = l2d_validation(l2dcnn, device, optimizer, valid_loader,
                                                       epoch, d2lcnn_pretrained)
        # save results
        # epoch_train_losses.append(train_losses)
        epoch_test_losses.append(epoch_test_loss)

        torch.save(l2dcnn.state_dict(), 'l2dcnn.pth')

    torch.save(l2dcnn, 'L2DCNN_model.pth')


def l2d2lcnn_test(path, dataset):
    print('===================L2DCNN===========================')
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    full_dataset = L2D_Dataset(path, dataset)
    valid_loader = DataLoader(dataset=full_dataset, batch_size=32, shuffle=True, num_workers=4)

    # start training
    d2lcnn_pretrained = torch.load('D2LCNN_model.pth', map_location='cuda:0')
    l2dcnn_pretrained = torch.load('L2DCNN_model.pth', map_location='cuda:0')

    epoch_num = 50
    for epoch in range(epoch_num):  # train entire dataset 5 times
        epoch_test_loss = l2d2l_validation(l2dcnn_pretrained, d2lcnn_pretrained, device, valid_loader,
                                                       epoch)


if __name__ == '__main__':
     # d2lcnn_test('data', 'dataset_all.mat')
    # l2dcnn_test('data', 'dataset_all.mat')
    l2d2lcnn_test('data_test', 'dataset_test_all.mat')