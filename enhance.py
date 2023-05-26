import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import sys
import cv2
import torchvision
import torch.optim as optim
from tqdm import tqdm
import json

from model import vgg

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # dir_path = os.path.join(self.root_dir, 'NIR_face_dataset')
        # self.root_dir = dir_path
        # self.image_list = os.listdir(dir_path)



    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label = int(img_name.split('-')[0]) - 1
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label



def contrast_enhancement(image, enhancement_factor):
    #实现对比度增强
    enhancer = ImageEnhance.Contrast(image)
    enhanced_img = enhancer.enhance(enhancement_factor)
    return enhanced_img


def matrix_transformations(matrix):
    # Horizontal symmetric matrix
    hor_sym_matrix = np.flipud(matrix)

    # Vertical symmetric matrix
    ver_sym_matrix = np.fliplr(matrix)

    # 90-degree rotation matrices
    rot_90_matrix_1 = np.rot90(matrix, 1)
    rot_90_matrix_2 = np.rot90(matrix, 2)
    rot_90_matrix_3 = np.rot90(matrix, 3)

    return hor_sym_matrix, ver_sym_matrix, rot_90_matrix_1, rot_90_matrix_2, rot_90_matrix_3

# root_dir = 'D:\Google_download\Chips_Thermal_Face_Dataset'
# data_set = YOLODataset(root_dir=root_dir)
# image, label = data_set[1210]
# print(label)

def main():
    # 数据预处理器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = r'D:\Google_download\data1'
    data_set = YOLODataset(root_dir=root_dir, transform=transform)

    #data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    '''
    image, label = data_set[210]
    print(type(image))
    np_img = image.numpy()
    print(type(np_img))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.show()
    '''
    train_size = int(0.8 * len(data_set))
    val_size = int(0.2 * len(data_set))
    test_size = len(data_set) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        data_set,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0)
    )
    cla_dict = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8'}
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_num = len(train_dataset)
    # print(train_num)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("using {} device.".format(device))
    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=202, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = tuple(map(int, labels))
            # print(labels)
            labels = torch.tensor(labels)
            # print(labels)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            #output 出来的东西是一系列八维矩阵，labels应该也要是一系列八维矩阵，这样才能在crossentropyloss
            #此处的label是一个数字，并且还是有group数据群的
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_labels = tuple(map(int, val_labels))
                val_labels = torch.tensor(val_labels)
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()