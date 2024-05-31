import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from skimage.io import imread 
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from collections import Counter

class CustomDataset(Dataset):
    def __init__(self, data, bins, transform=None):
        
        self.data = data
        self.transform = transform
        self.bins = bins
        # self.encode = {
        #     0: 0,
        #     150: 1,
        #     180: 2,
        #     210: 3,
        #     240: 4,
        #     270: 5,
        #     300: 6,
        #     330: 7,
        #     360: 8
        # }
        self.encode = {
            0: 0,
            90: 1,
            180: 2,
            270: 3,
            360: 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = np.array(Image.open(self.data[index]).convert('RGB'))
        img = cv2.resize(img, (1546,1003))
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.0001)
        img = np.reshape(img, (3, 1546,1003))

        assert not np.isnan(img).any()

        img = Image.open(self.data[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if '_500_' in self.data[index]:
            raw_label = int(self.data[index].split('_500_')[1].split('.png')[0])
        elif '_0_250' in self.data[index]:
            raw_label = 0
        else:
            raise Exception
        
        binned_label_int = self.find_nearest(int(raw_label))
        encoded_label = self.encode[binned_label_int]
        
        return img, torch.tensor(raw_label)
        # return img, binned_label_int
    
    def viz(self):
        img = imread(self.data.iloc[0, 0])
        plt.imshow(img)

    def find_nearest(self, value):
        idx = (np.abs(self.bins-np.ceil(value))).argmin()
        return self.bins[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation((0,180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


mode = 'test'
root = "/home/kashis/Desktop/Capstone/dataset/Pivot GIS Project/images_classification/" + mode

classes = [0, 90, 180, 270, 360]
num_classes = len(classes)

dataset = CustomDataset(glob.glob(root+"/*.png"), bins = classes, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

model = models.resnet101(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

# labels = [label for _, label in dataset]  # Count unique labels
# counter = Counter(labels)
# encode = {
#             0: 0,
#             90: 1,
#             180: 2,
#             270: 3,
#             360: 4
#         }
# labels = [classes[encode[key]] for key in sorted(encode.keys())]
# values = [counter[encode[key]] for key in sorted(encode.keys())]
# plt.bar(labels, values)
# plt.bar(labels, values, width=20)
# plt.xticks(classes) 


if mode == 'train':

        model.load_state_dict(torch.load('resnet101_custom_dataset_regression.pth'))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        num_epochs = 150
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(-1)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), 'resnet101_custom_dataset_regression_retrained.pth')

if mode == 'test':
    model = models.resnet101(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.load_state_dict(torch.load('resnet101_custom_dataset_regression_retrained.pth'))
    model.eval()

    device = torch.device('cuda')
    model.to(device)
    out_arr = []
    gt_label = []

    correct = 0

    angle_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    accuracy_list = []

    for angle_thresh in angle_thresholds: 
        for i, (images, labels) in enumerate(dataloader):

            x = images.to(device).to(torch.float32).to(device)
            y = labels.cpu().numpy()
        
            out = model(x)
            

            pred = out.detach().cpu().numpy()
            predicted = np.array(pred, dtype=int)

            #TODO: plot
            correct += np.sum(np.abs(predicted.flatten() - y.flatten()) <= angle_thresh)

            out_arr.extend(predicted)
            gt_label.extend(y)


        accuracy = 100 * correct / len(dataset)
        # f1 = f1_score(gt_label, out_arr, average='weighted')
        # precision = precision_score(gt_label, out_arr, average='weighted')
        # recall = recall_score(gt_label, out_arr, average='weighted')
        # matrix = multilabel_confusion_matrix(gt_label, out_arr)
        # print('Test Accuracy: {:.2f}%'.format(accuracy))
        # print('Precision, Recall, F1: ', precision, recall, f1)
        # print(matrix)
        accuracy_list.append(accuracy)
        correct = 0

    print(accuracy_list)

    
