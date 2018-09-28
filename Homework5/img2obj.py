import torch
from torch import nn, optim
from torchvision import datasets, transforms
import cv2
from time import time
import os,sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, img):
        ret = torch.nn.functional.relu(self.conv1(img))
        ret = torch.nn.functional.max_pool2d(ret, 2)
        ret = torch.nn.functional.relu(self.conv2(ret))
        ret = torch.nn.functional.max_pool2d(ret, 2)
        ret = ret.view(ret.size(0), -1)
        ret = torch.nn.functional.relu(self.fc1(ret))
        ret = torch.nn.functional.relu(self.fc2(ret))
        return self.fc3(ret)

class Img2Obj:
    def __init__(self):

        self.labels = ['apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                       'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle',
                       'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'kangaroo',
                       'couch', 'crocodile', 'cups', 'crab', 'dinosaur', 'elephant', 'dolphin', 'flatfish', 'forest',
                       'girl', 'fox', 'hamster', 'house', 'computer keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion',
                       'lizard', 'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak',
                       'oranges', 'orchids', 'otter', 'palm', 'pears', 'pickup truck', 'pine', 'plain', 'plates',
                       'poppies', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea',
                       'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
                       'streetcar', 'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 'television', 'tiger',
                       'tractor', 'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman',
                       'worm']

        self.outcomes = len(self.labels)
        learning_rate = 0.0001
        self.epochs = 25
        self.batch = 20

        self.input_size = 3 * 32 * 32

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./CIFAR100', train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), normalize])),
            batch_size=self.batch, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./CIFAR100', train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), normalize])),
            batch_size=self.batch, shuffle=True)

        self.model = LeNet5()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0005)

        self.loss = nn.CrossEntropyLoss()
        self.train_loss = 0
        self.valid_loss = 0

    def train(self):
        def train_epoch():
            training_loss = 0
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                training_loss += loss.data.item()
                loss.backward()
                self.train_loss += float(loss.data)
                self.optimizer.step()
            return training_loss

        def validation():
            self.model.eval()
            true_positive = 0
            self.validation_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                onehot_target = torch.zeros(self.batch, self.outcomes)
                for n in range(len(target)):
                    onehot_target[n][target[n]] = 1
                output = self.model(data)
                loss = self.loss(output, target)
                value, index = torch.max(output, 1)

                self.validation_loss += float(loss.data)

                for n in range(len(target)):
                    if index[n] == target[n]:
                        true_positive += 1

            return float(true_positive)/len(self.test_loader.dataset), self.validation_loss

        for i in range(self.epochs):
            self.train_loss = 0
            self.valid_loss = 0
            start = time()
            train_epoch()
            end = time()
            total_time = end - start
            training_loss = train_epoch()
            accuracy, validation_loss = validation()

            print("Epoch = " + str(i + 1) + ", Training loss = " + str(float(training_loss) / (len(self.train_loader.dataset) / self.batch)) + ", time =  " + str(float(total_time)))
            print("Epoch = " + str(i + 1) + ", Validation loss = " + str(float(validation_loss) / (len(self.test_loader.dataset) / self.batch)) + " , Validation Accuracy = " + str(float(accuracy)))

    def forward(self, img):
        img = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        self.model.eval()
        output = self.model(img)
        value, index = torch.max(output, 1)
        return self.labels[index.data[0]]

    def view(self, img):
        pred_class = self.forward(img)
        image_numpy = np.transpose(img.numpy(), (1, 2, 0))

        cv2.namedWindow(pred_class, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pred_class, 800, 800)
        cv2.imshow(pred_class, image_numpy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cam(self, idx=0):
        cap_obj = cv2.VideoCapture(idx)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cap_obj.set(3, 800)
        cap_obj.set(4, 800)

        def preprocess(image):
            scaled_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
            preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
            return preprocess(scaled_image)

        while 1:
            read, frame = cap_obj.read()
            if read:
                norm_image_tensor = preprocess(frame)
                pred_class = self.forward(norm_image_tensor)
                cv2.putText(frame, pred_class, (225, 100), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.imshow('Cam', frame)
            else:
                print('\nError is reading video frame from the webcam..Exiting..')
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        cap_obj.release()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    nnimg2obj = Img2Obj()
    nnimg2obj.train()
    nnimg2obj.cam()
