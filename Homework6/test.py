import torch
from torchvision import transforms
from torch.autograd import Variable
import os, sys
import cv2
import argparse

parser = argparse.ArgumentParser(description="Test AlexNet on webcam")
parser.add_argument('--model', type=str, help='path to saved model')
args = parser.parse_args()
sys.argv = [sys.argv[0]]

from train import AlexNet


class TestModel:
    def __init__(self):
        self.model = AlexNet()

        filename = 'alexnet_model.pth.tar'
        load_checkpoint_file = os.path.join(args.model, filename)
        if os.path.isfile(load_checkpoint_file):
            checkpoint = torch.load(load_checkpoint_file)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.class_names = checkpoint['numeric_class_names']
            self.tiny_class = checkpoint['tiny_class']

            print ("Loading model")
        else:
            print ("Saved model not found")
            sys.exit(0)

    def forward(self, img):
        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)

        self.model.eval()

        output = self.model(input_image)
        value, pred_label = torch.max(output, 1)

        label = self.tiny_class[self.class_names[pred_label.data[0]]]
        return label

    def cam(self, idx=0):
        cap_obj = cv2.VideoCapture(idx)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cap_obj.set(3, 800)
        cap_obj.set(4, 800)

        def preprocess(image):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_transform = transforms.Compose([transforms.ToPILImage(), transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
            return img_transform(image)

        while 1:
            read, frame = cap_obj.read()
            if read:
                norm_image_tensor = preprocess(frame)
                pred_class = self.forward(norm_image_tensor)
                cv2.putText(frame, pred_class, (225, 100), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.imshow('Cam', frame)
            else:
                print ("Error")
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        cap_obj.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    alex = TestModel()
    alex.cam()
