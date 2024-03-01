import torch.nn as nn
import torchvision.models as models

# class CAD_MENBI_Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(CAD_MENBI_Classifier, self).__init__()
#         self.resnet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
#         # num_features = self.resnet.fc.in_features
#         # self.resnet.fc = nn.Linear(num_features, num_classes)

#     def forward(self, x):
# #         return self.resnet(x)
    
# class CAD_MENBI_Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(CAD_MENBI_Classifier, self).__init__()
#         self.resnet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
#         num_features = self.resnet.classifier[3].in_features  # Adjust this line to access the correct layer
#         self.resnet.classifier[3] = nn.Linear(num_features, num_classes)  # Replace the final layer

#     def forward(self, x):
#         return self.resnet(x)

class CAD_MENBI_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(CAD_MENBI_Classifier, self).__init__()
        # Load a pre-trained DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        # Replace the classifier with a new linear layer for your number of classes
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
    
if __name__ == '__main__':
    model = CAD_MENBI_Classifier(4)