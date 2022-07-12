import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = torchvision.models.densenet121(pretrained=True)

        # Get the input dimension of last layer
        model_output_features = self.model.classifier.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.model.classifier = nn.Sequential(nn.Linear(model_output_features, num_classes),
                                              nn.Sigmoid())

    def forward(self, inputs):
        """
        the forward pass through the network using supplied inputs
        """
        return self.model(inputs)


class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = torchvision.models.inception_v3(weights='Inception_V3_Weights.DEFAULT')

        # Get the input dimension of last layer
        model_output_features = self.model.fc.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.model.fc = nn.Sequential(nn.Linear(model_output_features, 4),
                                    nn.Sigmoid())

    def forward(self, inputs):
        """
        the forward pass through the network using supplied inputs
        """
        return self.model(inputs)