import torch
import torch.nn as nn
import timm


class EfficientNetV2Backbone(nn.Module):
    def __init__(self, nc=3, model_name='tf_efficientnetv2_b0'):
        super(EfficientNetV2Backbone, self).__init__()

        # Load the pretrained EfficientNetV2 model from timm
        self.efficientnet = timm.create_model(model_name, pretrained=True, in_chans=nc, num_classes=0)

        # Remove the global pooling layer to preserve spatial dimensions
        self.features = nn.Sequential(*list(self.efficientnet.children())[:-2])  # Exclude the pooling and head layers

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.efficientnet.num_features, out_channels=512, kernel_size=(3, 3),
                               stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x


def test():
    net = EfficientNetV2Backbone(1)  # Example with single-channel input
    inputs = torch.randn(32, 1, 64, 170)  # Batch size 32, single channel, 128x128 image
    print("Input stats: min={}, max={}, mean={}, std={}".format(
        inputs.min(), inputs.max(), inputs.mean(), inputs.std()))
    y = net(inputs)
    print(y.size())
    if torch.isnan(y).any():
        raise Exception("There are nan in the output of the backbone!")


if __name__ == '__main__':
    test()
