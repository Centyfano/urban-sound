from torch import nn
from torchsummary import summary
import torch


class Conv(nn.Module):

    def __init__(self):
        super().__init__()
        self.convs, self.last_shape = self.conv_blocks(4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # spectogram as grayscale
                      out_channels=16,  # 16 filters
                      kernel_size=3,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,  # spectogram as grayscale
                      out_channels=32,  # 16 filters
                      kernel_size=3,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,  # spectogram as grayscale
                      out_channels=64,  # 16 filters
                      kernel_size=3,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,  # spectogram as grayscale
                      out_channels=128,  # 16 filters
                      kernel_size=3,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*5*4, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_data):
        x = 1
        for i, conv in enumerate(self.convs):
            if i == 0:
                x = conv(input_data)
            else:
                x = conv(x)

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = Conv()
    cnn = cnn.cuda() if torch.cuda.is_available() else cnn
    summary(cnn, (1, 64, 44))