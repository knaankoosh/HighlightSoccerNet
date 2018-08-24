import torch
import torch.nn as nn
import torch.functional as F
from torchvision.models import resnet

class crNN(nn.Module):
    def __init__(self, video_embedd):
        super(crNN, self).__init__()

        self.video_embedd = video_embedd
        self.ResNet = resnet.resnet18(pretrained=False, num_classes=self.video_embedd)
        self.RNN = nn.GRU(self.video_embedd + 1, 10, 2, batch_first=True)

        self.cnn_params = self.ResNet.parameters()
        self.rnn_params = self.RNN.parameters()

    def forward(self, frame, audio):
        # c_in = x.view(batch_size * timesteps, H, W, C)
        cnn_out = self.ResNet(frame)
        rnn_in = torch.cat((audio, frame), 1)
        rnn_out = self.RNN(rnn_in)
        return F.log_softmax(rnn_out, dim=1)

