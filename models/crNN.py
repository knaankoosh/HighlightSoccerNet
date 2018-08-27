import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

class crNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(crNN, self).__init__()

        self.max_len = 32
        self.video_embedd = input_size
        self.ResNet = resnet.resnet18(pretrained=False, num_classes=self.video_embedd)
        self.RNN = nn.GRU(self.video_embedd + 1, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(self.max_len, 1)

    def forward(self, frame, audio):
        # c_in = x.view(batch_size * timesteps, H, W, C)
        cnn_out = self.ResNet(frame)
        rnn_in = torch.cat((audio.unsqueeze(1), cnn_out), 1)
        rnn_out, rnn_hidden = self.RNN(rnn_in.unsqueeze(0))
        rnn_out = rnn_out[:, -1, -1]
        # rnn_out = torch.cat((rnn_out, torch.zeros((1,self.max_len-rnn_out.shape[-1])).cuda()), 1)
        # output = F.sigmoid(self.fc(rnn_out))
        output = F.sigmoid(rnn_out)
        return rnn_out
