import torch
import torch.nn as nn
from torch.nn import Conv3d
import torch.nn.functional as F
import torch.nn.utils as Utils
from torch.autograd import Variable
from torchvision.models import resnet

class crNN(nn.Module):
    def __init__(self, embedd_size, hidden_size, output_size, n_layers=1):
        super(crNN, self).__init__()

        self.max_len = 32
        self.embedd_size = embedd_size
        self.ResNet = resnet.resnet18(pretrained=False, num_classes=self.embedd_size)
        self.bn = nn.BatchNorm1d(self.embedd_size, momentum=0.01)
        self.RNN = nn.LSTM(self.embedd_size + 1, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(self.max_len, 1)

    def forward(self, frame, audio):
        cnn_out = self.bn(self.ResNet(frame))
        rnn_in = torch.cat((audio.unsqueeze(1), cnn_out), 1)
        rnn_out, rnn_hidden = self.RNN(rnn_in.unsqueeze(0))
        rnn_out = rnn_out[:, -1, -1]
        output = F.sigmoid(rnn_out)
        return rnn_out

class crNN_audio(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(crNN_audio, self).__init__()
        self.activation = F.sigmoid

        self.input_size = input_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.lstm_n_layers = n_layers

        self.RNN = nn.LSTM(self.input_size, self.lstm_hidden_size, self.lstm_n_layers, batch_first=True)
        self.hidden_to_pred = nn.Linear(self.lstm_hidden_size, self.output_size)


    def forward(self, volumes, seq_lengths):
        # Pack batch
        in_pack = Utils.rnn.pack_padded_sequence(volumes.unsqueeze(2), seq_lengths, batch_first=True)

        # now run through LSTM
        out_pack, hidden = self.RNN(in_pack)

        # Unpack batch
        out, hidden = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)

        ############ Before addition of fully-connected layer ###############
        # out = out[:,:,-1]
        # out_tensor = torch.zeros(len(seq_lengths)).cuda()
        # for i, length in enumerate(seq_lengths):
        #     out_tensor[i] = out[i,length-1]
        # out_tensor = self.activation(out_tensor)
        # out_tensor = self.activation(out_tensor)

        # Reshape data to run through fc layer
        X = out.contiguous()
        X = X.view(-1, X.shape[2])

        # Run through fc layer
        X = self.hidden_to_pred(X)

        # Calculate probs
        X = F.softmax(X, dim=1)

        # View back as (batch, seq_len, num_classes)
        X = X.view(len(seq_lengths), seq_lengths[0], self.output_size)

        # Pull result of last timestep for each sequence
        out_tensor = torch.zeros((len(seq_lengths),self.output_size)).cuda()
        for i, length in enumerate(seq_lengths):
            out_tensor[i] = X[i,length-1]

        return out_tensor


