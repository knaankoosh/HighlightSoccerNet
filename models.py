import torch
import torch.nn as nn
from torch.nn import Conv3d
import torch.nn.functional as F
import torch.nn.utils as Utils
from torch.autograd import Variable
from torchvision.models import resnet

class C3RNN(nn.Module):
    def __init__(self, combine_size, input_size, hidden_size, n_layers=2):
        super(C3RNN, self).__init__()

        self.combine_size = combine_size

        self.video_net = crNN_video(combine_size).cuda()
        self.audio_net = crNN_audio(input_size, hidden_size, combine_size, n_layers).cuda()

        self.fc1 = nn.Linear(2*combine_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, frames, volumes, seq_lengths):
        audio_out = self.audio_net(volumes, seq_lengths, combine=1)
        video_out = self.video_net(frames, combine=1)

        combined_in = torch.cat((audio_out,video_out), dim=1).cuda()
        out = self.relu(self.fc1(combined_in))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        pred = self.softmax(out)
        return pred

class crNN_audio(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(crNN_audio, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.lstm_n_layers = n_layers

        self.RNN = nn.LSTM(self.input_size, self.lstm_hidden_size, self.lstm_n_layers, batch_first=True)
        self.hidden_to_pred = nn.Linear(self.lstm_hidden_size, self.output_size)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, volumes, seq_lengths, combine=0):
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

        # Reshape data to run through fc layer
        X = out.contiguous()
        X = X.view(-1, X.shape[2])

        # Run through fc layer
        X = self.hidden_to_pred(X)

        if (combine==0):
            # Calculate probs
            X = self.softmax(X)
        if (combine==1):
            X = self.relu(X)

        # View back as (batch, seq_len, num_classes)
        X = X.view(len(seq_lengths), seq_lengths[0], self.output_size)

        # Pull result of last timestep for each sequence
        out_tensor = torch.zeros((len(seq_lengths),self.output_size)).cuda()
        for i, length in enumerate(seq_lengths):
            out_tensor[i] = X[i,length-1]

        return out_tensor

class crNN_video(nn.Module):
    def __init__(self, combine_size=32):
        super(crNN_video, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0))

        self.fc6 = nn.Linear(32768, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # combination layer
        self.fc_comb = nn.Linear(4096, combine_size)

    def forward(self, frame_stack, combine=0):
        out = self.relu(self.conv1(frame_stack))
        out = self.pool1(out)

        out = self.relu(self.conv2(out))
        out = self.pool2(out)

        out = self.relu(self.conv3a(out))
        out = self.relu(self.conv3b(out))
        out = self.pool3(out)

        out = self.relu(self.conv4a(out))
        out = self.relu(self.conv4b(out))
        out = self.pool4(out)

        out = self.relu(self.conv5a(out))
        out = self.relu(self.conv5b(out))
        out = self.pool5(out)

        out = out.view(-1, 32768)
        out = self.relu(self.fc6(out))
        out = self.dropout(out)
        out = self.relu(self.fc7(out))
        out = self.dropout(out)

        if (combine==0):
            # Calculate probabilities
            out = self.softmax(self.fc8(out))
        else:
            # Run through fully-connected layer for later concat with audio
            out = self.relu(self.fc_comb(out))

        return out
