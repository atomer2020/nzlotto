import pyTorch
import pyTorch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out, (hn, cn) = self.lstm(x.unsqueeze(1), (hidden, cell))
        out = self.fc(out.squeeze(1))
        return out, hn, cn

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size, num_layers)

    def forward(self, src, target_len):
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, target_len, src.size(2)).to(src.device)
        hidden, cell = self.encoder(src)
        dec_input = src[:, -1, :]  # 使用最后一个输入作为解码器的初始输入
        for t in range(target_len):
            output, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t, :] = output
            dec_input = output
        return outputs
