import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from dibhttp import get_lotto_from_db


def convert_data(datas):
    X_train, y_train, history_numbers = [], [], []
    for parts in datas:
        draw_id = parts[0]
        main_numbers = list(map(int, parts[2:8]))
        bonus_number = int(parts[8])
        powerball_number = int(parts[9])
        features = main_numbers
        label = main_numbers + [bonus_number, powerball_number]
        X_train.append(features)
        y_train.append(label)
        history_numbers.append(features + [bonus_number, powerball_number])
    return np.array(X_train), np.array(y_train), history_numbers



# 假设数据格式为：['draw_id', 'date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus', 'powerball']
data = get_lotto_from_db("select * from lotto order by id")
last_data = data[-1]
lastResult=last_data[2:10]
data=data[0:len(data) - 1]

X_train, y_train, numbers = convert_data(data)
X_train = np.expand_dims(X_train, axis=1)
class LottoDataset(Dataset):
    def __init__(self, data, sequence_length=5):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.sequence_length], dtype=torch.float),
                torch.tensor(self.data[idx + self.sequence_length], dtype=torch.float))

sequence_length = 5
dataset = LottoDataset(numbers, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
        dec_input = src[:, -1, :]
        for t in range(target_len):
            output, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t, :] = output
            dec_input = output
        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderDecoder(input_size=8, hidden_size=64, output_size=8, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs, targets.size(1))
        loss = criterion(outputs, targets.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    recent_data = torch.tensor(numbers[-sequence_length:], dtype=torch.float).unsqueeze(0).to(device)
    predictions = []
    for _ in range(10):
        prediction = model(recent_data, target_len=1).cpu().numpy()
        predictions.append(prediction[0, 0, :])
        print('Predicted numbers:', prediction[0, 0, :6].astype(int))
        print('Predicted bonus number:', prediction[0, 0, 6].astype(int))
        print('Predicted powerball number:', prediction[0, 0, 7].astype(int))

    # 打印所有预测结果
    for i, prediction in enumerate(predictions, 1):
        print(f'Prediction {i}:')
        print('Main numbers:', prediction[:6].astype(int))
        print('Bonus number:', prediction[6].astype(int))
        print('Powerball number:', prediction[7].astype(int))