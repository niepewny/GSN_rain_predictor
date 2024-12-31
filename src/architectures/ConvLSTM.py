import torch.nn as nn
import torch
from src.data_modules.SEVIR import SEVIRDataset

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, conv_output.size(1) // 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            self.cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size))

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        h = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cell_list[layer](input_t, h[layer], c[layer])
                input_t = h[layer]
            outputs.append(h[-1])
        return torch.stack(outputs, dim=1)


import torch.optim as optim

# Initialize dataset and dataloader
dataset = SEVIRDataset('path_to_sevir_data.h5')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
model = ConvLSTM(input_channels=1, hidden_channels=64, kernel_size=3, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs in dataloader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Adjust target as per your task
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
