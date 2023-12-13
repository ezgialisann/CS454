import torch
import torchvision
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
data_dir = 'data'

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True,transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True,transform=transform)
m = len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(20 * 6 * 64, encoded_space_dim)
        )
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 20 * 6 * 64),
            nn.ReLU(True),
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(64, 20, 6))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

loss_fn = torch.nn.MSELoss()

lr = 0.0015
torch.manual_seed(0)
d = 4

encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

encoder.to(device)
decoder.to(device)


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []
    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        width = image_batch.size(-1)
        left_images = image_batch[:, :, :, :width//2]  # Extract left side
        right_images = image_batch[:, :, :, width//2:]  # Extract right side
        encoded_data = encoder(left_images)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, right_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            width = image_batch.size(-1)
            left_images = image_batch[:, :, :, :width//2]  # Extract left side
            right_images = image_batch[:, :, :, width//2:]  # Extract right side
            encoded_data = encoder(left_images)
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(right_images.cpu())
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data



def plot_ae_outputs(encoder, decoder, n=10):
    targets = test_dataset.targets.numpy()
    t_idx = {i: random.sample(list(np.where(targets == i)[0]), 5) for i in range(10)}
    for j in range(5):
        plt.figure(figsize=(16, 8))
        for i in range (n):
            ax = plt.subplot(3, n, i + 1)
            img = test_dataset[t_idx[i][j]][0].unsqueeze(0).to(device)
            width = img.size(-1)
            left = img[:, :, :, :width//2]
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                rec_img = decoder(encoder(left))
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title('Original images')
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(left.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title('Left side of the original images')
            ax = plt.subplot(3, n, i + 1 + n + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title('Reconstructed images')
        plt.show()

num_epochs = 25
diz_loss = {'Epoch': [],'train_loss': [], 'val_loss': []}
trainMSE = []
valMSE = []
for epoch in range(num_epochs):
    train_loss = train_epoch(encoder, decoder, device,
                             train_loader, loss_fn, optim)
    val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
    diz_loss['Epoch'].append(epoch + 1)
    diz_loss['train_loss'].append(train_loss)
    val_loss_value = val_loss.item()
    diz_loss['val_loss'].append(val_loss_value)
    plot_ae_outputs(encoder, decoder, n=10)

plt.figure()
plt.scatter(diz_loss['Epoch'], diz_loss['train_loss'])
plt.plot(diz_loss['Epoch'],diz_loss['train_loss'], marker = 'o', color='red', label = 'Train Error')
plt.scatter(diz_loss['Epoch'], diz_loss['val_loss'])
plt.plot(diz_loss['Epoch'],diz_loss['val_loss'], marker = 'o', color='blue', label = 'Test Error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

print(diz_loss['Epoch'])
print(diz_loss['train_loss'])
print(diz_loss['val_loss'])