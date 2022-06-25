import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# To use the following code, you need to copy them to your "models.py" file.
# Author: Qidong Yang
# Last Edit Date: 06-24-2022


'''

Learnable Basis Module

'''


class BasisGenerator(nn.Module):
    def __init__(self, shape=(16, 4, 4), positivity=True, device='cuda'):
        super(BasisGenerator, self).__init__()

        self.shape = shape
        self.positivity = positivity
        self.size = shape[0] * shape[1] * shape[2]
        self.device = device

        self.basis_generator = nn.Linear(1, self.size, bias=False)

        # Generate learnable basis with sum one on shape[0] dimension
        # shape: the basis output shape without batch dimension [tuple] (reduce shape[0] to encourage sparsity)
        # positivity: whether basis elementwise positive [boolean]
        # device: the place to put the basis

    def forward(self):

        input = torch.ones(1, 1).to(self.device)
        basis = self.basis_generator(input)
        # (1, size)
        basis = basis.reshape((1, self.shape[0], self.shape[1], self.shape[2]))[0]
        # (shape[0], shape[1], shape[2])

        if self.positivity:
            basis = torch.exp(basis)

        # Normalization
        sums = torch.sum(basis, dim=(1, 2))
        basis = basis / sums.reshape((-1, 1, 1))

        return basis


'''

MotifNet with learnable basis example
Note: the following code serves as an example to show how to use the above learnable basis module,
which means the following code has not been tested yet!!

'''


class MotifNetLearnBasis(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1):
        super(MotifNetLearnBasis, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2))

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Constraint layers
        self.softmax = nn.Softmax(dim=1)
        self.mult_in = MultIn()
        self.basis_generator = BasisGenerator()

    def forward(self, x):

        out = self.conv1(x[:, 0, ...])
        out = self.conv2(out)

        for layer in self.res_blocks:
            out = layer(out)

        out = self.conv3(out)

        # Softmax constraining
        out = self.softmax(out)
        # (n_batch, 16, 32, 32)
        out = self.mult_in(out, x[:, 0, ...])
        # (n_batch, 16, 32, 32)

        # Generate basis
        basis = self.basis_generator()
        # (16, 4, 4)

        # Kron operation
        output = torch.tensor(np.zeros((out.shape[0], out.shape[1], out.shape[2] * basis.shape[-2], out.shape[3] * basis.shape[-1])), requires_grad=True).to('cuda')
        for ii in range(basis.shape[0]):
            output[:, ii, :, :] = torch.kron(out[:, ii, :, :], basis[ii, :, :])
        # (n_batch, 16, 32*4, 32*4)

        # Add channels
        output = torch.sum(output, dim=1)
        output = output.unsqueeze(1)
        output = output.unsqueeze(1)

        return output
