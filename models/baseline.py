
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()

        # Define the main branch (left branch)
        self.first_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=17, stride=4, padding=7),
        )

        # Define the skip connection (right branch)
        self.skip_connection = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
        )

        self.second_branch = nn.Sequential(
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout_rate)
        )

    def forward(self, lower_out, upper_out):
        first_out = self.first_branch(lower_out)

        residual = self.skip_connection(upper_out)

        upper_out = first_out + residual

        lower_out = self.second_branch(upper_out)

        return lower_out, upper_out

class StackedResidual(nn.Module):
    def __init__(self, channels, num_blocks, dropout_rate=0.2):
        super().__init__()

        # Create a list of residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = channels[i]
            out_channels = in_channels + 64
            self.blocks.append(
                ResidualBlock(in_channels, out_channels, dropout_rate=dropout_rate)
            )

    def forward(self, x):
        lower, upper = x, x
        for i, block in enumerate(self.blocks):
            lower, upper = block(lower, upper)

        return lower

class ResnetBaseline(nn.Module):
    def __init__(self, n_classes, num_blocks = 4, channels = [64, 128, 192, 256], dropout_rate = 0.2):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=17, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 64, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
        )

        self.residual_blocks = StackedResidual(
            channels, num_blocks, dropout_rate=dropout_rate
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1280, n_classes)

    def forward(self, x):
        x = self.input_layer(x)

        x = self.residual_blocks(x)

        x = self.flatten(x)

        logits = self.linear(x)

        return {'logits': logits, 'signal_embedding': x}

### IMPLEMENTAÇÃO ANTONIO ###
import torch.nn as nn
import numpy as np

def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding

def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample

class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""
    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        # if kernel_size % 2 == 0:
        #     raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y

class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x