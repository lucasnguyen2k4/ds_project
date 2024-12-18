from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch

class CustomGRU(nn.Module):
    def __init__(self, input_size, output_size, seq_len=4, label_scaler=None):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.label_scaler = label_scaler
        
        # fully connected layers to generate initial hidden state for GRU layers
        self.init_nn = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # GRU layers
        self.flatten = nn.Flatten(1, -1)
        self.normalize = nn.LayerNorm(input_size * seq_len)
        self.gru1 = nn.GRU(input_size, 256, batch_first=True)
        self.gru2 = nn.GRU(256, 128, batch_first=True)
        self.gru3 = nn.GRU(128, 64, batch_first=True)
        
        # Final fully connected layer
        self.linear = nn.Linear(64, output_size)

    def forward(self, inp, rescale=False):
        X, init_data = inp
        X = self.flatten(X)
        X = self.normalize(X).reshape((-1, self.seq_len, self.input_size))
        init_data = self.init_nn(init_data.unsqueeze(0))
        X, _ = self.gru1(X, init_data)
        X, _ = self.gru2(X)
        X, _ = self.gru3(X)
        X = self.linear(X)
        # Rescale if needed with a standard scaler (for actual prediction)
        if rescale:
            X = self.label_scaler.inverse_transform(X)
        return X
    
    def predict(self, inp, numpy_output=True):
        X, init_data = inp
        inp = (Tensor(X), Tensor(init_data))
        self.eval()
        with torch.no_grad():
            output = self(inp, rescale=True)
        if numpy_output:
            output = output.numpy()
        return output[:, -1]
    
class StandardScaler(torch.nn.Module):
    """Scaler for normalizing and revert data to original."""
    def __init__(self):
        super(StandardScaler, self).__init__()
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True)

    def forward(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean

"""
All credits goes to https://github.com/ndrplz/ConvLSTM_pytorch 
"""

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              #padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvLSTMTimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim, input_width, output_width):
        super().__init__()
        self.conv_lstm = ConvLSTM(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            kernel_size = (1 , input_width),
            num_layers = len(hidden_dim),
            batch_first = True
            )
        self.linear = nn.Linear(hidden_dim[-1] * input_width, input_dim*output_width)
        self.flatten = nn.Flatten(1, -1)

    def forward(self, X):
        _, last_states = self.conv_lstm(X)
        X = last_states[0][0]
        X = self.flatten(X)
        X = self.linear(X)

        return X

    def predict(self, X, numpy_output=True):
        self.eval()
        with torch.no_grad():
            output = self(X)
        if numpy_output:
            output = output.numpy()
        return output
    
class TimeSeries3DDataset(Dataset):
    def __init__(self, target, features, n_provinces, sequence_length=3, custom_scaler=None, predict=False):
        if custom_scaler:
            self.features_scaler, self.target_scaler = custom_scaler
        else:
            self.target_scaler = preprocessing.StandardScaler()
            self.target_scaler.fit(target.values)
            self.features_scaler = preprocessing.StandardScaler()
            self.features_scaler.fit(features.values)

        self.features = self.features_scaler.transform(features.values)

        self.X = torch.tensor(self.features.reshape(n_provinces, len(features)//n_provinces, len(features.columns))).float()
        self.predict = predict
        if self.predict:
            self.y = None
        else:
            self.target = self.target_scaler.transform(target.values)
            self.y = torch.tensor(self.target.reshape(n_provinces, len(target)//n_provinces, len(target.columns))).float()
            self.target_length = self.target.shape[-1]

        self.sequence_length = sequence_length
        self.features_length = self.features.shape[-1]
    



    def __len__(self):
        return self.X.shape[1]

    def _mirror_padding(self, x, sequence_length, padding_needed):
        mirrored_part = torch.flip(x[:, :padding_needed, :], dims=[1])
        padded_x = torch.cat([mirrored_part, x], dim=1)
        return padded_x

    def _get_window(self, X, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = X[:, i_start:(i + 1), :]
        else:
            padding_needed = self.sequence_length - i - 1
            x = self._mirror_padding(X, self.sequence_length, padding_needed)
            x = x[:, :self.sequence_length, :]
        return x

    def __getitem__(self, i):
        x_window = self._get_window(self.X, i)
        x_window = x_window.permute(1, 0, 2).unsqueeze(2)
        if self.predict: 
            return x_window
        return x_window, self.y[:, i, :].flatten()