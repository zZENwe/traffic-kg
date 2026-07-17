import numpy as np
import torch
import torch.nn as nn
from dcrnn_cell import DCGRUCell


class DCGRUEncoder(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes,
                 num_rnn_layers, filter_type, use_gc_for_ru=True, extra_supports=None):
        super().__init__()
        self._num_nodes = num_nodes
        self._num_rnn_layers = num_rnn_layers
        self._num_units = num_units

        self.cells = nn.ModuleList()
        for i in range(num_rnn_layers):
            in_dim = input_dim if i == 0 else num_units
            self.cells.append(DCGRUCell(in_dim, num_units, adj_mx, max_diffusion_step,
                                        num_nodes, filter_type, use_gc_for_ru,
                                        extra_supports=extra_supports))

    def forward(self, inputs):
        # inputs: (B, seq_len, num_nodes, input_dim)
        batch_size, seq_len = inputs.size(0), inputs.size(1)
        inputs = inputs.view(batch_size, seq_len, -1)  # (B, seq_len, N*input_dim)

        state = [torch.zeros(batch_size, self._num_nodes * self._num_units, device=inputs.device)
                 for _ in range(self._num_rnn_layers)]

        for t in range(seq_len):
            x = inputs[:, t, :]
            for i, cell in enumerate(self.cells):
                x, state[i] = cell(x, state[i])

        return state


class DCRNNModel(nn.Module):
    def __init__(self, adj_mx, batch_size, seq_len=12, horizon=12, input_dim=2,
                 output_dim=1, num_nodes=207, num_rnn_layers=2, rnn_units=64,
                 max_diffusion_step=2, filter_type="dual_random_walk",
                 use_curriculum_learning=True, cl_decay_steps=2000,
                 extra_supports=None):
        super().__init__()
        self._num_nodes = num_nodes
        self._output_dim = output_dim
        self._rnn_units = rnn_units
        self._num_rnn_layers = num_rnn_layers
        self._horizon = horizon
        self._use_curriculum_learning = use_curriculum_learning
        self._cl_decay_steps = cl_decay_steps

        self.encoder = DCGRUEncoder(
            input_dim, rnn_units, adj_mx, max_diffusion_step, num_nodes,
            num_rnn_layers, filter_type, use_gc_for_ru=True,
            extra_supports=extra_supports
        )

        # Decoder cells: last one has projection to output_dim
        self.decoder_cells = nn.ModuleList()
        for i in range(num_rnn_layers):
            in_dim = output_dim if i == 0 else rnn_units
            proj = output_dim if i == num_rnn_layers - 1 else None
            self.decoder_cells.append(DCGRUCell(in_dim, rnn_units, adj_mx,
                                                max_diffusion_step, num_nodes,
                                                filter_type, use_gc_for_ru=True,
                                                num_proj=proj,
                                                extra_supports=extra_supports))

    def _compute_sampling_threshold(self, global_step):
        k = self._cl_decay_steps
        return k / (k + np.exp(global_step / k))

    def forward(self, inputs, labels=None, global_step=None):
        # inputs: (B, seq_len, N, input_dim)
        # labels: (B, horizon, N, output_dim) — used in training for scheduled sampling
        batch_size = inputs.size(0)
        device = inputs.device

        enc_states = self.encoder(inputs)

        go_symbol = torch.zeros(batch_size, self._num_nodes * self._output_dim, device=device)
        outputs = []

        prev = go_symbol
        for i in range(self._horizon):
            states_i = []
            x = prev
            for j, cell in enumerate(self.decoder_cells):
                x, new_s = cell(x, enc_states[j])
                states_i.append(new_s)

            outputs.append(x.view(batch_size, self._num_nodes, self._output_dim))
            enc_states = states_i  # carry forward decoder states

            if self.training and self._use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                threshold = self._compute_sampling_threshold(global_step)
                if c < threshold:
                    prev = labels[:, i, :, :self._output_dim].reshape(batch_size, -1)
                else:
                    prev = x
            else:
                prev = x

        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N, output_dim)
        return outputs
