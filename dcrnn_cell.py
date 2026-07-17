import math
import torch
import torch.nn as nn

from utils import build_supports


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes,
                 filter_type="dual_random_walk", use_gc_for_ru=True, num_proj=None,
                 extra_supports=None):
        super().__init__()
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru

        # Precompute diffusion kernel
        if extra_supports is not None:
            supports = extra_supports
        else:
            supports = build_supports(adj_mx, filter_type)
        kernels = [torch.eye(num_nodes)]
        for support in supports:
            s0 = torch.eye(num_nodes)
            s1 = support
            kernels.append(s1)
            if max_diffusion_step >= 2:
                s_prev, s_cur = s0, s1
                for k in range(2, max_diffusion_step + 1):
                    s_next = 2 * support @ s_cur - s_prev
                    kernels.append(s_next)
                    s_prev, s_cur = s_cur, s_next

        self.register_buffer('_diff_kernel', torch.stack(kernels, dim=0))  # (M, N, N)
        self._num_matrices = len(kernels)

        gconv_input_size = input_dim + num_units
        self.gate_weights = nn.Parameter(
            torch.empty(gconv_input_size * self._num_matrices, 2 * num_units))
        self.gate_bias = nn.Parameter(torch.zeros(2 * num_units))
        self.candidate_weights = nn.Parameter(
            torch.empty(gconv_input_size * self._num_matrices, num_units))
        self.candidate_bias = nn.Parameter(torch.zeros(num_units))
        self.reset_parameters()

        if num_proj is not None:
            self.proj_weight = nn.Parameter(torch.empty(num_units, num_proj))
            self.reset_parameters_proj()

    def reset_parameters(self):
        for w in [self.gate_weights, self.candidate_weights]:
            nn.init.xavier_uniform_(w)
        nn.init.constant_(self.gate_bias, 1.0)

    def reset_parameters_proj(self):
        nn.init.xavier_uniform_(self.proj_weight)

    def _gconv(self, inputs, state, weights, bias, output_size):
        B = inputs.size(0)
        N = self._num_nodes
        M = self._num_matrices

        inputs = inputs.view(B, N, -1)
        state = state.view(B, N, -1)
        x = torch.cat([inputs, state], dim=2)  # (B, N, C)
        C = x.size(2)

        # Batched diffusion: (M, N, N) @ (M, N, C*B) -> (M, N, C*B)
        x0 = x.permute(1, 2, 0).reshape(N, C * B)  # (N, C*B)
        x0 = x0.unsqueeze(0).expand(M, N, C * B)     # (M, N, C*B)
        x_diffused = torch.bmm(self._diff_kernel, x0)  # (M, N, C*B)

        x = x_diffused.view(M, N, C, B).permute(3, 1, 2, 0)  # (B, N, C, M)
        x = x.reshape(B * N, C * M)
        x = x @ weights + bias
        return x.view(B, N * output_size)

    def forward(self, inputs, state):
        if self._use_gc_for_ru:
            value = torch.sigmoid(
                self._gconv(inputs, state, self.gate_weights, self.gate_bias, 2 * self._num_units))
        else:
            value = self._fc(inputs, state, 2 * self._num_units)

        value = value.view(-1, self._num_nodes, 2 * self._num_units)
        r, u = torch.split(value, self._num_units, dim=-1)
        r = r.reshape(-1, self._num_nodes * self._num_units)
        u = u.reshape(-1, self._num_nodes * self._num_units)

        c = self._gconv(inputs, r * state, self.candidate_weights, self.candidate_bias, self._num_units)
        c = torch.tanh(c)

        new_state = u * state + (1 - u) * c
        output = new_state.view(-1, self._num_nodes, self._num_units)

        if self._num_proj is not None:
            output = output.reshape(-1, self._num_units) @ self.proj_weight
            output = output.view(-1, self._num_nodes * self._num_proj)

        return output, new_state
