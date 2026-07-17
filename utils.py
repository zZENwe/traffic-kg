import numpy as np
import pickle
import scipy.sparse as sp
import torch


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(d, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mx).tocoo()


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    lap = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()
    return lap


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        from scipy.sparse.linalg import eigsh
        lambda_max, _ = eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def build_supports(adj_mx, filter_type):
    if filter_type == "laplacian":
        supports = [calculate_scaled_laplacian(adj_mx, lambda_max=None)]
    elif filter_type == "random_walk":
        supports = [calculate_random_walk_matrix(adj_mx).T]
    elif filter_type == "dual_random_walk":
        supports = [calculate_random_walk_matrix(adj_mx).T,
                    calculate_random_walk_matrix(adj_mx.T).T]
    else:
        supports = [calculate_scaled_laplacian(adj_mx)]
    # Convert to dense torch tensors (207x207 is tiny)
    return [torch.FloatTensor(s.toarray()) for s in supports]


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader:
    def __init__(self, xs, ys, batch_size, device, shuffle=False):
        self.batch_size = batch_size
        self.device = device
        if len(xs) % batch_size != 0:
            num_pad = (batch_size - (len(xs) % batch_size))
            xs = np.concatenate([xs, np.repeat(xs[-1:], num_pad, axis=0)], axis=0)
            ys = np.concatenate([ys, np.repeat(ys[-1:], num_pad, axis=0)], axis=0)
        self.xs = torch.FloatTensor(xs)
        self.ys = torch.FloatTensor(ys)
        self.size = len(xs)
        self.num_batch = self.size // batch_size
        self.shuffle = shuffle

    def get_iterator(self):
        if self.shuffle:
            perm = torch.randperm(self.size)
            self.xs = self.xs[perm]
            self.ys = self.ys[perm]
        for i in range(self.num_batch):
            start = i * self.batch_size
            end = start + self.batch_size
            yield (self.xs[start:end].to(self.device),
                   self.ys[start:end].to(self.device))


def load_dataset(dataset_dir, batch_size, test_batch_size, device):
    data = {}
    for cat in ['train', 'val', 'test']:
        cat_data = np.load(f'{dataset_dir}/{cat}.npz')
        data[f'x_{cat}'] = cat_data['x']
        data[f'y_{cat}'] = cat_data['y']

    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std()
    )
    for cat in ['train', 'val', 'test']:
        data[f'x_{cat}'][..., 0] = scaler.transform(data[f'x_{cat}'][..., 0])
        data[f'y_{cat}'][..., 0] = scaler.transform(data[f'y_{cat}'][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, device, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, device, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, device, shuffle=False)
    data['scaler'] = scaler
    return data


def masked_mae_np(preds, labels, null_val=0.):
    mask = (labels != null_val).astype('float32')
    mask /= np.mean(mask)
    return np.mean(np.abs(preds - labels) * mask)


def masked_mape_np(preds, labels, null_val=0.):
    mask = (labels > 1e-5).astype('float32')
    mask /= np.mean(mask)
    return np.mean(np.abs((preds - labels) / labels) * mask) * 100


def masked_rmse_np(preds, labels, null_val=0.):
    mask = (labels != null_val).astype('float32')
    mask /= np.mean(mask)
    return np.sqrt(np.mean((preds - labels) ** 2 * mask))


def masked_mae_loss(preds, labels, null_val=0.):
    mask = (labels != null_val).float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels) * mask
    return loss.mean()
