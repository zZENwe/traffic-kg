import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # (num_samples, num_nodes, 1)
    data_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)

    data = np.concatenate(data_list, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_df_filename", type=str, default="data/metr-la.h5")
    parser.add_argument("--output_dir", type=str, default="data/METR-LA")
    args = parser.parse_args()

    print(f"Reading {args.traffic_df_filename}...")
    df = pd.read_hdf(args.traffic_df_filename)
    print(f"Data shape: {df.shape}")

    # seq_len=12: 11 previous steps + current step
    x_offsets = np.sort(np.arange(-11, 1, 1))  # [-11, -10, ..., 0]
    y_offsets = np.sort(np.arange(1, 13, 1))   # [1, 2, ..., 12] — predict next hour

    x, y = generate_graph_seq2seq_io_data(df, x_offsets=x_offsets, y_offsets=y_offsets,
                                          add_time_in_day=True)
    print(f"x: {x.shape}, y: {y.shape}")

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    os.makedirs(args.output_dir, exist_ok=True)

    splits = [
        ('train', x[:num_train], y[:num_train]),
        ('val', x[num_train:num_train + num_val], y[num_train:num_train + num_val]),
        ('test', x[-num_test:], y[-num_test:])
    ]
    for cat, xs, ys in splits:
        np.savez_compressed(os.path.join(args.output_dir, f"{cat}.npz"), x=xs, y=ys)
        print(f"{cat}: x {xs.shape}, y {ys.shape}")

    print("Done! Data saved to", args.output_dir)


if __name__ == "__main__":
    main()
