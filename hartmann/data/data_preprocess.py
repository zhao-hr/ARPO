import numpy as np
import os
import argparse

def hartmann(x0, x1, x2):
    n_samples = x0.shape[0]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = np.array([[3689, 1170, 2673],
                  [4699, 4387, 7470],
                  [1091, 8732, 5547],
                  [381, 5743, 8828]]) * 1e-4
    x = np.array([[x0, x1, x2]])
    x = np.transpose(x, [2,0,1])
    x = np.repeat(x, repeats=4, axis=1)
    P = np.repeat(np.array([P]), repeats=n_samples, axis=0)
    exponent = np.einsum("ijk,jk->ij", (x - P) ** 2, A)
    y = -np.einsum("ij,j->i", np.exp(-exponent), alpha)
    return -y

def create_random_mesh(n_points=10):
    x0 = np.random.random(n_points)
    x1 = np.random.random(n_points)
    x2 = np.random.random(n_points)
    # x0 = np.concatenate([x0, np.array([0.114614])], axis=0)
    # x1 = np.concatenate([x1, np.array([0.555649])], axis=0)
    # x2 = np.concatenate([x2, np.array([0.852547])], axis=0)
    y = hartmann(x0, x1, x2)
    return x0, x1, x2, y

def create_norm_mesh(n_points=10, std=0.1):
    opt_x0 = 0.114614
    opt_x1 = 0.555649
    opt_x2 = 0.852547
    x0 = np.random.randn(n_points) * std + opt_x0
    x1 = np.random.randn(n_points) * std + opt_x1
    x2 = np.random.randn(n_points) * std + opt_x2
    limits = (x0 > 0) & (x0 < 1) & (x1 > 0) & (x1 < 1) & (x2 > 0) & (x2 < 1) & \
        ((x0 - opt_x0) ** 2 + (x1 - opt_x1) ** 2 + (x2 - opt_x2) ** 2 > (std * 0.5) ** 2)
    x0, x1, x2 = x0[limits], x1[limits], x2[limits]
    y = hartmann(x0, x1, x2)
    return x0, x1, x2, y

def init_random_data(n_points=100):
    x0, x1, x2, y = create_random_mesh(n_points)
    data = {}
    data["x"] = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=-1)
    data["y"] = y.reshape(-1, 1)
    return data

def init_top_data(n_points=100, param=0.5):
    x0, x1, x2, y = create_random_mesh(int(n_points / param))
    data = {}
    x = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=-1)
    y = y.reshape(-1, 1)
    indices = np.argsort(y, axis=0).reshape(-1)
    data["x"] = x[indices[-n_points:]]
    data["y"] = y[indices[-n_points:]]
    return data

def init_bottom_data(n_points=100, param=0.5):
    x0, x1, x2, y = create_random_mesh(int(n_points / param))
    data = {}
    x = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=-1)
    y = y.reshape(-1, 1)
    indices = np.argsort(y, axis=0).reshape(-1)
    data["x"] = x[indices[:n_points]]
    data["y"] = y[indices[:n_points]]
    return data

def init_norm_data(n_points=100, param=0.2):
    x0, x1, x2, y = create_norm_mesh(int(n_points * 2), std=param)
    data = {}
    # print(y.shape)
    x = np.concatenate([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=-1)
    y = y.reshape(-1, 1)
    indices = np.argsort(y, axis=0).reshape(-1)
    data["x"] = x[indices[-n_points:]]
    data["y"] = y[indices[-n_points:]]
    return data

def main(args):
    if args.method == "top": data = init_top_data(n_points=12000, param=args.param)
    elif args.method == "bottom": data = init_bottom_data(n_points=12000, param=args.param)
    elif args.method == "random": data = init_random_data(n_points=12000)
    else: data = init_norm_data(n_points=12000, param=args.param)
    # print(data["x"][np.argmax(data["y"])], data["y"].max(), data["y"].min())
    print(data['x'].shape, data['y'].shape)
    os.makedirs(args.method, exist_ok=True)
    np.save(os.path.join(args.method, 'x.npy'), data['x'])
    np.save(os.path.join(args.method, 'y.npy'), data['y'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, default="norm")
    parser.add_argument("-p", "--param", type=float, default=0.5)
    args = parser.parse_args()
    main(args)