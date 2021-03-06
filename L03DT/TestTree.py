import os
import sys

import time

from Tree import *
import numpy as np


class DataUtil:
    naive_sets = {
        "mushroom", "balloon", "mnist", "cifar", "test"
    }

    @staticmethod
    def is_naive(name):
        for naive_dataset in DataUtil.naive_sets:
            if naive_dataset in name:
                return True
        return False

    @staticmethod
    def get_dataset(name, path, n_train=None, tar_idx=None, shuffle=True,
                    quantize=False, quantized=False, one_hot=False, **kwargs):
        x = []
        with open(path, "r", encoding="utf8") as file:
            if DataUtil.is_naive(name):
                for sample in file:
                    x.append(sample.strip().split(","))
            elif name == "bank1.0":
                for sample in file:
                    sample = sample.replace('"', "")
                    x.append(list(map(lambda c: c.strip(), sample.split(";"))))
            else:
                raise NotImplementedError
        if shuffle:
            np.random.shuffle(x)
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        if quantized:
            x = np.asarray(x, dtype=np.float32)
            y = y.astype(np.int8)
            if one_hot:
                y = (y[..., None] == np.arange(np.max(y) + 1))
        else:
            x = np.asarray(x)
        if quantized or not quantize:
            if n_train is None:
                return x, y
            return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])
        x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, **kwargs)
        if one_hot:
            y = (y[..., None] == np.arange(np.max(y) + 1)).astype(np.int8)
        if n_train is None:
            return x, y, wc, features, feat_dicts, label_dict
        return (
            (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:]),
            wc, features, feat_dicts, label_dict
        )

    @staticmethod
    def get_one_hot(y, n_class):
        one_hot = np.zeros([len(y), n_class])
        one_hot[range(len(y)), y] = 1
        return one_hot

    @staticmethod
    def gen_xor(size=100, scale=1, one_hot=True):
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        if one_hot:
            return np.c_[x, y].astype(np.float32), z
        return np.c_[x, y].astype(np.float32), np.argmax(z, axis=1)

    @staticmethod
    def gen_spiral(size=50, n=7, n_class=7, scale=4, one_hot=True):
        xs = np.zeros((size * n, 2), dtype=np.float32)
        ys = np.zeros(size * n, dtype=np.int8)
        for i in range(n):
            ix = range(size * i, size * (i + 1))
            r = np.linspace(0.0, 1, size + 1)[1:]
            t = np.linspace(2 * i * pi / n, 2 * (i + scale) * pi / n, size) + np.random.random(size=size) * 0.1
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = i % n_class
        if not one_hot:
            return xs, ys
        return xs, DataUtil.get_one_hot(ys, n_class)

    @staticmethod
    def gen_random(size=100, n_dim=2, n_class=2, scale=1, one_hot=True):
        xs = np.random.randn(size, n_dim).astype(np.float32) * scale
        ys = np.random.randint(n_class, size=size).astype(np.int8)
        if not one_hot:
            return xs, ys
        return xs, DataUtil.get_one_hot(ys, n_class)

    @staticmethod
    def gen_two_clusters(size=100, n_dim=2, center=0, dis=2, scale=1, one_hot=True):
        center1 = (np.random.random(n_dim) + center - 0.5) * scale + dis
        center2 = (np.random.random(n_dim) + center - 0.5) * scale - dis
        cluster1 = (np.random.randn(size, n_dim) + center1) * scale
        cluster2 = (np.random.randn(size, n_dim) + center2) * scale
        data = np.vstack((cluster1, cluster2)).astype(np.float32)
        labels = np.array([1] * size + [0] * size)
        indices = np.random.permutation(size * 2)
        data, labels = data[indices], labels[indices]
        if not one_hot:
            return data, labels
        return data, DataUtil.get_one_hot(labels, 2)

    @staticmethod
    def gen_simple_non_linear(size=120, one_hot=True):
        xs = np.random.randn(size, 2).astype(np.float32) * 1.5
        ys = np.zeros(size, dtype=np.int8)
        mask = xs[..., 1] >= xs[..., 0] ** 2
        xs[..., 1][mask] += 2
        ys[mask] = 1
        if not one_hot:
            return xs, ys
        return xs, DataUtil.get_one_hot(ys, 2)

    @staticmethod
    def gen_nine_grid(size=120, one_hot=True):
        x, y = np.random.randn(2, size).astype(np.float32)
        labels = np.zeros(size, np.int8)
        xl, xr = x <= -1, x >= 1
        yf, yc = y <= -1, y >= 1
        x_mid_mask = ~xl & ~xr
        y_mid_mask = ~yf & ~yc
        mask2 = x_mid_mask & y_mid_mask
        labels[mask2] = 2
        labels[(x_mid_mask | y_mid_mask) & ~mask2] = 1
        xs = np.vstack([x, y]).T
        if not one_hot:
            return xs, labels
        return xs, DataUtil.get_one_hot(labels, 3)

    @staticmethod
    def gen_x_set(size=1000, centers=(1, 1), slopes=(1, -1), gaps=(0.1, 0.1), one_hot=True):
        xc, yc = centers
        x, y = (2 * np.random.random([size, 2]) + np.asarray(centers) - 1).T.astype(np.float32)
        l1 = (-slopes[0] * (x - xc) + y - yc) > 0
        l2 = (-slopes[1] * (x - xc) + y - yc) > 0
        labels = np.zeros(size, dtype=np.int8)
        mask = (l1 & ~l2) | (~l1 & l2)
        labels[mask] = 1
        x[mask] += gaps[0] * np.sign(x[mask] - centers[0])
        y[~mask] += gaps[1] * np.sign(y[~mask] - centers[1])
        xs = np.vstack([x, y]).T
        if not one_hot:
            return xs, labels
        return xs, DataUtil.get_one_hot(labels, 2)

    @staticmethod
    def gen_noisy_linear(size=10000, n_dim=100, n_valid=5, noise_scale=0.5, test_ratio=0.15, one_hot=True):
        x_train = np.random.randn(size, n_dim)
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        idx = np.random.permutation(n_dim)[:n_valid]
        w = np.random.randn(n_valid, 1)
        y_train = (x_train[..., idx].dot(w) > 0).astype(np.int8).ravel()
        y_test = (x_test[..., idx].dot(w) > 0).astype(np.int8).ravel()
        if not one_hot:
            return (x_train_noise, y_train), (x_test, y_test)
        return (x_train_noise, DataUtil.get_one_hot(y_train, 2)), (x_test, DataUtil.get_one_hot(y_test, 2))

    @staticmethod
    def gen_noisy_poly(size=10000, p=3, n_dim=100, n_valid=5, noise_scale=0.5, test_ratio=0.15, one_hot=True):
        p = int(p)
        assert p > 1, "p should be greater than 1"
        x_train = np.random.randn(size, n_dim)
        x_train_list = [x_train] + [x_train ** i for i in range(2, p + 1)]
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        x_test_list = [x_test] + [x_test ** i for i in range(2, p + 1)]
        idx_list = [np.random.permutation(n_dim)[:n_valid] for _ in range(p)]
        w_list = [np.random.randn(n_valid, 1) for _ in range(p)]
        o_train = [x[..., idx].dot(w) for x, idx, w in zip(x_train_list, idx_list, w_list)]
        o_test = [x[..., idx].dot(w) for x, idx, w in zip(x_test_list, idx_list, w_list)]
        y_train = (np.sum(o_train, axis=0) > 0).astype(np.int8).ravel()
        y_test = (np.sum(o_test, axis=0) > 0).astype(np.int8).ravel()
        if not one_hot:
            return (x_train_noise, y_train), (x_test, y_test)
        return (x_train_noise, DataUtil.get_one_hot(y_train, 2)), (x_test, DataUtil.get_one_hot(y_test, 2))

    @staticmethod
    def gen_special_linear(size=10000, n_dim=10, n_redundant=3, n_categorical=3,
                           cv_ratio=0.15, test_ratio=0.15, one_hot=True):
        x_train = np.random.randn(size, n_dim)
        x_train_redundant = np.ones([size, n_redundant]) * np.random.randint(0, 3, n_redundant)
        x_train_categorical = np.random.randint(3, 8, [size, n_categorical])
        x_train_stacked = np.hstack([x_train, x_train_redundant, x_train_categorical])
        n_test = int(size * test_ratio)
        x_test = np.random.randn(n_test, n_dim)
        x_test_redundant = np.ones([n_test, n_redundant]) * np.random.randint(3, 6, n_redundant)
        x_test_categorical = np.random.randint(0, 5, [n_test, n_categorical])
        x_test_stacked = np.hstack([x_test, x_test_redundant, x_test_categorical])
        w = np.random.randn(n_dim, 1)
        y_train = (x_train.dot(w) > 0).astype(np.int8).ravel()
        y_test = (x_test.dot(w) > 0).astype(np.int8).ravel()
        n_cv = int(size * cv_ratio)
        x_train_stacked, x_cv_stacked = x_train_stacked[:-n_cv], x_train_stacked[-n_cv:]
        y_train, y_cv = y_train[:-n_cv], y_train[-n_cv:]
        if not one_hot:
            return (x_train_stacked, y_train), (x_cv_stacked, y_cv), (x_test_stacked, y_test)
        return (
            (x_train_stacked, DataUtil.get_one_hot(y_train, 2)),
            (x_cv_stacked, DataUtil.get_one_hot(y_cv, 2)),
            (x_test_stacked, DataUtil.get_one_hot(y_test, 2))
        )

    @staticmethod
    def quantize_data(x, y, wc=None, continuous_rate=0.1, separate=False):
        if isinstance(x, list):
            xt = map(list, zip(*x))
        else:
            xt = x.T
        features = [set(feat) for feat in xt]
        if wc is None:
            wc = np.array([len(feat) >= int(continuous_rate * len(y)) for feat in features])
        else:
            wc = np.asarray(wc)
        feat_dicts = [
            {_l: i for i, _l in enumerate(feats)} if not wc[i] else None
            for i, feats in enumerate(features)
        ]
        if not separate:
            if np.all(~wc):
                dtype = np.int
            else:
                dtype = np.float32
            x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                          for sample in x], dtype=dtype)
        else:
            x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                          for sample in x], dtype=np.float32)
            x = (x[:, ~wc].astype(np.int), x[:, wc])
        label_dict = {l: i for i, l in enumerate(set(y))}
        y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
        label_dict = {i: l for l, i in label_dict.items()}
        return x, y, wc, features, feat_dicts, label_dict

    @staticmethod
    def transform_data(x, y, wc, feat_dicts, label_dict):
        if np.all(~wc):
            dtype = np.int
        else:
            dtype = np.float32
        label_dict = {l: i for i, l in label_dict.items()}
        x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                      for sample in x], dtype=dtype)
        y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
        return x, y


def main(visualize=True):
    # x, y = DataUtil.get_dataset("balloon1.0(en)", "../_Data/balloon1.0(en).txt")
    print(
        "===============================\n"
        "{}\n"
        "-------------------------------\n".format('test'), end='\t')
    x, y = DataUtil.get_dataset("test", "data/test.txt")
    fit_time = time.time()
    tree = CartTree(whether_continuous=[False] * 4)
    tree.fit(x, y, train_only=True)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()
    print(
        "===============================\n"
        "{}\n"
        "-------------------------------\n".format('mushroom'), end='\t')
    train_num = 6000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "data/mushroom.txt", tar_idx=0, n_train=train_num)
    fit_time = time.time()
    tree = C45Tree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x_train, y_train)
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()

    x, y = DataUtil.gen_xor(one_hot=False)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x, y, train_only=True)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y, n_cores=1)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize2d(x, y, dense=1000)
        tree.visualize()

    wc = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in continuous_lst:
        wc[_cl] = True
    print(
        "===============================\n"
        "{}\n"
        "-------------------------------\n".format('bank1.0'), end='\t')
    train_num = 2000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "bank1.0", "data/bank1.0.txt", n_train=train_num, quantize=True)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()


if __name__ == '__main__':
    main(True)
