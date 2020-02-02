import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import pi, exp

from Bases import ClassifierBase

sqrt_pi = (2 * pi) ** 0.5


class NBFunctions:
    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        def func(_c):
            def sub(x):
                return NBFunctions.gaussian(x, mu[_c], sigma[_c])

            return sub

        return [func(_c=c) for c in range(n_category)]


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


class NaiveBayes(ClassifierBase):

    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__(**kwargs)
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dict = self._feat_dicts = None

    def feed_data(self, x, y, sample_weight=None):
        pass

    def feed_sample_weight(self, sample_weight=None):
        pass

    def get_prior_probability(self, lb=1):
        return [(c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for c_num in self._cat_counter]

    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._func = self._fit(lb)

    def _fit(self, lb):
        pass

    def predict_one(self, x, get_raw_result=False):
        if type(x) is np.ndarray:
            x = x.tolist()
        else:
            x = x[:]
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_result:
            return self.label_dict[m_arg]
        return m_probability

    def predict(self, x, get_raw_result=False, **kwargs):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    def _transfer_x(self, x):
        return x


class MergedNB(NaiveBayes):

    def __init__(self, **kwargs):
        super(MergedNB, self).__init__(**kwargs)
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()

        wc = kwargs.get("whether_continuous")
        if wc is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.asarray(wc)
            self._whether_discrete = ~self._whether_continuous

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(
            x, y, wc=self._whether_continuous, separate=True)
        if self._whether_continuous is None:
            self._whether_continuous = wc
            self._whether_discrete = ~self._whether_continuous
        self.label_dict = label_dict
        discrete_x, continuous_x = x
        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter
        labels = [y == value for value in range(len(cat_counter))]

        labelled_x = [discrete_x[ci].T for ci in labels]
        self._multinomial._x, self._multinomial._y = x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dicts = [dic for i, dic in enumerate(feat_dicts) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features)
                                              if self._whether_discrete[i]]
        self._multinomial.label_dict = label_dict

        labelled_x = [continuous_x[label].T for label in labels]
        self._gaussian._x, self._gaussian._y = continuous_x.T, y
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels
        self._gaussian._cat_counter, self._gaussian.label_dict = cat_counter, label_dict

        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._gaussian.feed_sample_weight(sample_weight)

    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        p_category = self._multinomial.get_prior_probability(lb)
        discrete_func, continuous_func = self._multinomial["func"], self._gaussian["func"]

        def func(input_x, tar_category):
            input_x = np.asarray(input_x)
            return discrete_func(
                input_x[self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[self._whether_continuous], tar_category) / p_category[tar_category]

        return func

    def _transfer_x(self, x):
        feat_dicts = self._multinomial["feat_dicts"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            if not discrete:
                x[d] = float(x[d])
            else:
                x[d] = feat_dicts[idx][x[d]]
            if discrete:
                idx += 1
        return x


class GaussianNB(NaiveBayes):

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x = np.array([list(map(lambda c: float(c), sample)) for sample in x])
        labels = list(set(y))
        label_dict = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dict[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[label].T for label in labels]

        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dict = cat_counter, {i: l for l, i in label_dict.items()}
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weights = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weights[label]

    def _fit(self, lb):
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        data = [
            NBFunctions.gaussian_maximum_likelihood(
                self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]

        return func

    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dict) for i in range(len(self.label_dict))])
        colors = {cat: color for cat, color in zip(self.label_dict.values(), colors)}
        for j in range(len(self._x)):
            tmp_data = self._x[j]
            x_min, x_max = np.min(tmp_data), np.max(tmp_data)
            gap = x_max - x_min
            tmp_x = np.linspace(x_min - 0.1 * gap, x_max + 0.1 * gap, 200)
            title = "$j = {}$".format(j + 1)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.plot(tmp_x, [self._data[j][c](xx) for xx in tmp_x],
                         c=colors[self.label_dict[c]], label="class: {}".format(self.label_dict[c]))
            plt.xlim(x_min - 0.2 * gap, x_max + 0.2 * gap)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))


class MultinomialNB(NaiveBayes):

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, _, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, wc=np.array([False] * len(x[0])))
        cat_counter = np.bincount(y)
        n_possibilities = [len(feats) for feats in features]
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[ci].T for ci in labels]

        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dicts, self._n_possibilities = cat_counter, feat_dicts, n_possibilities
        self.label_dict = label_dict
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=p) for xx in self._labelled_x])
            else:
                local_weights = sample_weight * len(sample_weight)
                self._con_counter.append([
                    np.bincount(xx[dim], weights=local_weights[label], minlength=p)
                    for label, xx in self._label_zip])

    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)

        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                 for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            return rs * p_category[tar_category]

        return func

    def _transfer_x(self, x):
        for j, char in enumerate(x):
            x[j] = self._feat_dicts[j][char]
        return x

    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dict) for i in range(len(self.label_dict))])
        colors = {cat: color for cat, color in zip(self.label_dict.values(), colors)}
        rev_feat_dicts = [{val: key for key, val in feat_dict.items()} for feat_dict in self._feat_dicts]
        for j in range(len(self._n_possibilities)):
            rev_dict = rev_feat_dicts[j]
            sj = self._n_possibilities[j]
            tmp_x = np.arange(1, sj + 1)
            title = "$j = {}; S_j = {}$".format(j + 1, sj)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.bar(tmp_x - 0.35 * c, self._data[j][c, :], width=0.35,
                        facecolor=colors[self.label_dict[c]], edgecolor="white",
                        label=u"class: {}".format(self.label_dict[c]))
            plt.xticks([i for i in range(sj + 2)], [""] + [rev_dict[i] for i in range(sj)] + [""])
            plt.ylim(0, 1.0)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))


if __name__ == '__main__':
    import time

    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst:
        whether_continuous[cl] = True

    train_num = 40000
    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "bank1.0", "data/bank1.0.txt", n_train=train_num)
    data_time = time.time() - data_time
    learning_time = time.time()
    nb = MergedNB(whether_continuous=whether_continuous)
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print(
        "Data cleaning   : {:12.6} s\n"
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            data_time, learning_time, estimation_time,
            data_time + learning_time + estimation_time
        )
    )
