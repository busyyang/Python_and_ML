import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class NaiveBayes:

    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__(**kwargs)
        self._x = self._y = self._data = None
        self._n_possibilities = self._p_category = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dict = self._feat_dicts = None

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, '_' + item)

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
        self._fit(lb)

    def _fit(self, lb):
        pass

    def _func(self, x, i):
        pass

    def predict(self, x, get_raw_result=False, **kwargs):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = [xx[:] for xx in x]
        x = self._transfer_x(x)
        m_arg, m_probability = np.zeros(len(x), dtype=np.int8), np.zeros(len(x))
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            mask = p > m_probability
            m_arg[mask], m_probability[mask] = i, p[mask]
        if not get_raw_result:
            return np.array([self.num_to_label_dict[arg] for arg in m_arg])
        return m_probability

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        print('Acc={:12.6} %'.format(100 * np.sum(y_pred == y) / len(y)))

    def _transfer_x(self, x):
        return x


class DataUtil:

    def get_dataset(name, path, n_train=None, tar_idx=None, shuffle=True):
        x = []
        with open(path, "r", encoding="utf8") as file:
            if "balloon" in name or 'mushroom' in name:
                for sample in file:
                    x.append(sample.strip().split(","))
        if shuffle:
            np.random.shuffle(x)
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        x = np.asarray(x)
        if n_train is None:
            return x, y
        return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


class MultinomialNB(NaiveBayes):

    def feed_data(self, x, y, sample_weight=None):
        if isinstance(x, list):
            features = list(map(list, zip(*x)))
        else:
            features = x.T
        features = [set(feat) for feat in features]
        feat_dicts = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dict = {_l: i for i, _l in enumerate(set(y))}
        num_to_label_dict = {i: _l for i, _l in enumerate(set(y))}

        x = np.array([[feat_dicts[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dict[yy] for yy in y])

        cat_counter = np.bincount(y)
        n_possibilities = [len(feats) for feats in features]

        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[ci].T for ci in labels]

        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dicts, self._n_possibilities = cat_counter, feat_dicts, n_possibilities
        self.label_dict = label_dict
        self.num_to_label_dict = num_to_label_dict
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=p) for xx in self._labelled_x])
            else:
                self._con_counter.append([
                    np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=p)
                    for label, xx in self._label_zip])

    def _fit(self, lb=1):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb)

        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                 for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

    def _func(self, x, i):
        x = np.atleast_2d(x).T
        rs = np.ones(x.shape[1])
        for d, xx in enumerate(x):
            rs *= self._data[d][i][xx]
        return rs * self._p_category[i]

    def _transfer_x(self, x):
        for i, sample in enumerate(x):
            for j, char in enumerate(sample):
                x[i][j] = self._feat_dicts[j][char]
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
                plt.bar(tmp_x - 0.35 * c, self._data[j][c, :], width=0.35, edgecolor="white",
                        label=u"class: {}".format(self.num_to_label_dict[c]))
            plt.xticks([i for i in range(sj + 2)], [""] + [rev_dict[i] for i in range(sj)] + [""])
            plt.ylim(0, 1.0)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))


def run_balloon():
    import time

    for dateset in ('balloon1.0.txt', 'balloon1.5.txt'):
        print(
            "===============================\n"
            "{}\n"
            "-------------------------------\n".format(dateset), end='')
        _x, _y = DataUtil.get_dataset(dateset, 'data/{}'.format(dateset))
        learning_time = time.time()
        nb = MultinomialNB()
        nb.fit(_x, _y)
        learning_time = time.time() - learning_time
        estimation_time = time.time()
        nb.evaluate(_x, _y)
        estimation_time = time.time() - estimation_time
        print(
            "Model building  : {:12.6} s\n"
            "Estimation      : {:12.6} s\n"
            "Total           : {:12.6} s".format(
                learning_time, estimation_time,
                learning_time + estimation_time
            )
        )
        # nb.show_timing_log()
        nb.visualize()


def run_mushroom():
    import time

    dateset = 'mushroom.txt'
    print(
        "===============================\n"
        "{}\n"
        "-------------------------------\n".format(dateset), end='\t')
    (_x, _y), (_x_val, _y_val) = DataUtil.get_dataset(dateset, 'data/{}'.format(dateset), tar_idx=0, n_train=7000)
    learning_time = time.time()
    nb = MultinomialNB()
    nb.fit(_x, _y)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(_x, _y)
    nb.evaluate(_x_val, _y_val)
    estimation_time = time.time() - estimation_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    # nb.show_timing_log()
    nb.visualize()


if __name__ == '__main__':
    run_mushroom()
