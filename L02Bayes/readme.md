<font size=6>贝叶斯分类器</font>

内容与代码来源自：Python与机器学习实战（何宇健）作者git [repo](https://github.com/carefree0910/MachineLearning)
我的git [repo](https://github.com/busyyang/Python_and_ML)
# 理论
贝叶斯决策理论和传统统计学理论有区别，最大的区别在于对于概率的定义：
 - 统计学的频率强调“自然属性”，认为应该使用事件在重复试验中发生的频率最为其发生的概率的估计；
 - 贝叶斯决策理论的频率不强调客观随机性，认为仅仅是观察者不知道时间发生的结果。对于“知情者”来说，该事件其实不具备随机性。

不管是哪种说法，最大的问题都是如何从已知的样本中获取信息并估计目标模型的参数，目前比较有名的就是"频率近似概率"的方法。下面表述了两种参数估计方法，在本质上也归结于它。
## 极大似然估计

对于参数$\theta$的情况下，$x_n$时间发生的概率为$p(x_n|\theta)$,那么可以得到
$$p(\overline{X}|\theta)=p(x_1|\theta)p(x_2|\theta)...p(x_n|\theta)\tag{1}$$其中，$\overline{X}=[x_1,x_2,...,x_n]$。
如果想要$\overline{X}$这个事件发生的概率最大，也就是要找到一个参数$\widetilde{\theta}$使得
$$\widetilde{\theta}=arg\underset{\theta}{max}p(\overline{X}|\theta)=arg\underset{\theta}{max}\prod_{i=1}^Np(x_i\vert \theta)\tag{2}$$极大似然估计可以看做一个带估参数未知但是固定的量，是传统统计学的做法。
## 极大后验概率估计
核心数学公式：
$$p(\overline{X}|\theta)p(\theta)=p(\theta,\overline{X})=p(\theta|\overline{X})p(\overline{X})\tag{3}$$直观理解就是：样本$\overline{X}$在参数$\theta$下的概率乘以参数$\theta$的概率=样本与参数的联合概率$p(\theta,\overline{X})$=参数$\theta$在样本$\overline{X}$下的概率乘以样本本身的概率。考虑到独立性假设，可以得知样本$\overline{X}$的概率就是它组成部分$X_i$的概率的炼乘积:
$$p(\overline{X}|\theta)=\prod_{i=1}^Np(X_i|\theta)\tag{4}$$ $$p(\overline{X})=\prod_{i=1}^N[\int_{\Theta}p(X_i|\theta)p(\theta)d\theta]\tag{5}$$

# 朴素贝叶斯
朴素贝叶斯的“朴素”二字对应的是“独立性假设”。也就是样本空间中的样本是相互独立的。常用的朴素贝叶斯有：
 - 离散朴素贝叶斯(MultinomialNB):所有维度特征都是离散随机变量；
 - 连续朴素贝叶斯(GaussianNB):所有维度特征都是连续随机变量；
 - 混合朴素贝叶斯(MergedNB):既有离散的也有联系的随机变量。

## 离散朴素贝叶斯

### 数学基础
首先朴素贝叶斯的模型参数即类别的选择空间：
$$\Theta=\{y=c_1,y=c_2,...,y=c_k\}\tag{6}$$贝叶斯的总的参数空间$\overline{\Theta}$包含了先验概率$p(\theta_k)=p(y=c_k)$，样本空间在模型参数下的条件概率$p(X|\theta_k)=p(X|y=c_k)$和样本空间本身的概率$p(X)$。由于我们选择样本空间的子集$\overline{X}$作为训练集，所以$p(X)=p(\overline{X})$为常数，所以我们只关心模型空间的先验概率和样本空间在模型参数下的条件概率：
$$\overline{\Theta}=\{p(\theta),p(X|\theta):\theta \in \Theta\}\tag{7}$$模型的决策就是让后验概率最大：
$$\delta(\overline{X})=\overline{\theta}=arg\underset{\overline{\theta}\in \overline{\Theta}}{max}p(\overline{\theta}|\overline{X})\tag{8}$$在确定$\overline{\theta}$后，模型的决策可以写成：
$$f(x^*)=argmax\overline{p}(c_k|X=x^*)=arg\underset{c_k}{max}\overline{p}(X^{(j)}=x^*|y=c_k)\tag{9}$$在离散朴素贝叶斯中的损失函数为：
$$L(\theta,\delta(\overline{X}))=\sum_{i=1}^N\overline{L}(y_i,f(x_i))=\sum_{i=1}^NI(y_i\neq(f(x_i)))\tag{10}$$其中：
$$I(y_i\neq(f(x_i))=\left\{\begin{array}{l}1,如果y_i\neq f(x_i)\\0, 如果y_i= f(x_i)\end{array}\right.\tag{11}$$离散朴素贝叶斯的方法为：
 - 有数据集$D=\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$;
 - 计算先验概率$p(y=c_k)$的极大似然估计
  $$\overline{p}(y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)}{N},k=1,2,...,K\tag{12}$$
 - 计算条件概率$p(X^{(j)}=a_{jl}|y=c_k)$的极大似然估计(假设每一个单独输入的$n$维向量$x_i$的第$j$维的特征$x^{(j)}$可能取值的集合为$\{a_{j1},...,a_{js_j}\}$)
  $$p(X^{(j)}=a_{jl}|y=c_k)=\frac{\sum {I(x_i^{(j)}=a_{jl},y_i=c_k})}{\sum I(y_i=c_k)}\tag{13}$$
 - 输出（利用MAP估计）：
  $$y=f(x^*)=arg\underset{c_k}{max}\overline{p}(y=c_k)\prod_{j=1}^N\overline{p}(X^{(j)}=x^{*(j)}|y=c_k)\tag{14}$$

在训练时候，由于是使用的事件的一个子集作为训练集的，可能数据集选择有问题（不均衡），导致一些先验概率不存在，发生错误，所以又加入了一个平滑项：
 - 计算先验概率$p_{\lambda}(y=c_k)$
  $$p_{\lambda}(y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)+\lambda}{N+K\lambda},k=1,2,...,K\tag{15}$$
 - 计算条件概率$p_{\lambda}(X^{(j)}=a_{jl}|y=c_k)$：
  $$p_{\lambda}(X^{(j)}=a_{jl}|y=c_k)=\frac{\sum {I(x_i^{(j)}=a_{jl},y_i=c_k})+\lambda}{\sum I(y_i=c_k)+S_j\lambda}\tag{16}$$
  
当$\lambda=0$时候，就是极大似然估计，当$\lambda=1$时，就是拉普拉斯平滑。
**将公式15和公式16带入公式14就是使用了平滑的贝叶斯估计概率。**做一个二分类问题的话，那就看哪个的概率更大。
### 代码实现
以下是实现代码。完整的脚本为：
~~~py
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
    # run_balloon()
    run_mushroom()

~~~

## 连续朴素贝叶斯
回忆离散朴素贝叶斯，无非就是计算先验概率$p_{\lambda}(y=c_k)$与条件概率$p(X^{(j)}=a_{jl}|y=c_k)$,如果是连续变量，假设变量服从正态分布(高斯分布)，用极大似然估计来计算条件概率。
连续变量的条件概率：
$$p(X^{(j)}=a_{jl}|y=c_k)=\frac{1}{\sqrt{2\pi}\sigma_{jk}}e^{-\frac{(a_{jl}-\mu_{jk})^2}{2\sigma_{jk}^2}}\tag{17}$$这里面又2个参数$\mu_{jk}$和$\sigma_{jk}$,用极大似然估计法得到：
$$\widetilde{\mu}_{jk}=\frac{1}{N_k}\sum_{i=1}^Nx_i^{(j)}I(y_i=c_k)\tag{18}$$$$\widetilde{\sigma}^2_{jk}=\frac{1}{N_k}(x_i^{(j)}-\mu_{jk})^2I(y_i=c_k)\tag{19}$$其中，$N_k=\sum_{i=1}^NI(y_i=c_k)$是类别$c_k$的样本数。
再离散朴素贝叶斯的基础上还需要定义一个计算高斯分布的极大似然估计的类：
~~~py
import numpy as np
from math import pi

sqrt_pi = (2 * pi) ** 0.5

class NBFunctions:
    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

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
~~~
`@staticmethod`表示静态方法，类似C++中的static关键字。对于GaussianNB来说，只有条件概率的相关计算改变了，也就是只需要改变这部分代码就可以了。
~~~py
import numpy as np
from math import pi
import matplotlib.pyplot as plt

sqrt_pi = (2 * pi) ** 0.5

class NBFunctions:
    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

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
            tmp_x = np.linspace(x_min-0.1*gap, x_max+0.1*gap, 200)
            title = "$j = {}$".format(j + 1)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.plot(tmp_x, [self._data[j][c](xx) for xx in tmp_x],
                         c=colors[self.label_dict[c]], label="class: {}".format(self.label_dict[c]))
            plt.xlim(x_min-0.2*gap, x_max+0.2*gap)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))


def run_mushroom():
    import time

    dateset = 'data.txt'
    print(
        "===============================\n"
        "{}\n"
        "-------------------------------\n".format(dateset), end='\t')
    (_x, _y), (_x_val, _y_val) = DataUtil.get_dataset(dateset, 'data/{}'.format(dateset), tar_idx=0, n_train=7000)
    learning_time = time.time()
    nb = GaussianNB()
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
~~~
这里只要有了数值化的数据集就可以了。可以发现建模速度比较快，因为建模比较简单，但是预测的速度比较慢了，由于预测时候需要计算大量正态分布密度。
## 混合朴素贝叶斯
混合朴素贝叶斯完全可以将离散与连续的计算结合起来：
$$y=f(x^*)=arg\underset{c_k}{max}p(y=c_k)\prod_{j \in S_1}p(X^{(j)}=x^{*(j)}|y=c_k)\prod_{j \in S_2}p(X^{(j)}=x^{*(j)}|y=c_k)$$其中$S_1,S_2$表示离散和连续维度的集合，条件概率由公式16和17得到。理论基础已经差不多了，但是代码实现，用书上代码的耦合度比较高，还没有完全分类理解，先贴再这里,有时间了再梳理一下：
~~~py
# MergedNB.py
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

~~~
还有一个工具文件：
~~~py
# Bases.py
import io
import cv2
import time
import math
import ctypes
import multiprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D


class ModelBase:
    """
        Base for all models
        Magic methods:
            1) __str__     : return self.name; __repr__ = __str__
            2) __getitem__ : access to protected members
        Properties:
            1) name  : name of this model, self.__class__.__name__ or self._name
            2) title : used in matplotlib (plt.title())
        Static method:
            1) disable_timing  : disable Timing()
            2) show_timing_log : show Timing() records
    """

    def __init__(self, **kwargs):
        self._plot_label_dict = {}
        self._title = self._name = None
        self._metrics, self._available_metrics = [], {
            "acc": ClassifierBase.acc
        }
        self._params = {
            "sample_weight": kwargs.get("sample_weight", None)
        }

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name

    @property
    def title(self):
        return str(self) if self._title is None else self._title


    # Handle animation

    @staticmethod
    def _refresh_animation_params(animation_params):
        animation_params["show"] = animation_params.get("show", False)
        animation_params["mp4"] = animation_params.get("mp4", False)
        animation_params["period"] = animation_params.get("period", 1)

    def _get_animation_params(self, animation_params):
        if animation_params is None:
            animation_params = self._params["animation_params"]
        else:
            ClassifierBase._refresh_animation_params(animation_params)
        show, mp4, period = animation_params["show"], animation_params["mp4"], animation_params["period"]
        return show or mp4, show, mp4, period, animation_params

    def _handle_animation(self, i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period,
                          name=None, img=None):
        if draw_ani and x.shape[1] == 2 and (i + 1) % ani_period == 0:
            if img is None:
                img = self.get_2d_plot(x, y, **animation_params)
            if name is None:
                name = str(self)
            if show_ani:
                cv2.imshow(name, img)
                cv2.waitKey(1)
            if make_mp4:
                ims.append(img)


    def get_2d_plot(self, x, y, padding=1, dense=200, draw_background=False, emphasize=None, extra=None, **kwargs):
        pass

    # Visualization

    def scatter2d(self, x, y, padding=0.5, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        indices = [labels == i for i in range(np.max(labels) + 1)]
        scatters = []
        plt.figure()
        plt.title(title)
        for idx in indices:
            scatters.append(plt.scatter(axis[0][idx], axis[1][idx], c=colors[idx]))
        plt.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                   ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def scatter3d(self, x, y, padding=0.1, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        labels, n_label = transform_arr(labels)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]
        indices = [labels == i for i in range(n_label)]
        scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in indices:
            scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                  ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.show()

    # Util

    def predict(self, x, get_raw_results=False, **kwargs):
        pass


class ClassifierBase(ModelBase):
    """
        Base for classifiers
        Static methods:
            1) acc, f1_score           : Metrics
            2) _multi_clf, _multi_data : Parallelization
    """

    def __init__(self, **kwargs):
        super(ClassifierBase, self).__init__(**kwargs)
        self._params["animation_params"] = kwargs.get("animation_params", {})
        ClassifierBase._refresh_animation_params(self._params["animation_params"])

    # Metrics

    @staticmethod
    def acc(y, y_pred, weights=None):
        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if weights is not None:
            return np.average((y == y_pred) * weights)
        return np.average(y == y_pred)

    # noinspection PyTypeChecker
    @staticmethod
    def f1_score(y, y_pred):
        tp = np.sum(y * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y) * y_pred)
        fn = np.sum(y * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    # Parallelization

    # noinspection PyUnusedLocal
    @staticmethod
    def _multi_clf(x, clfs, task, kwargs, stack=np.vstack, target="single"):
        if target != "parallel":
            return np.array([clf.predict(x) for clf in clfs], dtype=np.float32).T
        n_cores = kwargs.get("n_cores", 2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = np.array([clf.predict(x, n_cores=1) for clf in clfs], dtype=np.float32).T
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(clfs) / n_cores)
            clfs = [clfs[i*batch_size:(i+1)*batch_size] for i in range(n_cores)]
            x_size = np.prod(x.shape)  # type: int
            shared_base = multiprocessing.Array(ctypes.c_float, int(x_size))
            shared_matrix = np.ctypeslib.as_array(shared_base.get_obj()).reshape(x.shape)
            shared_matrix[:] = x
            matrix = stack(
                pool.map(task, ((shared_matrix, clfs, n_cores) for clfs in clfs))
            ).T.astype(np.float32)
        return matrix

    # noinspection PyUnusedLocal
    def _multi_data(self, x, task, kwargs, stack=np.hstack, target="single"):
        if target != "parallel":
            return task((x, self, 1))
        n_cores = kwargs.get("n_cores", 2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = task((x, self, n_cores))
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(x) / n_cores)
            batch_base, batch_data, cursor = [], [], 0
            x_dim = x.shape[1]
            for i in range(n_cores):
                if i == n_cores - 1:
                    batch_data.append(x[cursor:])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, (len(x) - cursor) * x_dim))
                else:
                    batch_data.append(x[cursor:cursor + batch_size])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, batch_size * x_dim))
                cursor += batch_size
            shared_arrays = [
                np.ctypeslib.as_array(shared_base.get_obj()).reshape(-1, x_dim)
                for shared_base in batch_base
            ]
            for i, data in enumerate(batch_data):
                shared_arrays[i][:] = data
            matrix = stack(
                pool.map(task, ((x, self, n_cores) for x in shared_arrays))
            )
        return matrix.astype(np.float32)

    # Training

    @staticmethod
    def _get_train_repeat(x, batch_size):
        train_len = len(x)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        return 1 if not do_random_batch else int(train_len / batch_size) + 1

    def _batch_work(self, *args):
        pass

    def _batch_training(self, x, y, batch_size, train_repeat, *args):
        pass

    # Visualization

    def get_2d_plot(self, x, y, padding=1, dense=200, title=None,
                    draw_background=False, emphasize=None, extra=None, **kwargs):
        axis, labels = np.asarray(x).T, np.asarray(y)
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])  # type: float
        y_min, y_max = np.min(axis[1]), np.max(axis[1])  # type: float
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)
        z = self.predict(base_matrix).reshape((nx, ny))

        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        buffer_ = io.BytesIO()
        plt.figure()
        if title is None:
            title = self.title
        plt.title(title)
        if draw_background:
            xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")

        plt.savefig(buffer_, format="png")
        plt.close()
        buffer_.seek(0)
        image = Image.open(buffer_)
        canvas = np.asarray(image)[..., :3]
        buffer_.close()
        return canvas

    def visualize2d(self, x, y, padding=0.1, dense=200, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None, **kwargs):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = self.predict(base_matrix, **kwargs).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        if show_org:
            plt.figure()
            plt.scatter(axis[0], axis[1], c=colors)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()

        plt.figure()
        plt.title(title)
        if draw_background:
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def visualize3d(self, x, y, padding=0.1, dense=100, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None, **kwargs):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))

        def decision_function(xx):
            return self.predict(xx, **kwargs)

        nx, ny, nz, padding = dense, dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def get_base(_nx, _ny, _nz):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            _zf = np.linspace(z_min, z_max, _nz)
            n_xf, n_yf, n_zf = np.meshgrid(_xf, _yf, _zf)
            return _xf, _yf, _zf, np.c_[n_xf.ravel(), n_yf.ravel(), n_zf.ravel()]

        xf, yf, zf, base_matrix = get_base(nx, ny, nz)

        t = time.time()
        z_xyz = decision_function(base_matrix).reshape((nx, ny, nz))
        p_classes = decision_function(x).astype(np.int8)
        _, _, _, base_matrix = get_base(10, 10, 10)
        z_classes = decision_function(base_matrix).astype(np.int8)
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        z_xy = np.average(z_xyz, axis=2)
        z_yz = np.average(z_xyz, axis=1)
        z_xz = np.average(z_xyz, axis=0)

        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        yz_xf, yz_yf = np.meshgrid(yf, zf, sparse=True)
        xz_xf, xz_yf = np.meshgrid(xf, zf, sparse=True)

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        labels, n_label = transform_arr(labels)
        p_classes, _ = transform_arr(p_classes)
        z_classes, _ = transform_arr(z_classes)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])
        if extra is not None:
            ex0, ex1, ex2 = np.asarray(extra).T
        else:
            ex0 = ex1 = ex2 = None

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        if show_org:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(axis[0], axis[1], axis[2], c=colors[labels])
            plt.show()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        plt.title(title)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.set_title("Org")
        ax2.set_title("Pred")
        ax3.set_title("Boundary")

        ax1.scatter(axis[0], axis[1], axis[2], c=colors[labels])
        ax2.scatter(axis[0], axis[1], axis[2], c=colors[p_classes], s=15)
        if extra is not None:
            ax2.scatter(ex0, ex1, ex2, s=80, zorder=25, facecolors="red")
        xyz_xf, xyz_yf, xyz_zf = base_matrix[..., 0], base_matrix[..., 1], base_matrix[..., 2]
        ax3.scatter(xyz_xf, xyz_yf, xyz_zf, c=colors[z_classes], s=15)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        def _draw(_ax, _x, _xf, _y, _yf, _z):
            if draw_background:
                _ax.pcolormesh(_x, _y, _z > 0, cmap=plt.cm.Pastel1)
            else:
                _ax.contour(_xf, _yf, _z, c='k-', levels=[0])

        def _emphasize(_ax, axis0, axis1, _c):
            _ax.scatter(axis0, axis1, c=_c)
            if emphasize is not None:
                indices = np.array([False] * len(axis[0]))
                indices[np.asarray(emphasize)] = True
                _ax.scatter(axis0[indices], axis1[indices], s=80,
                            facecolors="None", zorder=10)

        def _extra(_ax, axis0, axis1, _c, _ex0, _ex1):
            _emphasize(_ax, axis0, axis1, _c)
            if extra is not None:
                _ax.scatter(_ex0, _ex1, s=80, zorder=25, facecolors="red")

        colors = colors[labels]

        ax1.set_title("xy figure")
        _draw(ax1, xy_xf, xf, xy_yf, yf, z_xy)
        _extra(ax1, axis[0], axis[1], colors, ex0, ex1)

        ax2.set_title("yz figure")
        _draw(ax2, yz_xf, yf, yz_yf, zf, z_yz)
        _extra(ax2, axis[1], axis[2], colors, ex1, ex2)

        ax3.set_title("xz figure")
        _draw(ax3, xz_xf, xf, xz_yf, zf, z_xz)
        _extra(ax3, axis[0], axis[2], colors, ex0, ex2)

        plt.show()

        print("Done.")

    # Util

    def get_metrics(self, metrics):
        if len(metrics) == 0:
            for metric in self._metrics:
                metrics.append(metric)
            return metrics
        for i in range(len(metrics) - 1, -1, -1):
            metric = metrics[i]
            if isinstance(metric, str):
                try:
                    metrics[i] = self._available_metrics[metric]
                except AttributeError:
                    metrics.pop(i)
        return metrics

    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x, **kwargs)
        y = np.asarray(y)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs
~~~
