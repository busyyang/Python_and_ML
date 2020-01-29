<font size=6>贝叶斯分类器</font>
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