<font size=6>集成学习</font>
集成学习的思想在于：通过适当的方式集成多个“个体模型”，得到最终的模型，希望最终模型优于个体模型。
~~~mermaid
graph LR
A(个体模型G1) --> B(集成模块)
C(个体模型G2) --> B(集成模块)
D(个体模型Gn) --> B(集成模块)
B(集成模块) --> Z(最终模型)
~~~
所以问题变成了如何选择，生成个体模型，以及如何进行集成。有不同的设计思路：
 - 将不同的个体模型进行集成；
 - 将类型相同但参数不同的模型进行集成；
 - 将类型相同但训练集不同的模型进行集成。

第一种方式的应用并不十分广泛，第二种方式又被称为并行方式，代表算法为Bagging,Bagging中应用最广泛的是随机森林(Random Forest, RF)。第三种方式又叫串行方法，最具有代表性的是Boosting，(AdaBoost)。
# Bagging与随机森林
在了解Bagging之前要先了解Bootstrap理论，假设有样本集$X=\{x_1,...,x_N\}$:
 - 从$X$中随机抽取一个样本（抽出$x_1,..,x_N$的概率是一样的）；
 - 将该样本放入拷贝数据集$X_j$中；
 - 将样本放回$X$中。

重复$N$次使得$X_j$有N个样本。将上述过程重复M次，得到M个有N个元素的数据集$X_j,(j=1,..,M)$，其中$X_j=\{x_{j1},...,x_{jN}\}$。简单来说，就是一个有放回的随机抽样过程。

Bagging就是Bootstrap Aggregating，处理方法为：
 - 用Bootstrap生成$M$个数据集；
 - 用这$M$个数据集训练出$M$个弱分类器(个体模型)；
 - 最终模型就是这M个弱分类器的简单组合。
   - 对于分类问题，简单投票表决；
   - 对于回归问题，进行简单取平均。
  
**随机森林**不仅在训练集上使用Bootstrap算法，对每个节点生成算法时都会随机挑出一个可选特征空间的子空间作为决策树的可选特征空间。生成的模型不剪枝，保留原始形式。
 - 用Bootstrap生成M个训练集；
 - 用这M个训练集训练M棵不剪枝的决策树，且在每棵决策树生成的过程中，每次对节点划分时候，都从可选特征空间(一共d个)中随机选择k个特征($d\geqslant k$)，然后依据信息增益选择当前节点;
 - 最终模型就是这M个弱分类器的简单组合。

如果$k=d$，那么这样训练的决策树和一般决策树是一样的。如果$k=1$,生成决策树时候的每一步都是随机的。一般建议$k=\log_2d$。

### 随机森林的编程实现
由于随机森林可以看做是决策树的累加实现的，所以可以先直接把[决策树的代码](https://github.com/busyyang/Python_and_ML/tree/master/L03DT)拷贝过来，主要是Node.py, Tree.py,  Cluster.py。

~~~py
from Tree import *
from Bases import ClassifierBase
from Util import DataUtil
from ProgressBar import ProgressBar


def rf_task(args):
    x, trees, n_cores = args
    return [tree.predict(x, n_cores=n_cores) for tree in trees]

class RandomForest(ClassifierBase):
    cvd_trees = {
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree
    }

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__(**kwargs)
        self._tree, self._trees = "", []

        self._params["tree"] = kwargs.get("tree", "Cart")
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["feature_bound"] = kwargs.get("feature_bound", "log")

    @property
    def title(self):
        return "Tree: {}; Num: {}".format(self._tree, len(self._trees))

    @staticmethod
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)
        return u[np.argmax(c)]

    def fit(self, x, y, sample_weight=None, tree=None, epoch=None, feature_bound=None, **kwargs):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if tree is None:
            tree = self._params["tree"]
        if epoch is None:
            epoch = self._params["epoch"]
        if feature_bound is None:
            feature_bound = self._params["feature_bound"]
        x, y = np.atleast_2d(x), np.asarray(y)
        n_sample = len(y)
        self._tree = tree
        bar = ProgressBar(max_value=epoch, name="RF")
        for _ in range(epoch):
            tmp_tree = RandomForest.cvd_trees[tree](**kwargs)
            indices = np.random.randint(n_sample, size=n_sample)
            if sample_weight is None:
                local_weight = None
            else:
                local_weight = sample_weight[indices]
                local_weight /= local_weight.sum()
            tmp_tree.fit(x[indices], y[indices], sample_weight=local_weight, feature_bound=feature_bound)
            self._trees.append(deepcopy(tmp_tree))
            bar.update()

    def predict(self, x, get_raw_results=False, bound=None, **kwargs):
        trees = self._trees if bound is None else self._trees[:bound]
        matrix = self._multi_clf(x, trees, rf_task, kwargs, target=kwargs.get("target", "parallel"))
        return np.array([RandomForest.most_appearance(rs) for rs in matrix])

    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        kwargs["target"] = "single"
        super(RandomForest, self).evaluate(x, y, metrics, tar, prefix, **kwargs)
~~~
RandomForest继承自ClassifierBase，主要定义了一些常用的类的属性。[原作者的代码的Bases.py](https://github.com/carefree0910/MachineLearning/blob/master/Util/Bases.py)的ClassifierBase中，直接运行会报错，因为强行将将结果p和e这样的字符串转化为float32。可以直接将`dtype=np.float32`去掉。

~~~py
    @staticmethod
    def _multi_clf(x, clfs, task, kwargs, stack=np.vstack, target="single"):
        if target != "parallel":
          return np.array([clf.predict(x) for clf in clfs], dtype=np.float32).T #去掉, dtype=np.float32 
~~~
或者改为
~~~py
    @staticmethod
    def _multi_clf(x, clfs, task, kwargs, stack=np.vstack, target="single"):
        if target != "parallel":
            try:
                return np.array([clf.predict(x) for clf in clfs], dtype=np.float32).T
            except Exception:
                return np.array([clf.predict(x) for clf in clfs], dtype=np.str_).T
~~~
其他代码参考：git [repo:Python_and_ML:L04Emsemble](https://github.com/busyyang/Python_and_ML/tree/master/L04Emsemble)


# AdaBoost
AdaBoost要解决的问题是：
 - 如何根据弱模型的表现更新训练集的权重；
 - 如何根据弱模型的表现决定弱模型的话语权。

假设有一个二分类数据集：
$$D=\{(x_1,y_1),...,(x_n,y_n)\}$$每个样本都是由实例$x_i$和类别$y_i$组成，且：
$$x_i\in \bold{X} \subseteq \bold{R^N},y_i\in \bold{Y}\subseteq \{-1,+1\}$$AdaBoost会使用如下方法从训练集中训练一系列的弱分类器，并且集成为一个强分类器：
输入：训练数据(包含N条数据)，弱学习算法及对应的弱分类器，迭代次数M
1. 初始化训练数据的权值分布：$$W_0=(w_{01},...,w_{0N})$$
2. 对$k=0,1,...,M-1$:
   1. 使用权值分布为$W_k$的训练数据集训练弱分类器$$g_{k+1}(x):X\rightarrow {-1,+1}$$
   2. 计算$g_{k+1}(x)$在训练集上的加权错误率：$$e_{k+1}=\sum_{i=1}^Nw_{ki}I(g_{k+1}(x_i)\neq y_i)$$
   3. 根据加权错误率计算$g_{k+1}(x)$的话语权；$$\alpha _{k+1}=\frac{1}{2}\ln \frac{1-e_{k+1}}{e_{k+1}}$$
   4. 根据$g_{k+1}(x)$的表现更新训练集的权值分布，被$g_{k+1}(x)$误分的样本($y_ig_{k+1}(x_i)<0$的样本)要相对地（以$e^{\alpha_{k+1}}为比例$）增大其权重，反之则要（以$e^{-\alpha_{k+1}}为比例$）减少其权重；$$W_{k+1,i}=\frac{W_{ki}}{Z_k}·\exp{(-\alpha_{k+1}y_ig_{k+1}(x_i))}\\W_{k+1}=(W_{k+1,1},...,W_{k+1,N})$$这里的$Z_k$是规范化因子，
   $$Z_k=\sum_{i=1}^Nw_{ki}·\exp{(-\alpha_{k+1}y_ig_{k+1}(x_i))}$$
   作用是将$W_{k+1}$归一化为概率分布。
3. 加权集成：
   $$f(x)=\sum_{k=1}^N\alpha_kg_k(x)$$

输出：最终分类器$g(x)$：$$g(x)=sign(f(x))=sign(\sum_{k=1}^M\alpha_kg_k(x))$$

代码参考：git [repo:Python_and_ML:L04Emsemble](https://github.com/busyyang/Python_and_ML/tree/master/L04Emsemble)
