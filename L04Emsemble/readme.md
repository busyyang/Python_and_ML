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