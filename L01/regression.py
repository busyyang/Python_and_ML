import numpy as np
import matplotlib.pyplot as plt

x, y = [], []
for sample in open('data.txt', 'r'):
    _x, _y = sample.split(',')
    x.append(float(_x))
    y.append(float(_y))

x, y = np.array(x), np.array(y)
x = (x - np.mean(x)) / np.std(x)

plt.scatter(x, y, c='g', s=20)
# plt.savefig('images/visualization.png')
# plt.show()

x0 = np.linspace(-2, 4, 100)


def get_model(deg):
    return lambda input_x0=x0: np.polyval(np.polyfit(x, y, deg), input_x0)


def loss(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()


test_set = (1, 4, 10)
for d in test_set:
    plt.plot(x0, get_model(d)(), label='degree={}'.format(d))
    print(loss(d, x, y))
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
plt.legend()
plt.savefig('images/cf.png')
plt.show()
