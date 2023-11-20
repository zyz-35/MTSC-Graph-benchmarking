from matplotlib import pyplot as plt
import numpy as np

# fig, ax = plt.subplots()
# colors = ["r", "g", "b"]
# names = ["red", "green", "blue"]
# x = [25, 50, 75]
# y = [25, 50, 75]
# ax.plot([0, 100], [0, 100], c="black")
# for i in range(len(colors)):
#     ax.scatter(x[i], y[i], c=colors[i], label=names[i])
#     ax.annotate(names[i], (x[i]+1, y[i]-1), xytext=(x[i]+10, y[i]-10),
#                 c=colors[i], fontsize=12,
#                 arrowprops=dict(edgecolor=colors[i], arrowstyle="->", head_width=10))
# ax.text(10, 80, 'Hello World', fontsize=12)
# # ax.legend()
# ax.margins(0)
# plt.show()
import matplotlib.pyplot as plt

plt.annotate('',
             xy=(0.8, 0.5),
             xytext=(0.2, 0.5),
             arrowprops=dict(arrowstyle='->,head_width=0.6,head_length=0.6', linewidth=1.5, mutation_scale=20))

plt.show()