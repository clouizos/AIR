import numpy as np
import matplotlib.pyplot as plt

# scipt for visualizing the bar graphs

x = [0.293, 0.437, 0.493, 0.578]
y = [0.828, 0.640, 0.553, 0.163]
z = [0.639, 0.514, 0.440, 0.121]
t = [0.673, 0.543, 0.466, 0.127]
labels = ['', '1', '', '3', '', '5', '', '10', '']

wh = np.arange(4)
ax = plt.subplot(111)
ax.bar(wh-0.2, x, width=0.2, color='b', align='center', label='LambdaMART')
ax.bar(wh, z, width=0.2, color='r', align='center', label='DCIGMM')
ax.bar(wh+0.2, y, width=0.2, color='g', align='center', label='DCIGMM KernelPCA')
ax.bar(wh+0.4, t, width=0.2, color='m', align='center', label='DCIGMM Sparse')
ax.set_xticklabels(labels, fontdict=None, minor=False)
plt.title('NDCG scores')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

fig = plt.gcf()
fig.savefig('res.png', dpi=100)
plt.show()
