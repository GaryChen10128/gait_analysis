import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
 1.參數設定
"""
xmin, xmax, A, N = 0, 4*np.pi, 4, 100
x = np.linspace(xmin, xmax, N)
y = A*np.sin(x)

"""
 2.繪圖
"""
fig = plt.figure(figsize=(7, 6), dpi=100)
ax = fig.gca()
line, = ax.plot(x, y, color='blue', linestyle='-', linewidth=3)
dot, = ax.plot([], [], color='red', marker='o', markersize=10, markeredgecolor='black', linestyle='')
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)

def update(i):
    dot.set_data(x[i], y[i])
    return dot,

def init():
    dot.set_data(x[0], y[0])
    return dot,

ani = animation.FuncAnimation(fig=fig, func=update, frames=N, init_func=init, interval=1000/N, blit=True, repeat=True)
plt.show()
ani.save('MovingPointMatplotlib.gif', writer='imagemagick', fps=1/0.04)