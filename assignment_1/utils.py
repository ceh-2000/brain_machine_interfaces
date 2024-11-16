from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import cond_color

def plot_2D_trajectories(Xs, Ys, C, reduce_time=0):
    fig, ax = plt.subplots(figsize=(10, 4))  # Initialize figures and axes

    for c in range(0, C):
        if reduce_time > 0:
            xs = Xs[c, :-reduce_time]
            ys = Ys[c, :-reduce_time]
        else:
            xs = Xs[c, :]
            ys = Ys[c, :]

        colors = cond_color.get_colors(xs, ys)
        cond_color.plot_start(xs[0], ys[0], colors[0], markersize=200, ax=ax)
        cond_color.plot_end(xs[-1], ys[-1], colors[-1], markersize=50, ax=ax)

        # Create the segments for the line
        segments = []
        for i in range(len(xs) - 1):
            segment = [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])]
            segments.append(segment)

        # Create a LineCollection from the segments and assign colors
        lc = LineCollection(segments, colors=colors, linewidth=2)
        ax.add_collection(lc)

    return fig, ax