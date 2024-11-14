import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from matplotlib.collections import LineCollection
from scipy.cluster.vq import whiten
from sklearn.decomposition import PCA

import cond_color

data = np.load('data/psths.npz')
X, times = data['X'], data['times'] # N x C x T and T x 1
N, C, T = X.shape[0], X.shape[1], X.shape[2]

# Trial average spike counts have been divided by the bin width, which is 10 ms
# So, X[i, c, t] is the average firing rate of neuron i, in t'th time bin, in condition c
# Units are Hertz or spikes per second
print(f'Average firing rate of neuron 0, in 0th time bin, for condition 0: {round(X[0][0][0], 2)} Hz')

# Start (in milliseconds) of different time bins
# print(times)


####################################################################################################
# Create a heat map for a few different neurons (1)

num_neurons = 20
c = 100
data_subset = X[:num_neurons, :, c]

# Create the heatmap
plt.figure(figsize=(10, 3))
sns.heatmap(data_subset, yticklabels=range(1, num_neurons+1),
            cmap='viridis', cbar=True)

# Customize the plot
plt.xticks(ticks=np.arange(0, T, 10), labels=times[::10])
plt.xlabel('Time Bins')
plt.ylabel(f'Neurons (1-{num_neurons})')
plt.title(f'PSTH for Condition c = {c}')

plt.savefig(f'outputs/plotting_raw_psths_condition_{c}.pdf', format='pdf', dpi=300, bbox_inches='tight')

####################################################################################################
# Population average across neuron and condition (1)

avg_across_neurons_conditions = np.mean(X, axis=(0, 1))
data_subset = avg_across_neurons_conditions.reshape(1, T)

plt.figure(figsize=(10, 0.5))
sns.heatmap(data_subset, cmap='viridis', yticklabels=False, cbar=True)

plt.xticks(ticks=np.arange(0, T, 10), labels=times[::10])
plt.xlabel('Time Bins')
plt.title('Population average firing rate')

plt.savefig(f'outputs/population_average_firing_rate.pdf', format='pdf', dpi=300, bbox_inches='tight')

####################################################################################################
# Plotting maximum firing rates of neurons across time and conditions (2a)

X_max = np.max(X, axis=(1, 2)) # Maximum across conditions and time

plt.figure(figsize=(10, 3))

sns.histplot(X_max, edgecolor='black', linewidth=0.5, color='skyblue')
plt.xlabel('Maximum firing rate (Hz)')
plt.ylabel('Neuron count')
plt.title('Neurons\' maximum firing rates across time and condition')

plt.savefig(f'outputs/neurons_maximum_firing_rates.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Question is "why normalize?" We don't want certain principal components to be selected simply because their
# variances are on a different scale and thus appear to explain more of the data. If their variances are scaled to one,
# then larger covariances will correspond to the patterns exhibited in more of the data.

####################################################################################################
# Normalization (2a)

X_min = np.min(X, axis = (1, 2)) # Minimum across conditions and time

X_min = X_min[:, np.newaxis, np.newaxis]
X_max = X_max[:, np.newaxis, np.newaxis]

X = (X - X_min) / (X_max - X_min + 5)

####################################################################################################
# Mean centering (2b)

X_avg_condition = np.average(X, axis=1)
X_avg_condition = X_avg_condition[:, np.newaxis, :]

X = X - X_avg_condition

assert np.all(np.average(X, axis=1) < 0.0001), "Average across conditions exceeds the threshold of 0.0001"

####################################################################################################
# PCA (2c)

# Subset X on −150ms to +300ms
mask = (times >= -150) & (times <= 300)
X = X[:, :, mask]
T = X.shape[2] # Now only 46 time bins for this reduced interval

assert X.shape == (N, C, T) # Ensure T redefined correctly

# Reshape to N x C*T
X = X.reshape(N, C * T)

# PCA from Equation 18 in Lecture Notes 2
S = (1/(C*T))*np.matmul(X, X.T) # Sample covariance matrix
assert S.shape == (N, N)

M = 12 # Number of principal components

eigenvalues, eigenvectors = np.linalg.eig(S)

# Sort the eigenvalues in descending order, and get the sorted indices
sorted_indices = np.argsort(eigenvalues)[::-1]

sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
V_M = sorted_eigenvectors[:, :M]

Z = np.matmul(V_M.T, X)

####################################################################################################
# Plotting PCA space trajectories (3)

Z = Z.reshape(M, C, T)

PC_1 = Z[0, :]
PC_2 = Z[1, :]

assert PC_1.shape == (C, T)

# Plot each condition as its own line and the time from PC_1 vs PC_2
# Use cond_color
fig, ax = plt.subplots(figsize=(10,4)) # Initialize figures and axes

for c in range(0, C):
    xs = PC_1[c, :]
    ys = PC_2[c, :]
    colors = cond_color.get_colors(xs, ys)
    print(colors[0])
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

plt.title('Plot of trajectories in PC1-PC2 plane')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.savefig(f'outputs/pca_dim_1_dim_2.pdf', format='pdf', dpi=300, bbox_inches='tight')

