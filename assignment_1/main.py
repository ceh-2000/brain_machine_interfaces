import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

import cond_color
import utils

data = np.load('data/psths.npz')
X, times = data['X'], data['times']  # N x C x T and T x 1
N, C, T = X.shape[0], X.shape[1], X.shape[2]
print(f'N: {N}, C: {C}, T: {T}')

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
sns.heatmap(data_subset, yticklabels=range(1, num_neurons + 1),
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

X_max = np.max(X, axis=(1, 2))  # Maximum across conditions and time

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

X_min = np.min(X, axis=(1, 2))  # Minimum across conditions and time

X_min = X_min[:, np.newaxis, np.newaxis]
X_max = X_max[:, np.newaxis, np.newaxis]

X = (X - X_min) / (X_max - X_min + 5)

####################################################################################################
# Mean centering (2b)

X_avg_condition = np.average(X, axis=1)
X_avg_condition = X_avg_condition[:, np.newaxis, :]

X = X - X_avg_condition

assert np.all(np.average(X, axis=1) < 0.0001), 'Average across conditions exceeds the threshold of 0.0001.'

####################################################################################################
# PCA (2c)

# Subset X on −150ms to +300ms
mask = (times >= -150) & (times <= 300)
X = X[:, :, mask]
T = X.shape[2]  # Now only 46 time bins for this reduced interval

assert X.shape == (N, C, T), 'Shape of X is wrong.'  # Ensure T redefined correctly

# Reshape to N x C*T
X = X.reshape(N, C * T)

# PCA from Equation 18 in Lecture Notes 2
S = (1 / (C * T)) * np.matmul(X, X.T)  # Sample covariance matrix
assert S.shape == (N, N)

M = 12  # Number of principal components

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
fig, ax = utils.plot_2D_trajectories(PC_1, PC_2, C)

plt.title('Plot of trajectories in PC1-PC2 plane')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.savefig(f'outputs/pca_dim_1_dim_2.pdf', format='pdf', dpi=300, bbox_inches='tight')

####################################################################################################
# Computing log-likelihood of a linear model (4a, 4b, 4c)

# Computing A in terms of beta and H
test_M = 4
A = np.array([
    [0, 2, 3, 4],
    [-2, 0, 7, 8],
    [-3, -7, 0, 12],
    [-4, -8, -12, 0]
], dtype=np.float64)


def compute_K(M):
    return int((M - 1) * (M) / 2)

test_K = compute_K(test_M)

def construct_H(M, K):
    H = np.zeros(shape=(K, M, M))

    a = 0
    i = 0
    start_j = 1

    while a < K:
        j = start_j
        while j < M:
            H[a, i, j] = 1
            H[a, j, i] = -1
            j += 1
            a += 1
        i += 1
        start_j += 1

    return H

# Test our new function
beta_test = np.array([2, 3, 4, 7, 8, 12], dtype=np.float64)

H = construct_H(test_M, test_K)
new_A = np.tensordot(beta_test, H, axes=1)
assert np.all(A == new_A), 'Equation 3 not satisfied.'

def construct_W(H, Z):
    return np.tensordot(H, Z, axes=([2], [0]))

# Now find A from example Z data
def compute_A(full_Z):
    M, C, T = full_Z.shape
    Z_T = full_Z[:, :, :T-1]
    Z_T_1 = full_Z[:, :, 1:T]
    delta_Z = Z_T_1 - Z_T

    assert delta_Z.shape == (M, C, T-1), 'Delta Z not of shape M x C x (T-1)'

    # Now reshape Z_T and delta_Z as 2D arrays
    Z_T = Z_T.reshape(M, C*(T-1))
    delta_Z = delta_Z.reshape(M, C*(T-1))

    # Get the intermediate values and matrices
    K = compute_K(M) # Number of elements above diagonal
    H = construct_H(M, K) # Use to construct matrices A and W
    W = construct_W(H, Z_T) # Use to compute beta in final equation

    # Formula: beta = (delta_Z)(W_T)(W W_T)^-1
    W_W_T = np.tensordot(W, W.T, axes=([2, 1], [0, 1])) # Specify which axes to contract together
    inverse_W_W_T = np.linalg.inv(W_W_T)
    W_T_inverse_W_W_T = np.tensordot(W.T, inverse_W_W_T, axes=([2], [0]))
    beta = np.tensordot(delta_Z, W_T_inverse_W_W_T, axes=([1, 0], [0, 1]))

    assert beta.shape == (K,), f'beta is the wrong shape. Should be a vector of size {K}'

    A = np.tensordot(beta, H, axes=1)

    assert A.shape == (M, M), f'A should have shape {M}x{M}.'
    assert np.allclose(A, -A.T), 'A is not antisymmetric.'

    return A

A = compute_A(Z)

####################################################################################################
# Color plot of A (4d)

# Plot A
plt.figure(figsize=(5, 5))
plt.imshow(A, cmap='viridis')
plt.colorbar()
plt.title('Color plot of A')
plt.savefig(f'outputs/color_plot_of_A.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.figure(figsize=(5, 5))
plt.imshow(np.abs(A), cmap='viridis')
plt.colorbar()
plt.title('Absolute value color plot of A')
plt.savefig(f'outputs/absolute_value_color_plot_of_A.pdf', format='pdf', dpi=300, bbox_inches='tight')

####################################################################################################
# Test A calculation (4e)

test_data = np.load('data/test.npz')
Z_test, A_test = test_data['Z_test'], test_data['A_test']

A_estimate = compute_A(Z_test)
assert np.allclose(A_estimate, A_test), 'Our estimate of A and A_test do not match.'

####################################################################################################
# Find complex eigenvectors and imaginary eigenvalues of A (5a)

eigenvalues, eigenvectors = np.linalg.eig(A)

assert np.allclose(eigenvalues[0::2], np.conj(eigenvalues[1::2])), 'Adjacent eigenvalues are not complex conjugates of each other.'

####################################################################################################
# Project Z in plane with fastest rotation (P_FR) (5b)

P_FR, P_FR_project = utils.compute_FP_proj(eigenvalues, eigenvectors, M, Z, 0)

####################################################################################################
# Plot 2D trajectories in fastest plane (5c)

fig, ax = utils.plot_2D_trajectories(P_FR_project[0], P_FR_project[1], C, 10)

plt.title('Plot of trajectories projected into fasted 2D plane')
ax.set_xlabel('P_FR_1')
ax.set_ylabel('P_FR_2')

plt.savefig(f'outputs/traj_projected_fastest_2D_plane.pdf', format='pdf', dpi=300, bbox_inches='tight')

####################################################################################################
# Plot 2D trajectories in second and third fastest planes (5d)

# Second fastest
_, P_project_2nd = utils.compute_FP_proj(eigenvalues, eigenvectors, M, Z, 1)

fig, ax = utils.plot_2D_trajectories(P_project_2nd[0], P_project_2nd[1], C, 10)

plt.title('Plot of trajectories projected into second fastest 2D plane')
ax.set_xlabel('P_1')
ax.set_ylabel('P_2')

plt.savefig(f'outputs/traj_projected_second_fastest_2D_plane.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Third fastest
_, P_project_3rd = utils.compute_FP_proj(eigenvalues, eigenvectors, M, Z, 2)

fig, ax = utils.plot_2D_trajectories(P_project_3rd[0], P_project_3rd[1], C, 10)

plt.title('Plot of trajectories projected into third fastest 2D plane')
ax.set_xlabel('P_1')
ax.set_ylabel('P_2')

plt.savefig(f'outputs/traj_projected_third_fastest_2D_plane.pdf', format='pdf', dpi=300, bbox_inches='tight')

####################################################################################################
# Apply the projections obtained for the interval [-150ms, 300ms] to the interval [-800ms, -150ms] (6)

P_FR_V_M_T = np.matmul(P_FR, V_M.T)

assert P_FR_V_M_T.shape == (2, N), f'Shape of 2 x N projection matrix is {P_FR_V_M_T.shape}, and should be {(2, N)}'

# Get a fresh version of X
X = data['X']

# Subset X on -800 ms to -150 ms
mask = (times >= -800) & (times <= -150)
X = X[:, :, mask]
T = X.shape[2]  # Now only 66 time bins for this reduced interval

assert X.shape == (N, C, T), 'Shape of X is wrong.'  # Ensure T redefined correctly

print(P_FR_V_M_T.shape, X.shape)

# TODO: Project X with P_FR_V_M_T