import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

####################################################################################################
# Normalization

X_min = np.min(X, axis=(1,2)) # Minimum across conditions and time
X_psth = np.min(X, axis=1) # Average across conditions only

print(X_psth.shape)

