# Import necessary libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp  # GPU-accelerated computations
from scipy.spatial.distance import pdist, squareform
from scipy.stats import powerlaw
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Constants for particle assembly simulation
NUM_NEURONS = 1000
ASSEMBLY_TIME = 5.0    # seconds
DT_ASSEMBLY = 1e-4     # time step for assembly simulation
TEMPERATURE = 300      # Kelvin
BOLTZMANN_CONSTANT = 1.38e-23  # J/K
VISCOSITY = 0.001      # Pa·s (water at room temperature)
PARTICLE_RADIUS = 1e-6  # meters

# Acoustic force parameters
FREQUENCY = 1e6        # Hz
ACOUSTIC_PRESSURE = 1e5  # Pa

# Constants for neuronal simulation
SIM_TIME = 2.0         # seconds
DT_NEURON = 1e-5       # time step for neuronal simulation

# Hodgkin-Huxley model parameters
C_m = 1e-9             # Membrane capacitance (F)
g_Na = 120e-6          # Sodium conductance (S)
g_K = 36e-6            # Potassium conductance (S)
g_L = 0.3e-6           # Leak conductance (S)
E_Na = 50e-3           # Sodium reversal potential (V)
E_K = -77e-3           # Potassium reversal potential (V)
E_L = -54.387e-3       # Leak reversal potential (V)

# Synaptic parameters
E_exc = 0e-3           # Excitatory reversal potential (V)
E_inh = -80e-3         # Inhibitory reversal potential (V)
tau_exc = 5e-3         # Excitatory synaptic time constant (s)
tau_inh = 10e-3        # Inhibitory synaptic time constant (s)

# Advanced STDP parameters (Triplet Model)
A2_plus = 6e-3
A3_plus = 0.0
A2_minus = -7e-3
A3_minus = 0.0
tau_plus = 16.8e-3
tau_minus = 33.7e-3
w_max = 0.1e-6         # Maximum synaptic weight (S)
w_min = 0.0

# Homeostatic plasticity parameters
TARGET_RATE = 5.0      # Hz
TAU_HOMEOSTASIS = 1000.0  # seconds

# Performance optimization settings
USE_GPU = True         # Set to False if GPU is unavailable

# Helper functions for particle assembly simulation
def initialize_positions(num_neurons):
    # Random initial positions in 3D space within a cube
    positions = np.random.uniform(-1e-3, 1e-3, (num_neurons, 3))
    return positions

def compute_acoustic_force(positions):
    # Simplified model of acoustic radiation force
    # Using Gor'kov potential for a standing wave field
    k = 2 * np.pi * FREQUENCY / 1500  # Wave number (assuming speed of sound in water)
    forces = -ACOUSTIC_PRESSURE * np.sin(k * positions)
    return forces

def particle_assembly_simulation(num_neurons, dt, total_time):
    num_steps = int(total_time / dt)
    positions = initialize_positions(num_neurons)
    velocities = np.zeros_like(positions)
    mass = (4/3) * np.pi * PARTICLE_RADIUS**3 * 1000  # Assuming particle density equal to water
    gamma = 6 * np.pi * VISCOSITY * PARTICLE_RADIUS   # Stokes drag coefficient
    sqrt_2kT_gamma_dt = np.sqrt(2 * BOLTZMANN_CONSTANT * TEMPERATURE * gamma / dt)
    target_positions = assemble_target_positions(num_neurons)
    for step in tqdm(range(num_steps), desc="Assembly Simulation"):
        # Compute forces
        acoustic_forces = compute_acoustic_force(positions)
        spring_forces = -1e-4 * (positions - target_positions)  # Guiding force towards target
        random_forces = np.random.normal(0, sqrt_2kT_gamma_dt, positions.shape)
        total_forces = acoustic_forces + spring_forces + random_forces
        # Update velocities and positions
        accelerations = total_forces / mass
        velocities += accelerations * dt
        velocities *= np.exp(-gamma * dt / mass)  # Damping due to drag
        positions += velocities * dt
    return positions

def assemble_target_positions(num_neurons):
    # Generate target positions following a fractal or hierarchical pattern
    # For simplicity, place neurons in clusters with preferential connectivity
    positions = np.zeros((num_neurons, 3))
    num_clusters = int(np.sqrt(num_neurons))
    cluster_centers = np.random.uniform(-0.5e-3, 0.5e-3, (num_clusters, 3))
    cluster_size = num_neurons // num_clusters
    for i in range(num_clusters):
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size
        positions[start_idx:end_idx] = cluster_centers[i] + np.random.normal(0, 1e-5, (cluster_size, 3))
    return positions

def create_network(positions):
    # Create a network based on proximity with probability decreasing with distance
    distance_matrix = squareform(pdist(positions))
    probability_matrix = np.exp(-distance_matrix / 1e-4)  # Decaying function with length scale
    adjacency_matrix = np.random.rand(*probability_matrix.shape) < probability_matrix
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-connections
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    # Assign synaptic weights and delays
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.01e-6, 0.05e-6)  # Conductance in Siemens
        G[u][v]['delay'] = np.random.uniform(1e-3, 5e-3)        # Synaptic delay in seconds
    return G

# Define the neuron class with Hodgkin-Huxley model
class Neuron:
    def __init__(self, neuron_id, is_excitatory=True):
        self.id = neuron_id
        self.is_excitatory = is_excitatory
        self.V_m = E_L  # Membrane potential
        self.n = 0.3177  # Potassium activation
        self.m = 0.0529  # Sodium activation
        self.h = 0.5961  # Sodium inactivation
        self.spike_times = []
        self.I_syn_exc = 0  # Excitatory synaptic current
        self.I_syn_inh = 0  # Inhibitory synaptic current
        self.firing_rate = 0  # For homeostatic plasticity

# Functions for Hodgkin-Huxley equations
def alpha_n(V): return 0.01*(V*1e3 + 55)/(1 - np.exp(-0.1*(V*1e3 + 55)))
def beta_n(V): return 0.125*np.exp(-0.0125*(V*1e3 + 65))
def alpha_m(V): return 0.1*(V*1e3 + 40)/(1 - np.exp(-0.1*(V*1e3 + 40)))
def beta_m(V): return 4.0*np.exp(-0.0556*(V*1e3 + 65))
def alpha_h(V): return 0.07*np.exp(-0.05*(V*1e3 + 65))
def beta_h(V): return 1/(1 + np.exp(-0.1*(V*1e3 + 35)))

# Function to run the neuronal simulation
def run_neuronal_simulation(G, neurons):
    num_neurons = len(neurons)
    num_steps = int(SIM_TIME / DT_NEURON)
    time = np.arange(0, SIM_TIME, DT_NEURON)
    # Initialize variables
    spikes = np.zeros((num_neurons, num_steps), dtype=bool)
    pre_spike_trace = np.zeros(num_neurons)
    post_spike_trace = np.zeros(num_neurons)
    # Precompute adjacency and weights
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    syn_weights = np.zeros((num_neurons, num_neurons))
    syn_delays = np.zeros((num_neurons, num_neurons))
    for u, v in G.edges():
        syn_weights[u, v] = G[u][v]['weight']
        syn_delays[u, v] = int(G[u][v]['delay'] / DT_NEURON)
    # Convert to GPU arrays if enabled
    if USE_GPU:
        cp.cuda.Stream.null.synchronize()
        adjacency_matrix = cp.asarray(adjacency_matrix)
        syn_weights = cp.asarray(syn_weights)
        syn_delays = cp.asarray(syn_delays)
        pre_spike_trace = cp.asarray(pre_spike_trace)
        post_spike_trace = cp.asarray(post_spike_trace)
        spikes = cp.asarray(spikes)
    # Main simulation loop
    for t_idx in tqdm(range(num_steps), desc="Neuronal Simulation"):
        t = t_idx * DT_NEURON
        for neuron in neurons:
            # Update gating variables
            V = neuron.V_m
            n = neuron.n
            m = neuron.m
            h = neuron.h
            dn = (alpha_n(V)*(1 - n) - beta_n(V)*n) * DT_NEURON
            dm = (alpha_m(V)*(1 - m) - beta_m(V)*m) * DT_NEURON
            dh = (alpha_h(V)*(1 - h) - beta_h(V)*h) * DT_NEURON
            n += dn
            m += dm
            h += dh
            # Compute currents
            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K = g_K * n**4 * (V - E_K)
            I_L = g_L * (V - E_L)
            I_syn_exc = neuron.I_syn_exc
            I_syn_inh = neuron.I_syn_inh
            I_total = - (I_Na + I_K + I_L) + I_syn_exc + I_syn_inh
            # Update membrane potential
            V += (I_total / C_m) * DT_NEURON
            # Check for spike
            if V >= 0:
                neuron.spike_times.append(t)
                spikes[neuron.id, t_idx] = True
                V = E_L  # Reset potential
                # Update post-synaptic trace
                post_spike_trace[neuron.id] += 1
            # Update synaptic currents
            neuron.I_syn_exc *= np.exp(-DT_NEURON / tau_exc)
            neuron.I_syn_inh *= np.exp(-DT_NEURON / tau_inh)
            # Update neuron variables
            neuron.V_m = V
            neuron.n = n
            neuron.m = m
            neuron.h = h
        # Pre-synaptic trace decay
        pre_spike_trace *= np.exp(-DT_NEURON / tau_minus)
        # Synaptic updates and transmission
        for neuron in neurons:
            if spikes[neuron.id, t_idx]:
                pre_spike_trace[neuron.id] += 1
                # Transmit spikes to post-synaptic neurons
                post_ids = np.where(adjacency_matrix[neuron.id, :] > 0)[1]
                for post_id in post_ids:
                    delay = int(syn_delays[neuron.id, post_id])
                    arrival_idx = t_idx + delay
                    if arrival_idx < num_steps:
                        weight = syn_weights[neuron.id, post_id]
                        if neuron.is_excitatory:
                            neurons[post_id].I_syn_exc += weight
                        else:
                            neurons[post_id].I_syn_inh += weight
                        # STDP updates
                        delta_t = t - neurons[post_id].spike_times[-1] if neurons[post_id].spike_times else np.inf
                        if delta_t != np.inf:
                            if delta_t > 0:
                                dw = A2_plus * np.exp(-delta_t / tau_plus)
                            else:
                                dw = A2_minus * np.exp(delta_t / tau_minus)
                            syn_weights[neuron.id, post_id] += dw
                            syn_weights[neuron.id, post_id] = np.clip(syn_weights[neuron.id, post_id], w_min, w_max)
        # Homeostatic plasticity
        for neuron in neurons:
            neuron.firing_rate = len(neuron.spike_times) / (t + DT_NEURON)
            homeo_factor = (TARGET_RATE - neuron.firing_rate) / TARGET_RATE
            syn_weights[:, neuron.id] *= (1 + homeo_factor * DT_NEURON / TAU_HOMEOSTASIS)
            syn_weights[:, neuron.id] = np.clip(syn_weights[:, neuron.id], w_min, w_max)
    if USE_GPU:
        spikes = cp.asnumpy(spikes)
    return spikes, time

def analyze_criticality(spikes):
    # Compute avalanche size distribution
    total_activity = np.sum(spikes, axis=0)
    is_active = total_activity > 0
    avalanche_sizes = []
    avalanche_durations = []
    avalanche_size = 0
    avalanche_duration = 0
    for active in is_active:
        if active:
            avalanche_size += total_activity[active]
            avalanche_duration += 1
        elif avalanche_size > 0:
            avalanche_sizes.append(avalanche_size)
            avalanche_durations.append(avalanche_duration)
            avalanche_size = 0
            avalanche_duration = 0
    # Fit power-law distribution
    if avalanche_sizes:
        fit = powerlaw.Fit(avalanche_sizes, discrete=True)
        alpha = fit.alpha
        xmin = fit.xmin
        print(f"Avalanche size power-law exponent: {alpha:.2f}")
        # Plot
        plt.figure()
        fit.plot_pdf(color='b', linewidth=2)
        fit.power_law.plot_pdf(color='r', linestyle='--')
        plt.xlabel('Avalanche Size')
        plt.ylabel('Probability Density')
        plt.title('Avalanche Size Distribution')
        plt.legend(['Empirical Data', 'Power-law Fit'])
        plt.show()
    else:
        print("No avalanches detected.")

def simulate_cognitive_task(neurons, spikes, time):
    # Implement a pattern recognition task using PCA or clustering
    from sklearn.decomposition import PCA
    spike_counts = np.sum(spikes, axis=1)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(spikes)
    plt.figure()
    sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=spike_counts)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Neuronal Population Activity')
    plt.show()

def main():
    # Assembly simulation
    positions = particle_assembly_simulation(NUM_NEURONS, DT_ASSEMBLY, ASSEMBLY_TIME)
    G = create_network(positions)
    # Assign neuron types
    neurons = []
    for i in G.nodes():
        is_excitatory = np.random.rand() < 0.8  # 80% excitatory neurons
        neurons.append(Neuron(i, is_excitatory=is_excitatory))
    # Neuronal simulation
    spikes, time = run_neuronal_simulation(G, neurons)
    # Analyze criticality
    analyze_criticality(spikes)
    # Simulate cognitive task
    simulate_cognitive_task(neurons, spikes, time)
    # Plot raster plot
    plt.figure(figsize=(12, 6))
    for neuron_id in range(len(neurons)):
        spike_times = time[np.where(spikes[neuron_id])]
        plt.vlines(spike_times, neuron_id - 0.5, neuron_id + 0.5, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron ID')
    plt.title('Raster Plot of Network Activity')
    plt.show()

if __name__ == "__main__":
    main()
