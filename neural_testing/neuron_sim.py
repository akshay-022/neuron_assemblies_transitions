from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import linear
import numpy as np

# Constants
NUM_NEURONS = 200
TIME_STEPS = 10000
DT_NEURON = 0.5e-5
REFRACTORY_PERIOD_MEAN = 8
REFRACTORY_PERIOD_VARIABILITY = 2
V_THRESHOLD = -45.0
V_RESET = -65.0
V_REST = -70.0
V_DECAY = 0.98

# STDP parameters
A_PLUS = 0.015  # Learning rate for potentiation (synapse strengthening)
A_MINUS = -0.015  # Learning rate for depression (synapse weakening)
TAU_PLUS = 20.0  # Time constant for potentiation (ms)
TAU_MINUS = 20.0  # Time constant for depression (ms)
BASELINE_CURRENT = 15.0  # Constant external input to ensure neurons remain active
WEIGHT_DECAY = 0.9995  # Decay factor for synaptic weights

# Synaptic boundaries
MIN_WEIGHT = -1.5  # Minimum synaptic weight (inhibitory)
MAX_WEIGHT = 1.0   # Maximum synaptic weight (excitatory)

# Homeostatic Plasticity Parameters
TARGET_SPIKE_RATE = 0.01  # Target spike rate for homeostasis (5% of timesteps)
HOMEOSTATIC_LEARNING_RATE = 0.005  # Rate at which neurons adjust their excitability

# Conductance-based synapse parameters
G_MAX = 0.024  # Maximum conductance
SYNAPTIC_TAU = 5.0  # Time constant for synaptic conductance decay
SYNAPTIC_TAU_INHIBITORY = 10.0  # Inhibitory synaptic decay for more realistic balancing

# Initialize membrane potential, external input, and conductance
v = np.random.uniform(-70.0, -50.0, NUM_NEURONS)  # # Adjust the range of external input
I_ext = np.random.normal(50.0, 20.0, (NUM_NEURONS, TIME_STEPS))  # Lower mean input for less excitation

g_syn = np.zeros(NUM_NEURONS)  # Synaptic conductance for each neuron

# Classify neurons: 80% excitatory, 20% inhibitory
EXCITATORY_RATIO = 0.8
num_excitatory = int(EXCITATORY_RATIO * NUM_NEURONS)
num_inhibitory = NUM_NEURONS - num_excitatory
neuron_types = np.zeros(NUM_NEURONS, dtype=int)  # 0: inhibitory, 1: excitatory
neuron_types[:num_excitatory] = 1  # First 80% excitatory, rest inhibitory

# Shuffle neuron types for randomness
np.random.shuffle(neuron_types)

# Initialize synaptic connections (random weights: positive for excitatory, negative for inhibitory)
synaptic_weights = np.zeros((NUM_NEURONS, NUM_NEURONS))
for i in range(NUM_NEURONS):
    if neuron_types[i] == 1:  # Excitatory neuron
        synaptic_weights[i] = np.random.uniform(0, 0.5, NUM_NEURONS)  # Decrease max excitatory weight
    else:  # Inhibitory neuron
        synaptic_weights[i] = np.random.uniform(-1, -0.2, NUM_NEURONS)  # Strengthen inhibitory weights

# Synaptic input and homeostatic plasticity tracking
synaptic_input = np.zeros(NUM_NEURONS)
firing_rate = np.zeros(NUM_NEURONS)  # Track how often each neuron spikes

# Spike matrix and fatigue
spikes = np.zeros((NUM_NEURONS, TIME_STEPS), dtype=bool)
fatigue = np.zeros(NUM_NEURONS, dtype=int)
refractory_periods = np.random.randint(REFRACTORY_PERIOD_MEAN - REFRACTORY_PERIOD_VARIABILITY,
                                       REFRACTORY_PERIOD_MEAN + REFRACTORY_PERIOD_VARIABILITY, NUM_NEURONS)

# Last spike time for each neuron
last_spike_times = np.full(NUM_NEURONS, -np.inf)

# STDP Update Rule with synaptic bounds and weight decay
def update_stdp(pre_id, post_id, delta_t):
    if delta_t > 0:  # Pre spikes before post (potentiation)
        dw = A_PLUS * np.exp(-delta_t / TAU_PLUS)
    else:  # Pre spikes after post (depression)
        dw = A_MINUS * np.exp(delta_t / TAU_MINUS)
    
    # Update the synaptic weight
    synaptic_weights[pre_id, post_id] += dw
    # Keep weights within bounds and apply decay
    synaptic_weights[pre_id, post_id] = np.clip(synaptic_weights[pre_id, post_id] * WEIGHT_DECAY, MIN_WEIGHT, MAX_WEIGHT)

# Homeostatic Plasticity: Adjust neuron excitability to maintain target firing rate
def homeostatic_plasticity(neuron_id, timestep):
    actual_rate = firing_rate[neuron_id] / timestep  # Current firing rate
    excitability_adjustment = HOMEOSTATIC_LEARNING_RATE * (TARGET_SPIKE_RATE - actual_rate)
    
    # Adjust threshold to maintain target spike rate (homeostatic adjustment)
    global V_THRESHOLD
    V_THRESHOLD -= excitability_adjustment

# Simulate neuron dynamics over time
for t in range(TIME_STEPS):
    synaptic_input.fill(0)  # Reset synaptic input for this timestep
    g_syn *= np.exp(-DT_NEURON / SYNAPTIC_TAU)  # Apply conductance decay

    # Calculate synaptic input from neurons that spiked in the previous timestep
    if t > 0:
        for i in range(NUM_NEURONS):
            if spikes[i, t-1]:  # If neuron i spiked in the last timestep
                synaptic_input += synaptic_weights[i]  # Add its contribution to other neurons' input
                g_syn += synaptic_weights[i] * G_MAX  # Update synaptic conductance

    # For each neuron, check if it can spike
    for neuron_id in range(NUM_NEURONS):
        if fatigue[neuron_id] == 0:
            # Update membrane potential with external input, synaptic input, and conductance
            synaptic_current = g_syn[neuron_id] * (V_REST - v[neuron_id])  # Conductance-based synaptic current
            v[neuron_id] += (I_ext[neuron_id, t] + synaptic_input[neuron_id] + synaptic_current + BASELINE_CURRENT) * DT_NEURON
            v[neuron_id] *= V_DECAY  # Apply decay

            # Check if the neuron spikes
            if v[neuron_id] >= V_THRESHOLD:
                spikes[neuron_id, t] = 1
                v[neuron_id] = V_RESET
                fatigue[neuron_id] = refractory_periods[neuron_id]
                firing_rate[neuron_id] += 1  # Track firing rate for homeostatic plasticity

                # Update STDP for synapses
                for pre_id in range(NUM_NEURONS):
                    if spikes[pre_id, t-1]:  # If the presynaptic neuron spiked in the previous timestep
                        delta_t = (t * DT_NEURON * 1000) - last_spike_times[neuron_id]  # Time difference in ms
                        update_stdp(pre_id, neuron_id, delta_t)

                # Update last spike time
                last_spike_times[neuron_id] = t * DT_NEURON * 1000

                # Apply homeostatic plasticity
                homeostatic_plasticity(neuron_id, t)
        else:
            fatigue[neuron_id] -= 1  # Decrease fatigue

# Set up Bokeh plot
p = figure(width=800, height=600, title="Raster Plot with Homeostatic Plasticity and Conductance-Based Synapses", x_axis_label="Time (s)", y_axis_label="Neuron ID")
source = ColumnDataSource(data=dict(x=[], y=[]))
p.scatter(x="x", y="y", source=source, size=2, color="black", alpha=0.6)

# Update function for Bokeh
def update_raster(step):
    if step >= TIME_STEPS:
        return
    spike_times = []
    neuron_ids = []
    for neuron_id in range(NUM_NEURONS):
        spike_time = np.where(spikes[neuron_id, :step])[0] * DT_NEURON
        spike_times.extend(spike_time)
        neuron_ids.extend([neuron_id] * len(spike_time))
    source.data = dict(x=spike_times, y=neuron_ids)

@linear()
def update(step):
    update_raster(step)

curdoc().add_root(column(p))
curdoc().add_periodic_callback(update, 50)
