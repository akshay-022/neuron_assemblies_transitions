from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import linear
import numpy as np
from skimage import io, color, transform

# Constants
NUM_NEURONS = 100  # Number of neurons (rows or columns for the image-based input)
TIME_STEPS = 1000  # Number of time steps in the simulation
DT_NEURON = 0.5e-3  # Time step for neuronal simulation (0.5 ms)
REFRACTORY_PERIOD_MEAN = 5
REFRACTORY_PERIOD_VARIABILITY = 2
V_THRESHOLD = -45.0  # Membrane potential threshold for spiking (mV)
V_RESET = -65.0  # Membrane potential after spiking (mV)
V_REST = -70.0  # Resting membrane potential (mV)
V_DECAY = 0.98  # Membrane potential decay factor

# Load the Lena image
lena_image_url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
lena_image = io.imread(lena_image_url)

# Convert the image to grayscale
lena_gray = color.rgb2gray(lena_image)

# Resize the image to fit the neuron grid (100 x 100)
lena_resized = transform.resize(lena_gray, (NUM_NEURONS, NUM_NEURONS))

# Normalize the pixel values to range between 0 and 1 (for easier neuron input scaling)
lena_normalized = lena_resized / np.max(lena_resized)

# Initialize membrane potential and input
v = np.ones(NUM_NEURONS) * V_REST

# Use the pixel values as input (with small variability added to simulate noise)
I_ext = np.zeros((NUM_NEURONS, TIME_STEPS))

# Random sampling across the image over time
for t in range(TIME_STEPS):
    # Randomly sample different rows from the Lena image for each timestep
    random_row_indices = np.random.choice(np.arange(NUM_NEURONS), NUM_NEURONS, replace=True)
    I_ext[:, t] = lena_normalized[random_row_indices, t % NUM_NEURONS] * 50  # Scale input to neuronal response

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
        synaptic_weights[i] = np.random.uniform(0, 1, NUM_NEURONS)  # Positive weights
    else:  # Inhibitory neuron
        synaptic_weights[i] = np.random.uniform(-1, 0, NUM_NEURONS)  # Negative weights

# Synaptic input
synaptic_input = np.zeros(NUM_NEURONS)

# Spike matrix and fatigue
spikes = np.zeros((NUM_NEURONS, TIME_STEPS), dtype=bool)
fatigue = np.zeros(NUM_NEURONS, dtype=int)
refractory_periods = np.random.randint(REFRACTORY_PERIOD_MEAN - REFRACTORY_PERIOD_VARIABILITY,
                                       REFRACTORY_PERIOD_MEAN + REFRACTORY_PERIOD_VARIABILITY, NUM_NEURONS)

# Last spike time for each neuron
last_spike_times = np.full(NUM_NEURONS, -np.inf)

# STDP Update Rule with synaptic bounds and weight decay
def update_stdp(pre_id, post_id, delta_t):
    A_PLUS = 0.01  # Potentiation factor
    A_MINUS = -0.012  # Depression factor
    TAU_PLUS = 20.0  # Time constant for potentiation (ms)
    TAU_MINUS = 20.0  # Time constant for depression (ms)
    WEIGHT_DECAY = 0.999  # Synaptic weight decay factor

    if delta_t > 0:  # Pre spikes before post (potentiation)
        dw = A_PLUS * np.exp(-delta_t / TAU_PLUS)
    else:  # Pre spikes after post (depression)
        dw = A_MINUS * np.exp(delta_t / TAU_MINUS)
    
    # Update the synaptic weight and apply decay
    synaptic_weights[pre_id, post_id] += dw
    synaptic_weights[pre_id, post_id] = np.clip(synaptic_weights[pre_id, post_id] * WEIGHT_DECAY, -1.0, 1.0)

# Simulate neuron dynamics over time
for t in range(TIME_STEPS):
    synaptic_input.fill(0)  # Reset synaptic input for this timestep

    # Calculate synaptic input from neurons that spiked in the previous timestep
    if t > 0:
        for i in range(NUM_NEURONS):
            if spikes[i, t-1]:  # If neuron i spiked in the last timestep
                synaptic_input += synaptic_weights[i]  # Add its contribution to other neurons' input

    # For each neuron, check if it can spike
    for neuron_id in range(NUM_NEURONS):
        if fatigue[neuron_id] == 0:
            # Update membrane potential with external input and synaptic input
            v[neuron_id] += (I_ext[neuron_id, t] + synaptic_input[neuron_id]) * DT_NEURON
            v[neuron_id] *= V_DECAY  # Apply decay

            # Check if the neuron spikes
            if v[neuron_id] >= V_THRESHOLD:
                spikes[neuron_id, t] = 1
                v[neuron_id] = V_RESET
                fatigue[neuron_id] = refractory_periods[neuron_id]

                # Update STDP for synapses
                for pre_id in range(NUM_NEURONS):
                    if spikes[pre_id, t-1]:  # If the presynaptic neuron spiked in the previous timestep
                        delta_t = (t * DT_NEURON * 1000) - last_spike_times[neuron_id]  # Time difference in ms
                        update_stdp(pre_id, neuron_id, delta_t)

                # Update last spike time
                last_spike_times[neuron_id] = t * DT_NEURON * 1000
        else:
            fatigue[neuron_id] -= 1  # Decrease fatigue

# Set up Bokeh plot
p = figure(width=800, height=600, title="Raster Plot with Lena Image as Input Stimuli", x_axis_label="Time (s)", y_axis_label="Neuron ID")
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
