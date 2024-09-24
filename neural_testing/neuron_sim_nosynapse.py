from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import linear
import numpy as np

# Constants
NUM_NEURONS = 1000  # Keep it low for debugging
TIME_STEPS = 1000  # Keep it low for now
DT_NEURON = 0.5e-3  # Time step for simulation (0.5 ms)
REFRACTORY_PERIOD_MEAN = 5  # Mean refractory period
REFRACTORY_PERIOD_VARIABILITY = 2  # Add variability to the refractory period
V_THRESHOLD = -45.0  # Increase the threshold for spiking (mV) for more variation
V_RESET = -65.0  # Membrane potential after spiking (mV)
V_REST = -70.0  # Resting membrane potential (mV)
V_DECAY = 0.98  # Slightly reduce decay to allow more accumulation

# DEBUG: Print out basic constants for sanity check
print(f"NUM_NEURONS: {NUM_NEURONS}, TIME_STEPS: {TIME_STEPS}, DT_NEURON: {DT_NEURON}, REFRACTORY_PERIOD_MEAN: {REFRACTORY_PERIOD_MEAN}")

# Initialize membrane potential for each neuron
v = np.ones(NUM_NEURONS) * V_REST

# External input (increase variability for each neuron at each time step)
I_ext = np.random.normal(100.0, 50.0, (NUM_NEURONS, TIME_STEPS))  # Higher variability

# Simulated spikes: neurons spike when membrane potential exceeds threshold
spikes = np.zeros((NUM_NEURONS, TIME_STEPS), dtype=bool)

# "Fatigue" state: tracks how long each neuron has to wait after spiking before it can fire again
# Add variability to the refractory period
fatigue = np.zeros(NUM_NEURONS, dtype=int)
refractory_periods = np.random.randint(REFRACTORY_PERIOD_MEAN - REFRACTORY_PERIOD_VARIABILITY,
                                       REFRACTORY_PERIOD_MEAN + REFRACTORY_PERIOD_VARIABILITY, NUM_NEURONS)

# Simulate neuron dynamics over time
for t in range(TIME_STEPS):
    # For each neuron, check if it can spike (i.e., not fatigued)
    for neuron_id in range(NUM_NEURONS):
        if fatigue[neuron_id] == 0:  # Neuron can fire
            # Update membrane potential: increase due to external input
            v[neuron_id] += I_ext[neuron_id, t] * DT_NEURON
            
            # Decay membrane potential slightly to prevent it from accumulating indefinitely
            v[neuron_id] *= V_DECAY
            
            # Check if neuron spikes
            if v[neuron_id] >= V_THRESHOLD:
                spikes[neuron_id, t] = 1  # Record spike
                v[neuron_id] = V_RESET  # Reset membrane potential
                fatigue[neuron_id] = refractory_periods[neuron_id]  # Set neuron in refractory period
        else:
            fatigue[neuron_id] -= 1  # Decrease the fatigue counter

# DEBUG: Print out the spikes matrix to make sure it has values
print("Spikes matrix:")
print(spikes.astype(int))  # Print it as integers (0 or 1) for easier reading

# Set up Bokeh figure
p = figure(width=800, height=600, title="Raster Plot of Network Activity", x_axis_label="Time (s)", y_axis_label="Neuron ID")

# Initialize the data source for the plot
source = ColumnDataSource(data=dict(x=[], y=[]))
p.scatter(x="x", y="y", source=source, size=2, color="black", alpha=0.6)

# Update function for Bokeh
def update_raster(step):
    # Make sure we only use steps within the bounds of the matrix
    if step >= TIME_STEPS:
        return
    
    spike_times = []
    neuron_ids = []
    for neuron_id in range(NUM_NEURONS):
        # Get spike times up to the current step
        spike_time = np.where(spikes[neuron_id, :step])[0] * DT_NEURON
        spike_times.extend(spike_time)
        neuron_ids.extend([neuron_id] * len(spike_time))

    # DEBUG: Print to ensure we're sending data
    print(f"Step: {step}, Spikes added: {len(spike_times)}")

    source.data = dict(x=spike_times, y=neuron_ids)

# Use a linear driver to update in real time
@linear()
def update(step):
    update_raster(step)

# Periodic callback to update every 50 ms
curdoc().add_root(column(p))
curdoc().add_periodic_callback(update, 50)
