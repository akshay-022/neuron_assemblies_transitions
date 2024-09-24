from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import linear
import numpy as np

# Constants
NUM_NEURONS = 100
SIM_TIME = 1.0        # seconds
DT_NEURON = 0.5e-3    # Time step for neuronal simulation (0.5 ms)
TIME_STEPS = int(SIM_TIME / DT_NEURON)

# Izhikevich model parameters
a, b, c, d = 0.02, 0.2, -65, 8  # Regular spiking neurons
v = np.random.uniform(-70.0, -50.0, NUM_NEURONS)  # Membrane potential
u = b * v  # Recovery variable
spikes = np.zeros((NUM_NEURONS, TIME_STEPS))

# External input
I_ext = np.random.normal(100.0, 13.0, (NUM_NEURONS, TIME_STEPS))

# Simulate spiking behavior using the Izhikevich model
for t in range(TIME_STEPS):
    fired = v >= 30.0
    spikes[:, t] = fired
    v[fired] = c
    u[fired] += d
    v += DT_NEURON * (0.04 * v**2 + 5.0 * v + 140 - u + I_ext[:, t])
    u += DT_NEURON * a * (b * v - u)

# Set up Bokeh figure
p = figure(width=800, height=600, title="Raster Plot of Network Activity", x_axis_label="Time (s)", y_axis_label="Neuron ID")

# Initialize the data source for the plot
source = ColumnDataSource(data=dict(x=[], y=[]))
p.scatter(x="x", y="y", source=source, size=2, color="black", alpha=0.6)

# Update function
def update_raster(step):
    spike_times = []
    neuron_ids = []
    for neuron_id in range(NUM_NEURONS):
        spike_time = np.where(spikes[neuron_id, :step])[0] * DT_NEURON
        spike_times.extend(spike_time)
        neuron_ids.extend([neuron_id] * len(spike_time))

    source.data = dict(x=spike_times, y=neuron_ids)

# Use a linear driver to update in real time
@linear()
def update(step):
    update_raster(step)

# Periodic callback to update every 50 ms
curdoc().add_root(column(p))
curdoc().add_periodic_callback(update, 50)
