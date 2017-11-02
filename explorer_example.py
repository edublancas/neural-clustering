%load_ext autoreload
%autoreload 2

from neural_noise.explorer import Explorer
import matplotlib.pyplot as plt

explorer = Explorer('ej49_data1_set1.bin', 'ej49_geometry1.txt', dtype='int16', window_size=15,
                    n_channels=49, neighbor_radius=70)

explorer.read_waveform(1560)

explorer.plot_waveform(1560, [0,2,3,4,6], line_at_t=True)

explorer.plot_waveform(100000, [0,2,3,4,6], line_at_t=True, overlay=True)

explorer.plot_waveform(100000, [0,2,3,4,6], line_at_t=True, overlay=False)

plt.show()

explorer.plot_waveform_around_channel(100000, 10)
plt.show()