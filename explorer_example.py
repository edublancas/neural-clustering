%load_ext autoreload
%autoreload 2

from neural_noise.explorer import Explorer
import matplotlib.pyplot as plt

explorer = Explorer('standarized.bin', dtype='float', window_size=15,
                    n_channels=7)

explorer.read_waveform(15)

explorer.plot_waveforms(15, [0,2,3,4,6], line_at_t=True)

explorer.plot_waveforms(15, [0,2,3,4,6], line_at_t=True, overlay=True)

plt.show()