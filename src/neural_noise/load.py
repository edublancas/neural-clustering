import numpy as np


def data(path, n_channels, data_format='long', dtype='int16'):
    """Load MEAs readings

    Parameters
    ----------
    path: str
        Path to data file
    n_channels: int
        Number of channels
    data_format: str
        'long' [observations per channel, number of channels] or 'wide'
        [number of channels, observations per channel]
    dtype: str
        Numpy dtype

    Returns
    -------
    data: data in long format
    """
    data = np.fromfile(path, dtype=dtype)
    n_obs = len(data)
    obs_per_channel = int(n_obs/n_channels)

    dims = ((obs_per_channel, n_channels) if data_format == 'long'
            else (n_channels, obs_per_channel))

    data_reshaped = data.reshape(dims)

    return data_reshaped if data_reshaped == 'long' else data_reshaped.T
