import numpy as np

def step_forcing(t, t_on, magnitude):
    return np.where(t >= t_on, magnitude, 0.0)

def ramp_forcing(t, t_start, t_end, mag_start, mag_end):
    frac = np.clip((t - t_start) / (t_end - t_start), 0.0, 1.0)
    return mag_start + frac * (mag_end - mag_start)

def sinusoid_forcing(t, amplitude, freq, offset):
    return offset + amplitude * np.sin(2 * np.pi * freq * t)

def random_forcing_generator(rng, kind=None, t_span=(0.0, 20.0), **kwargs):
    if kind is None:
        kind = rng.choice(['step', 'ramp', 'sinusoid'])

    if kind == 'step':
        t_on = rng.uniform(t_span[0], t_span[1])
        magnitude = rng.uniform(1.0, 3.0)
        return lambda t: step_forcing(t, t_on, magnitude)

    elif kind == 'ramp':
        t_start = rng.uniform(t_span[0], t_span[1] * 0.6)
        t_end = t_start + rng.uniform(2.0, 8.0)
        mag_start = rng.uniform(1.0, 2.0)
        mag_end = rng.uniform(1.0, 3.0)
        return lambda t: ramp_forcing(t, t_start, t_end, mag_start, mag_end)

    elif kind == 'sinusoid':
        offset = rng.uniform(1.5, 2.5)
        amplitude = rng.uniform(0.3, offset * 0.9) 
        freq = rng.uniform(0.05, 0.15)
        return lambda t: sinusoid_forcing(t, amplitude, freq, offset)

    else:
        raise ValueError(f"unknown forcing kind: {kind}")