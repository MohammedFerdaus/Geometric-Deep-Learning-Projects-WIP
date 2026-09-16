import numpy as np

def tau_E(n, T, I, B, C, a, b, c, d):
    tau = C * I**a * B**b * n**c * T**d
    return tau

def derivatives(state, t, P_heat_fn, params):
    n, T = state
    P_heat = P_heat_fn(t)

    dn_dt = params['S_p'] - n / params['tau_p']

    tau = tau_E(n, T, params['I'], params['B'], params['C'],
                params['a'], params['b'], params['c'], params['d'])

    dT_dt = P_heat / (n * params['cv']) - T / tau

    return np.array([dn_dt, dT_dt])

def rk4_step(state, t, dt, P_heat_fn, params):
    k1 = derivatives(state, t, P_heat_fn, params)
    k2 = derivatives(state + dt/2 * k1, t + dt/2, P_heat_fn, params)
    k3 = derivatives(state + dt/2 * k2, t + dt/2, P_heat_fn, params)
    k4 = derivatives(state + dt * k3, t + dt, P_heat_fn, params)

    next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

def simulate(initial_state, t_span, dt, P_heat_fn, params):
    n_steps = int((t_span[1] - t_span[0]) / dt ) + 1
    t_array = np.linspace(t_span[0], t_span[1], n_steps)

    n_array = np.zeros(n_steps)
    T_array = np.zeros(n_steps)
    state = np.array(initial_state, dtype=float)

    n_array[0], T_array[0] = state
    for i in range(1, n_steps):
        state = rk4_step(state, t_array[i-1], dt, P_heat_fn, params)
        n_array[i], T_array[i] = state

    return t_array, n_array, T_array

def default_params():
    return {
        'S_p': 1.0,
        'tau_p': 1.0,
        'cv': 1.0,
        'I': 1.0,
        'B': 1.0,
        'C': 1.0,
        'a': 0.0, 
        'b': 0.0,
        'c': 0.3,
        'd': -1.5,
    }