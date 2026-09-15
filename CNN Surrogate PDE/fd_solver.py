import numpy as np

def stability_limit(alpha, dx, dy=None):
    if dy is None:
        dy = dx   
    max_dt = 1/(2 * alpha * (1/dx**2 + 1/dy**2))
    return max_dt

def choose_stable_dt(alpha, dx, dy=None, safety_factor=0.5):
    limit = stability_limit(alpha, dx, dy)
    return safety_factor * limit

def laplacian_stencil(field, dx, dy=None):
    if dy is None:
        dy = dx
    lap = np.zeros_like(field)
    center = field[1:-1, 1:-1]
    up     = field[0:-2, 1:-1]
    down   = field[2:  , 1:-1]
    left   = field[1:-1, 0:-2]
    right  = field[1:-1, 2:  ]

    d2x = (left + right - 2*center) / dx**2
    d2y = (up + down - 2*center) / dy**2

    lap[1:-1, 1:-1] = d2x + d2y
    return lap

def apply_dirichlet_bc(field, boundary_value):
    new_field = field.copy()

    new_field[0, :] = boundary_value
    new_field[-1, :] = boundary_value
    new_field[:, 0] = boundary_value
    new_field[:, -1] = boundary_value

    return new_field

def fd_step(field, dt, dx, alpha, boundary_value, dy=None):
    lap = laplacian_stencil(field, dx, dy)
    
    new_field = field.copy()
    new_field[1:-1, 1:-1] = field[1:-1, 1:-1] + alpha * dt * lap[1:-1, 1:-1]
    new_field = apply_dirichlet_bc(new_field, boundary_value)

    return new_field

def fd_rollout(initial_field, boundary_value, n_steps, dt, dx, alpha, dy=None):
    field = apply_dirichlet_bc(initial_field, boundary_value)
    
    for i in range(n_steps):
        field = fd_step(field, dt, dx, alpha, boundary_value, dy)

    return field

def generate_random_initial_condition(H, W, rng, n_gaussians=3):
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    field = np.zeros((H, W))

    for i in range(n_gaussians):
        x0 = rng.uniform(0, W)
        y0 = rng.uniform(0, H)
        amplitude = rng.uniform(0.2, 1.0)
        sigma = rng.uniform(W * 0.08, W * 0.2)

        bump = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        field += bump

    return field