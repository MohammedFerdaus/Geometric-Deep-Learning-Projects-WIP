import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def pairwise_displacements(x):
    return x[:, None, :] - x[None, :, :]

def lj_energy(x, epsilon=1.0, sigma=1.0):
    N = x.shape[0]
    diff = pairwise_displacements(x) 

    r2 = jnp.sum(diff ** 2, axis=-1)
    eye = jnp.eye(N)
    r2_safe = r2 + eye

    s2 = sigma ** 2 / r2_safe
    s6 = s2 ** 3
    s12 = s6 ** 2

    pair = 4 * epsilon * (s12 - s6)
    pair = pair * (1.0 - eye)

    return 0.5 * jnp.sum(pair)

def lj_forces_analytic(x, epsilon=1.0, sigma=1.0):
    N = x.shape[0]
    diff = pairwise_displacements(x) 

    r2 = jnp.sum(diff ** 2, axis=-1)
    eye = jnp.eye(N)
    r2_safe = r2 + eye

    s2 = sigma ** 2 / r2_safe
    s6 = s2 ** 3
    s12 = s6 ** 2

    coeff = 24 * epsilon * (2 * s12 - s6) / r2_safe
    coeff = coeff * (1.0 - eye)

    weighted = coeff[:, :, None] * diff

    return jnp.sum(weighted, axis=1)

def lj_forces_autodiff(x, epsilon=1.0, sigma=1.0):
    return -jax.grad(lj_energy)(x, epsilon, sigma)

def check_forces_agree(x, epsilon=1.0, sigma=1.0):
    f_analytic = lj_forces_analytic(x, epsilon, sigma)
    f_autodiff = lj_forces_autodiff(x, epsilon, sigma)

    max_diff = jnp.max(jnp.abs(f_analytic - f_autodiff))
    print(f"max absolute difference: {max_diff}")

    assert max_diff < 1e-10, f"forces do not agree. Max diff: {max_diff}"
    print("forces agree")

def kinetic_energy(v, mass=1.0):
    return 0.5 * mass * jnp.sum(v ** 2)

def velocity_verlet_step(x, v, force_fn, dt, mass=1.0):
    F = force_fn(x)
    v_half = v + 0.5 * dt * F / mass

    x_new = x + dt * v_half
    F_new = force_fn(x_new)
    v_new = v_half + 0.5 * dt * F_new / mass

    return x_new, v_new

def run_md(x0, v0, force_fn, dt, n_steps):

    def step(carry, _):
        x, v = carry
        x_new, v_new = velocity_verlet_step(x, v, force_fn, dt)

        pe = lj_energy(x_new)
        ke = kinetic_energy(v_new)
        e_total = pe + ke

        new_carry = (x_new, v_new)
        recorded = (x_new, v_new, e_total)

        return new_carry, recorded

    final_carry, recorded = jax.lax.scan(step, (x0, v0), None, length=n_steps)
    xs, vs, energies = recorded

    e0 = lj_energy(x0) + kinetic_energy(v0)

    xs = jnp.concatenate([x0[None], xs], axis=0)
    vs = jnp.concatenate([v0[None], vs], axis=0)
    energies = jnp.concatenate([jnp.array([e0]), energies], axis=0)

    return xs, vs, energies

def initial_cluster(n_atoms=13, rng_key=None):
    phi = (1 + jnp.sqrt(5)) / 2

    vertices = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            vertices.append([0, s1 * 1, s2 * phi])
            vertices.append([s1 * 1, s2 * phi, 0])
            vertices.append([s2 * phi, 0, s1 * 1])

    x = jnp.array(vertices) 
    
    center = jnp.zeros((1, 3))
    x = jnp.concatenate([center, x], axis=0) 

    x = x * (1.1 / jnp.sqrt(1 + phi ** 2))

    if rng_key is not None:
        noise = jax.random.normal(rng_key, shape=x.shape)
        x = x + 0.05 * noise

    lr = 0.002
    for i in range(1000):
        F = lj_forces_autodiff(x)
        x = x + lr * F
        if i % 200 == 0:
            print(f"relax step {i}: E = {lj_energy(x)}")

    return x