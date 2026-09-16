import jax
import jax.numpy as jnp

def init_lstm_params(rng, input_dim, hidden_dim, output_dim):   
    keys = jax.random.split(rng, 9)

    def glorot(key, shape):
        fan_in, fan_out = shape
        limit = jnp.sqrt(6 / (fan_in + fan_out))   

        return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
    
    params = {
        'W_f_x': glorot(keys[0], (input_dim, hidden_dim)),
        'W_f_h': glorot(keys[1], (hidden_dim, hidden_dim)),
        'b_f': jnp.zeros(hidden_dim),
    
        'W_i_x': glorot(keys[2], (input_dim, hidden_dim)),
        'W_i_h': glorot(keys[3], (hidden_dim, hidden_dim)),
        'b_i': jnp.zeros(hidden_dim),
    
        'W_c_x': glorot(keys[4], (input_dim, hidden_dim)),
        'W_c_h': glorot(keys[5], (hidden_dim, hidden_dim)),
        'b_c': jnp.zeros(hidden_dim),
    
        'W_o_x': glorot(keys[6], (input_dim, hidden_dim)),
        'W_o_h': glorot(keys[7], (hidden_dim, hidden_dim)),
        'b_o': jnp.zeros(hidden_dim),
    
        'W_out': glorot(keys[8], (hidden_dim, output_dim)),
        'b_out': jnp.zeros(output_dim),}
    
    return params

def lstm_cell(params, x_t, h_prev, c_prev):
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    f_t = sigmoid(x_t @ params['W_f_x'] + h_prev @ params['W_f_h'] + params['b_f'])
    i_t = sigmoid(x_t @ params['W_i_x'] + h_prev @ params['W_i_h'] + params['b_i'])
    c_tilde = jnp.tanh(x_t @ params['W_c_x'] + h_prev @ params['W_c_h'] + params['b_c'])
    c_t = f_t * c_prev + i_t * c_tilde

    o_t = sigmoid(x_t @ params['W_o_x'] + h_prev @ params['W_o_h'] + params['b_o'])
    h_t = o_t * jnp.tanh(c_t)

    return h_t, c_t
    
def lstm_forward(params, X_seq, h0, c0):
    def step_fn(carry, x_t):
        h_prev, c_prev = carry
        h_t, c_t = lstm_cell(params, x_t, h_prev, c_prev)
        return  (h_t, c_t), h_t
    
    (h_final, c_final), outputs = jax.lax.scan(step_fn, (h0, c0), X_seq)
    return outputs, h_final, c_final

def predict(params, X_seq):
    hidden_dim = params['W_f_h'].shape[0]
    h0 = jnp.zeros(hidden_dim)
    c0 = jnp.zeros(hidden_dim)

    outputs, h_final, c_final = lstm_forward(params, X_seq, h0, c0)
    Y_pred = h_final @ params['W_out'] + params['b_out']
    return Y_pred