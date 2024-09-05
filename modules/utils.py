# Copyright 2024 LEVINTHAL Biotechnology Co. Ltd
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (the "License");
# you may not use this work except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, this work is provided on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
from jax import random
import numpy as np
import jax

def rotation_vector_to_matrix(vectors):
    assert vectors.shape[1] == 3

    theta = jnp.linalg.norm(vectors, axis=1)

    small_theta = theta < 1e-6
    axis = jnp.where(small_theta[:, None], jnp.zeros_like(vectors), vectors / theta[:, None])

    axis = vectors / theta[:, None]
    
    cross_product_matrix = jnp.stack([
        jnp.stack([jnp.zeros_like(theta), -axis[:, 2], axis[:, 1]], axis=1),
        jnp.stack([axis[:, 2], jnp.zeros_like(theta), -axis[:, 0]], axis=1),
        jnp.stack([-axis[:, 1], axis[:, 0], jnp.zeros_like(theta)], axis=1)
    ], axis=1)
    outer_product = axis[:, :, None] * axis[:, None, :]

    a = jnp.cos(theta)[:, None, None]
    b = (1 - jnp.cos(theta))[:, None, None]
    c = jnp.sin(theta)[:, None, None]
    R = a * jnp.eye(3) + c * cross_product_matrix + b * outer_product

    identity = jnp.eye(3)
    R = jnp.where(small_theta[:, None, None], identity, a * identity + c * cross_product_matrix + b * outer_product)

    return R


def uniform_random_rotation(key, n_samples=1):
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, shape=(n_samples, 3)) 
    x /= jnp.linalg.norm(x, axis=-1, keepdims=True)   
    
    # sample and apply rotation
    angles = jax.random.uniform(key2, (n_samples,), minval=0.02, maxval=jnp.pi)
    rot_vecs = x * angles[:, None] 
    
    return rotation_vector_to_matrix(rot_vecs)


def centre_random_augmentation(xs, x_mask, x_center_mask, key, s_trans=1.0):
    '''
    xs: (Natom,3)
    x_mask: (Natom,)
    '''
    # Step 1: Center the coordinates by subtracting the mean
    mean_xs = jnp.sum(xs * (x_mask * x_center_mask)[...,None], axis=0) / ((x_mask * x_center_mask).sum()+1e-6)
    xs_centered = (xs - mean_xs) * x_mask[...,None]

    # Step 2: Apply a uniform random rotation
    R = uniform_random_rotation(key)[0]
    
    # Step 3: Generate a random translation vector
    key, subkey = random.split(key)
    t = s_trans * random.normal(subkey, (3,))

    # Step 4: Apply the rotation and translation to the centered coordinates
    xs_transformed = jnp.dot(xs_centered, R.T) + t
    xs_transformed *= x_mask[...,None]  
    return xs_transformed

