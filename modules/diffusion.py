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
import jax
import numpy as np


class EDM(object):
    """A class for adding Gaussian noise."""
    def __init__(self, num_step, shape, sigma_data=16.0, psigma_mean=-1.2, psigma_std=1.5):
        self.num_step = num_step
        self.shape = shape
        self.sigma_data = sigma_data
        self.psigma_mean = psigma_mean
        self.psigma_std = psigma_std
        self.noise_schedule = self.get_noise_level_from_float(jnp.linspace(0.001, 0.999999, num_step))
        

    def get_noise_level_from_float(self, t):
        # t~[0,1)
        noise_level = self.sigma_data * jnp.exp(self.psigma_mean + self.psigma_std * jnp.asarray(jax.scipy.stats.norm.ppf(t)))
        return noise_level
    
    def forward(self, x0, t, key, mask=None):
        z = jax.random.normal(key, shape=self.shape) # ~N(0,I3)
        xt = x0 + z * self.get_noise_level_from_float(t)

        # Apply masking if provided
        if mask is not None: xt = jnp.where(mask, xt, x0)   
        return xt
    
    def add_additional_noise(self, xt, t, key, mask=None, noise_scale=1.003,
                             gamma=0.4, t_min=0.25, t_max=1.0):
        assert 0.0 <= t_min <= 1.0
        assert 0.0 <= t_max <= 1.0
        assert t_min < t_max
        
        # sampling guassian noise:
        z = jax.random.normal(key, shape=self.shape)

        # get sigma_t & gama:
        sigma_in = self.get_noise_level_from_float(t)  
        sigma_next = self.get_noise_level_from_float(t-(1/self.num_step)) 
        sigma_next = jnp.where(t-(1/self.num_step)>0, sigma_next, 0.0)
        gamma = jnp.where(t <= t_max, gamma, 0.0)
        gamma = jnp.where(t >= t_min, gamma, 0.0)
        
        # Increase noise temporarily.
        t_hat = sigma_in * (1+gamma+1e-6)
        e = noise_scale * jnp.sqrt(t_hat**2-sigma_in**2) * z
        xt_noisy = xt + e
        if mask is not None: xt_noisy = jnp.where(mask, xt_noisy, xt)
        return xt_noisy, t_hat, sigma_next

    def reverse(self, x0, xt_noisy, t_hat, sigma_next, mask=None, step_scale=1.5):
        """
        adapted from "StochasticSampler"
        t: ~(1, self.T) 0=low noise, 1=high noise;
        """
        # ODE step:
        score = (xt_noisy - x0) / t_hat                   
        step = score * step_scale * (sigma_next - t_hat)  

        # get x_t_minus_1:
        new_xt = xt_noisy + step

        # Apply masking if provided
        if mask is not None: new_xt = jnp.where(mask, new_xt, x0)   
        return new_xt


def get_index_embedding(indices, embed_size, max_len=2056):
  """Creates sine / cosine positional embeddings from a prespecified indices.

  Args:
      indices: offsets of size [..., N_edges] of type integer
      max_len: maximum length.
      embed_size: dimension of the embeddings to create

  Returns:
      positional embedding of shape [N, embed_size]
  """
  K = jnp.arange(embed_size//2)
  pos_embedding_sin = jnp.sin(
      indices[..., None] * jnp.pi / (max_len**(2*K[None]/embed_size)))
  pos_embedding_cos = jnp.cos(
      indices[..., None] * jnp.pi / (max_len**(2*K[None]/embed_size)))
  pos_embedding = jnp.concatenate([
      pos_embedding_sin, pos_embedding_cos], axis=-1)
  return pos_embedding
