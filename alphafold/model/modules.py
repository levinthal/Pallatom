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

import functools
from alphafold.common import residue_constants
from alphafold.model import common_modules
from alphafold.model import layer_stack
from alphafold.model import mapping
from alphafold.model import prng
from alphafold.model import utils
import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange
from haiku.initializers import VarianceScaling

# utils:
def stable_softmax(logits: jax.Array) -> jax.Array:
  """Numerically stable softmax for (potential) bfloat 16."""
  if logits.dtype == jnp.float32:
    output = jax.nn.softmax(logits)
  elif logits.dtype == jnp.bfloat16:
    # Need to explicitly do softmax in float32 to avoid numerical issues
    # with large negatives. Large negatives can occur if trying to mask
    # by adding on large negative logits so that things softmax to zero.
    output = jax.nn.softmax(logits.astype(jnp.float32)).astype(jnp.bfloat16)
  else:
    raise ValueError(f'Unexpected input dtype {logits.dtype}')
  return output


def matmul_single_slice(a, b):
    return jnp.tensordot(a, b, axes=1)
  
  
def segment_mean(data, segment_ids, num_segments):
    # Computes the sum within segments of an array.
    segment_sum = jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)
    segment_count = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments=num_segments)
    return segment_sum / (segment_count + 1e-5)
  

def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss)


def glorot_uniform():
  return hk.initializers.VarianceScaling(scale=1.0,
                                         mode='fan_avg',
                                         distribution='uniform')


def rbf_kernel(x, mu, sigma=0.4):
    """Radial basis function (RBF) kernel."""
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def rbf_transform(dist_matrix, num_centers=39, min_val=3.25, max_val=50.75, sigma=5.0, scale_val=1.0):
    """Apply RBF kernel transformation to a distance matrix with different centers."""
    # Create an array of RBF centers
    centers = jnp.linspace(min_val, max_val, num_centers)
    # Apply the RBF kernel
    rbf_values = rbf_kernel(dist_matrix, centers/scale_val, sigma/scale_val)
    return rbf_values


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
  """Compute distogram from amino acid positions.

  Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.

  Returns:
    Distogram with the specified number of bins.
  """

  def squared_difference(x, y):
    return jnp.square(x - y)

  lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
  lower_breaks = jnp.square(lower_breaks)
  upper_breaks = jnp.concatenate([lower_breaks[1:],
                                  jnp.array([1e8], dtype=jnp.float32)], axis=-1)
  dist2 = jnp.sum(
      squared_difference(
          jnp.expand_dims(positions, axis=-2),
          jnp.expand_dims(positions, axis=-3)),
      axis=-1, keepdims=True)

  dgram = ((dist2 > lower_breaks).astype(jnp.float32) *
           (dist2 <= upper_breaks).astype(jnp.float32))
  return dgram


def diag_zero_distogram(distogram):
    n = distogram.shape[0]
    distogram *= (1-jax.numpy.identity(n))[...,None] 
    return distogram
  

def dgram_from_dsgram(dsgram, dgram_mask, min_bin=3.25, max_bin=50.75, num_bins=39, dtype=jnp.bfloat16):
    lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
    lower_breaks = jnp.square(lower_breaks)
    upper_breaks = jnp.concatenate([lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1)
    dist2 = jnp.square(dsgram)
    dgram = ((dist2 > lower_breaks).astype(dtype) * (dist2 <= upper_breaks).astype(dtype))
    dgram = diag_zero_distogram(dgram)
    return (dgram * dgram_mask).astype(dtype) 
  

def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
  """Applies dropout to a tensor."""
  if is_training and rate != 0.0:
    shape = list(tensor.shape)
    if broadcast_dim is not None:
      shape[broadcast_dim] = 1
    keep_rate = 1.0 - rate
    keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
    return keep * tensor / keep_rate
  else:
    return tensor

def dropout_wrapper(module,
                    input_act,
                    mask,
                    safe_key,
                    global_config,
                    output_act=None,
                    is_training=True,
                    **kwargs):
  """Applies module + dropout + residual update."""
  if output_act is None:
    output_act = input_act

  gc = global_config
  residual = module(input_act, mask, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

  # Will override `is_training` to True if want to use dropout.
  should_apply_dropout = True if gc.eval_dropout else is_training

  if module.config.shared_dropout:
    if module.config.orientation == 'per_row':
      broadcast_dim = 0
    else:
      broadcast_dim = 1
  else:
    broadcast_dim = None

  residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=dropout_rate,
                           is_training=should_apply_dropout,
                           broadcast_dim=broadcast_dim)

  new_act = output_act + residual

  return new_act

def dropout_wrapper_with_bias(module,
                    input_act,
                    bias_act,
                    mask,
                    safe_key,
                    global_config,
                    output_act=None,
                    is_training=True,
                    **kwargs):
  """Applies module + dropout + residual update."""
  if output_act is None:
    output_act = input_act

  gc = global_config
  residual = module(input_act, bias_act, mask, is_training=is_training, **kwargs)
  dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

  # Will override `is_training` to True if want to use dropout.
  should_apply_dropout = True if gc.eval_dropout else is_training

  if module.config.shared_dropout:
    if module.config.orientation == 'per_row':
      broadcast_dim = 0
    else:
      broadcast_dim = 1
  else:
    broadcast_dim = None

  residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=dropout_rate,
                           is_training=should_apply_dropout,
                           broadcast_dim=broadcast_dim)

  new_act = output_act + residual

  return new_act

def _layer_norm(axis=-1, name='layer_norm'):
  return common_modules.LayerNorm(
      axis=axis,
      create_scale=True,
      create_offset=True,
      eps=1e-5,
      use_fast_variance=True,
      scale_init=hk.initializers.Constant(1.),
      offset_init=hk.initializers.Constant(0.),
      param_axis=axis,
      name=name)


class TemplatePairStack(hk.Module):
  """Pair stack for the templates.

  Jumper et al. (2021) Suppl. Alg. 16 "TemplatePairStack"
  """

  def __init__(self, config, global_config, name='template_pair_stack'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, pair_mask, is_training, safe_key=None):
    """Builds TemplatePairStack module.

    Arguments:
      pair_act: Pair activations for single template, shape [N_res, N_res, c_t].
      pair_mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: Safe key object encapsulating the random number generation key.

    Returns:
      Updated pair_act, shape [N_res, N_res, c_t].
    """

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    gc = self.global_config
    c = self.config

    if not c.num_block:
      return pair_act

    def block(x):
      """One block of the template pair stack."""
      pair_act, safe_key = x

      dropout_wrapper_fn = functools.partial(
          dropout_wrapper, is_training=is_training, global_config=gc)

      safe_key, *sub_keys = safe_key.split(6)
      sub_keys = iter(sub_keys)

      pair_act = dropout_wrapper_fn(
          TriangleAttention(c.triangle_attention_starting_node, gc,
                            name='triangle_attention_starting_node'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          TriangleAttention(c.triangle_attention_ending_node, gc,
                            name='triangle_attention_ending_node'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                 name='triangle_multiplication_outgoing'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                 name='triangle_multiplication_incoming'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          Transition(c.pair_transition, gc, name='pair_transition'),
          pair_act,
          pair_mask,
          next(sub_keys))

      return pair_act, safe_key

    res_stack = layer_stack.layer_stack(c.num_block)(block)
    pair_act, safe_key = res_stack((pair_act, safe_key))
    return pair_act

class Transition(hk.Module):
    """Algorithm 11 Transition layer using swish"""

    def __init__(self, config, global_config, n=4, name='transition_block'):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config
        self.n = n if n is not None else self.config.num_intermediate_factor

    def transition_module(self, act, num_intermediate):
        # Linear transformations
        a = common_modules.Linear(num_intermediate, initializer='linear', use_bias=False, name='transition1')(act)
        b = common_modules.Linear(num_intermediate, initializer='linear', use_bias=False, name='transition2')(act)
        
        # Combine and project back to original size
        x = common_modules.Linear(act.shape[-1], initializer='linear', use_bias=False, name='transition3')(jax.nn.swish(a) * b)
        return x

    def __call__(self, act, mask, is_training):
        """
        Applies the transition layer.

        Arguments:
          act: Tensor of shape [batch_size, N_res, N_channel], input activations.
          mask: Tensor of shape [batch_size, N_res], input mask.
          is_training: bool, whether the module is in training mode.

        Returns:
          Tensor of shape [batch_size, N_res, N_channel], output activations.
        """
        num_intermediate = int(act.shape[-1] * self.n)
        mask = jnp.expand_dims(mask, axis=-1)
        
        # Apply mask to activations
        act *= mask

        # Layer normalization
        act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='input_layer_norm')(act)
        
        act = mapping.inference_subbatch(
            self.transition_module,
            self.global_config.subbatch_size,
            batched_args=[act, num_intermediate],
            nonbatched_args=[],
            low_memory=False)
        
        return act
      
class AdaLN(hk.Module):
    '''
    Algorithm 26 Adaptive LayerNorm
    '''
    def __init__(self, name='adaptive_layernorm'):
        super().__init__(name=name)

    def __call__(self, a, s):
        # Layer normalization on input tensor `a` without scale and offset
        a = common_modules.LayerNorm(axis=[-1], create_scale=False, create_offset=False, name='norm1')(a)
        
        # Layer normalization on condition embedding `s` with scale but without offset
        s = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=False, name='norm2')(s)
        
        # Linear transformation with sigmoid activation on `s`
        a = jax.nn.sigmoid(common_modules.Linear(s.shape[-1], name='linear1')(s)) * a \
            + common_modules.Linear(s.shape[-1], use_bias=False, name='linear2')(s) 
        
        return a

class ConditionedTransitionBlock(hk.Module):
    '''
    Algorithm 25 Conditioned Transition Block:
    Fusion of a single condition embedding and current single representation using a 
    SwiGLU Transition block with adaptive layer normalization (AdaLN).
    '''
    
    def __init__(self, name='condition_transition'):
        '''Initializes the ConditionedTransitionBlock with a name.'''
        super().__init__(name=name)

    def __call__(self, a, s, n=2):
        # Adaptive layer normalization
        adaln = AdaLN(name='condition_adaln')
        a = adaln(a, s)
        
        # SwiGLU transformation
        b = jax.nn.swish(
            common_modules.Linear(int(a.shape[-1] * n), use_bias=False, name='linear1')(a)
        ) * common_modules.Linear(int(a.shape[-1] * n), use_bias=False, name='linear2')(a)
        
        # Output projection
        a = jax.nn.sigmoid(
            common_modules.Linear(a.shape[-1], bias_init=-2.0, name='linear3')(s)
        ) * common_modules.Linear(a.shape[-1], use_bias=False, name='linear4')(b)
        
        return a

class SingleTemplateEmbedding(hk.Module):
  """Embeds a single template.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9+11
  """

  def __init__(self, config, global_config, name='single_template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, zij, batch, mask_2d, is_training):
    """Build the single template embedding.
    """
    assert mask_2d.dtype == zij.dtype
    dtype = zij.dtype
    num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
    template_mask_2d = batch['template_cb_atom_mask'][:, None] * batch['template_cb_atom_mask'][None, :]  
    template_mask_2d = template_mask_2d.astype(dtype)

    # template_distogram: using batch feature here.
    to_concat = [batch['template_distogram'].astype(dtype)]               

    # add adj_block features:
    to_concat.append(batch['template_adj_feat'].astype(jnp.int32))        
    
    # add time embedding 2d:
    time_embedding2d = batch['template_time_embedding_2d'].astype(dtype)  
    to_concat.append(time_embedding2d)                                   
    
    # template 2d mask:
    to_concat.append(template_mask_2d[:, :, None])                       
    
    # to activation:
    act = jnp.concatenate(to_concat, axis=-1)              
    act *= template_mask_2d[..., None].astype(dtype)       

    # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 9
    vij = common_modules.Linear(num_channels, initializer='relu', name='embedding2d', use_bias=False)(act)
    raw_embedding = vij
    
    # AF3: zij
    zij = common_modules.LayerNorm([-1], True, True, name='zij_norm')(zij)
    zij = common_modules.Linear(num_channels, initializer='linear', name='embedding_zij', use_bias=False)(zij)
    vij += zij
    
    # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 11
    vij = TemplatePairStack(self.config.template_pair_stack, self.global_config)(vij, mask_2d, is_training)
    uij = common_modules.LayerNorm([-1], True, True, name='output_layer_norm')(vij)
    return uij, raw_embedding

class TemplateEmbedding(hk.Module):
  """Embeds a set of templates.
  Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
  Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
  AF3: Algorithm 16 Template embedder
  """

  def __init__(self, config, global_config, name='template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_batch, mask_2d, is_training):
    """Build TemplateEmbedding module.

    Arguments:
      query_embedding: Query pair representation, shape [N_res, N_res, c_z]. pair_repr
      template_batch: A batch of template features.
      mask_2d: Padding mask (Note: this doesn't care if a template exists,
        unlike the template_pseudo_beta_mask).
      is_training: Whether the module is in training mode.

    Returns:
      A template embedding [N_res, N_res, c_z].
    """
    dtype = query_embedding.dtype
    template_mask = template_batch['template_mask']
    template_mask = template_mask.astype(dtype)
    
    # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
    template_embedder = SingleTemplateEmbedding(self.config, self.global_config)
    def map_fn(batch):
      return template_embedder(query_embedding, batch, mask_2d, is_training)
    embedding, raw_embedding = mapping.sharded_map(map_fn, in_axes=0)(template_batch) 

    embedding = jnp.mean(embedding, axis=0)
    embedding = common_modules.Linear(self.config.pair_channel,
                                    initializer="linear",
                                    name='template_projection',
                                    use_bias=False)(jax.nn.relu(embedding))

    # No gradients if no templates.
    embedding *= (jnp.sum(template_mask) > 0.).astype(embedding.dtype)

    return embedding, raw_embedding

class TriangleAttention(hk.Module):
  """Triangle Attention.

  Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
  Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
  """

  def __init__(self, config, global_config, name='triangle_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, pair_mask, is_training=False):
    """Builds TriangleAttention module.

    Arguments:
      pair_act: [N_res, N_res, c_z] pair activations tensor
      pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    c = self.config

    assert len(pair_act.shape) == 3
    assert len(pair_mask.shape) == 2
    assert c.orientation in ['per_row', 'per_column']

    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)
      pair_mask = jnp.swapaxes(pair_mask, -1, -2)

    bias = (1e9 * (pair_mask - 1.))[:, None, None, :]  
    assert len(bias.shape) == 4

    pair_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=pair_act.dtype,
        init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)

    attn_mod = Attention(
        c, self.global_config, pair_act.shape[-1])
    pair_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[pair_act, pair_act, bias],
        nonbatched_args=[nonbatched_bias],
        low_memory=not is_training)

    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)

    return pair_act

class TriangleMultiplication(hk.Module):
  """Triangle multiplication layer ("outgoing" or "incoming").

  Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
  Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
  """

  def __init__(self, config, global_config, name='triangle_multiplication'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, left_act, left_mask, is_training=False):
    """Builds TriangleMultiplication module.

    Arguments:
      left_act: Pair activations, shape [N_res, N_res, c_z]
      left_mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Outputs, same shape/type as left_act.
    """
    del is_training

    if self.config.fuse_projection_weights:
      return self._fused_triangle_multiplication(left_act, left_mask)
    else:
      return self._triangle_multiplication(left_act, left_mask)

  @hk.transparent
  def _triangle_multiplication(self, left_act, left_mask):
    """Implementation of TriangleMultiplication used in AF2 and AF-M<2.3."""
    c = self.config
    gc = self.global_config

    mask = left_mask[..., None]

    act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True,
                       name='layer_norm_input')(left_act)
    input_act = act

    left_projection = common_modules.Linear(
        c.num_intermediate_channel, use_bias=False,
        name='left_projection')
    left_proj_act = mask * left_projection(act)

    right_projection = common_modules.Linear(
        c.num_intermediate_channel, use_bias=False,
        name='right_projection')
    right_proj_act = mask * right_projection(act)

    left_gate_values = jax.nn.sigmoid(common_modules.Linear(
        c.num_intermediate_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        use_bias=False,
        name='left_gate')(act))

    right_gate_values = jax.nn.sigmoid(common_modules.Linear(
        c.num_intermediate_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        use_bias=False,
        name='right_gate')(act))

    left_proj_act *= left_gate_values
    right_proj_act *= right_gate_values

    # "Outgoing" edges equation: 'ikc,jkc->ijc'
    # "Incoming" edges equation: 'kjc,kic->ijc'
    # Note on the Suppl. Alg. 11 & 12 notation:
    # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
    # For the "incoming" edges, it's swapped:
    #   b = left_proj_act and a = right_proj_act
    act = jnp.einsum(c.equation, left_proj_act, right_proj_act)

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='center_layer_norm')(
            act)

    output_channel = int(input_act.shape[-1])

    act = common_modules.Linear(
        output_channel,
        initializer=utils.final_init(gc),
        use_bias=False,
        name='output_projection')(act)

    gate_values = jax.nn.sigmoid(common_modules.Linear(
        output_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        use_bias=False,
        name='gating_linear')(input_act))
    act *= gate_values

    return act

  @hk.transparent
  def _fused_triangle_multiplication(self, left_act, left_mask):
    """TriangleMultiplication with fused projection weights."""
    mask = left_mask[..., None]
    c = self.config
    gc = self.global_config

    left_act = _layer_norm(axis=-1, name='left_norm_input')(left_act)

    # Both left and right projections are fused into projection.
    projection = common_modules.Linear(
        2*c.num_intermediate_channel, name='projection', use_bias=False)
    proj_act = mask * projection(left_act)

    # Both left + right gate are fused into gate_values.
    gate_values = common_modules.Linear(
        2 * c.num_intermediate_channel,
        use_bias=False,
        name='gate',
        bias_init=1.,
        initializer=utils.final_init(gc))(left_act)
    proj_act *= jax.nn.sigmoid(gate_values)

    left_proj_act = proj_act[:, :, :c.num_intermediate_channel]
    right_proj_act = proj_act[:, :, c.num_intermediate_channel:]
    act = jnp.einsum(c.equation, left_proj_act, right_proj_act)

    act = _layer_norm(axis=-1, name='center_norm')(act)

    output_channel = int(left_act.shape[-1])

    act = common_modules.Linear(
        output_channel,
        use_bias=False,
        initializer=utils.final_init(gc),
        name='output_projection')(act)

    gate_values = common_modules.Linear(
        output_channel,
        bias_init=1.,
        use_bias=False,
        initializer=utils.final_init(gc),
        name='gating_linear')(left_act)
    act *= jax.nn.sigmoid(gate_values)

    return act

class DistogramHead(hk.Module):
  """Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self, config, global_config, name='distogram_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, representations, batch, is_training):
    """Builds DistogramHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = common_modules.Linear(
        self.config.num_bins,
        initializer=utils.final_init(self.global_config),
        name='half_logits')(representations['pair'])

    # symm.
    logits = half_logits + jnp.swapaxes(half_logits, -2, -3) 

    # return bins.
    breaks = jnp.linspace(self.config.first_break, self.config.last_break, self.config.num_bins - 1)

    return dict(logits=logits, bin_edges=breaks)

class TriangleMulBlock(hk.Module):
  """Single iteration (block) of TriangleMulBlock.
  """

  def __init__(self, config, global_config, name='triangle_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, pair_mask, is_training=True, safe_key=None):
    """Builds TriangleMulBlock module.

    Arguments:
      activations: Dictionary containing activations:
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    c = self.config
    gc = self.global_config
    
    if safe_key is None: safe_key = prng.SafeKey(hk.next_rng_key())

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=gc)

    safe_key, *sub_keys = safe_key.split(10)
    sub_keys = iter(sub_keys)

    # triangle_attention
    pair_act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_starting_node, gc,
                          name='triangle_attention_starting_node'),
        pair_act,
        pair_mask,
        next(sub_keys))
    pair_act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_ending_node, gc,
                          name='triangle_attention_ending_node'),
        pair_act,
        pair_mask,
        next(sub_keys))

    # triangle_multiplication
    pair_act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                               name='triangle_multiplication_outgoing'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    
    pair_act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                               name='triangle_multiplication_incoming'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    pair_act = dropout_wrapper_fn(
        Transition(c.pair_transition, gc, name='pair_transition'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))


    return pair_act

class AffineTriangleAttention(hk.Module):
  """Triangle Attention.

  Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
  Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
  """

  def __init__(self, config, global_config, name='affine_triangle_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, affine_act, pair_mask, is_training=False):
    """Builds TriangleAttention module.

    Arguments:
      pair_act: [N_res, N_res, c_z] pair activations tensor.
      affine_act: [N_res, N_res, c_z] affine activations tensor for bias.
      pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    c = self.config
    assert len(pair_act.shape) == 3
    assert len(pair_mask.shape) == 2
    assert c.orientation in ['per_row', 'per_column']

    # no input layernorm.
    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)
      pair_mask = jnp.swapaxes(pair_mask, -1, -2)

    # input pair norm:
    pair_act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='pair_layernorm')(pair_act)
    affine_act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='affine_act_layernorm')(affine_act)

    dtype = pair_act.dtype
    bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4
    
    TRUNCATED_NORMAL_STDDEV_FACTOR = .87962566103423978
    # init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=pair_act.dtype,
        init=hk.initializers.RandomNormal(stddev= 1.0 / TRUNCATED_NORMAL_STDDEV_FACTOR))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', affine_act, weights).astype(dtype)

    attn_mod = Attention(
        c, self.global_config, pair_act.shape[-1])
    pair_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[pair_act, pair_act, bias],
        nonbatched_args=[nonbatched_bias, None],
        low_memory=not is_training)

    if c.orientation == 'per_column': pair_act = jnp.swapaxes(pair_act, -2, -3)

    return pair_act

class AffineTriangleBlock(hk.Module):

  def __init__(self, config, global_config, name='affine_triangle_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, affine_act, pair_mask, is_training=False, safe_key=None):
    """Builds TriangleMulBlock module.

    Arguments:
      activations: Dictionary containing activations:
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    c = self.config
    gc = self.global_config

    if safe_key is None: safe_key = prng.SafeKey(hk.next_rng_key())

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper_with_bias,
        is_training=is_training,
        global_config=gc)

    safe_key, *sub_keys = safe_key.split(10)
    sub_keys = iter(sub_keys)
    
    # Triangle_Attention
    pair_act = dropout_wrapper_fn(
        AffineTriangleAttention(c.triangle_attention_starting_node, gc,
                          name='affine_triangle_attention_starting_node'),
        input_act=pair_act, 
        bias_act=affine_act,
        mask=pair_mask,
        safe_key=next(sub_keys))  

    pair_act = dropout_wrapper_fn(
        AffineTriangleAttention(c.triangle_attention_ending_node, gc,
                          name='affine_triangle_attention_ending_node'),
        input_act=pair_act, 
        bias_act=affine_act,
        mask=pair_mask,
        safe_key=next(sub_keys))  

    # Triangle_Multiplication
    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        is_training=is_training,
        global_config=gc)

    # Transition
    pair_act = dropout_wrapper_fn(
        Transition(c.pair_transition, gc, name='pair_transition'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))  
    
    return pair_act

class Attention(hk.Module):
  """Sliced Multihead attention."""

  def __init__(self, config, global_config, output_dim, name='slice_attention'):
    super().__init__(name=name)

    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim
  
  def __call__(self, q_data, m_data, padding_mask, nonbatched_bias=None, slice_idx=None):
    """Builds Attention module.
    """
    # Sensible default for when the config keys are missing
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    dtype = q_data.dtype
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head

    if self.config.use_q_bias:
      # q has bias
      q_weights = hk.get_parameter(
          'query_w', shape=(q_data.shape[-1], num_head, key_dim),
          dtype=q_data.dtype,
          init=glorot_uniform())
      b_weights = hk.get_parameter(
          'query_b', shape=(num_head, key_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      q = (jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) + b_weights) * key_dim**(-0.5)
      
    else:
      # q has no-bias
      q_weights = hk.get_parameter(
          'query_w', shape=(q_data.shape[-1], num_head, key_dim),
          dtype=q_data.dtype,
          init=glorot_uniform())
      q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], num_head, value_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    
    if slice_idx is not None:
      if nonbatched_bias is not None:
        pair_bias = jnp.expand_dims(nonbatched_bias, axis=0).astype(dtype) 
      else:
        pair_bias = jnp.zeros_like(padding_mask).astype(dtype)
        
      q_indices, k_indices, local_mask = slice_idx
      logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) + padding_mask + local_mask + pair_bias
    else:
      logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) + padding_mask 

      if nonbatched_bias is not None:
        logits += jnp.expand_dims(nonbatched_bias, axis=0) 

    weights = stable_softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)  

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      gate_values = jnp.einsum('bqc, chv->bqhv', q_data, gating_weights)
      gate_values = jax.nn.sigmoid(gate_values)
      weighted_avg *= gate_values

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        dtype=q_data.dtype,
        init=init)

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) 

    return output

class AttentionPairBias(hk.Module):
  '''
  Algorithm 24 DiffusionAttention with pair bias and mask
  '''
  def __init__(self, config, global_config, name='attention_pair_bias'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,a, a_mask, z, s=None, slice_idx=None):
    c = self.config
    assert len(z.shape) == 3
    assert len(a.shape) == 3
    assert len(a_mask.shape) == 2
    adaln = AdaLN(name='a_ada_layernorm')
    
    # input LN:
    if s is not None: a = adaln(a, s)
    else: a = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='a_layer_norm')(a)
    z = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='z_layer_norm')(z) # LN
    
    # l mask:
    padding_mask = (1e9 * (a_mask - 1.))[:, None, None, :] 
    assert len(padding_mask.shape) == 4

    # Input projections:
    TRUNCATED_NORMAL_STDDEV_FACTOR = .87962566103423978
    
    dtype = z.dtype
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(z.shape[-1], c.num_head),
        dtype=z.dtype,
        init=hk.initializers.RandomNormal(stddev= 1.0 / TRUNCATED_NORMAL_STDDEV_FACTOR))
    
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', z, weights).astype(dtype) 

    attn_mod = Attention(c, self.global_config, output_dim=a.shape[-1])
    if slice_idx is not None:
      a = mapping.inference_subbatch(
          attn_mod,
          self.global_config.subbatch_size,
          batched_args=[a, a, padding_mask],
          nonbatched_args=[nonbatched_bias, slice_idx],
          low_memory=False)
    else:
      a = mapping.inference_subbatch(
          attn_mod,
          self.global_config.subbatch_size,
          batched_args=[a, a, padding_mask],
          nonbatched_args=[nonbatched_bias, None],
          low_memory=False)

    if s is not None:
      a = jax.nn.sigmoid(
        common_modules.Linear(a.shape[-1], bias_init=-2.0, name='sigmoid_linear')(s)
        ) * a  
      
    return a

class SingleAttentionWithPairBias(hk.Module):
  """MSA per-row attention biased by the pair representation.

  Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
  """

  def __init__(self, config, global_config,
               name='msa_row_attention_with_pair_bias'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act,
               msa_mask,
               pair_act,
               is_training=False):
    """Builds MSARowAttentionWithPairBias module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      pair_act: [N_res, N_res, c_z] pair representation.
      is_training: Whether the module is in training mode.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2

    mask = msa_mask[:, None, None, :]
    assert len(mask.shape) == 4

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    pair_act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='feat_2d_norm')(
            pair_act)

    TRUNCATED_NORMAL_STDDEV_FACTOR = .87962566103423978
    dtype = pair_act.dtype
    # init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=pair_act.dtype,
        init=hk.initializers.RandomNormal(stddev= 1.0 / TRUNCATED_NORMAL_STDDEV_FACTOR))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights).astype(dtype)

    attn_mod = Attention(
        c, self.global_config, msa_act.shape[-1])
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, mask],
        nonbatched_args=[nonbatched_bias],
        low_memory=not is_training)

    return msa_act
  
  
class NodeUpdate(hk.Module):

  def __init__(self, config, global_config, name='node_update'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, a, ti, pair, a_masks, safe_key, is_training=False):
    c = self.config
    gc = self.global_config

    safe_key, *sub_keys = safe_key.split(5)
    sub_keys = iter(sub_keys)

    # define class:
    atten_pair_bias = AttentionPairBias(self.config.row_attention_with_pair_bias, gc)
    residual = atten_pair_bias(a[None], a_masks, pair, ti[None], slice_idx=None)
    should_apply_dropout = True if gc.eval_dropout else is_training
    residual = apply_dropout(tensor=residual,
                            safe_key=next(sub_keys),
                            rate=c.row_attention_with_pair_bias.dropout_rate,
                            is_training=should_apply_dropout,
                            broadcast_dim=0) # per-row dropout;
    a += residual[0]  

    # row_transition: 
    dropout_wrapper_fn = functools.partial(dropout_wrapper,
        is_training=is_training, global_config=gc)
       
    a = dropout_wrapper_fn(
        Transition(c.row_transition, gc, name='row_transition'),
        a[None],
        a_masks,
        safe_key=next(sub_keys))

    return a[0], residual[0] 


class DiffusionTransformer(hk.Module):
  '''
  Algorithm 23 Diffusion Transformer
  '''
  def __init__(self, config, global_config, num_block=3, name='diffusion_transformer'):
    super().__init__(name=name)
    self.config = config
    self.gc = global_config
    self.num_block = num_block

  def __call__(self, a, s, z, a_mask, slice_idx=None):
    # define class:
    atten_pair_bias = AttentionPairBias(self.config, self.gc)
    condition_transition = ConditionedTransitionBlock()
    
    def stack_fn(x):
      a, a_mask, z, s, slice_idx = x
      b = atten_pair_bias(a, a_mask, z, s=s, slice_idx=slice_idx) 
      a = b + condition_transition(a, s, n=2)
      return a, a_mask, z, s, slice_idx

    # run stack_blocks:
    _stack = layer_stack.layer_stack(self.num_block)(stack_fn)
    
    inputs = (a[None], a_mask, z, s[None], slice_idx) 
    outpus = _stack(inputs)
    return outpus[0][0]  

class AtomTransformer(hk.Module):
  '''
  Implements Algorithm 7 Atom Transformer from AlphaFold 3.
  '''
  def __init__(self, config, global_config, num_block=3, num_head=4, name='atom_transformer'):
    super().__init__(name=name)
    self.config = config
    self.gc = global_config
    self.num_block = num_block
    self.num_head = num_head

  def __call__(self, ql, cl, plm, masks):
    ql_mask = masks['atom']
    slice_idx = masks['slice_idx']

    ## Apply the diffusion transformer
    diff_transformer = DiffusionTransformer(self.config.atom_transformer, self.gc, self.num_block)
    ql = diff_transformer(ql, cl, plm, ql_mask, slice_idx) 
    return ql


# Embedder:
class AtomFeatureEncoder(hk.Module):
  '''
  Based on Algorithm 5 Atom attention encoder from the AF3 algorithm:
  Used to extract and integrate features of ref-conformer and batch, 
  focusing only on the extraction of internal atomic information at the token level;
  reference-embedding.
  '''
  def __init__(self, config, global_config, name='ref_atom_attention_embedder'):
    super().__init__(name=name)
    self.config = config
    self.gc = global_config

  def __call__(self, batch, rl, s_trunk, z_trunk, masks):
    c = self.config
    
    # (that is simple ala ref-conformer)
    ref_space_uid = batch['ref_space_uid']
    ref_element = batch['ref_element']                                                 
    ref_pos = batch['ref_pos']                                                    
    ref_name_char = jnp.tile(jax.nn.one_hot(jnp.arange(14), 14)[None], (20, 1, 1))
    std_features_hstack = jnp.concatenate([ref_pos,ref_element,ref_name_char],-1)
    
    pseudo_seq = batch['pseudo_seq']   # all ALA
    batched_matmul = jax.vmap(matmul_single_slice, in_axes=(0, None))
    features_hstack = batched_matmul(pseudo_seq, std_features_hstack).reshape(-1, std_features_hstack.shape[-1])  
    _ref_pos = batched_matmul(pseudo_seq, ref_pos).reshape(-1, ref_pos.shape[-1])                                 
    
    cl = common_modules.Linear(c.atom_channel, name='atom_embed', use_bias=False)(features_hstack)   
    ref_cl = cl

    mask_atom = (batch['seq_mask'][ref_space_uid])      
    atom_mask_2d = (mask_atom[None] * mask_atom[:,None])

    vlm = (jnp.equal(ref_space_uid[:, None], ref_space_uid[None, :]).astype(cl.dtype) * atom_mask_2d)[...,None]    
    dlm = (_ref_pos[None] - _ref_pos[:,None,:]).astype(cl.dtype) * atom_mask_2d[...,None]              
    plm = common_modules.Linear(c.atompair_channel, name='atom_pair_embed', use_bias=False)(dlm)*vlm   
    
    dist_sq = 1/(1+jnp.linalg.norm(plm+1e-6, axis=-1)**2)
    plm += common_modules.Linear(c.atompair_channel, name='inverse_squared_embed', use_bias=False)(dist_sq[...,None]) * vlm  
    plm += common_modules.Linear(c.atompair_channel, name='valid_mask', use_bias=False)(vlm) *vlm                

    ql = cl

    left_cl = common_modules.Linear(c.atompair_channel, name='left_cl', use_bias=False)(jax.nn.relu(cl))    
    right_cl = common_modules.Linear(c.atompair_channel, name='right_cl', use_bias=False)(jax.nn.relu(cl))  
    plm += left_cl[None] + right_cl[:,None,:]  

    # # assign ref_plm.
    ref_plm = plm
    
    # Add the noisy positions to ql;
    rl_scaled = rl.astype(jnp.float32)/jnp.sqrt((batch['t_hat']**2).astype(jnp.float32) + self.config.r3_edmp.sigma_data**2).astype(jnp.float32)
    ql += common_modules.Linear(c.atom_channel, name='add_rl', use_bias=False)(rl_scaled.astype(cl.dtype))  
    
    # Add condition from single/pair:
    # Broadcast the single embedding to atom-level:
    cl += common_modules.Linear(c.atom_channel, name='add_s_trunk', use_bias=False)(
      common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='s_trunk_layer_norm')(s_trunk)
      )[ref_space_uid]  
    
    # Broadcast the pair embedding to atom-level:
    pair_plm = common_modules.Linear(c.atompair_channel, name='add_z_trunk', use_bias=False)(
      common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='z_trunk_layer_norm')(z_trunk)
      )[ref_space_uid[:, None], ref_space_uid[None, :]]    
    plm += pair_plm
    
    plm_0 = plm
    for _ in range(3):
      plm_0 = common_modules.Linear(c.atompair_channel, name='pair_mlp_activation', use_bias=False, initializer='relu')(jax.nn.relu(plm_0))
    plm += plm_0

    # Cross attention transformer on {ql}, {cl}, {plm};
    atom_transformer = AtomTransformer(self.config, self.gc, num_block=3)
    ql = atom_transformer(ql, cl, plm, masks)
    
    # Aggregate per-atom representation to per-token representation
    a = common_modules.Linear(c.token_channel, name='aggregate_atoms', use_bias=False, initializer='relu')(ql) 
    a = jax.nn.relu(a)  
    a = segment_mean(a, batch['ref_space_uid'], num_segments=batch['seq_mask'].shape[0]) 
    return a, ql, cl, plm, ref_plm, ref_cl

class RelposEmbedder(hk.Module):
  '''Algorithm 3 Relative position encoding"'''
  def __init__(self, config, name='edge_embedder'):
    super().__init__(name=name)
    self.config = config

  def __call__(self, pos):
    # Relative position encoding.
    c = self.config
    # Add one-hot-encoded clipped residue distances to the pair activations.
    offset = pos[:, None] - pos[None, :]
    rel_pos = jax.nn.one_hot(
        jnp.clip(offset + c.max_relative_feature, 
                 a_min=0, a_max=2 * c.max_relative_feature), 
        2 * c.max_relative_feature + 1)
    pair_act = common_modules.Linear(c.pair_channel, name='pair_activiations', use_bias=False)(rel_pos)
    return pair_act

class AtomAttentionDecoder(hk.Module):
  def __init__(self, config, global_config, num_block=3, name='atom_attention_decoder'):
    super().__init__(name=name)
    self.config = config
    self.gc = global_config
    self.num_block = num_block

  def __call__(self, rl, ql, plm, cl, s_trunk, z_trunk, ref_space_uid, masks, ref_cl):
    c = self.config
    # Broadcast per-token activiations to per-atom activations and add the skip connection
    ql_brocast = common_modules.Linear(c.atom_channel, name='broadcast_single_feature', use_bias=False)(
      common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='s_trunk_layer_norm')(s_trunk)
      )[ref_space_uid] # [Natom,128]

    ql += ql_brocast

    # Broadcast the pair embedding to atom-level:
    pair_plm = common_modules.Linear(c.atompair_channel, name='broadcast_pair_feature', use_bias=False)(
      common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='z_trunk_layer_norm')(z_trunk)
      )[ref_space_uid[:, None], ref_space_uid[None, :]]    
    
    # Run a small MLP on the atom-pair activations.
    plm += pair_plm
    plm_0 = plm
    for _ in range(3):
      plm_0 = common_modules.Linear(c.atompair_channel, name='pair_mlp_activation', use_bias=False, initializer='relu')(jax.nn.relu(plm_0))
    plm += plm_0

    # Cross attention transformer on {ql}, {cl};
    atom_transformer = AtomTransformer(self.config, self.gc, num_block=self.num_block) 
    ql = atom_transformer(ql, cl, plm, masks)
    
    # Map to positions update.
    rl_update = common_modules.Linear(c.r_cannel, name='update_rl', use_bias=False, initializer='zeros_like')(
      common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='ql_layernorm')(ql)
      )
    
    # update cl using ref_cl & a for next block:
    cl = ref_cl + common_modules.Linear(c.atom_channel, name='broadcast_single_feature2', use_bias=False)(
      common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='s_trunk_layer_norm2')(s_trunk)
      )[ref_space_uid]  
    
    return ql, plm, cl, rl_update, ql_brocast

class TimeFourierEmbedding(hk.Module):
  def __init__(self, config):
    super().__init__()
    self.c = config

  def __call__(self, t_hat):
    w_init = VarianceScaling(1.0, "fan_avg", "uniform")
    b_init = VarianceScaling(1.0, "fan_avg", "uniform")

    w = hk.get_parameter("w", shape=(t_hat.shape[-1], self.c.single_channel), dtype=t_hat.dtype, init=w_init)
    b = hk.get_parameter("b", shape=(self.c.single_channel,), dtype=t_hat.dtype, init=b_init)
    
    # embedding:
    _t_hat = 0.25*jnp.log(t_hat/self.c.r3_edmp.sigma_data)
    _t_hat = jnp.cos(2 * jnp.pi * (jnp.dot(t_hat, w) + b))
    
    # linear norm()
    _t_hat = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='t_norm')(_t_hat)
    _t_hat = common_modules.Linear(self.c.single_channel, name='t_linear', use_bias=False)(_t_hat)               

    return _t_hat 


# main trunk:
class MainTrunk(hk.Module):
  def __init__(self, config, global_config, is_training, name='main_trunk'):
    super().__init__(name=name)
    self.config = config
    self.gc = global_config
    self.is_training = is_training
  
  def pred_seq_from_ql(self, ql, ref_centre_uid, fix_size):
    pred_aafeat = segment_mean(
      jax.nn.relu(
        common_modules.Linear(self.config.atom_channel, name='ql_to_seqfeat')(
          common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='ql2seq_layernorm')(ql)
        )
      ), 
      ref_centre_uid, 
      fix_size
    )
    seq_logits = common_modules.Linear(self.config.num_letters, name='ql_pred_seq', use_bias=False, initializer='zeros_like')(pred_aafeat)
    return seq_logits
  

  def affine2triangle(self, pair_act, xyz_cb, pair_masks, safe_key, sigma_data):
    # affine2rbf
    dtype = pair_act.dtype

    # current dsm rbf:
    dsm = jnp.sqrt(jnp.square(xyz_cb[:,None,:] - xyz_cb[None,:,:]).sum(-1) + 1e-6)[...,None].astype(dtype) 
    rbf_dsm = rbf_transform(dsm, scale_val=sigma_data).astype(dtype)
    # encoder rl_dsm 
    affine_pair = common_modules.Linear(self.config.pair_channel, name='affine_project', use_bias=False)(rbf_dsm)      
    dgram = dgram_from_dsgram(dsm, pair_masks[...,None], min_bin=3.25, max_bin=50.75, num_bins=39, dtype=dtype)  

    # triangle update:
    safe_key, safe_subkey = safe_key.split()
    affine_triagnle = AffineTriangleBlock(self.config.triangle_block, self.gc)
    pair_act = affine_triagnle(pair_act, affine_pair, pair_masks, safe_key=safe_subkey, is_training=self.is_training)
    return pair_act, affine_pair, dgram


  def __call__(self, activations, masks, safe_key=None):
    if safe_key is None: safe_key = prng.SafeKey(hk.next_rng_key())
    a = activations['a']
    rl = activations['rl']
    rl_input = activations['rl_input']
    rl_updates = activations['rl_update']
    si = activations['si']
    cl = activations['cl']
    ti = activations['ti']
    zij = activations['zij']
    ref_ql = activations['ref_ql']
    ref_plm = activations['ref_plm']
    ref_cl = activations['ref_cl']
    ref_space_uid = activations['ref_space_uid']
    ref_centre_uid = masks['ref_center_mask']
    t_hat = activations['t_hat']
    sigma_data = self.config.r3_edmp.sigma_data
    dtype = si.dtype
    
    # define fns:
    atom_decoder = AtomAttentionDecoder(self.config, self.gc, num_block=1)
    node_update_module = NodeUpdate(self.config, self.gc, name='node_update')

    # update node with selfcond pair.
    safe_key, safe_subkey1, safe_subkey2  = safe_key.split(3)
    a, residual_a = node_update_module(a, ti, zij, masks['msa'], safe_subkey1, is_training=self.is_training)
    
    # deencode to atom features:
    ql, plm, cl, rl_update, ql_brocast = atom_decoder(rl, ref_ql, ref_plm, cl, a, zij, ref_space_uid, masks, ref_cl)
    
    # accumulate rl_updates between blocks:
    rl_updates += rl_update
    
    # update rl using rl_updates & rescaled:      
    rl = sigma_data**2/(sigma_data**2 + t_hat.astype(jnp.float32)**2) * rl_input + \
      sigma_data * t_hat.astype(jnp.float32)/jnp.sqrt(sigma_data**2 + t_hat.astype(jnp.float32)**2) * rl_updates  
    rl = rl.astype(ql.dtype)      

    # update pair repr using scaled:
    cb_xyz = segment_mean(rl, ref_centre_uid, a.shape[0])

    # update pair using updated rl;
    safe_key, safe_subkey = safe_key.split()
    zij, affine_pair, dgram = self.affine2triangle(zij, cb_xyz, masks['pair'], safe_subkey, 1.0)  
    
    # pred seq from sequence:
    pred_seq_logits = self.pred_seq_from_ql(ql, ref_space_uid, a.shape[0]) 
    
    return {
        "a": a.astype(dtype),                              
        "si": si.astype(dtype),                       
        "ti": ti.astype(dtype),                       
        "zij": zij.astype(dtype),                     
        'rl': rl.astype(dtype),                       
        'ref_ql': ref_ql.astype(dtype),               
        'ref_plm': ref_plm.astype(dtype),             
        'ql': ql.astype(dtype),                       
        'plm': plm.astype(dtype),                     
        'cl': cl.astype(dtype),                       
        'ref_cl': ref_cl.astype(dtype),               
        'ref_space_uid': ref_space_uid,               
        't_hat': t_hat,
        'rl_update': rl_updates,
        'rl_input': rl_input,
      }, ql, plm, cl, a, si, rl, zij, affine_pair, dgram, ql_brocast, residual_a, pred_seq_logits


# main function:
class EmbeddingsAndDenoise(hk.Module):
  """Embeds the input data and runs denoise blocks."""

  def __init__(self, config, global_config, name='model'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, batch, is_training, safe_key=None):
    c = self.config
    gc = self.global_config
    dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32
    
    # fns:
    main_iteration = MainTrunk(c, gc, is_training, name='main_iteration')
    relpos_embedder = RelposEmbedder(c, name='relpos_embedder')
    atom_feature_encoder = AtomFeatureEncoder(c, gc, name='atom_embedder')

    if safe_key is None: safe_key = prng.SafeKey(hk.next_rng_key())
    
    with utils.bfloat16_context():
      # masking;
      mask_1d = batch['seq_mask'][None].astype(dtype)                    
      mask_atom = (batch['seq_mask'][batch['ref_space_uid']])[None]      
      mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]  
      mask_2d = mask_2d.astype(dtype)

      # slice_idx
      slice_idx = (batch['q_indices'], batch['k_indices'], batch['local_mask'])

      # update mask:
      masks = {'msa': mask_1d, 'pair': mask_2d, 'atom': mask_atom, 
               'ref_center_mask': batch['ref_center_mask'], 'slice_idx': slice_idx}
      
      # init si
      s_init = jnp.concatenate([batch['residx_embedding'], batch['ss_feat']],axis=-1)
      si = common_modules.Linear(c.single_channel, name='single_activations', use_bias=False)(s_init.astype(dtype))                
      
      # time embedding:
      ti = TimeFourierEmbedding(self.config)(batch['time_embedding'][...,None]) 
      si += ti
      
      # 2d embedding:
      z_init = relpos_embedder(batch['residue_index']).astype(dtype)
      template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
      z_trunk, z_raw_embedding = TemplateEmbedding(c.template, gc)(z_init,template_batch,mask_2d,is_training=is_training)
      zij = z_init.astype(dtype) + z_trunk.astype(dtype)
      
      # current allatom positions:
      rl = batch['all_atom_positions'] 

      # AtomEncoder:
      a, ql, cl, plm, ref_plm, ref_cl = atom_feature_encoder(batch, rl, si, zij, masks)

      # stack main-trunk blocks:
      inputs = {
        "a": a.astype(dtype),                          
        "si": si.astype(dtype),                        
        "ti": ti.astype(dtype),                        
        "zij": zij.astype(dtype),                      
        'rl': jnp.zeros_like(rl).astype(dtype),        
        'ql': ql.astype(dtype),                        
        'ref_ql': ql.astype(dtype),                    
        'plm': plm.astype(dtype),                      
        'ref_plm': ref_plm.astype(dtype),              
        'cl': cl.astype(dtype),                        
        'ref_cl': ref_cl.astype(dtype),                
        'ref_space_uid': batch['ref_space_uid'],       
        't_hat': batch['t_hat'],                       
        'rl_update': jnp.zeros_like(rl).astype(dtype), 
        'rl_input': rl.astype(dtype),                  
      }
      
      # Main trunk of the score network      
      def main_fn(x):
        activations, safe_key = x
        safe_key, safe_subkey = safe_key.split()

        # single denoise layer:
        output, ql_acts, plm_acts, cl_acts, single_acts, si_acts, block_rl, pair_acts, affine_pair, dgram, ql_brocast, residual_a, pred_seq_logits = main_iteration(activations, masks, safe_key=safe_subkey)
        return (output, safe_key), (ql_acts, plm_acts, cl_acts, single_acts, si_acts, block_rl, pair_acts, affine_pair, dgram, ql_brocast, residual_a, pred_seq_logits)

      # run denoise blocks:
      num_block = 8
      main_stack = layer_stack.layer_stack(num_block, with_state=True)(main_fn) 
      (outputs, safe_key), (ql_acts, plm_acts, cl_acts, single_acts, si_acts, block_rl, pair_acts, affine_pair, dgram, ql_brocast, residual_a, pred_seq_logits) = main_stack((inputs, safe_key))            
      
      output = {
          'final_atom_positions': outputs['rl'],
          'final_atom_mask': mask_atom,
          'a': a,
          'ql': outputs['ql'],
          'cl': outputs['cl'],
          'plm': outputs['plm'],
          'ref_ql': ql,
          'ref_plm': ref_plm,
          'z_init': z_init,
          's_init': s_init,
          'si': si,
          'single': outputs['a'],
          'pair': outputs['zij'],
          'z_trunk': z_trunk,
          'raw_template': z_raw_embedding[0],
          'time_act': ti,
          'seq_logits': pred_seq_logits[-1],
          'states': {'ql':ql_acts, 'plm': plm_acts, 'cl': cl_acts, 'a':single_acts, 'rl':block_rl, 
                     'pair': pair_acts, 'affine_pair': affine_pair, 'dgram': dgram, 'si': si_acts,
                     'ql_brocast': ql_brocast, 'residual_a': residual_a, 'guide_seq': pred_seq_logits}
      }

    return output


class Pallatom(hk.Module):
  """
  Run the Pallatom model.
  
  Arguments:
    batch: Dictionary with inputs to the Pallatom model.
    is_training: Whether the system is in training or inference mode.
  """

  def __init__(self, config, name='pallatom'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(self,
               batch,
               is_training):

    def slice_batch(i):
      b = {k: v[i] for k, v in batch.items()}
      return b

    denoise_module = EmbeddingsAndDenoise(self.config.pallatom, self.global_config)
    batch0 = slice_batch(0)
    representations = denoise_module(batch0, is_training)
    
    # DistogramHead:
    dm_head = DistogramHead(self.config.heads.distogram, self.global_config, name='distogram_head')
    representations['distogram'] = dm_head(representations, batch0, is_training)
    
    return representations
