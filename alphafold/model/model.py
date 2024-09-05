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

"""Code for constructing the model."""
from typing import Any, Mapping, Optional, Union

from absl import logging
from alphafold.model import modules
import haiku as hk
import jax
import ml_collections
import numpy as np


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
               is_training=True):
    self.config = config
    self.params = params

    def _forward_fn(batch):
      model = modules.Pallatom(self.config.model)
      return model(
          batch,
          is_training=is_training)
    self.apply = jax.jit(hk.transform(_forward_fn).apply)
