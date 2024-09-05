from typing import Dict, Optional
from alphafold.model import utils
import numpy as np


def atom37_to_atom14_np(
    atom37_data: np.ndarray,  # (N, 37, ...)
    batch: Dict[str, np.ndarray]) -> np.ndarray:  # (N, 14, ...)
  """Convert atom14 to atom37 representation."""
  assert len(atom37_data.shape) in [2, 3]
  assert 'residx_atom14_to_atom37' in batch
  assert 'atom14_atom_exists' in batch

  atom14_data = utils.batched_gather_np(atom37_data,
                                     batch['residx_atom14_to_atom37'],
                                     batch_dims=1)
  if len(atom37_data.shape) == 2:
    atom14_data *= batch['atom14_atom_exists'].astype(atom14_data.dtype)
  elif len(atom37_data.shape) == 3:
    atom14_data *= batch['atom14_atom_exists'][:, :, None].astype(atom14_data.dtype)
  return atom14_data


def atom14_to_atom37_np(atom14_data: np.ndarray,  # (N, 14, ...)
                     batch: Dict[str, np.ndarray]
                    ) -> np.ndarray:  # (N, 37, ...)
  """Convert atom14 to atom37 representation."""
  assert len(atom14_data.shape) in [2, 3]
  assert 'residx_atom37_to_atom14' in batch
  assert 'atom37_atom_exists' in batch

  atom37_data = utils.batched_gather_np(atom14_data,
                                     batch['residx_atom37_to_atom14'],
                                     batch_dims=1)
  if len(atom14_data.shape) == 2:
    atom37_data *= batch['atom37_atom_exists']
  elif len(atom14_data.shape) == 3:
    atom37_data *= batch['atom37_atom_exists'][:, :, None].astype(atom37_data.dtype)
  return atom37_data









