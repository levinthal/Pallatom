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

from pathlib import Path
import numpy as np
import jax.numpy as jnp
import os
from rdkit import Chem
from alphafold.common.residue_constants import (restype_atom37_mask,
                                                restype_order, restype_1to3,
                                                restype_name_to_atom14_names)
from alphafold.model.tf.data_transforms import make_atom14_masks
from alphafold.model.all_atom import atom37_to_atom14_np, atom14_to_atom37_np

# padding atoms postisions located at CA
restype_order_reverse = {idx: res for res, idx in restype_order.items()}
amino_acid_atom_indices = {}
for restype in restype_name_to_atom14_names:
    len_atoms = (np.asarray(restype_name_to_atom14_names[restype]) != '').sum()
    amino_acid_atom_indices[restype] = list(range(len_atoms)) + [999] * (14 - len_atoms)


current_directory = os.path.dirname(os.path.abspath(__file__))

restype3_sdf_file_path = {res: str(Path(f'{current_directory}/residues_sdf').joinpath(f'{res}_ideal.sdf')) for res in restype_name_to_atom14_names.keys()}


def generate_sdf_amino_acid_specific_ref_features(restype3, name_chars_dim=64):
    # read SDF files
    file_path = restype3_sdf_file_path[restype3]
    supplier = Chem.SDMolSupplier(file_path)
    molecule = supplier[0]
    specific_atom_indices = amino_acid_atom_indices[restype3]

    if molecule is None:
        raise ValueError(f"Failed to read molecule from SDF file: {file_path}")

    # get 3D positions
    conformer = molecule.GetConformer()

    coords = []
    ref_pos = []
    ref_element = []
    ref_charge = []
    ref_atom_name_chars = []
    ref_mask = []

    # Find the alpha carbon atom index and its position (assuming CA is at index 1 for simplicity)
    alpha_carbon_idx = specific_atom_indices[1]  # CA
    alpha_pos = conformer.GetAtomPosition(alpha_carbon_idx)

    # Iterate over the specific atoms and get their properties
    for idx, atom_idx in enumerate(specific_atom_indices):
        if atom_idx != 999:
            atom = molecule.GetAtomWithIdx(atom_idx)
            pos = conformer.GetAtomPosition(atom_idx)
            element_name = atom.GetSymbol()
            shifted_x = pos.x - alpha_pos.x
            shifted_y = pos.y - alpha_pos.y
            shifted_z = pos.z - alpha_pos.z
            coords.append((element_name, atom_idx, shifted_x, shifted_y, shifted_z))

            ref_pos.append([shifted_x, shifted_y, shifted_z])
            ref_element.append(atom.GetAtomicNum())

            ref_charge.append(atom.GetFormalCharge())
            atom_name = element_name + str(idx)

            # ref_atom_name_chars
            char_indices = np.array([ord(c) - 32 for c in atom_name.ljust(4)[:4]])
            ref_atom_name_chars_one_hot = np_one_hot(np.array(char_indices), name_chars_dim)
            ref_atom_name_chars.append(ref_atom_name_chars_one_hot)
            ref_mask.append(1)
            
        else:
            # padding:
            ref_pos.append([0.0, 0.0, 0.0])
            ref_element.append(0) # zeros is fake atom. which is aligned to Ca atom
            ref_charge.append(0)
            atom_name = 'V0'
            
            # ref_atom_name_chars
            char_indices = np.array([ord(c) - 32 for c in atom_name.ljust(4)[:4]])
            ref_atom_name_chars_one_hot = np_one_hot(np.array(char_indices), name_chars_dim)
            ref_atom_name_chars.append(ref_atom_name_chars_one_hot)
            ref_mask.append(0)

    return {
        "ref_pos": np.array(ref_pos).astype(np.float32),
        "ref_element": np.array(ref_element).astype(np.int32),
        "ref_charge": np.array(ref_charge).astype(np.float32),
        "ref_atom_name_chars": np.array(ref_atom_name_chars).astype(np.int32),
        "ref_mask": np.array(ref_mask).astype(np.int32),
    }


def get_entangled_ref_features(name_chars_dim=64):
    restypes = ['ALA']
    entangled_ref_features_batch = {}
    for aatype in restypes:
        entangled_ref_features = generate_sdf_amino_acid_specific_ref_features(aatype, name_chars_dim=name_chars_dim)
        for k in entangled_ref_features.keys():
            if k not in entangled_ref_features_batch.keys(): entangled_ref_features_batch[k] = entangled_ref_features[k][None]
            else: entangled_ref_features_batch[k] = np.concatenate([entangled_ref_features_batch[k],entangled_ref_features[k][None]], axis=0)
    return entangled_ref_features_batch


def genertate_entangled_ref_features(L, element_dim=128, name_chars_dim=64, dtype=jnp.float32):
    """
    aatype: protein.aatype, used restype_order {int: restype1}
    """
    entangled_ref_features = get_entangled_ref_features(name_chars_dim=name_chars_dim) # 21 aatype ref_feature.
    
    entangled_ref_features.update({'ref_pos': np.tile(entangled_ref_features['ref_pos'], (1, L, 1))})             # (E, L*14, 3)
    ref_element = np.array([np_one_hot(np.array(c), element_dim) for c in entangled_ref_features['ref_element']])
    entangled_ref_features.update({'ref_element': np.tile(ref_element, (1, L, 1))})                               # (E, L*14, element_dim=128)
    entangled_ref_features.update({'ref_charge': np.tile(entangled_ref_features['ref_charge'], (1, L))})          # (E, L*14)
    entangled_ref_features.update({'ref_atom_name_chars': np.tile(entangled_ref_features['ref_atom_name_chars'], (1, L, 1, 1))})  # (E, L*14, 4, 64)
    entangled_ref_features.update({'ref_mask': np.tile(entangled_ref_features['ref_mask'], (1, L))})              # (E, L*14)

    return entangled_ref_features


def np_one_hot(arr, num_classes):
    one_hot = np.zeros((len(arr), num_classes))
    one_hot[np.arange(len(arr)), arr] = 1
    return one_hot

    
def make_atom14_masks_np(aatype):
    """
    aatype: protein.aatype, array, (L,)    
    """    
    mask_dict = {}
    mask_dict['aatype'] = aatype

    mask_dict = make_atom14_masks(mask_dict)
    tf2np_keys = ['atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists']
    for k in tf2np_keys: mask_dict[k] = mask_dict[k].numpy()    
    return mask_dict
    

def atom14_to_atom37(aatype, all_atom_positions):
    mask_dict = make_atom14_masks_np(aatype)
    atom37_pos = atom14_to_atom37_np(all_atom_positions, mask_dict)
    return atom37_pos


def create_local_mask(q_idx, v_idx, Natom):
  mask = -jnp.inf * jnp.ones((Natom, Natom))
  for q, v in zip(q_idx, v_idx):
    mask = mask.at[q:q+32, v:v+128].set(0)
  return mask