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

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import dynamic_update_slice
from alphafold.model.modules import dgram_from_positions, segment_mean
from Bio import PDB, BiopythonWarning
from warnings import simplefilter
pdb_io = PDB.PDBIO()
pdb = PDB.PDBParser()
simplefilter('ignore', BiopythonWarning)


def create_null_batch(token_size=128, atom_size=1792, N_template=1, dtype=jnp.bfloat16):
    '''
    create a batch used by alphafold.model().
    '''
    batch = {
        # sequence:
        "aatype": jnp.zeros([1, token_size]).astype(np.int32),                                   
        "restype": jnp.zeros([1, token_size, 32]).astype(np.int32),

        # data mask:
        "seq_mask": jnp.zeros([1, token_size]).astype(dtype),                                    
        "motif_mask": jnp.zeros([1,token_size]).astype(dtype),                                   
        "noise_mask": jnp.zeros([1,token_size]).astype(dtype),
        "seq_mask_2d": jnp.zeros([1,N_template,token_size,token_size,1]).astype(dtype),          
        "seq_atom_mask": jnp.zeros([1, atom_size]).astype(dtype),                                
        
        # index feature:
        "token_index": jnp.zeros([1,token_size]).astype(np.int32),
        "residue_index": jnp.zeros([1, token_size]).astype(np.int32),                            
        "residx_embedding": jnp.zeros([1,token_size,32]).astype(dtype),                          
        "q_indices": jnp.zeros([1,int(atom_size/32)]).astype(np.int32),
        "k_indices": jnp.zeros([1,int(atom_size/32)]).astype(np.int32),
        "local_mask": jnp.zeros([1, atom_size, atom_size]).astype(np.int32),
        
        # ref feature:
        "ref_pos": jnp.zeros([1, 20, 14, 3]).astype(dtype),
        "ref_mask": jnp.zeros([1, 20, 14]).astype(np.int32),
        "ref_element": jnp.zeros([1, 20, 14, 128]).astype(np.int32),
        "ref_atom_name_chars": jnp.zeros([1, 20, 14, 4, 64]).astype(dtype),
        "ref_space_uid": jnp.zeros([1, atom_size]).astype(np.int32) -1,
        'ref_center_mask': jnp.zeros([1, atom_size]).astype(np.int32) -1,
        "center_mask": jnp.zeros([1, atom_size]).astype(np.int32),
        "pseudo_seq": jax.nn.one_hot(jnp.zeros(token_size), 20)[None].astype(np.float32),
        
        # templates feature:
        "template_cb_atom_mask": jnp.zeros([1,N_template,token_size]).astype(dtype),                      
        "template_distogram": jnp.zeros([1,N_template,token_size,token_size,39]).astype(dtype),           
        "template_adj_feat": jnp.zeros([1,N_template,token_size,token_size,3]),                           
        "template_time_embedding_2d": jnp.zeros([1,N_template,token_size,token_size,1]).astype(dtype),    
        "template_mask": jnp.zeros([1,N_template]).astype(dtype),                                         
        
        'all_atom_positions': jnp.zeros([1,atom_size,3]).astype(np.float32),                     
        "all_atom_masks": jnp.zeros([1,atom_size]).astype(dtype),                                
        
        # ss feature:
        "ss_feat": jnp.zeros([1,token_size,9]).astype(dtype),                                    
        "ss_mask": jnp.zeros([1, token_size]).astype(dtype),                                     
        
        # diffusion feature:
        "seq_time_embedding": jnp.zeros([1,token_size]).astype(dtype),
        "time_embedding": jnp.zeros([1,token_size]).astype(dtype),
        "t_hat": jnp.zeros([1,1]).astype(dtype),
    }
    return batch


def update_1d_2d_seqmask(batch, dtype=jnp.bfloat16):
    # LxLx1 2d_seq_mask:
    batch['seq_mask_2d'] = (batch['seq_mask'][0][None,]*batch['seq_mask'][0][:,None])[None,None,...,None]    
    
    # Atom mask:
    batch['seq_atom_mask'] = ((batch['seq_mask'][0])[batch['ref_space_uid'][0]])[None]      
    return batch



def diag_zero_distogram(distogram):
    "distogram: (L,L,1)"
    n = distogram.shape[0]
    distogram *= (1-jax.numpy.identity(n))[...,None] 
    return distogram


def update_template_features(atom_positions, atom_mask, batch, axis=0, dtype=jnp.bfloat16, insert_motif=False):
    '''
    update template feature by N_template axis dim.
    atom_positions:  (Natom,3)
    atom_mask: (L,)
    '''
    # get center repr atom_xyz:
    center_atom_xyz = segment_mean(atom_positions, batch['ref_center_mask'], batch['seq_mask'][0].shape[0]) 

    # update template_distogram, shape:(E,N,L,L,39)
    template_dgram = dgram_from_positions(center_atom_xyz, min_bin=3.25, max_bin=50.75, num_bins=39).astype(dtype) * batch['seq_mask_2d'][0,0] 
    batch['template_distogram'] = dynamic_update_slice(batch['template_distogram'], template_dgram[None,None,...].astype(dtype), (0,axis,0,0,0)) 
    
    # update template mask:
    batch["template_cb_atom_mask"] = dynamic_update_slice(batch["template_cb_atom_mask"], atom_mask[None,None,...].astype(dtype), (0, axis, 0)) 
    template_mask = (jnp.sum(batch['template_cb_atom_mask'][0,axis]) > 0.).astype(dtype) * jnp.ones_like(batch["template_mask"][0,axis]) 
    batch['template_mask'] = batch['template_mask'].at[0,axis].set(template_mask).astype(dtype)
    
    return batch


def pdb_to_string(pdb_filename):
    # read pdbfile:
    lines = []
    for line in open(pdb_filename,"r"):
        if line[:6] == "HETATM" and line[17:20] == "MSE": line = "ATOM  "+line[6:17]+"MET"+line[20:]
        if line[:4] == "ATOM": lines.append(line)
    return "".join(lines)


def aatype_to_seq3(seq_aatype, alphabet='ARNDCQEGHILKMFPSTWYVX-Z'):
    '''
    swicth one-hot vector to sequences string.
    '''
    alphabet_1_to_3  = {'A':'ALA', 'C':'CYS', 'D':"ASP", "E":"GLU", "F":"PHE", "G":"GLY",
                        "H":"HIS", "I":"ILE", "K":"LYS", "L":"LEU", "M":"MET", "N":"ASN",
                        "P":"PRO", "Q":"GLN", "R":"ARG", "S":"SER", "T":"THR", "V":"VAL", 
                        "W":"TRP", "Y":"TYR", "X":"UNK", '-':"UNK", 'Z':"UNK"}
    restypes = [aa for aa in alphabet]
    aa2type = {idx:aa for idx, aa in enumerate(restypes)}
    seq1 = [aa2type[id_] for id_ in seq_aatype.tolist()]
    seq3 = [alphabet_1_to_3[aa] for aa in seq1]
    return seq3


def get_atomline(atom_index, residue_index, xyz_array, name, element, b_factor=0.0, res_name_3="VAL", chain_ids='A'):
    # PDB is a columnar format, every space matters here!
    xyz_array = xyz_array.tolist()
    alt_loc, insertion_code, charge, record_type, occupancy = '', '', '', 'ATOM', 1.00
    atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                f'{res_name_3:>3} {chain_ids:>1}'
                f'{residue_index:>4}{insertion_code:>1}   '
                f'{xyz_array[0]:>8.3f}{xyz_array[1]:>8.3f}{xyz_array[2]:>8.3f}'
                f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                f'{element:>2}{charge:>2}\n')
    return atom_line



def save_all_pdb(savepath, aatype, atom_positions, prefix, plddt_array=None):
    '''
    much faster than alphafold.protein object. but only save mainchain.
    args:
    - atom_positions.shape = (L,37,3)
    - plddt_array.shape = (L,)
    - aatype.shape = (L,)
    '''
    assert atom_positions.shape != 3

    restype_name_to_atom14_names = {
        'ALA': ['N', 'CA', 'C', 'O', 'CB'],
        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2'],
        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2'],
        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2'],
        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2'],
        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2'],
        'GLY': ['N', 'CA', 'C', 'O'],
        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2'],
        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2'],
        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ'],
        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE'],
        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD'],
        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH'],
        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
        'UNK': ['N', 'CA', 'C', 'O', 'CB'],
    }

    # ATOM37:
    atom_types = [
        'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        'CZ3', 'NZ', 'OXT'
    ]
    element_type = ['N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'O', 'O', 'S', 'C',
                    'C', 'C', 'N', 'N', 'O', 'O', 'S', 'C', 'C', 'C', 'C',
                    'N', 'N', 'N', 'O', 'O', 'C', 'N', 'N', 'O', 'C', 'C',
                    'C', 'N', 'O']

    atomtype2atom37index = {atomtype:idx for idx, atomtype in enumerate(atom_types)}
    atomtype2element = dict(zip(atom_types, element_type))
    
    # get name3:
    res_name_3_list = aatype_to_seq3(aatype) 

    # get atomlines:
    atomlines = []
    if len(atom_positions.shape) == 4:
        atom_positions = atom_positions - atom_positions.mean(axis=1)[:,None]
        L = atom_positions.shape[1]
        for t in range(1, atom_positions.shape[0]+1):
            atomlines.append(f'MODEL     {t}\n')
            residue_idx = 1
            for l in range(L):
                res_name_3 = res_name_3_list[l]
                atom_types = restype_name_to_atom14_names[res_name_3]
                for atom_idx, _ in enumerate(atom_types):
                    name_ = atom_types[atom_idx]
                    element_ = atomtype2element[name_]
                    coord_ = atom_positions[t-1, l, atomtype2atom37index[name_]]
                    
                    plddt_ = plddt_array[residue_idx-1] if plddt_array is not None else 0.0
                    atom_line = get_atomline(atom_idx+1, residue_idx, coord_, name_, element_, plddt_, res_name_3)
                    atomlines.append(atom_line)
                residue_idx += 1
            atomlines.append('ENDMDL\n')
            
    elif len(atom_positions.shape) == 3:
        L = atom_positions.shape[0]
        residue_idx = 1
        for l in range(L):
            res_name_3 = res_name_3_list[l]
            atom_types = restype_name_to_atom14_names[res_name_3]
            for atom_idx, _ in enumerate(atom_types):
                name_ = atom_types[atom_idx]
                element_ = atomtype2element[name_]
                coord_ = atom_positions[l, atomtype2atom37index[name_]]
                
                plddt_ = plddt_array[residue_idx-1] if plddt_array is not None else 0.0
                atom_line = get_atomline(atom_idx+1, residue_idx, coord_, name_, element_, plddt_, res_name_3)
                atomlines.append(atom_line)
            residue_idx += 1

    # write to file:
    save_pdbname = f'{savepath}/{prefix}.pdb'
    with open(save_pdbname, 'w') as f: 
        for line in atomlines: f.write(line)
        
    return save_pdbname



