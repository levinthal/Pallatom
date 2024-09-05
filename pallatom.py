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

import argparse
import jax
import os
import numpy as np
from alphafold.common.residue_constants import restypes_wo_x
from modules.feature import save_all_pdb
from modules.ref_features import atom14_to_atom37
from modules.sampling import Sampler
from tqdm import trange


def main(data_dir, model_name, savepath, L, cuda_devices, t_min, t_max, gamma, step_scale, T, rounds):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
    print('Run denoising.....')

    # Ensure savepath exists
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Create or open the fasta file for writing sequences
    fasta_file_path = os.path.join(savepath, 'sample_seq.fasta')
    with open(fasta_file_path, 'w') as fasta_file:

        # Initialize sampler
        sampler = Sampler(
            T=T, sample_len=L, use_selfcond=True, 
            add_noise_level=[t_min, t_max, gamma], step_scale=step_scale,
            is_training=False, params_dir=data_dir, model_name=model_name)

        # Seed random key
        seed = np.random.randint(0, 2147483647)
        key = jax.random.PRNGKey(seed)

        aatype_tores = {i: res for i, res in enumerate(restypes_wo_x)}

        for round_idx in trange(rounds):
            key, key1, key2 = jax.random.split(key, 3)  # Update key
            
            # Prepare batch & Sample noise
            batch = sampler.prepare_batch(L)
            batch = sampler.SampleReference(batch, key1)
            
            # Denoise
            results, out_traj = sampler.Sample(batch, key=key2)
            print(round_idx)
            
            prefix = f'L{L}_denoised_{round_idx}'
            mask_atom = batch['seq_mask'][0][batch['ref_space_uid'][0]]
            mask_seq = batch['seq_mask'][0]
            final_aa = np.argmax(out_traj['seq_logits'][:,-1, :], axis=-1)[:int(mask_seq.sum()), ...]
            
            # Write sequence to fasta file
            seq = ''.join([aatype_tores[x] for x in final_aa.tolist()])
            fasta_file.write(f'>{prefix}\n{seq}\n')
            # save sample proteins as pdbs
            final_atoms = np.array(out_traj['px0'][:,-1,:])[:int(mask_atom.sum()), ...]
            final_atoms = atom14_to_atom37(final_aa, final_atoms.reshape(-1, 14, 3))
            save_all_pdb(savepath, final_aa, final_atoms, plddt_array=None, prefix=prefix)       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Pallatom model sampling process.')
    parser.add_argument('--data_dir', type=str, default='./', help='Directory where model parameters are stored')
    parser.add_argument('--model_name', type=str, default='Pallatom', help='Name of the model to use')
    parser.add_argument('--savepath', type=str, default='./results', help='Directory where results will be saved')
    parser.add_argument('--L', type=int, default=100, help='Length of the sequence to sample')
    parser.add_argument('--cuda_devices', type=str, default='3', help='CUDA visible devices')
    parser.add_argument('--t_min', type=float, default=0.01, help='Minimum noise level for add_noise_level')
    parser.add_argument('--t_max', type=float, default=1.0, help='Maximum noise level for add_noise_level')
    parser.add_argument('--gamma', type=float, default=0.2, help='Gamma value for add_noise_level')
    parser.add_argument('--step_scale', type=float, default=2.25, help='Scale of the step')
    parser.add_argument('--T', type=int, default=200, help='Number of steps for the sampling process')
    parser.add_argument('--rounds', type=int, default=10, help='Number of rounds to run')

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        model_name=args.model_name,
        savepath=args.savepath,
        L=args.L,
        cuda_devices=args.cuda_devices,
        t_min=args.t_min,
        t_max=args.t_max,
        gamma=args.gamma,
        step_scale=args.step_scale,
        T=args.T,
        rounds=args.rounds
    )
