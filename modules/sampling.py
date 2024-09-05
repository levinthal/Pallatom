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

import jax
import jax.numpy as jnp
import numpy as np
from modules.app import PallatomLauncher
from modules.feature import diag_zero_distogram, create_null_batch, update_1d_2d_seqmask, update_template_features
from jax.lax import dynamic_update_slice
from modules.ref_features import genertate_entangled_ref_features, create_local_mask
from modules.diffusion import EDM, get_index_embedding
from modules.utils import centre_random_augmentation
import warnings
warnings.filterwarnings("ignore")


class SelfCondManager(object):
    """Manage the self-conditioning in a structured prediction model."""

    def __init__(self, dtype) -> None:
        """Initialize the self-conditioning manager with configuration flags."""
        self.dtype = dtype

    def _update_template_features(self, batch, t_norm):
        # t: is normalized timestep.
        ############################ self.conditioning ########################
        # get prevs:
        dtype = batch['seq_mask'].dtype

        # update template feature using prevs_xyz/prevs_mask.
        batch = update_template_features(self.prev_pos, batch['seq_mask'][0], batch, dtype=dtype)

        # update time embedding:
        batch['template_time_embedding_2d'] = (batch['seq_mask_2d'] * t_norm).astype(dtype)

        return batch

    def update_selcond(self, out, batch, t_norm):
        """Update self-conditioning based on the outputs and the current batch."""
        # get model outputs
        self.prev_pos = out['final_atom_positions']            # (Natom,3)
        self.prev_mask = out['final_atom_mask']                # (Natom)
        batch = self._update_template_features(batch, t_norm)
        
        return batch

class Sampler(object):
    def __init__(self, params_dir, model_name, sample_len, T=200, 
                 use_selfcond=True, add_noise_level=[0.25, 1.0, 0.4], 
                 step_scale=1.5, psigma_mean=-1.2, psigma_std=1.5,
                 use_bfloat16=False, is_training=False) -> None:
        
        self.pallatom = PallatomLauncher(params_dir, model_name=model_name, use_bfloat16=use_bfloat16, is_training=is_training)
        self.pallatom.setup_models()
        self.cfg = self.pallatom.cfg
        self.pallatom.cfg.model.global_config.subbatch_size = 4
        self.pallatom.cfg.model.global_config.use_remat = False
        self.fix_size = int(jnp.ceil(sample_len/32) * 32)
        self.atom_fix_size = int(self.fix_size * 14)
        self.add_noise_level = add_noise_level
        self.r3_step_scale = step_scale

        # set optional flags:
        self.use_selfcond = use_selfcond
        self.dtype = jnp.bfloat16 if use_bfloat16 == True else jnp.float32
        self.T = T
        
        # setup dppm sampler:
        r3_edmp_config = self.pallatom.cfg.model.pallatom.r3_edmp
        self.r3_sampler = EDM(num_step=self.T, shape=(self.atom_fix_size,3), 
                               sigma_data=r3_edmp_config.sigma_data, 
                               psigma_mean=psigma_mean, 
                               psigma_std=psigma_std)  # (Natom,3)
        
        # selfcond:
        self.selfcond_manger = SelfCondManager(dtype=self.dtype)
        
    def prepare_batch(self, L):
        # create null batch:
        batch = create_null_batch(token_size=self.fix_size, atom_size=self.atom_fix_size, dtype=self.dtype)
        ref_batch = genertate_entangled_ref_features(1, element_dim=128, name_chars_dim=64, dtype=np.float32)
        
        # update ref features for protein:
        batch["ref_pos"] = dynamic_update_slice(batch["ref_pos"], ref_batch['ref_pos'][None], (0, 0, 0, 0))                          # (E,atom_size,3)
        batch["ref_mask"] = dynamic_update_slice(batch["ref_mask"], ref_batch['ref_mask'][None], (0, 0, 0))                          # (E,atom_size)
        batch["ref_element"] = dynamic_update_slice(batch["ref_element"], ref_batch['ref_element'][None].astype(jnp.int32), (0, 0, 0, 0))              # (E,atom_size,128)

        ref_space_uid_concat = []
        ref_center_mask_concat = []
        for i in range(L):
            ref_space_uid_concat.append(np.ones(14) * i)
            ref_center_mask_concat.append(np.array([-1, -1, -1, i, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])) 
        ref_space_uid_concat = np.concatenate(ref_space_uid_concat, axis=0)[None]       # (E, L*14)
        ref_center_mask_concat = np.concatenate(ref_center_mask_concat, axis=0)[None]   # (E, L*14)
        
        batch["ref_space_uid"] = dynamic_update_slice(batch["ref_space_uid"], ref_space_uid_concat.astype(jnp.int32), (0, 0))         # (E,atom_size)
        batch["ref_center_mask"] = dynamic_update_slice(batch["ref_center_mask"], ref_center_mask_concat, (0, 0))   # (E,atom_size)
        
        # update masks:
        seq_mask = jnp.zeros([1, self.fix_size]).at[...,:L].set(1.0).astype(self.dtype)
        batch.update({"seq_mask": seq_mask.astype(self.dtype)})
        batch = update_1d_2d_seqmask(batch, dtype=self.dtype)
        batch['center_mask'] = batch['seq_atom_mask'][0][None]
        
        # update aatype | sequence feature;
        batch["residue_index"] = dynamic_update_slice(batch["residue_index"], jnp.arange(L)[None,...].astype(jnp.int32), (0, 0))  # (E,L)
        
        # update residx_embedding:
        batch['residx_embedding'] = get_index_embedding(jnp.arange(batch['seq_mask'].shape[1]), 32, max_len=2056)[None] * batch['seq_mask'][...,None] # (E,L,32)

        # slice_idx for AtomTransofmer:
        q_indices = jnp.arange(0, (self.atom_fix_size), 32)
        k_indices = jnp.append(jnp.array([0,0]), 16 + 32*jnp.arange((self.atom_fix_size/32)-2).astype(int))
        local_mask = create_local_mask(q_indices, k_indices, self.atom_fix_size)
        batch['q_indices'] = q_indices[None]
        batch['k_indices'] = k_indices[None]
        batch['local_mask'] = local_mask[None]

        # SS_feat default as X.
        batch['ss_feat'] = (jax.nn.one_hot(jnp.ones((self.fix_size))*8,num_classes=9)[None]).astype(self.dtype)   # (E,L,9)
        batch['input_ss_feat'] = batch['ss_feat']
        batch['ss_mask'] = batch['seq_mask']
            
        return batch

    def SampleReference(self, batch, key):
        # random small perturb on t:
        key, key1, key2 = jax.random.split(key, 3)
        t_perturb = jax.random.uniform(key, shape=()) * 1/self.T
        t = 1.0 - t_perturb   # perturbed t.
        
        # x.T noise:
        t_hat = self.r3_sampler.get_noise_level_from_float(t)
        z = jax.random.normal(key1, shape=(self.atom_fix_size, 3)) * t_hat      # (atom_size,3)
        batch["all_atom_positions"] = z[None]*batch["seq_atom_mask"][...,None]
        batch["all_atom_masks"] = batch["seq_atom_mask"]                        

        return batch

    def SampleIter(self, batch, key, time_idx):
        # get new key:
        key, key1, key2, key3, key4 = jax.random.split(key, 5)
        
        # random small perturb on t:
        t_perturb = jax.random.uniform(key1, shape=()) * 1/self.T
        t = time_idx/(self.T-1) - t_perturb   # perturbed t.
        t = jnp.where(t<0, 0.0, t)
        _t_hat = self.r3_sampler.get_noise_level_from_float(t)
        batch['t_hat'] = jnp.array([_t_hat])[None].astype(self.dtype)
        
        # update time_embedding(....)
        batch['time_embedding'] = (_t_hat * batch['seq_mask']).astype(self.dtype)  # (E,L)

        # CentreRandomAugmentation for x_{t-1} # (E,Natom,3), centred by all atom mask
        batch['all_atom_positions'] = centre_random_augmentation(batch['all_atom_positions'][0], batch["all_atom_masks"][0], 
                                                                x_center_mask=batch['center_mask'][0], key=key1, s_trans=1.0)[None].astype(self.dtype)

        # add additional noise:
        xt = batch["all_atom_positions"][0]    # (Natom,3)
        xt_noisy, r3_t_hat, r3_sigma_next = self.r3_sampler.add_additional_noise(xt, t, key2, 
                                                                           mask=batch['all_atom_masks'][0][...,None], 
                                                                           gamma=self.add_noise_level[-1], 
                                                                           t_min=self.add_noise_level[0], 
                                                                           t_max=self.add_noise_level[1]) # (Natom,3)
        # update x_noisy to batch
        batch["all_atom_positions"] = xt_noisy[None]
        
        # run prediction:
        out = self.pallatom.model_runner.apply(self.pallatom.model_params, key4, batch)

        # get px0:
        px0 = out['final_atom_positions']                        # (Natom,3)

        # reverse & update: x0, xt, t_idx, key, mask=None,
        xt_minus1 = self.r3_sampler.reverse(px0, xt_noisy, r3_t_hat, r3_sigma_next, 
                                            mask=batch['all_atom_masks'][0][...,None], 
                                            step_scale=self.r3_step_scale) # (Natom,3)
        batch['all_atom_positions'] = xt_minus1[None]

        # update selfconditioning features:
        if self.use_selfcond: batch = self.selfcond_manger.update_selcond(out, batch, t)

        return batch, {"px0": px0,                                       # (Natom,3)
                       "xt": xt_minus1,                                  # (Natom,3),
                       "seq_logits": jax.nn.softmax(out['seq_logits']/0.1, -1),
                       }, out

    def Sample(self, batch, key):
        timesteps = jnp.arange(int(self.T)-1, -1, -1)  

        # aux
        aux = {}
        for t in timesteps:
            # update random key:
            key, key0 = jax.random.split(key, 2)
            
            # forward:
            batch, out_metrics, _ = self.SampleIter(batch, key0, t)
            
            # update aux:
            for k in out_metrics.keys():
                if k not in aux.keys(): aux[k] = out_metrics[k][:,None,...]
                else: aux[k] = jnp.concatenate([aux[k], out_metrics[k][:,None,...]], axis=1) 
        
        return batch, aux
