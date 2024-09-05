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
import numpy as np
from alphafold.model import model, data
from alphafold.model import config

class Key():
    '''random key generator'''
    def __init__(self, key=None, seed=None):
        if key is None:
            self.seed = np.random.randint(0,2147483647) if seed is None else seed
            self.key = jax.random.PRNGKey(self.seed) 
        else:
            self.key = key

    def get(self, num=1):
        if num > 1:
            self.key, *sub_keys = jax.random.split(self.key, num=(num+1))
            return sub_keys
        else:
            self.key, sub_key = jax.random.split(self.key)
            return sub_key
      

class PallatomLauncher(object):
    def __init__(self, data_dir, model_name="model_1_ptm", savepath='./', is_training=True, use_bfloat16=False):
        self.data_dir = data_dir
        self.savepath = savepath
        self.is_training = is_training
        self.use_bfloat16 = use_bfloat16

        self.model_name = model_name 
        self.key = Key()

    def setup_models(self):
        # check params_dir
        assert self.data_dir is not None

        # setup AF3 models configs;
        self.cfg = config.CONFIG
            
        # floats:
        self.cfg.model.global_config.bfloat16 = self.use_bfloat16

        # get basic model_params
        self.model_params = data.get_model_haiku_params(self.model_name, data_dir=self.data_dir)
        self.model_runner = model.RunModel(self.cfg, self.model_params, is_training=self.is_training)

        # print model details:
        print(f"MODELS: {self.model_name}")
                