# P(allatom): A New Path for Protein Design

## Overview

Pallatom is an innovative protein generation model that produces protein structures with all-atom coordinates. By learning and modeling the joint distribution $P(\text{structure}, \text{seq})$, with a focus on $P(\text{all-atom})$, Pallatom effectively addresses the interdependence between sequence and structure in protein generation. This project introduces a novel network architecture designed specifically for all-atom protein generation, employing a dual-track framework that tokenizes proteins into token-level and atomic-level representations. Pallatom excels in key metrics of protein design, including designability, diversity, and novelty, paving the way for future applications in more complex systems.

## Installation

To set up the environment for running Pallatom, follow these steps:

1. **Create and activate a conda environment:**

   ```bash
   conda create --name pallatom python=3.7.16
   conda activate pallatom
   ```

2. **Install JAX:**

   First, install the specific version of JAX needed for this project:

   ```bash
   pip install jax==0.3.25
   pip install "jax[cuda]"==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

3. **Install other dependencies:**

   Finally, install the additional required packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Pallatom model sampling process, use the `pallatom.py` script. Below is an example of how to use the script with command-line arguments:

```bash
python pallatom.py --savepath ./results --L 120 --batch_num 4 --cuda_devices 0 --t_min 0.01 --t_max 1.0 --gamma 0.2 --step_scale 2.25 --T 200 --rounds 10
```

### Parameters:

- **`data_dir`**: Directory where model parameters are stored (default: `./`)
- **`model_name`**: Name of the model to use (default: `Pallatom`)
- **`savepath`**: Directory where results will be saved (default: `./results`)
- **`L`**: Length of the sequence to sample (default: `120`)
- **`batch_num`**: Number of batches to run (default: `4`)
- **`cuda_devices`**: CUDA visible device (default: `0`)
- **`t_min`**: Minimum noise level for `add_noise_level` (default: `0.01`)
- **`t_max`**: Maximum noise level for `add_noise_level` (default: `1.0`)
- **`gamma`**: Gamma value for `add_noise_level` (default: `0.2`)
- **`step_scale`**: Scale of the step (default: `2.25`)
- **`T`**: Number of steps for the sampling process (default: `200`)
- **`rounds`**: Number of rounds to run (default: `1`)

### Output

The results, including the generated sequences in FASTA format and protein structures in PDB format, will be saved in the specified `savepath` directory.


## Citation

If you find Pallatom useful in your research, please consider citing our work:

```bibtex
@article {Qu2024.08.16.608235,
	author = {Qu, Wei and Guan, Jiawei and Ma, Rui and Zhai, Ke and Wu, Weikun and Wang, Haobo},
	title = {P(all-atom) Is Unlocking New Path For Protein Design},
	year = {2024},
	doi = {10.1101/2024.08.16.608235},
	journal = {bioRxiv}
}
```

## Copyright and License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
