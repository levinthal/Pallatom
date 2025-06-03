import os
import math
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import (
    PDBParser,
    DSSP,
    NeighborSearch,
    PDBIO,
    PPBuilder,
    Select,
    is_aa,
    Polypeptide,
)
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from itertools import groupby
import numpy as np
from Bio.PDB import PDBParser, DSSP, is_aa
import logging


def calculate_sse_metrics(
    pdb_file: str, 
    min_strand: int = 3, 
    min_helix: int = 4
) -> dict:
    """
    Calculate comprehensive secondary structure metrics from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        min_strand: Minimum residues to count as β-strand (default: 3)
        min_helix: Minimum residues to count as α-helix (default: 4)
    
    Returns:
        dict: {
            'composition': (helix_ratio, strand_ratio, loop_ratio),
            'total_residues': int,
            'ss_string': str,
            'helices': [(start,end)],       # All helix regions
            'strands': [(start,end)],      # All strand regions
            'loops': [(start,end)],        # All loop regions
            'valid_helices': [(start,end)], # Helices meeting length threshold
            'valid_strands': [(start,end)], # Strands meeting length threshold
            'sse_count': int,              # Total valid SSEs
            'max_loop_length': int,
            'mean_loop_length': float,
            'helix_residues': int,         # Total residues in helices
            'strand_residues': int         # Total residues in strands
        }
        Returns empty dict if calculation fails
    """
    # DSSP secondary structure classification mapping
    SS_MAPPING = {
        'H': 'H', 'B': 'E', 'E': 'E', 'G': 'H', 
        'I': 'H', 'T': 'L', 'S': 'L', '-': 'L'
    }
    
    results = {
        'composition': (0.0, 0.0, 0.0),
        'total_residues': 0,
        'ss_string': '',
        'helices': [],
        'strands': [],
        'loops': [],
        'valid_helices': [],
        'valid_strands': [],
        'sse_count': 0,
        'max_loop_length': 0,
        'mean_loop_length': 0.0,
        'helix_residues': 0,
        'strand_residues': 0
    }

    try:
        # Parse structure and run DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("temp", pdb_file)
        model = structure[0]
        
        # Build DSSP lookup dictionary with proper key handling
        dssp = DSSP(model, pdb_file)
        dssp_keys = [key[1][1] for key in dssp.keys()]
        
        ss_string = []
        for residue in structure.get_residues():
            if residue.id[1] in dssp_keys:
                index = dssp_keys.index(residue.id[1])
                a_key = list(dssp.keys())[index]
                ss_class = dssp[a_key][2]
                ss_string.append(SS_MAPPING.get(ss_class, "L"))
            else:
                logger.warning(f"Missing DSSP data for residue {residue.id[1]} in chain {chain.id}")
                ss_string.append('L')
            
        ss_string = "".join(ss_string)
        total_residues = len(ss_string)
        
        # Calculate composition ratios
        helix_count = ss_string.count('H')
        strand_count = ss_string.count('E')
        loop_count = total_residues - helix_count - strand_count
        
        # Analyze structural features
        current_pos = 0
        loop_lengths = []
        
        for ss_type, group in groupby(ss_string):
            seg_len = sum(1 for _ in group)
            seg_range = (current_pos, current_pos + seg_len - 1)
            
            if ss_type == 'H':
                results['helices'].append(seg_range)
                results['helix_residues'] += seg_len
                if seg_len >= min_helix:
                    results['valid_helices'].append(seg_range)
                    results['sse_count'] += 1
                    
            elif ss_type == 'E':
                results['strands'].append(seg_range)
                results['strand_residues'] += seg_len
                if seg_len >= min_strand:
                    results['valid_strands'].append(seg_range)
                    results['sse_count'] += 1
                    
            else:  # Loops
                results['loops'].append(seg_range)
                loop_lengths.append(seg_len)
                results['max_loop_length'] = max(results['max_loop_length'], seg_len)
            
            current_pos += seg_len
        
        # Finalize results
        results.update({
            'composition': (
                helix_count / total_residues,
                strand_count / total_residues,
                loop_count / total_residues
            ),
            'total_residues': total_residues,
            'ss_string': ss_string,
            'mean_loop_length': float(np.mean(loop_lengths)) if loop_lengths else 0.0
        })
        
        return results

    except Exception as e:
        logging.error(f"SSE calculation failed for {pdb_file}: {str(e)}")
        return {}


def calculate_mean_plddt(structure) -> float:
    """
    Calculate the mean pLDDT score from backbone atom B-factors in a protein structure.
    
    Args:
        structure: A Bio.PDB structure object
        
    Returns:
        float: Mean pLDDT score (0-100 scale)
        
    Raises:
        ValueError: If no backbone atoms are found in the structure
    """
    BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
    
    try:
        # Use list comprehension for faster collection of B-factors
        bfactors = [
            atom.bfactor
            for model in structure
            for chain in model
            for residue in chain
            for atom in residue
            if atom.id in BACKBONE_ATOMS
        ]
        
        if not bfactors:
            raise ValueError("No backbone atoms found in structure")
            
        return float(np.mean(bfactors))
    
    except Exception as e:
        raise ValueError(f"Error calculating mean pLDDT: {str(e)}")


def calculate_radius_of_gyration(structure) -> float:
    """
    Calculate radius of gyration for a protein structure
    
    Args:
        structure: Bio.PDB structure object
    
    Returns:
        float: Radius of gyration in Å
    """
    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])
    centroid = np.mean(coords, axis=0)
    squared_dists = np.sum((coords - centroid)**2, axis=1)
    return np.sqrt(np.mean(squared_dists))


def calculate_packing_density(pdb_file: str, core_sasa_threshold=0.2) -> float:
    """
    Calculate core packing density using SASA values
    
    Args:
        pdb_file: Path to PDB file
    
    Returns:
        float: Ratio of core residues (SASA <= threshold)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("temp", pdb_file)
    
    try:
        dssp = DSSP(structure[0], pdb_file)
    except Exception as e:
        logger.error(f"Packing density calculation failed: {e}")
        return 0.0

    core_residues = 0
    total_residues = 0
    
    for residue in structure.get_residues():
        try:
            sasa = dssp[(residue.parent.id, residue.id)][3]
            total_residues += 1
            if sasa <= core_sasa_threshold:
                core_residues += 1
        except KeyError:
            continue

    return core_residues / total_residues if total_residues else 0.0


def calculate_average_degree(structure):
    """
    Filters a model by the average degree of its residues.

    Args:
        structure: The structure to filter.

    Returns:
        True if the average degree is greater than the cutoff, False otherwise.
    """
    atoms = [atom for atom in structure.get_atoms()]
    ns = NeighborSearch(atoms)
    degrees = {}
    search_radius = 10.0

    for residue in structure.get_residues():
        neighbors = ns.search(residue.center_of_mass(), search_radius, level='R')
        degrees[residue] = len(neighbors) - 1

    average_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    return average_degree


def detect_ca_clashes(structure, threshold=2.0):
    """Detect steric clashes between C-alpha (CA) atoms in a protein structure.
    
    Args:
        structure (Bio.PDB.Structure): Parsed protein structure object
        threshold (float, optional): Distance cutoff for clash detection in Ångströms.
                                    Defaults to 2.0 Å (typical van der Waals radius for carbon).

    Returns:
        list: List of tuples containing clashing CA atom pairs:
            [(Atom1, Atom2), ...] 
            where Atom1 and Atom2 are CA atoms from different residues.
    """
    
    # Extract all C-alpha atoms from the structure
    ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA']
    
    # Initialize neighbor search with CA atoms for fast spatial queries
    ns = NeighborSearch(ca_atoms)
    
    # Detect atomic clashes using spatial search
    clashes = []
    for atom in ca_atoms:
        # Find atoms within threshold distance of current atom
        neighbors = ns.search(atom.coord, threshold)
        
        for neighbor in neighbors:
            # Exclude self-comparison and intra-residue pairs
            is_same_atom = (atom == neighbor)
            same_residue = (atom.get_parent() == neighbor.get_parent())
            
            if not is_same_atom and not same_residue:
                clashes.append((atom, neighbor))
    
    return clashes


def process_pdb_file(pdb_file: str) -> dict:
    """
    Optimized pipeline for processing PDB files and extracting structural metrics.
    
    Args:
        pdb_file: Path to input PDB file (supports .pdb, .cif, etc.)
    
    Returns:
        Dictionary containing:
        {
            "pdb_name": str,               # PDB ID without extension
            "total_residues": int,         # Number of amino acid residues
            "radius_of_gyration": float,   # Protein compactness measure
            "packing_density": float,      # Core residue ratio (0-1)
            "helix_content": float,       # Fraction of helical residues
            "strand_content": float,      # Fraction of beta-strand residues
            "loop_content": float,        # Fraction of loop residues
            "sse_count": int,            # Number of secondary structure elements
            "max_loop_length": int,      # Longest loop region length
            "mean_loop_length": float,    # Average loop length
            "average_degree": float,     # Residue connectivity measure
            "mean_plddt": float          # Average model confidence score
        }
        Returns empty dict on failure
    """
    # Initialize metrics with default values
    metrics = {
        "pdb_name": os.path.splitext(os.path.basename(pdb_file))[0],
        "total_residues": 0,
        "radius_of_gyration": 0.0,
        "packing_density": 0.0,
        "helix_content": 0.0,
        "strand_content": 0.0,
        "loop_content": 0.0,
        "sse_count": 0,
        "max_loop_length": 0,
        "mean_loop_length": 0.0,
        "average_degree": 0.0,
        "mean_plddt": 0.0,
        "pdb_path": pdb_file,
    }

    try:
        # 1. Parse structure (single pass)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("temp", pdb_file)
        
        # 2. Calculate all metrics that require the structure object
        metrics.update({
            "total_residues": len(list(structure.get_residues())),
            "radius_of_gyration": calculate_radius_of_gyration(structure),
            "average_degree": calculate_average_degree(structure),
            "mean_plddt": calculate_mean_plddt(structure),
            "ca_clash": len(detect_ca_clashes(structure)),
        })

        # 3. Calculate DSSP-dependent metrics in one operation
        sse_metrics = calculate_sse_metrics(pdb_file)
        metrics.update({
            "H_content": sse_metrics.get('composition', (0, 0, 0))[0],
            "E_content": sse_metrics.get('composition', (0, 0, 0))[1],
            "L_content": sse_metrics.get('composition', (0, 0, 0))[2],
            "sse_count": sse_metrics.get("sse_count", 0),
            "max_loop_length": sse_metrics.get("max_loop_length", 0),
            "mean_loop_length": sse_metrics.get("mean_loop_length", 0.0),
            "packing_density": calculate_packing_density(pdb_file)
        })

        return metrics

    except FileNotFoundError:
        logger.error(f"PDB file not found: {pdb_file}")
    except ValueError as e:
        logger.error(f"Invalid PDB format in {pdb_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing {pdb_file}: {str(e)}", exc_info=True)
    
    return {}

    
def parallel_metric(input_dir: str) -> None:
    """
    parallel_metric function
    
    Args:
        input_dir: Directory containing PDB files
    """
    pdb_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".pdb")
    ]

    logger.info(f"Found {len(pdb_files)} PDB files for processing")

    # Parallel processing
    with Pool() as pool, tqdm(total=len(pdb_files)) as pbar:
        results = []
        for pdb_file in pdb_files:
            result = pool.apply_async(
                process_pdb_file,
                (pdb_file,),
                callback=lambda _: pbar.update(),
            )
            results.append(result)

        # Collect results
        processed_data = []
        for result in results:
            data = result.get()
            if data:
                processed_data.append(data)

    # Save results
    df = pd.DataFrame(processed_data)
    return df


if __name__ == "__main__":
    INPUT_DIR = "/data/pallatom/L_100/esmf"
    OUTPUT_FILE = "protein_metrics.csv"
    output_df = parallel_metric(INPUT_DIR, OUTPUT_FILE)
    logger.info(f"Saved results to {output_file}")
    df.to_csv(output_file, index=False)