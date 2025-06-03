import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import glob
import pandas as pd

class FoldseekCluster:
    """
    A wrapper class for Foldseek clustering operations.
    
    Example usage:
    >>> cluster = FoldseekCluster()
    >>> cluster.run(
    ...     input_path="/home/liwensuo/data/filtered/L128_HHH",
    ...     output_dir="./res",
    ...     tmp_dir="./tmp",
    ...     tmscore_threshold=0.8
    ... )
    """
    
    def __init__(self, foldseek_path: str = '/data/bin/foldseek', verbose: bool = True):
        """
        Initialize FoldseekCluster instance.
        
        :param foldseek_path: Path to Foldseek executable
        """
        self.foldseek_path = foldseek_path
        self.verbose = verbose
        self._validate_executable()

    def _validate_executable(self):
        """Verify Foldseek executable exists and is executable"""
        if not os.path.isfile(self.foldseek_path):
            raise FileNotFoundError(f"Foldseek executable not found at {self.foldseek_path}")
        if not os.access(self.foldseek_path, os.X_OK):
            raise PermissionError(f"Foldseek executable at {self.foldseek_path} is not executable")

    def _validate_paths(self, input_path: str, tmp_dir: str):
        """Validate input paths"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Path {input_path} does not exist")
        
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        

    def run(
        self,
        input_path: str,
        output_prefix: str,
        tmp_dir: str,
        alignment_type: int = 1,
        coverage_threshold: float = 0.0,
        cov_mode: int = 0,
        min_seq_id: float = 0,
        tmscore_threshold: float = 0.0,
        e_value: float = 0.001
    ) -> dict:
        """
        Run Foldseek clustering workflow.
        
        Args:
            input_path: Directory containing structure files (PDB/mmCIF[.gz])
            output_prefix: Output prefix for results
            tmp_dir: Temporary working directory
            alignment_type: 0: 3Di Gotoh-Smith-Waterman (local, not recommended), 1: TMalign (global, slow), 2: 3Di+AA Gotoh-Smith-Waterman (default: 1)
            coverage_threshold: List matches above this fraction of aligned (covered) residues (default: 0.0)
            cov_mode: 0: coverage of query and target, 1: coverage of target, 2: coverage of query (default: 0)
            min_seq_id: the minimum sequence identity to be clustered (default: 0)
            tmscore_threshold: accept alignments with an alignment TMscore > thr (default: 0.0)
            e_value: List matches below this E-value (range 0.0-inf, default: 0.001); increasing it reports more distant structures
        Returns:
            Dictionary containing execution details:
            {
                "status": "success"/"failed",
                "returncode": exit code,
                "stdout": command output,
                "stderr": error output,
                "command": executed command
            }
        """
        self._validate_paths(input_path, tmp_dir)
        
        # Build Foldseek command arguments
        cmd = [
            self.foldseek_path,
            "easy-cluster",
            input_path,
            output_prefix,
            tmp_dir,
            "--alignment-type", str(alignment_type),
            "-c", str(coverage_threshold),
            "--cov-mode", str(cov_mode),
            "--min-seq-id", str(min_seq_id),
            "--tmscore-threshold", str(tmscore_threshold),
            "-e", str(e_value),
        ]

        # Execute command with real-time output capture
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL if not self.verbose else None,
            stderr=subprocess.DEVNULL if not self.verbose else None,
            text=True,
            check=True
        )
        
        # Verify output file generation
        output_tsv = f"{output_prefix}_cluster.tsv"
        if not os.path.exists(output_tsv):
            raise FileNotFoundError(f"Cluster result file missing: {output_tsv}")

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd),
            "output_tsv": os.path.join(input_path, output_prefix + "_cluster.tsv")
        }


def easy_cluster(
    input_files: List[str],
    foldseek_path: Optional[str] = None,
    num_workers: int = 32,
    verbose: bool = True,
    **kwargs
) -> dict:
    """Convenience function for rapid clustering workflows.
    
    Handles automatic temporary directory management, parallel file processing,
    and result parsing.

    Example:
    >>> df = easy_cluster(
    ...     input_files=["struct1.pdb", "struct2.cif"],
    ...     tmscore_threshold=0.6,
    ...     coverage_threshold=0.8
    ... )
    >>> print(df.head())

    Args:
        input_files: List of paths to structure files for clustering
        foldseek_path: Optional custom Foldseek executable path
        **kwargs: Arguments forwarded to FoldseekCluster.run()

    Returns:
        DataFrame containing clustering results with columns:
        | cluster_center | pdb_name |
        |----------------|----------|
        | center_1       | struct1  |
        | center_1       | struct2  |

    Raises:
        RuntimeError: When file processing or clustering execution fails
    """
    # Initialize executor with optional custom path
    executor = FoldseekCluster(foldseek_path, verbose=verbose) if foldseek_path else FoldseekCluster(verbose=verbose)
    
    with tempfile.TemporaryDirectory(prefix="foldseek_") as tmp_root:
        # Prepare temporary workspace
        processing_dir = Path(tmp_root) / "processing"
        processing_dir.mkdir()

        def _safe_copy(src: Path, dest: Path) -> None:
            """Thread-safe file copy operation with collision prevention."""
            if not src.exists():
                raise FileNotFoundError(f"Source file missing: {src}")
            if dest.exists():
                raise ValueError(f"Filename collision: {dest.name}")
            shutil.copy(src, dest)

        # Parallel file copy
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = []
            for file_path in input_files:
                src = Path(file_path)
                dest = processing_dir / src.name
                futures.append(pool.submit(_safe_copy, src, dest))
            
            # Verify all copies completed successfully
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    raise RuntimeError(f"File preparation failed: {str(e)}")

        # Execute clustering workflow
        result = executor.run(
            input_path=str(processing_dir),
            output_prefix=str(processing_dir / "clusters"),
            tmp_dir=str(processing_dir / "tmp"),
            **kwargs
        )

        # Parse and process clustering results
        df = pd.read_csv(
            result['output_tsv'], 
            sep='\t', 
            header=None, 
            names=['cluster_center', 'pdb_name']
        )
        # Normalize file extensions for consistency
        df = df.replace(r'\.(pdb|cif|ent)(\.gz)?$', '', regex=True)
        
    return df
        