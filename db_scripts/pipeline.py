import argparse
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from metric import parallel_metric
from foldseek import easy_cluster

class DatabasePipeline:
    def __init__(self, args):
        self.args = args
        self.step_dfs = {}
        
        # Configure logging
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def run(self):
        try:
            # Pipeline execution flow
            self.logger.info("Starting protein clustering pipeline")
            
            # Step 1: Metric calculation and filtering
            step1_df = self.step1_filter_metrics()
            self.step_dfs['step1'] = step1_df
            
            # Step 2: Deduplication clustering
            step2_df = self.step2_deduplicate(step1_df)
            self.step_dfs['step2'] = step2_df
            
            # Step 3: Final clustering
            step3_df = self.step3_cluster(step2_df)
            self.step_dfs['step3'] = step3_df
            
            # Step 4: Output organization
            self.step4_output_db(step3_df)
            
            self.logger.info("Pipeline completed successfully")
            return 0
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def step1_filter_metrics(self):
        """Step 1: Calculate metrics and apply initial filters"""
        self.logger.info("Running Step 1: Metric calculation and filtering")
        
        output_dir = Path(self.args.output_dir)
        if output_dir.exists():
            self.logger.warning("Output directory exists, removing...")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        # Run metric calculation
        metric_df = parallel_metric(self.args.input_dir)
                
        # Apply filtering criteria
        filtered_df = metric_df[
            (metric_df['total_residues'] <= self.args.max_length) &
            (metric_df['total_residues'] >= self.args.min_length) &
            (metric_df['radius_of_gyration'] <= self.args.max_gyration) & 
            (metric_df['packing_density'] >= self.args.min_packing_density) &
            (metric_df['ca_clash'] == 0) &
            (metric_df['sse_count'] >= self.args.min_sse_count) &
            (metric_df['average_degree'] >= self.args.min_degree) &
            (metric_df['L_content'] <= self.args.max_loop_content) &
            (metric_df['max_loop_length'] <= self.args.max_loop_length)
        ]
        
        self.logger.info(f"Step 1 completed. Remaining structures: {len(filtered_df)}")
        return filtered_df

    def step2_deduplicate(self, step1_df):
        """Step 2: Cluster with high similarity threshold for deduplication"""
        self.logger.info("Running Step 2: Deduplication clustering")
        
        valid_paths = step1_df.pdb_path.tolist()
        cluster_df = easy_cluster(
            valid_paths,
            foldseek_path=self.args.foldseek_path,
            alignment_type=1,
            coverage_threshold=0.9,
            cov_mode=0,
            min_seq_id=0,
            tmscore_threshold=self.args.deduplicate_threshold,
            e_value=0.001,
            num_workers=self.args.workers,
            verbose=self.args.verbose,
        )
        cluster_df = cluster_df.rename(columns={'cluster_center': 'cluster_center_high'})
        
        # Merge and select best per cluster
        merged_df = pd.merge(step1_df, cluster_df, on='pdb_name')
        step2_df = merged_df.loc[merged_df.groupby('cluster_center_high')['mean_plddt'].idxmax()]
        
        self.logger.info(f"Step 2 completed. Unique clusters: {len(step2_df)}")
        return step2_df.reset_index(drop=True)

    def step3_cluster(self, step2_df):
        """Step 3: Final clustering with lower similarity threshold"""
        self.logger.info("Running Step 3: Final clustering")
        
        valid_paths = step2_df.pdb_path.tolist()
        cluster_df = easy_cluster(
            valid_paths,
            foldseek_path=self.args.foldseek_path,
            alignment_type=1,
            coverage_threshold=0.9,
            cov_mode=0,
            min_seq_id=0,
            tmscore_threshold=self.args.final_cluster_threshold,
            e_value=0.001,
            num_workers=self.args.workers,
            verbose=self.args.verbose,
        )
        cluster_df = cluster_df.rename(columns={'cluster_center': 'cluster_center_final'})
        
        # Merge and assign cluster IDs
        merged_df = pd.merge(step2_df, cluster_df, on='pdb_name')
        merged_df['cluster_idx'] = pd.factorize(merged_df['cluster_center_final'])[0]
        
        self.logger.info(f"Step 3 completed. Final clusters: {merged_df['cluster_idx'].nunique()}")
        return merged_df

    def step4_output_db(self, step3_df):
        """Step 4: Organize output files into cluster directories"""
        self.logger.info("Running Step 4: Output organization")
        
        output_dir = Path(self.args.output_dir) / 'pdbs'
        if output_dir.exists():
            self.logger.warning("Output directory exists, removing...")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        # Generate file list with new names
        step3_df['count'] = step3_df.groupby('cluster_idx').cumcount()
        step3_df.to_csv(Path(self.args.output_dir) / "output_db.csv")
        
        file_list = []
        for _, row in step3_df.iterrows():
            src = Path(row['pdb_path'])
            dst = output_dir / f"AFDB_{row['cluster_idx']:05d}_{row['count']:04d}.pdb"
            file_list.append((src, dst))
        
        # Parallel file copy
        self.logger.info(f"Copying {len(file_list)} files to {output_dir}")
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = []
            for src, dst in file_list:
                futures.append(executor.submit(self._safe_copy, src, dst))
                
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"File copy failed: {str(e)}")
        
        self.logger.info(f"Output database created at {output_dir}")

    def _safe_copy(self, src: Path, dst: Path):
        """Thread-safe file copy with validation"""
        if not src.exists():
            raise FileNotFoundError(f"Source file missing: {src}")
        if dst.exists():
            raise ValueError(f"Destination already exists: {dst}")
        
        shutil.copy(src, dst)
        self.logger.debug(f"Copied {src.name} to {dst}")

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Structure Clustering Pipeline")
    
    # Required arguments
    parser.add_argument("input_dir", help="Input directory containing PDB files")
    parser.add_argument("output_dir", help="Output directory for clustered structures")
    
    # Step 1 parameters
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum protein length")
    parser.add_argument("--max-length", type=int, default=128,
                       help="Maximum protein length")
    parser.add_argument("--max-gyration", type=float, default=20,
                       help="Maximum radius of gyration")
    parser.add_argument("--min-packing-density", type=float, default=0.30,
                       help="Minimum packing density")
    parser.add_argument("--min-sse-count", type=int, default=4,
                       help="Minimum secondary structure elements")
    parser.add_argument("--min-degree", type=float, default=20.0,
                       help="Minimum average residue degree")
    parser.add_argument("--max-loop-content", type=float, default=0.5,
                       help="Maximum loop content")
    parser.add_argument("--max-loop-length", type=int, default=15,
                       help="Maximum loop length")
    
    # Clustering parameters
    parser.add_argument("--deduplicate-threshold", type=float, default=0.8,
                       help="TM-score threshold for deduplication clustering")
    parser.add_argument("--final-cluster-threshold", type=float, default=0.6,
                       help="TM-score threshold for final clustering")
    
    # System parameters
    parser.add_argument("--foldseek-path", default="foldseek",
                       help="Path to Foldseek executable")
    parser.add_argument("--workers", type=int, default=32,
                       help="Number of parallel workers")
    parser.add_argument("--verbose", action='store_true',
                       help="Enable verbose foldseek output")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = DatabasePipeline(args)
    pipeline.run()