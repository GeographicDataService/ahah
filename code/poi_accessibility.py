"""
POI Accessibility Calculator

A module for calculating travel times from postcodes to Points of Interest (POI)
using a Valhalla routing cluster.

Features:
- Auto-detects Valhalla cluster size
- Processes single or multiple POI files
- Saves results as GeoParquet
- Supports resuming interrupted runs
- Parallel processing with load balancing

Usage:
    from poi_accessibility import POIAccessibilityCalculator
    
    calc = POIAccessibilityCalculator(
        postcodes_path="data/postcodes.parquet",
        results_dir="results/"
    )
    
    # Process single POI
    calc.calculate("data/poi/supermarkets.parquet")
    
    # Process multiple POIs
    calc.calculate_batch([
        "data/poi/supermarkets.parquet",
        "data/poi/pharmacies.parquet",
        "data/poi/gp_surgeries.parquet",
    ])
"""

import concurrent.futures
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass
class ClusterConfig:
    """Configuration for Valhalla cluster connection."""
    base_port: int = 8010
    host: str = "localhost"
    max_workers: int = 20  # Maximum workers to probe for
    timeout: int = 5
    
    # Detected at runtime
    active_ports: List[int] = field(default_factory=list)
    
    @property
    def hosts(self) -> List[str]:
        """List of active Valhalla host URLs."""
        return [f"http://{self.host}:{port}" for port in self.active_ports]
    
    @property
    def num_workers(self) -> int:
        """Number of active workers detected."""
        return len(self.active_ports)


class POIAccessibilityCalculator:
    """
    Calculate travel times from postcodes to Points of Interest.
    
    Parameters
    ----------
    postcodes_path : str or Path
        Path to the postcodes GeoParquet file.
        Must contain columns: geometry (Point), and an ID column (default: pcd7).
        
    results_dir : str or Path, default="./results"
        Directory for storing output GeoParquet files.
        Created if it doesn't exist.
        
    postcode_id_col : str, default="pcd7"
        Column name for postcode identifier.
        
    postcode_lsoa_col : str, optional
        Column name for LSOA codes (for summary statistics).
        
    base_port : int, default=8010
        Starting port for Valhalla cluster detection.
        
    host : str, default="localhost"
        Hostname for Valhalla cluster.
        
    batch_size : int, default=50
        Number of postcodes per Valhalla request.
        Higher values = fewer requests but more memory.
        
    k_nearest : int, default=4
        Number of nearest POIs to consider for each postcode.
        The minimum travel time across these is returned.
        
    costing : str, default="auto"
        Valhalla costing model: "auto", "pedestrian", "bicycle", etc.
        
    request_timeout : int, default=30
        Timeout in seconds for each Valhalla request.
        
    auto_detect_cluster : bool, default=True
        Whether to auto-detect the Valhalla cluster size on init.
        If False, must call detect_cluster() manually or set num_workers.
        
    num_workers : int, optional
        Override auto-detection with fixed number of workers.
        Workers are assumed on ports: base_port, base_port+1, ...
        
    Attributes
    ----------
    cluster : ClusterConfig
        Detected cluster configuration.
        
    gdf_postcodes : GeoDataFrame
        Loaded postcodes data.
        
    Examples
    --------
    Basic usage with auto-detection:
    
    >>> calc = POIAccessibilityCalculator("data/postcodes.parquet")
    >>> print(f"Detected {calc.cluster.num_workers} Valhalla workers")
    >>> calc.calculate("data/poi/supermarkets.parquet")
    
    Custom configuration:
    
    >>> calc = POIAccessibilityCalculator(
    ...     postcodes_path="data/postcodes.parquet",
    ...     results_dir="output/accessibility",
    ...     batch_size=100,
    ...     k_nearest=5,
    ...     costing="pedestrian",
    ... )
    """
    
    def __init__(
        self,
        postcodes_path: Union[str, Path],
        results_dir: Union[str, Path] = "./results",
        postcode_id_col: str = "pcd7",
        postcode_lsoa_col: Optional[str] = "lsoa21cd",
        postcode_lat_col: str = "lat",
        postcode_lon_col: str = "long",
        base_port: int = 8010,
        host: str = "localhost",
        batch_size: int = 50,
        k_nearest: int = 4,
        costing: str = "auto",
        request_timeout: int = 30,
        auto_detect_cluster: bool = True,
        num_workers: Optional[int] = None,
    ):
        self.postcodes_path = Path(postcodes_path)
        self.results_dir = Path(results_dir)
        self.postcode_id_col = postcode_id_col
        self.postcode_lsoa_col = postcode_lsoa_col
        self.postcode_lat_col = postcode_lat_col
        self.postcode_lon_col = postcode_lon_col
        self.batch_size = batch_size
        self.k_nearest = k_nearest
        self.costing = costing
        self.request_timeout = request_timeout
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cluster configuration
        self.cluster = ClusterConfig(base_port=base_port, host=host)
        
        if num_workers is not None:
            # Manual override
            self.cluster.active_ports = list(range(base_port, base_port + num_workers))
            print(f"Using {num_workers} workers (manual override)")
        elif auto_detect_cluster:
            self.detect_cluster()
        
        # Lazy-load postcodes
        self._gdf_postcodes = None
        self._postcode_coords = None
    
    @property
    def gdf_postcodes(self) -> gpd.GeoDataFrame:
        """Lazy-loaded postcodes GeoDataFrame."""
        if self._gdf_postcodes is None:
            print(f"Loading postcodes from {self.postcodes_path}...")
            self._gdf_postcodes = gpd.read_parquet(self.postcodes_path)
            
            # Extract postcode prefix for chunked processing
            self._gdf_postcodes["_prefix"] = (
                self._gdf_postcodes[self.postcode_id_col]
                .str.extract(r"^([A-Z]+)", expand=False)
            )
            
            print(f"  Loaded {len(self._gdf_postcodes):,} postcodes")
        return self._gdf_postcodes
    
    @property
    def postcode_coords(self) -> np.ndarray:
        """Postcode coordinates as numpy array [lon, lat]."""
        if self._postcode_coords is None:
            gdf = self.gdf_postcodes
            self._postcode_coords = np.column_stack([
                gdf.geometry.x, gdf.geometry.y
            ])
        return self._postcode_coords
    
    def detect_cluster(self, max_probe: Optional[int] = None) -> int:
        """
        Auto-detect the number of active Valhalla workers.
        
        Probes sequential ports starting from base_port until a connection
        fails or max_probe is reached.
        
        Parameters
        ----------
        max_probe : int, optional
            Maximum number of ports to probe. Defaults to cluster.max_workers.
            
        Returns
        -------
        int
            Number of active workers detected.
        """
        max_probe = max_probe or self.cluster.max_workers
        active_ports = []
        
        print(f"Detecting Valhalla cluster (probing ports {self.cluster.base_port}+)...")
        
        for i in range(max_probe):
            port = self.cluster.base_port + i
            try:
                resp = requests.get(
                    f"http://{self.cluster.host}:{port}/status",
                    timeout=self.cluster.timeout
                )
                if resp.status_code == 200:
                    active_ports.append(port)
                else:
                    break
            except requests.RequestException:
                break
        
        self.cluster.active_ports = active_ports
        
        if not active_ports:
            print("  ⚠️  No active Valhalla workers found!")
            print(f"     Check that Valhalla is running on port {self.cluster.base_port}")
        else:
            print(f"  ✅ Detected {len(active_ports)} active workers")
            print(f"     Ports: {active_ports[0]}-{active_ports[-1]}")
        
        return len(active_ports)
    
    def _query_valhalla(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        host: str,
    ) -> List[Optional[float]]:
        """
        Query Valhalla sources_to_targets endpoint.
        
        Returns minimum travel time (seconds) for each source.
        """
        payload = {
            "sources": [{"lat": float(s[1]), "lon": float(s[0])} for s in sources],
            "targets": [{"lat": float(t[1]), "lon": float(t[0])} for t in targets],
            "costing": self.costing,
        }
        
        try:
            resp = requests.post(
                f"{host}/sources_to_targets",
                json=payload,
                timeout=self.request_timeout,
            )
            if resp.status_code != 200:
                return [None] * len(sources)
            
            matrix = resp.json().get("sources_to_targets", [])
            
            # Get minimum time for each source
            return [
                min((x["time"] for x in row if x.get("time")), default=None)
                for row in matrix
            ]
        except Exception:
            return [None] * len(sources)
    
    def _process_batch(
        self,
        args: Tuple[np.ndarray, np.ndarray, str, cKDTree, np.ndarray],
    ) -> List[Dict]:
        """Process a batch of postcodes."""
        batch_coords, batch_ids, host, poi_tree, poi_coords = args
        
        # Find k nearest POIs for this batch
        _, nearest_idx = poi_tree.query(batch_coords, k=self.k_nearest)
        unique_poi_idx = list(set(nearest_idx.flatten()))
        targets = poi_coords[unique_poi_idx]
        
        # Query Valhalla
        times = self._query_valhalla(batch_coords, targets, host)
        
        return [
            {"id": str(pcd_id), "time_seconds": t}
            for pcd_id, t in zip(batch_ids, times)
        ]
    
    def _process_prefix(
        self,
        prefix: str,
        gdf_subset: gpd.GeoDataFrame,
        poi_tree: cKDTree,
        poi_coords: np.ndarray,
    ) -> pd.DataFrame:
        """Process all postcodes for a given prefix."""
        pc_coords = np.column_stack([
            gdf_subset.geometry.x, gdf_subset.geometry.y
        ])
        pc_ids = gdf_subset[self.postcode_id_col].values
        
        # Create batches with round-robin host assignment
        batches = []
        hosts = self.cluster.hosts
        
        for i in range(0, len(gdf_subset), self.batch_size):
            host = hosts[len(batches) % len(hosts)]
            batch_coords = pc_coords[i:i + self.batch_size]
            batch_ids = pc_ids[i:i + self.batch_size]
            batches.append((batch_coords, batch_ids, host, poi_tree, poi_coords))
        
        # Process batches in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            for batch_results in executor.map(self._process_batch, batches):
                results.extend(batch_results)
        
        return pd.DataFrame(results)
    
    def calculate(
        self,
        poi_path: Union[str, Path],
        output_name: Optional[str] = None,
        resume: bool = True,
        save_intermediate: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Calculate travel times from all postcodes to a POI dataset.
        
        Parameters
        ----------
        poi_path : str or Path
            Path to POI GeoParquet file.
            Must have Point geometry.
            
        output_name : str, optional
            Name for output file. If not provided, derived from POI filename.
            Output will be: {results_dir}/{output_name}.parquet
            
        resume : bool, default=True
            If True, skip prefixes that already have results.
            Enables resuming interrupted runs.
            
        save_intermediate : bool, default=True
            If True, save results per postcode prefix.
            Enables resuming and reduces memory usage.
            
        Returns
        -------
        GeoDataFrame
            Postcodes with travel times and geometry.
            Columns: postcode, duration_minutes, longitude, latitude,
                     lsoa21cd (if available), geometry, duration_minutes_missing
        """
        if not self.cluster.active_ports:
            raise RuntimeError(
                "No active Valhalla workers. "
                "Run detect_cluster() or check Valhalla is running."
            )
        
        poi_path = Path(poi_path)
        
        # Derive output name from POI filename
        if output_name is None:
            output_name = poi_path.stem
        
        # Create POI-specific intermediate directory
        intermediate_dir = self.results_dir / f"_intermediate_{output_name}"
        if save_intermediate:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Load POI data
        print(f"\nProcessing POI: {poi_path.name}")
        print(f"  Output name: {output_name}")
        
        gdf_poi = gpd.read_parquet(poi_path)
        poi_coords = np.column_stack([gdf_poi.geometry.x, gdf_poi.geometry.y])
        poi_tree = cKDTree(poi_coords)
        
        print(f"  Loaded {len(gdf_poi):,} POI locations")
        print(f"  Using {self.cluster.num_workers} Valhalla workers")
        print(f"  Batch size: {self.batch_size}, k_nearest: {self.k_nearest}")
        
        # Get postcode prefixes
        gdf = self.gdf_postcodes
        prefixes = sorted(gdf["_prefix"].dropna().unique())
        
        print(f"  Processing {len(prefixes)} postcode prefixes...")
        
        # Process each prefix
        for prefix in tqdm(prefixes, desc="Prefixes"):
            prefix_output = intermediate_dir / f"{prefix}.parquet"
            
            # Skip if already processed
            if resume and prefix_output.exists():
                continue
            
            # Process this prefix
            gdf_subset = gdf[gdf["_prefix"] == prefix]
            df_results = self._process_prefix(prefix, gdf_subset, poi_tree, poi_coords)
            
            if save_intermediate:
                df_results.to_parquet(prefix_output, index=False)
            
            missing = df_results["time_seconds"].isna().sum()
            tqdm.write(f"{prefix}: {len(df_results):,} postcodes, {missing:,} missing")
        
        # Combine all results
        print("\nCombining results...")
        result_files = list(intermediate_dir.glob("*.parquet"))
        
        if not result_files:
            raise RuntimeError("No results found. Check Valhalla connectivity.")
        
        df_times = pd.concat(
            [pd.read_parquet(f) for f in result_files],
            ignore_index=True
        )
        
        # Merge with postcode data
        merge_cols = [self.postcode_id_col]
        if self.postcode_lon_col:
            merge_cols.append(self.postcode_lon_col)
        if self.postcode_lat_col:
            merge_cols.append(self.postcode_lat_col)
        if self.postcode_lsoa_col and self.postcode_lsoa_col in gdf.columns:
            merge_cols.append(self.postcode_lsoa_col)
        
        df_times = df_times.rename(columns={"id": self.postcode_id_col})
        
        result = (
            df_times
            .merge(gdf[merge_cols], how="left", on=self.postcode_id_col)
            .rename(columns={
                self.postcode_id_col: "postcode",
                "time_seconds": "duration_seconds",
                self.postcode_lon_col: "longitude",
                self.postcode_lat_col: "latitude",
            })
        )
        
        # Calculate minutes
        result["duration_minutes"] = result["duration_seconds"] / 60
        
        # Rename LSOA column if present
        if self.postcode_lsoa_col and self.postcode_lsoa_col in result.columns:
            result = result.rename(columns={self.postcode_lsoa_col: "lsoa21cd"})
        
        # Select and order columns
        cols = ["postcode", "duration_minutes", "longitude", "latitude"]
        if "lsoa21cd" in result.columns:
            cols.append("lsoa21cd")
        result = result[cols].copy()
        
        # Convert to GeoDataFrame
        result = gpd.GeoDataFrame(
            result,
            geometry=gpd.points_from_xy(result["longitude"], result["latitude"]),
            crs="EPSG:4326",
        )
        
        # Add missing flag
        result["duration_minutes_missing"] = result["duration_minutes"].isna()
        
        # Save final output
        output_path = self.results_dir / f"{output_name}.parquet"
        result.to_parquet(output_path, index=False)
        
        # Summary
        print(f"\n✅ Saved to: {output_path}")
        print(f"\nSummary:")
        print(f"  Total postcodes: {len(result):,}")
        print(f"  Missing times: {int(result['duration_minutes_missing'].sum()):,}")
        print(f"  Mean time: {result['duration_minutes'].mean():.2f} min")
        print(f"  Median time: {result['duration_minutes'].median():.2f} min")
        
        return result
    
    def calculate_batch(
        self,
        poi_paths: List[Union[str, Path]],
        resume: bool = True,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Calculate travel times for multiple POI datasets.
        
        Parameters
        ----------
        poi_paths : list of str or Path
            Paths to POI GeoParquet files.
            
        resume : bool, default=True
            If True, skip POIs that already have complete results.
            
        Returns
        -------
        dict
            Mapping of POI names to result GeoDataFrames.
        """
        results = {}
        
        print(f"Processing {len(poi_paths)} POI datasets...")
        print("=" * 60)
        
        for poi_path in poi_paths:
            poi_path = Path(poi_path)
            output_name = poi_path.stem
            output_path = self.results_dir / f"{output_name}.parquet"
            
            # Check if already complete
            if resume and output_path.exists():
                print(f"\n⏭️  Skipping {output_name} (already complete)")
                results[output_name] = gpd.read_parquet(output_path)
                continue
            
            try:
                results[output_name] = self.calculate(
                    poi_path, 
                    output_name=output_name,
                    resume=resume,
                )
            except Exception as e:
                print(f"\n❌ Error processing {output_name}: {e}")
                continue
        
        print("\n" + "=" * 60)
        print(f"✅ Completed {len(results)}/{len(poi_paths)} POI datasets")
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"POIAccessibilityCalculator("
            f"postcodes={self.postcodes_path.name}, "
            f"workers={self.cluster.num_workers}, "
            f"batch_size={self.batch_size})"
        )


def main():
    """CLI entry point for POI accessibility calculations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate travel times from postcodes to Points of Interest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate accessibility to supermarkets
  python poi_accessibility.py data/postcodes.parquet data/poi/supermarkets.parquet
  
  # Process multiple POIs
  python poi_accessibility.py data/postcodes.parquet \\
      data/poi/supermarkets.parquet \\
      data/poi/pharmacies.parquet \\
      data/poi/gp_surgeries.parquet
  
  # Custom settings
  python poi_accessibility.py data/postcodes.parquet data/poi/banks.parquet \\
      --results-dir output/ \\
      --batch-size 100 \\
      --k-nearest 5 \\
      --costing pedestrian
        """
    )
    
    parser.add_argument(
        "postcodes",
        help="Path to postcodes GeoParquet file"
    )
    parser.add_argument(
        "poi",
        nargs="+",
        help="Path(s) to POI GeoParquet file(s)"
    )
    parser.add_argument(
        "--results-dir", "-o",
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=50,
        help="Postcodes per request (default: 50)"
    )
    parser.add_argument(
        "--k-nearest", "-k",
        type=int,
        default=4,
        help="Number of nearest POIs to consider (default: 4)"
    )
    parser.add_argument(
        "--costing", "-c",
        default="auto",
        choices=["auto", "pedestrian", "bicycle", "bus", "truck"],
        help="Valhalla costing model (default: auto)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8010,
        help="Base port for Valhalla cluster (default: 8010)"
    )
    parser.add_argument(
        "--num-workers", "-n",
        type=int,
        help="Override auto-detection with fixed worker count"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already-processed prefixes"
    )
    parser.add_argument(
        "--postcode-id-col",
        default="pcd7",
        help="Postcode ID column name (default: pcd7)"
    )
    
    args = parser.parse_args()
    
    calc = POIAccessibilityCalculator(
        postcodes_path=args.postcodes,
        results_dir=args.results_dir,
        postcode_id_col=args.postcode_id_col,
        base_port=args.port,
        batch_size=args.batch_size,
        k_nearest=args.k_nearest,
        costing=args.costing,
        num_workers=args.num_workers,
    )
    
    if len(args.poi) == 1:
        calc.calculate(args.poi[0], resume=not args.no_resume)
    else:
        calc.calculate_batch(args.poi, resume=not args.no_resume)


if __name__ == "__main__":
    main()
