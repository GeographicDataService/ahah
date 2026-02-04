"""
Valhalla Cluster Manager

A module for setting up and managing a cluster of Valhalla routing engines
with Docker, optimised for high-performance parallel routing calculations.

Architecture:
- Multiple Valhalla Workers: Independent routing engines sharing the same tile data
- Docker Compose: Manages container orchestration
- Shared Tile Storage: All workers use the same pre-built tiles (built once, shared by all)

Usage:
    from valhalla_cluster import ValhallaCluster
    
    cluster = ValhallaCluster(num_workers=8, tiles_dir="./valhalla_tiles")
    cluster.build_tiles()
    cluster.generate_compose()
    cluster.apply_optimisations()
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import docker
import yaml


class ValhallaCluster:
    """
    Manages a Valhalla routing cluster with multiple worker instances.
    
    Parameters
    ----------
    num_workers : int, default=8
        Number of Valhalla worker instances to run in parallel.
        More workers = higher throughput for concurrent requests.
        Recommended: 1 worker per 2 CPU cores, up to available RAM.
        
    base_worker_port : int, default=8010
        Starting port number for worker instances.
        Workers will use ports: base_worker_port, base_worker_port+1, ..., base_worker_port+num_workers-1
        Ensure these ports are available and not blocked by firewall.
        
    tiles_dir : str or Path, default="./valhalla_tiles"
        Directory for storing Valhalla tile data.
        Will contain: valhalla_tiles/, valhalla.json, valhalla_tiles.tar
        Requires ~15-25GB for UK data, ~100GB+ for Europe.
        
    project_dir : str or Path, optional
        Base project directory. Defaults to current working directory.
        Docker-compose.yml will be created here.
        
    valhalla_image : str, default="ghcr.io/gis-ops/docker-valhalla/valhalla:latest"
        Docker image to use for Valhalla instances.
        Alternatives:
        - "ghcr.io/valhalla/valhalla:latest" (official image)
        - "ghcr.io/gis-ops/docker-valhalla/valhalla:latest" (GIS-OPS with extras)
        
    tile_url : str, default="https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf"
        URL to download OpenStreetMap PBF data from.
        Find regional extracts at: https://download.geofabrik.de/
        
    build_elevation : bool, default=False
        Whether to include elevation data in tiles.
        Increases build time and storage but enables elevation-aware routing.
        
    build_admins : bool, default=True
        Whether to build administrative boundary data.
        Required for some routing features (e.g., country-aware restrictions).
        
    build_time_zones : bool, default=True
        Whether to include timezone data.
        Enables time-aware routing calculations.
        
    Attributes
    ----------
    config_path : Path
        Path to the valhalla.json configuration file.
        
    compose_file : Path
        Path to the generated docker-compose.yml file.
        
    Examples
    --------
    Basic setup with defaults:
    
    >>> cluster = ValhallaCluster()
    >>> cluster.build_tiles()  # One-time tile building (~20 mins)
    >>> cluster.generate_compose()
    >>> cluster.start()
    
    Custom configuration:
    
    >>> cluster = ValhallaCluster(
    ...     num_workers=4,
    ...     tiles_dir="/data/valhalla",
    ...     tile_url="https://download.geofabrik.de/europe/germany-latest.osm.pbf"
    ... )
    """
    
    def __init__(
        self,
        num_workers: int = 8,
        base_worker_port: int = 8010,
        tiles_dir: Union[str, Path] = "./valhalla_tiles",
        project_dir: Optional[Union[str, Path]] = None,
        valhalla_image: str = "ghcr.io/gis-ops/docker-valhalla/valhalla:latest",
        tile_url: str = "https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf",
        build_elevation: bool = False,
        build_admins: bool = True,
        build_time_zones: bool = True,
    ):
        self.num_workers = num_workers
        self.base_worker_port = base_worker_port
        self.valhalla_image = valhalla_image
        self.tile_url = tile_url
        self.build_elevation = build_elevation
        self.build_admins = build_admins
        self.build_time_zones = build_time_zones
        
        # Paths
        self.project_dir = Path(project_dir).resolve() if project_dir else Path.cwd()
        self.tiles_dir = Path(tiles_dir).resolve()
        self.tiles_dir_name = Path(tiles_dir).name  # Relative name for docker-compose
        self.config_path = self.tiles_dir / "valhalla.json"
        self.compose_file = self.project_dir / "docker-compose.yml"
        
        # Ensure tiles directory exists
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Docker client (lazy loaded)
        self._client = None
    
    @property
    def client(self) -> docker.DockerClient:
        """Lazy-loaded Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client
    
    @property
    def worker_ports(self) -> List[int]:
        """List of ports used by worker instances."""
        return list(range(self.base_worker_port, self.base_worker_port + self.num_workers))
    
    def tiles_exist(self) -> bool:
        """Check if tiles have already been built."""
        tiles_path = self.tiles_dir / "valhalla_tiles"
        return tiles_path.exists() and self.config_path.exists()
    
    def build_tiles(self, force: bool = False) -> bool:
        """
        Build Valhalla routing tiles from OSM data.
        
        This is a one-time operation that downloads OSM data and builds
        the routing graph tiles. Takes 10-30+ minutes depending on data size
        and hardware.
        
        Parameters
        ----------
        force : bool, default=False
            If True, rebuild tiles even if they already exist.
            
        Returns
        -------
        bool
            True if tiles were built successfully.
            
        Notes
        -----
        - Tiles only need to be built once
        - All workers share the same tiles
        - PBF files are cleaned up after building to save space
        """
        # Check if tiles already exist
        if self.tiles_exist() and not force:
            print("⚠️  Tiles and config already exist.")
            print("   Use force=True to rebuild.")
            return True
        
        print(f"Building tiles with {self.valhalla_image}...")
        print(f"Downloading from: {self.tile_url}")
        print("This will take 10-30+ minutes depending on your hardware.\n")
        
        # Pull image
        print("Pulling Docker image...")
        self.client.images.pull(self.valhalla_image)
        
        # Remove any existing builder container
        try:
            old = self.client.containers.get("valhalla_builder")
            print("Removing existing builder container...")
            old.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        # Environment for tile building
        env = {
            "tile_urls": self.tile_url,
            "build_elevation": str(self.build_elevation),
            "build_admins": str(self.build_admins),
            "build_time_zones": str(self.build_time_zones),
        }
        
        # Start builder container
        print("Starting builder container...")
        container = self.client.containers.run(
            self.valhalla_image,
            name="valhalla_builder",
            detach=True,
            volumes={str(self.tiles_dir): {"bind": "/custom_files", "mode": "rw"}},
            environment=env,
        )
        
        print(f"Builder container started (id={container.short_id})")
        print("\nMonitoring tile build progress (this will take a while):")
        print("Ctrl+C to stop monitoring (build continues in background)\n")
        
        # Patterns indicating tile build is complete
        completion_patterns = [
            r"Starting Valhalla service",
            r"tcp://\*:8002",
            r"Listening at",
            r"INFO\] Starting",
            r"valhalla_service.*started",
        ]
        completion_regex = re.compile("|".join(completion_patterns), re.IGNORECASE)
        
        # Progress patterns for user feedback
        progress_patterns = {
            r"Downloading": "📥 Downloading OSM data...",
            r"Parsing.*ways": "🔄 Parsing ways...",
            r"Parsing.*nodes": "🔄 Parsing nodes...",
            r"Parsing.*relations": "🔄 Parsing relations...",
            r"Building tiles": "🏗️  Building tiles...",
            r"Creating graph": "🔗 Creating graph...",
            r"Adding elevation": "⛰️  Adding elevation...",
            r"Creating.*tar": "📦 Creating tile archive...",
            r"Finished": "✅ Build step finished",
        }
        
        # Stream logs and watch for completion
        tiles_built = False
        last_status = None
        try:
            for line in container.logs(stream=True, follow=True):
                decoded = line.decode('utf-8').strip()
                
                # Skip empty lines and curl progress bars
                if not decoded or re.match(r'^\s*\d+\s+\d+M?\s+\d+', decoded):
                    continue
                
                print(decoded)
                
                # Show progress status updates
                for pattern, status in progress_patterns.items():
                    if re.search(pattern, decoded, re.IGNORECASE) and status != last_status:
                        print(f"\n{'='*60}")
                        print(f"  {status}")
                        print(f"{'='*60}\n")
                        last_status = status
                        break
                
                # Check if tile building is complete
                if completion_regex.search(decoded):
                    print("\n" + "=" * 60)
                    print("🎉 Tile building complete! Service is starting...")
                    print("=" * 60)
                    tiles_built = True
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️  Stopped monitoring logs. Container is still running.")
            print("   To check progress: docker logs -f valhalla_builder")
            print("   To stop: docker stop valhalla_builder && docker rm valhalla_builder")
            return False
        
        # Check if tiles exist
        if not tiles_built:
            time.sleep(2)
            if self.tiles_exist():
                print("\n" + "=" * 60)
                print("🎉 Detected tile files - build appears complete!")
                print("=" * 60)
                tiles_built = True
        
        if tiles_built:
            print("\n✅ Tiles built successfully!")
            
            # Stop and remove the builder container
            print("Stopping builder container...")
            try:
                container.stop(timeout=10)
            except Exception as e:
                print(f"  Note: {e}")
            
            # Clean up PBF files to save space
            print("Cleaning up PBF files...")
            for pbf in self.tiles_dir.glob("*.osm.pbf"):
                print(f"  Removing {pbf.name}")
                pbf.unlink()
            
            # Remove container
            try:
                container.remove()
                print("Builder container removed.")
            except Exception as e:
                print(f"  Note: {e}")
            
            # Verify outputs
            print("\nVerifying outputs:")
            if (self.tiles_dir / "valhalla_tiles").exists():
                print(f"  ✅ {self.tiles_dir}/valhalla_tiles/")
            if self.config_path.exists():
                print(f"  ✅ {self.config_path}")
            if (self.tiles_dir / "valhalla_tiles.tar").exists():
                print(f"  ✅ {self.tiles_dir}/valhalla_tiles.tar")
            
            return True
        else:
            # Check container status
            container.reload()
            if container.status == 'exited':
                result = container.wait()
                print(f"\n❌ Tile build failed with status code {result['StatusCode']}")
                print("Check logs: docker logs valhalla_builder")
            else:
                print(f"\n⚠️  Container still running (status: {container.status})")
                print("Logs may have been interrupted. Check: docker logs -f valhalla_builder")
            return False
    
    def generate_compose(self) -> Path:
        """
        Generate docker-compose.yml for the Valhalla cluster.
        
        Returns
        -------
        Path
            Path to the generated docker-compose.yml file.
            
        Notes
        -----
        The generated compose file:
        - Creates N worker containers (valhalla_worker_0, valhalla_worker_1, ...)
        - Maps each to a sequential port starting from base_worker_port
        - Mounts tile directory as read-only
        - Includes health checks for each worker
        - Uses a shared bridge network
        """
        
        compose = {
            'services': {},
            'networks': {'valhalla_net': {'driver': 'bridge'}}
        }
        
        # Calculate relative path from project dir to tiles dir
        try:
            rel_tiles = self.tiles_dir.relative_to(self.project_dir)
            tiles_mount = f"./{rel_tiles}"
        except ValueError:
            # Tiles dir is not under project dir, use absolute path
            tiles_mount = str(self.tiles_dir)
        
        for i in range(self.num_workers):
            worker_name = f"valhalla_worker_{i}"
            compose['services'][worker_name] = {
                'image': self.valhalla_image,
                'container_name': worker_name,
                'ports': [f"{self.base_worker_port + i}:8002"],
                'volumes': [f"{tiles_mount}:/custom_files:ro"],
                'entrypoint': ['/bin/bash', '-c'],
                'command': ['valhalla_service /custom_files/valhalla.json 1'],
                'networks': ['valhalla_net'],
                'restart': 'unless-stopped',
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8002/status'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '30s'
                }
            }
        
        with open(self.compose_file, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created: {self.compose_file}")
        print(f"   Workers: {self.num_workers}")
        print(f"   Ports: {self.base_worker_port}-{self.base_worker_port + self.num_workers - 1}")
        print(f"   Volume: {tiles_mount}:/custom_files:ro")
        
        return self.compose_file
    
    def apply_optimisations(
        self,
        search_cutoff: int = 1000,
        node_snap_tolerance: int = 100,
        street_side_tolerance: int = 100,
        max_locations: int = 1000,
        max_matrix_distance: int = 2_000_000,
        max_matrix_location_pairs: int = 5_000_000,
        timeout_seconds: int = 300,
        costmatrix_allow_second_pass: bool = True,
        costmatrix_check_reverse_connection: bool = False,
        source_to_target_algorithm: str = "costmatrix",
        max_reserved_locations_costmatrix: int = 1000,
        use_simple_mem_cache: bool = True,
        max_cache_size: int = 5_000_000_000,
    ) -> None:
        """
        Apply performance optimisations to the Valhalla configuration.
        
        These settings are tuned for high-throughput batch routing (e.g., AHAH)
        rather than interactive use.
        
        Parameters
        ----------
        search_cutoff : int, default=1000
            Maximum search radius in metres for snapping points to the road network.
            Lower values fail faster on bad points. Default 1km.
            
        node_snap_tolerance : int, default=100
            Tolerance in metres for snapping to road nodes.
            
        street_side_tolerance : int, default=100
            Tolerance in metres for snapping to street sides.
            
        max_locations : int, default=1000
            Maximum number of locations per routing request.
            
        max_matrix_distance : int, default=2_000_000
            Maximum total distance in metres for matrix calculations.
            
        max_matrix_location_pairs : int, default=5_000_000
            Maximum number of origin-destination pairs in a matrix request.
            
        timeout_seconds : int, default=300
            Request timeout in seconds.
            
        costmatrix_allow_second_pass : bool, default=True
            Whether to allow a second routing pass for failed routes.
            Set False for speed if you don't need fallback routing.
            
        costmatrix_check_reverse_connection : bool, default=False
            Whether to check if destination is accessible from wrong direction.
            Disable for speed if your destinations are well-formed.
            
        source_to_target_algorithm : str, default="costmatrix"
            Algorithm for many-to-many routing.
            "costmatrix": Faster, returns only times/distances.
            "timedistancematrix": Slower, can return more detail.
            
        max_reserved_locations_costmatrix : int, default=1000
            Pre-allocated memory for cost matrix locations.
            
        use_simple_mem_cache : bool, default=True
            Enable in-memory tile caching for faster repeated access.
            
        max_cache_size : int, default=5_000_000_000
            Maximum memory cache size in bytes (default 5GB).
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {self.config_path}\n"
            )
        
        config = json.loads(self.config_path.read_text())
        
        # LOKI: Snapping & Failure Limits
        if "loki" not in config:
            config["loki"] = {}
        if "service_defaults" not in config["loki"]:
            config["loki"]["service_defaults"] = {}
        
        config["loki"]["service_defaults"]["search_cutoff"] = search_cutoff
        config["loki"]["service_defaults"]["node_snap_tolerance"] = node_snap_tolerance
        config["loki"]["service_defaults"]["street_side_tolerance"] = street_side_tolerance
        
        # SERVICE LIMITS
        if "service_limits" not in config:
            config["service_limits"] = {}
        if "auto" not in config["service_limits"]:
            config["service_limits"]["auto"] = {}
        
        config["service_limits"]["auto"]["max_locations"] = max_locations
        config["service_limits"]["auto"]["max_matrix_distance"] = max_matrix_distance
        config["service_limits"]["auto"]["max_matrix_location_pairs"] = max_matrix_location_pairs
        
        # HTTPD: Timeout
        if "httpd" not in config:
            config["httpd"] = {}
        if "service" not in config["httpd"]:
            config["httpd"]["service"] = {}
        
        config["httpd"]["service"]["timeout_seconds"] = timeout_seconds
        
        # THOR: Routing Engine
        if "thor" not in config:
            config["thor"] = {}
        
        config["thor"]["costmatrix_allow_second_pass"] = costmatrix_allow_second_pass
        config["thor"]["costmatrix_check_reverse_connection"] = costmatrix_check_reverse_connection
        config["thor"]["source_to_target_algorithm"] = source_to_target_algorithm
        config["thor"]["max_reserved_locations_costmatrix"] = max_reserved_locations_costmatrix
        
        # MJOLNIR: Memory Cache
        if "mjolnir" not in config:
            config["mjolnir"] = {}
        
        config["mjolnir"]["use_simple_mem_cache"] = use_simple_mem_cache
        config["mjolnir"]["max_cache_size"] = max_cache_size
        
        # Backup original config
        backup_path = self.tiles_dir / "valhalla.json.backup"
        if not backup_path.exists():
            backup_path.write_text(self.config_path.read_text())
            print(f"✅ Backed up original to: {backup_path}")
        
        # Write optimised config
        self.config_path.write_text(json.dumps(config, indent=2))
        print(f"✅ Saved optimised config: {self.config_path}")
        
        # Summary
        print("\nOptimisations applied:")
        print("=" * 60)
        print(f"  LOKI - search_cutoff: {search_cutoff}m")
        print(f"  LOKI - snap tolerances: {node_snap_tolerance}m")
        print(f"  SERVICE LIMITS - max_locations: {max_locations}")
        print(f"  SERVICE LIMITS - max_matrix_pairs: {max_matrix_location_pairs:,}")
        print(f"  THOR - algorithm: {source_to_target_algorithm}")
        print(f"  THOR - second_pass: {costmatrix_allow_second_pass}")
        print(f"  MJOLNIR - cache: {use_simple_mem_cache}, {max_cache_size / 1e9:.1f}GB")
        print(f"  HTTPD - timeout: {timeout_seconds}s")
        print("=" * 60)
        print("\n⚠️  Restart the cluster for changes to take effect:")
        print("   docker-compose restart")
    
    def start(self) -> None:
        """Start the Valhalla cluster using docker-compose."""
        if not self.compose_file.exists():
            print("docker-compose.yml not found. Generating...")
            self.generate_compose()
        
        print("Starting Valhalla cluster...")
        subprocess.run(
            ["docker-compose", "-f", str(self.compose_file), "up", "-d"],
            cwd=self.project_dir,
            check=True
        )
        print(f"\n✅ Cluster started with {self.num_workers} workers")
        print(f"   Ports: {self.base_worker_port}-{self.base_worker_port + self.num_workers - 1}")
    
    def stop(self) -> None:
        """Stop the Valhalla cluster."""
        if not self.compose_file.exists():
            print("No docker-compose.yml found.")
            return
        
        print("Stopping Valhalla cluster...")
        subprocess.run(
            ["docker-compose", "-f", str(self.compose_file), "down"],
            cwd=self.project_dir,
            check=True
        )
        print("✅ Cluster stopped")
    
    def restart(self) -> None:
        """Restart the Valhalla cluster."""
        print("Restarting Valhalla cluster...")
        subprocess.run(
            ["docker-compose", "-f", str(self.compose_file), "restart"],
            cwd=self.project_dir,
            check=True
        )
        print("✅ Cluster restarted")
    
    def status(self) -> Dict[str, str]:
        """
        Check the status of all worker containers.
        
        Returns
        -------
        dict
            Mapping of worker names to their status.
        """
        statuses = {}
        for i in range(self.num_workers):
            worker_name = f"valhalla_worker_{i}"
            try:
                container = self.client.containers.get(worker_name)
                statuses[worker_name] = container.status
            except docker.errors.NotFound:
                statuses[worker_name] = "not found"
        
        return statuses
    
    def health_check(self, timeout: int = 5) -> Dict[str, bool]:
        """
        Check health of all workers by calling their /status endpoint.
        
        Parameters
        ----------
        timeout : int, default=5
            Request timeout in seconds.
            
        Returns
        -------
        dict
            Mapping of worker ports to health status (True/False).
        """
        import requests
        
        health = {}
        for port in self.worker_ports:
            try:
                resp = requests.get(f"http://localhost:{port}/status", timeout=timeout)
                health[port] = resp.status_code == 200
            except requests.RequestException:
                health[port] = False
        
        return health
    
    def __repr__(self) -> str:
        return (
            f"ValhallaCluster(num_workers={self.num_workers}, "
            f"ports={self.base_worker_port}-{self.base_worker_port + self.num_workers - 1}, "
            f"tiles_dir='{self.tiles_dir}')"
        )


def main():
    """CLI entry point for setting up a Valhalla cluster."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up and manage a Valhalla routing cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build tiles and set up cluster with defaults
  python valhalla_cluster.py setup
  
  # Custom number of workers
  python valhalla_cluster.py setup --workers 4
  
  # Use different OSM data
  python valhalla_cluster.py setup --tile-url https://download.geofabrik.de/europe/germany-latest.osm.pbf
  
  # Just generate docker-compose (tiles already built)
  python valhalla_cluster.py compose --workers 8
  
  # Apply optimisations to existing config
  python valhalla_cluster.py optimise
  
  # Start/stop/restart cluster
  python valhalla_cluster.py start
  python valhalla_cluster.py stop
  python valhalla_cluster.py restart
        """
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "build", "compose", "optimise", "start", "stop", "restart", "status"],
        help="Command to run"
    )
    parser.add_argument(
        "--workers", "-n",
        type=int,
        default=8,
        help="Number of worker instances (default: 8)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8010,
        help="Base port for workers (default: 8010)"
    )
    parser.add_argument(
        "--tiles-dir", "-d",
        type=str,
        default="./valhalla_tiles",
        help="Directory for tile storage (default: ./valhalla_tiles)"
    )
    parser.add_argument(
        "--tile-url", "-u",
        type=str,
        default="https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf",
        help="URL to download OSM PBF data"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="ghcr.io/gis-ops/docker-valhalla/valhalla:latest",
        help="Docker image for Valhalla"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild/regenerate even if files exist"
    )
    
    args = parser.parse_args()
    
    cluster = ValhallaCluster(
        num_workers=args.workers,
        base_worker_port=args.port,
        tiles_dir=args.tiles_dir,
        valhalla_image=args.image,
        tile_url=args.tile_url,
    )
    
    if args.command == "setup":
        print(f"Setting up Valhalla cluster: {cluster}")
        cluster.build_tiles(force=args.force)
        cluster.generate_compose()
        cluster.apply_optimisations()
        print("\n✅ Setup complete! Run 'docker-compose up -d' to start the cluster.")
        
    elif args.command == "build":
        cluster.build_tiles(force=args.force)
        
    elif args.command == "compose":
        cluster.generate_compose()
        
    elif args.command == "optimise":
        cluster.apply_optimisations()
        
    elif args.command == "start":
        cluster.start()
        
    elif args.command == "stop":
        cluster.stop()
        
    elif args.command == "restart":
        cluster.restart()
        
    elif args.command == "status":
        statuses = cluster.status()
        print("\nCluster Status:")
        print("-" * 40)
        for worker, status in statuses.items():
            icon = "✅" if status == "running" else "❌"
            print(f"  {icon} {worker}: {status}")
        
        print("\nHealth Check:")
        health = cluster.health_check()
        for port, healthy in health.items():
            icon = "✅" if healthy else "❌"
            print(f"  {icon} Port {port}: {'healthy' if healthy else 'unhealthy'}")


if __name__ == "__main__":
    main()
