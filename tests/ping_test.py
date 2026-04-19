#!/usr/bin/env python3
"""
Roundtrip time checker for GPU server via Tailscale
"""
import subprocess
import platform
import statistics
from typing import List, Tuple, Optional
import time


def ping_host(
    host: str,
    count: int = 4,
    timeout: int = 10,
) -> Tuple[bool, Optional[List[float]], Optional[str]]:
    """
    Ping a host and return roundtrip times in milliseconds.
    
    Args:
        host: IP address or hostname of the server
        count: Number of ping packets to send
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success: bool, times: List[float] or None, error: str or None)
        Times are in milliseconds
    """
    try:
        # Use platform-specific ping command
        if platform.system() == "Windows":
            # Windows ping
            cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), host]
        else:
            # macOS/Linux ping
            cmd = ["ping", "-c", str(count), "-W", str(timeout * 1000), host]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        
        if result.returncode != 0:
            return False, None, f"Ping failed: {result.stderr}"
        
        # Parse roundtrip times from output
        times = []
        for line in result.stdout.split("\n"):
            if "time=" in line or "time<" in line:
                # Extract time value (works for both "time=X.XXms" and "time<1ms")
                try:
                    if "time=" in line:
                        time_str = line.split("time=")[1].split("ms")[0].strip()
                        times.append(float(time_str))
                except (IndexError, ValueError):
                    continue
        
        if not times:
            return False, None, "No roundtrip times parsed from ping output"
        
        return True, times, None
        
    except subprocess.TimeoutExpired:
        return False, None, f"Ping command timed out (>{timeout}s)"
    except FileNotFoundError:
        return False, None, "ping command not found"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


def print_statistics(times: List[float], host: str) -> None:
    """Print ping statistics."""
    print(f"\n{'='*60}")
    print(f"Ping Statistics for {host}")
    print(f"{'='*60}")
    print(f"Packets sent:     {len(times)}")
    print(f"Min roundtrip:    {min(times):.2f} ms")
    print(f"Max roundtrip:    {max(times):.2f} ms")
    print(f"Mean roundtrip:   {statistics.mean(times):.2f} ms")
    if len(times) > 1:
        print(f"Std deviation:    {statistics.stdev(times):.2f} ms")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check roundtrip time to GPU server via Tailscale"
    )
    parser.add_argument(
        "host",
        nargs="?",
        default=None,
        help="GPU server IP or hostname (Tailscale IP). If not provided, will prompt.",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=4,
        help="Number of ping packets to send (default: 4)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=1,
        help="Continuous ping interval in seconds (0 for single ping)",
    )
    
    args = parser.parse_args()
    
    # Get host if not provided
    host = args.host
    if not host:
        host = input("Enter GPU server IP/hostname (Tailscale): ").strip()
        if not host:
            print("No host specified. Exiting.")
            return
    
    print(f"Pinging {host}...\n")
    
    if args.interval <= 0:
        # Single ping
        success, times, error = ping_host(host, count=args.count, timeout=args.timeout)
        
        if success:
            print_statistics(times, host)
        else:
            print(f"❌ Ping failed: {error}\n")
    else:
        # Continuous ping
        all_times = []
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"Iteration {iteration}...")
                success, times, error = ping_host(
                    host, count=args.count, timeout=args.timeout
                )
                
                if success:
                    all_times.extend(times)
                    print_statistics(times, host)
                else:
                    print(f"❌ Ping failed: {error}\n")
                
                if args.interval > 0:
                    time.sleep(args.interval)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            if all_times:
                print("\n📊 Overall Statistics:")
                print_statistics(all_times, host)


if __name__ == "__main__":
    main()
