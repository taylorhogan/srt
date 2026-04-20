import json
import subprocess
import sys
import time


def get_speed():
    """Run a speed test via speedtest-cli subprocess and return results as a dict.

    Returns:
        dict with keys 'download_mbps', 'upload_mbps', 'ping_ms', 'server'
        or None on failure.
    """
    try:
        print("Speed test: launching speedtest-cli --json …")
        result = subprocess.run(
            [sys.executable, "-m", "speedtest", "--json"],
            capture_output=True, text=True, timeout=120
        )
        print(f"Speed test: exit code {result.returncode}")
        if result.stderr:
            print(f"Speed test stderr: {result.stderr.strip()}")
        if result.returncode != 0:
            print(f"Speed test stdout: {result.stdout.strip()}")
            return None

        data = json.loads(result.stdout)
        return {
            "download_mbps": round(data["download"] / 1_000_000, 1),
            "upload_mbps":   round(data["upload"] / 1_000_000, 1),
            "ping_ms":       round(data["ping"], 1),
            "server":        f"{data['server']['sponsor']} ({data['server']['name']})",
        }
    except subprocess.TimeoutExpired:
        print("Speed test: timed out after 120s")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Speed test: failed to parse output: {e}")
        print(f"Speed test stdout was: {result.stdout[:500]}")
        return None
    except Exception as e:
        import traceback
        print(f"Speed test error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def run_speed_test():
    """Run a speed test and print results (standalone use)."""
    print("Running speed test (30-60 seconds)…")
    result = get_speed()
    if result:
        print(f"Download : {result['download_mbps']} Mbps")
        print(f"Upload   : {result['upload_mbps']} Mbps")
        print(f"Ping     : {result['ping_ms']} ms")
        print(f"Server   : {result['server']}")
    else:
        print("Speed test failed.")


if __name__ == "__main__":
    start_time = time.time()
    run_speed_test()
    print(f"\nCompleted in {time.time() - start_time:.1f} seconds.")
