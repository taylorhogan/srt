import speedtest
import time


def get_speed():
    """Run a speed test and return results as a dict.

    Returns:
        dict with keys 'download_mbps', 'upload_mbps', 'ping_ms', 'server'
        or None on failure.
    """
    try:
        print("Speed test: initializing Speedtest()")
        st = speedtest.Speedtest()
        print("Speed test: Speedtest() OK, getting best server…")
        server = st.get_best_server()
        print(f"Speed test: server selected: {server}")
        print("Speed test: running download…")
        download_mbps = st.download() / 1_000_000
        print(f"Speed test: download {download_mbps:.1f} Mbps, running upload…")
        upload_mbps = st.upload() / 1_000_000
        print(f"Speed test: upload {upload_mbps:.1f} Mbps — done")
        return {
            "download_mbps": round(download_mbps, 1),
            "upload_mbps":   round(upload_mbps, 1),
            "ping_ms":       round(st.results.ping, 1),
            "server":        f"{server['sponsor']} ({server['name']})",
        }
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