import speedtest
import time


def run_speed_test():
    try:
        print("Initializing speed test (this may take 30-60 seconds)...\n")

        st = speedtest.Speedtest()

        print("Finding the best server...")
        st.get_best_server()
        server_info = st.get_best_server()
        print(f"Best server: {server_info['sponsor']} ({server_info['name']}, {server_info['country']})")
        print(f"Initial ping to server: {st.results.ping:.2f} ms\n")

        print("Testing download speed...")
        download_bits = st.download()
        download_mbps = download_bits / 1_000_000
        print(f"Download speed: {download_mbps:.2f} Mbps")

        print("Testing upload speed...")
        upload_bits = st.upload()
        upload_mbps = upload_bits / 1_000_000
        print(f"Upload speed: {upload_mbps:.2f} Mbps")

        final_ping = st.results.ping
        print(f"\nFinal results:")
        print(f"   Download: {download_mbps:.2f} Mbps")
        print(f"   Upload:   {upload_mbps:.2f} Mbps")
        print(f"   Ping:     {final_ping:.2f} ms")

        # Optional: generate a shareable results image link
        try:
            share_url = st.results.share()
            print(f"\nShareable results image: {share_url}")
        except:
            print("\nCould not generate share link (sometimes blocked by network settings).")

    except speedtest.ConfigRetrievalError:
        print("Error: Could not retrieve speedtest configuration (check internet connection).")
    except speedtest.NoMatchedServers:
        print("Error: No suitable speedtest servers found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    start_time = time.time()
    run_speed_test()
    print(f"\nTest completed in {time.time() - start_time:.1f} seconds.")