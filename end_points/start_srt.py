from multiprocessing import Process
import os
import subprocess
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from cmd_processing import social_server
from cmd_processing.social_server import RESTART_EXIT_CODE
import scheduler_server

if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    from sentry.internet_classify import run_speed_test
    run_speed_test()

    while True:
        p1 = Process(target=social_server.main)
        p2 = Process(target=scheduler_server.main)

        p1.start()
        p2.start()

        p1.join()  # wait for social server to exit
        exit_code = p1.exitcode

        # Always clean up the scheduler when the social server exits
        if p2.is_alive():
            p2.terminate()
            p2.join(timeout=10)

        if exit_code == RESTART_EXIT_CODE:
            result = subprocess.run(
                ["git", "-C", project_root, "pull"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"git pull failed (exit {result.returncode}):\n{result.stderr}")
                break
            continue  # relaunch both processes
        else:
            break  # normal or unexpected exit — don't loop
