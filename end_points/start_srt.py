from multiprocessing import Process
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from cmd_processing import social_server
import scheduler_server

if __name__ == "__main__":
    # Create two processes
    p1 = Process(target=social_server.main)
    p2 = Process(target=scheduler_server.main)

    # Start both processes
    p1.start()
    p2.start()


    p1.join()
    p2.join()
