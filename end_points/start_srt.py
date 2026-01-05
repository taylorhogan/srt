from multiprocessing import Process
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
