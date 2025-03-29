import threading
import social_server
import scheduler_server

def call_servers():
    print ("start")
    t1 = threading.Thread(target=start_social_server)
    t2 = threading.Thread(target=start_sched_server)
    t1.start()
    t2.start()
    print ("end")

def start_social_server():
    print ("start social server")
    social_server.main()

def start_sched_server():
    print ("start calendar server")
    while True:
        pass

if __name__ == '__main__':
    call_servers()