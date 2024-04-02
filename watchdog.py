import threading


class Watchdog:
    def __init__(self, t, callback):
        self.time = t
        self.callback = callback
        self.timer_going = False

    def start(self):
        print("Watchdog")
        self.timer_going = True
        threading.Timer(self.time, self.callback).start()

    def is_timer_going(self):
        return self.timer_going

def my_done():
    print("i am done")


