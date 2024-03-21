import threading


class Watchdog:
    def __init__(self, t, callback):
        self.time = t
        self.callback = callback

    def start(self):
        print("Watchdog")
        threading.Timer(self.time, self.callback).start()


def my_done():
    print("mydone")


w = Watchdog(1, my_done)
w.start()
