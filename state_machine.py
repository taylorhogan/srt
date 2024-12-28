from enum import Enum


class State(Enum):
    WaitingForNoon = 0,
    WaitingForSunset= 1,
    WaitingForSunrise= 2,
    RunningActiveSequence=3,
    RunningTestSequence=4,



def determine_state ()->State:
    return State.WaitingForSunset


def do_state (State)->State:
    return State.WaitingForSunrise

if __name__ == '__main__':
    do_state(determine_state())

