import sys
import super_user_commands as su


def printarg(mystring):
    print(mystring)

def lights_on (arg):
    su.kasa_exterior_lights_on()


def shut_down(arg):
    su.shutdown_command()

if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])