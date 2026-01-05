import sys
from cmd_processing import social_server

if __name__ == "__main__":

    print (len(sys.argv))
    if len(sys.argv) < 2:
        print ("1")
        social_server.post_social_message("alive")
    else:
        if len(sys.argv) < 3:
            social_server.post_social_message(sys.argv[1])
        else:
            social_server.post_social_message(sys.argv[1], sys.argv[2])


