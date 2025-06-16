import sys


import paho.mqtt.client as paho

to_sched = "iris/to_sched"
from_sched = "iris/from_sched"

def message_handling(client, userdata, msg):
    print (msg.topic)
    if msg.topic == topic1:
        print ("r_condition: " + str(msg.payload))

    if msg.topic == topic2:
        print("s_condition: " + str(msg.payload))

    if msg.topic == topic3:
        f = open('receive.jpg', 'wb')
        f.write(msg.payload)
        f.close()
        print('image received')



client = paho.Client()
client.on_message = message_handling

if client.connect("localhost", 1883, 60) != 0:
    print("Couldn't connect to the mqtt broker")
    sys.exit(1)


client.publish(from_sched, "This is a test", 0)


try:
    print("Press CTRL+C to exit...")
    client.loop_forever()
except Exception:
    print("Caught an Exception, something went wrong...")
finally:
    print("Disconnecting from the MQTT broker")
    client.disconnect()