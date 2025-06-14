import sys
#https://medium.com/@potekh.anastasia/a-beginners-guide-to-mqtt-understanding-mqtt-mosquitto-broker-and-paho-python-mqtt-client-990822274923

import paho.mqtt.client as paho

topic1 = "iris/r_condition"
topic2 = "iris/s_condition"
topic3 = "iris/inside_jpg"
topic1q = "iris/from_sched"

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


client.publish(topic1q, "This is a test", 0)


try:
    print("Press CTRL+C to exit...")
    client.loop_forever()
except Exception:
    print("Caught an Exception, something went wrong...")
finally:
    print("Disconnecting from the MQTT broker")
    client.disconnect()