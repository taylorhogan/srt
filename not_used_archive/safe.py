import paho.mqtt.client as mqtt

def message_handling(client, userdata, msg):
    print(f"{msg.topic}: {msg.payload.decode()}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

broker = "raspberrypi.local"
port = 1883
username = "indi-allsky"
password = "1415"
client_id = ""


client = mqtt.Client()
#client = mqtt.Client()
client.username_pw_set(username, password)
client.on_message = message_handling
client.on_connect = on_connect


client.connect(broker, port)
status = client.subscribe('indi-allsky/temp')
print (status)

try:
    print("Press CTRL+C to exit...")
    client.loop_forever()
except Exception:
    print("Caught an Exception, something went wrong...")
finally:
    print("Disconnecting from the MQTT broker")
    client.disconnect()

