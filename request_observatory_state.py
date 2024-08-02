import sys
import time
import paho.mqtt.client as paho
import baseconfig as config
import vision_safety


def connect_to_broker(cfg):
    client = paho.Client()
    if client.connect("localhost", 1883, 60) != 0:
        print("Couldn't connect to the mqtt broker")
        sys.exit(1)
    return client

def do_something_with_returned_value (client, userdata, msg):
    cfg = config.FlowConfig().config
    picture_topic = cfg["mqtt"]["ota_picture"]
    roof_topic = cfg["mqtt"]["roof_state"]
    ota_topic = cfg["mqtt"]["ota_state"]
    date_topic = cfg["mqtt"]["picture_date"]

    if (msg.topic == picture_topic):
        print ("got picture")
        print(msg.topic)
        f = open (cfg["camera safety"]["out_picture"], "wb")
        f.write(msg.payload)
        f.close()
        cfg["camera safety"]["received_count"] = cfg["camera safety"]["received_count"] + 1

    if (msg.topic == date_topic):
        print("got date")
        cfg["camera safety"]["date_state"] = msg.payload
        cfg["camera safety"]["received_count"] = cfg["camera safety"]["received_count"] + 1

    if (cfg["camera safety"]["received_count"]==2):
        cfg["camera safety"]["valid_data"] = True
        client.disconnect()





def subscribe_to_get_state (client, cfg):

    client.on_message = do_something_with_returned_value
    picture_topic = cfg["mqtt"]["ota_picture"]
    client.subscribe(picture_topic)
    date_state = cfg["mqtt"]["picture_date"]
    client.subscribe(date_state)

def ask_for_state():

    cfg = config.FlowConfig().config
    cfg["camera safety"]["received_count"] = 0
    cfg["camera safety"]["valid_data"] = False

    client = connect_to_broker(cfg)
    subscribe_to_get_state(client, cfg)
    topic = cfg["mqtt"]["observatory_state?"]
    result = client.publish(topic, "ask", 0)
    msg_status = result[0]
    if msg_status == 0:
        print(f"message : Message sent to topic {topic}")
    else:
        print(f"Failed to send message to topic {topic}")

    try:
        print("Press CTRL+C to exit...")
        client.loop_forever()
    except Exception:
        print("Caught an Exception, something went wrong...")
    finally:
        print("Disconnecting from the MQTT broker")
        client.disconnect()




