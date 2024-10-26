import os
import sys

import cv2 as cv
import paho.mqtt.client as paho
import time
import baseconfig as config
import shutil


def connect_to_broker(cfg):
    client = paho.Client()
    if client.connect("localhost", 1883, 60) != 0:
        print("Couldn't connect to the mqtt broker")
        sys.exit(1)
    return client


def take_snapshot():
    print("taking picture")
    cfg = config.FlowConfig().config
    no_image = cfg["camera safety"]["no_image"]
    to_path = cfg["camera safety"]["scope_view"]
    shutil.copyfile(no_image, to_path)
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    if ret:
        img_src = frame
        cv.imwrite(to_path, img_src)
        return True
    else:
        print("no Image")
        return False


def send_picture_and_status_to_broker(client, userdata, msg):
    cfg = config.FlowConfig().config
    image_path = cfg["camera safety"]["in_picture"]
    print("1")
    take_snapshot(image_path)
    print("2")

    with open(image_path, 'rb') as file:
        filecontent = file.read()
    print("3")
    byteArr = bytearray(filecontent)


    picture_topic = cfg["mqtt"]["ota_picture"]
    result = client.publish(picture_topic, byteArr, 0)
    msg_status = result[0]
    if msg_status == 0:
        print(f"message : Message sent to topic {picture_topic}")
    else:
        print(f"Failed to send message to topic {picture_topic}")

    mod_date = time.ctime(os.path.getmtime(image_path))
    picture_date = cfg["mqtt"]["picture_date"]
    result = client.publish(picture_date, mod_date, 0)
    msg_status = result[0]
    if msg_status == 0:
        print(f"message : Message sent to topic {picture_date}")
    else:
        print(f"Failed to send message to topic {picture_date}")


def subscribe_to_get_state(client, cfg):
    client.on_message = send_picture_and_status_to_broker
    topic = cfg["mqtt"]["observatory_state?"]
    client.subscribe(topic)


if __name__ == '__main__':
    cfg = config.FlowConfig().config
    client = connect_to_broker(cfg)
    subscribe_to_get_state(client, cfg)
    try:
        client.loop_forever()
    except Exception:
        print("Caught an Exception, something went wrong...")
    finally:
        print("Disconnecting from the MQTT broker")
        client.disconnect()
