import paho.mqtt.client as mqtt
from enum import Enum
from typing import Dict, Tuple
import json
import time

# Define states
class State(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    STOPPED = "Stopped"

# State Machine (Subscriber/Listener)
class StateMachine:
    def __init__(self, broker="localhost", port=1883):
        self.current_state = State.IDLE
        self.transitions: Dict[Tuple[State, str], State] = {
            (State.IDLE, "start"): State.RUNNING,
            (State.RUNNING, "pause"): State.IDLE,
            (State.RUNNING, "stop"): State.STOPPED,
            (State.STOPPED, "reset"): State.IDLE,
            (State.IDLE, "reset"): State.IDLE,
        }
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port)
        self.client.loop_start()  # Start the network loop in a separate thread

    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to the broker."""
        if rc == 0:
            print("Connected to MQTT broker")
            # Subscribe to the topic where events are published
            self.client.subscribe("events/#")  # Wildcard for all event subtopics
        else:
            print(f"Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        """Callback when a message is received."""
        topic = msg.topic
        payload = msg.payload.decode("utf-8")
        try:
            event_data = json.loads(payload)
            event = event_data.get("event")
            print(f"Received from {topic}: {event}")
            self.update(event)
        except json.JSONDecodeError:
            print(f"Invalid message payload: {payload}")

    def update(self, event: str):
        """Handle event and transition states."""
        transition_key = (self.current_state, event)
        if transition_key in self.transitions:
            old_state = self.current_state
            self.current_state = self.transitions[transition_key]
            print(f"Transition: {old_state.value} -> {self.current_state.value} due to '{event}'")
        else:
            print(f"No transition for {self.current_state.value} with event '{event}'")

    def run(self):
        """Keep the state machine running."""
        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            self.client.loop_stop()
            self.client.disconnect()
            print("Disconnected from broker")

# Simulate an Event Source (Publisher)
def simulate_event_source(broker="localhost", port=1883):
    client = mqtt.Client()
    client.connect(broker, port)
    client.loop_start()

    # Simulate events
    events = [
        {"event": "start", "topic": "events/button"},
        {"event": "pause", "topic": "events/timer"},
        {"event": "stop", "topic": "events/sensor"},
        {"event": "reset", "topic": "events/button"},
        {"event": "unknown", "topic": "events/test"},
    ]

    for event in events:
        payload = json.dumps({"event": event["event"]})
        client.publish(event["topic"], payload)
        print(f"Published to {event['topic']}: {event['event']}")
        time.sleep(2)  # Delay between events

    client.loop_stop()
    client.disconnect()

# Run the example
if __name__ == "__main__":
    # Start the state machine (assumes a local MQTT broker like Mosquitto is running)
    sm = StateMachine(broker="localhost", port=1883)

    # Simulate event sources in a separate thread or process
    import threading
    threading.Thread(target=simulate_event_source, args=("localhost", 1883)).start()

    # Run the state machine
    sm.run()