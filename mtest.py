from mastodon import Mastodon
from mastodon.streaming import CallbackStreamListener
import time
# Create an instance of the Mastodon class
mastodon = Mastodon(
    access_token='M67r7ztzqc8yWiNcokctuz0T5dJsL_P7Bh-o1VTHm9U',
    api_base_url='https://mastodon.social/'
)

# Post a new status update
mastodon.status_post('Hello, Mastodon!')

# Define a function to handle mentions
def handle_mention(notification):
   if notification.type == "mention":
        print(notification.status)

# Start streaming for mentions
listener = CallbackStreamListener(notification_handler = handle_mention)
mastodon.stream_user(listener,run_async=True, reconnect_async=True)
while (True):
    time.sleep(10)