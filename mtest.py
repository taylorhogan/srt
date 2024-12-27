from mastodon import Mastodon

# Create an instance of the Mastodon class
mastodon = Mastodon(
    access_token='M67r7ztzqc8yWiNcokctuz0T5dJsL_P7Bh-o1VTHm9U',
    api_base_url='https://mastodon.social/'
)

# Post a new status update
mastodon.status_post('Hello, Mastodon!')

# Define a function to handle mentions
def handle_mention(status):
    if '@tmhobservatory' in status.content:
        mastodon.status_post('@' + status.account.username + ' Hello there!')

# Start streaming for mentions
mastodon.stream_user(handle_mention)