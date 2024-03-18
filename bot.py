from mastodon import Mastodon, StreamListener
import baseconfig as cfg



config = cfg.FlowConfig().config

def do_notification(notification, m):
    print(notification['type'])
    account = notification.account.acct
    command = notification.status.content.lower()
    if notification['type'] != 'mention':
        return

    if 'm31' in command:
        media = m.media_post("m31.jpg", description="m31")
        m.status_post("Here you go " + '@' + account, media_ids=media)
    else:
        m.status_post("I do not know that object! " + '@' + account)








class TheStreamListener(StreamListener):

    def __init__(self, m):
        self.m = m

    def on_update(self, status):
        print(f"Got update: {status['content']}")

    def on_notification(self, notification):
        do_notification(notification, self.m)

the_mastodon = Mastodon(
    access_token=config["mastodon"]["access_token"],
    api_base_url=config["mastodon"]["api_base_url"]
)

the_mastodon.status_post("I am awake")
user = the_mastodon.stream_user(TheStreamListener(the_mastodon))
print("end")
