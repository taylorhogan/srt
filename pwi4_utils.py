from pwi4_client import PWI4


print("Connecting to PWI4...")

pwi4 = PWI4()

s = pwi4.status()
print("Mount connected:", s.mount.is_connected)
print (s)
if not s.mount.is_connected:
    print("Connecting to mount...")
    s = pwi4.mount_connect()
    print("Mount connected:", s.mount.is_connected)

print("  RA/Dec: %.4f, %.4f" % (s.mount.ra_j2000_hours, s.mount.dec_j2000_degs))

