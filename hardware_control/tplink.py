import tplinkcloud as TPLinkCloud


# Replace these with your TP-Link Kasa credentials
email = "your_email@example.com"
password = "your_password"

# Connect to the TP-Link Cloud
tplink = TPLinkCloud(email, password)
tplink.login()

# List available devices on your Kasa account
devices = tplink.getDeviceList()
print("Available devices:")
for device in devices:
    print(f"Device name: {device['alias']}, Type: {device['deviceType']}")

# Find a specific camera
for device in devices:
    if 'cam' in device['deviceType'].lower():  # Detect camera device
        camera_id = device['deviceId']
        print(f"Found a camera: {device['alias']} (ID: {camera_id})")

        # Fetch basic camera properties
        camera = tplink.getDevice(camera_id)

        # Grab a snapshot (if supported by your camera model)
        snapshot = camera.getLiveViewSnapshot()
        with open("camera_snapshot.jpg", "wb") as file:
            file.write(snapshot)
        print("Saved a snapshot from the Kasa camera.")

        # Exit after interacting with the first camera
        break
else:
    print("No camera found!")