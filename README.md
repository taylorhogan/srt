# OAAS - Observatory As A Service
# Or
# Autonomous Social Robotic Observatory

Imagine you are an avid reader, and have a list of books you would like to read. If only your local libraries could read your list, and let you know which one are available right now.
This project is similar, but for Astronomy. You provide a list of objects that you would like to image, along with a recipe on how to do the imaging, and the Observatory tries to optimize each night
with what Deep Sky Objects are visible, and requested. Once complete, the system returns the image. 
All this interaction is done though social media (currently Mastodon). 
All the details of imaging have nice defaults but the user can override

#### A Journey through telescope Automation
- No imaging equipment, just sitting out under the stars, letting your eyes adjust to the wonder.
- Using a non guided, eyepiece only. Think binoculars or a Dobsonian. I have often been asked what is the best beginner telescope. My answer is a bottle of bourbon and a set of binoculars.
- Mounted telescope with goto capability, no camera
- Mounted telescope with a camera. Automation allows for the user to program one or more capture images for the night. No provision for understanding the weather or best sky objects for the evening. Automation software (think NINA or ASIAir) is very capable but complex
- User provides a list of objects and to acquire. Scope each night determines what are the best objects (if any) to capture, and returns an image fully calibrated and processed when some level of Signal to Noise ration has been achieved. User interaction is through a social network and all subscribers see all interaction

## Architecture
The system is a collection of distributed services. Currently, MQTT is the communication methodology between services.
It is up to the user to map services to hardware. 

## My Hardware Block Diagram
![block diagram](iris.png)
## Appreciation for others
This work rest heavily on others
- Allsky
- Astroplan
- Astropy
- Mastodon
- N.I.N.A
- Pixinsight




