# Autonomous Social Robotic Observatory


The hope for this projects is to allow control of a telescope through high level interaction through a social network. All the details of imaging have nice defaults but can be incrementally overriden by the use. At the highest level a user can request an image of a Deep Sky Object. If the imagine has been previously created, that will be returned with a link to a full resolution image available for download. If the image has not been taken it will be added to the queue. Each night the telescope will decide if it is safe to image an optimize the aquisition of the queue given the position of deep sky objects. After each session, intermediate images will be calibrated, and processed. Once a desired level of Signal to Noise ration has been achieved, the DSO will be removed from the queue and be available for download.

#### My Journey through telescope Automation
- No imaging equipment, just sitting out under the stars, letting your eyes adjust to the wonder.
- Using a non guided, eyepiece only. Think binoculars or a Dobsonian. I have often been asked what is the best beginner telescope. My answer is a bottle of bourbon and a set of binoculars.
- Mounted telescope with goto capability, no camera
- Mounted telescope with a camera. Automation allows for the user to program one or more capture images for the night. No provision for understanding the weather or best sky objects for the evening. Automation software (think NINA or ASIAir) is very capable but complex
- User provides a list of objects and to acquire. Scope each night determines what are the best objects (if any) to capture, and returns an image fully calibrated and processed when some level of Signal to Noise ration has been achieved. User interaction is through a social network and all subscribers see all interaction

## Architecture
The system is a collection of distributed services. Currently, MQTT is the communication methodology between services.
It is up to the user to map services to hardware. 
## Appreciation for others
This work rest heavily on others
- Allsky
- Astroplan
- Astropy
- Mastodon
- 




