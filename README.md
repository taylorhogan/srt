# SRT Social Robotic Telescope
The goal of this project is to remove/reduce human interaction in astronomy. Users interact with the system
through a social messaging services (Currently Mastodon). Users request images of Deep Sky Objects and the system
optimizes each night's imaging based on observability. When complete, the system adds the processed image to its catalog
and alerts the user that it has completed. 
My goal is to have all this automated, I never want to be in the observatory. This all just happens. 
For now this is very very specific to my observatory, but I hope to generalize it in the future.

- **Architecture**

    - Social Server 
      - Interaction with Social Media (Currently Mastodon)
    - Scheduler
      - Manages the day and evening tasks of the observatory
    - Vision Safety
      - Cameras look for the state of the roof and telescope
    - Audio Safety
      - Microphones compare known good sounds to what is being heard to understand if there is a problem
    - Image Processing
      - Transforming raw images to a pretty picture
    - Servers communicate through mqtt
- 

## Commands

## My Hardware Block Diagram
![block diagram](doc/iris.png)
## Appreciation for others
This work rest heavily on others
- Allsky
- Astroplan
- Astropy
- Mastodon
- N.I.N.A
- Pixinsight

## How to build
Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh

exit shell and reopen shell

Download the repo

git clone https://github.com/taylorhogan/srt.git

cd srt

Create a virtual environment (only needed once)

uv venv venv

Activate the virtual environment

follow the commands from uv above, something like ./venv/bin/activate

uv pip install -r requirements.txt

cd configs

cp configs_blank_private.py configs_private.py






