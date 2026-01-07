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

## How to build on linux/mac
Install uv ( a package manager for Python, similar to pip) 
https://uv.readthedocs.io/en/latest/installation.html 

`curl -LsSf https://astral.sh/uv/install.sh | sh`

exit shell and reopen new shell, otherwise uv will not be in your path 
Download the repo in a location of your choosing

`git clone https://github.com/taylorhogan/srt.git`  

go into the repo

`cd srt`  

Create a virtual environment (only needed once)  

`uv venv venv`   

Activate the virtual environment   
follow the commands from uv above to activate the virtual environment

`source venv/bin/activate`

Install the requirements (dragons be here)

`uv pip install -r requirements.txt`

copy the blank private config file, utlimately this will be a private config file

`cd configs`

`cp configs_blank_private.py configs_private.py`

to test the sound compare feature

`cd sentry`

`python audio_compare.py`

now make some noise for 10 seconds







