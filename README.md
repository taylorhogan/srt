# SRT Social Robotic Telescope
The goal of this project is to remove/reduce human interaction in astronomy. Users interact with the system
through a social messaging services (Currently Mastodon). Users request images of Deep Sky Objects and the system
optimizes each night's imaging based on observability. When complete, the system adds the processed image to its catalog
and alerts the user that it has completed. 
My goal is to have all this automated, I never want to be in the observatory. This all just happens. 
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




