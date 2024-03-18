# Autonomous Social Robotic Observatory
#### Using distrubuted systems with a theme of Flow Based programming


https://en.wikipedia.org/wiki/Flow-based_programming


The hope for this projects is to allow control of a telescope through high level interaction through a social network. All the details of imaging have nice defaults but can be incrementally overriden by the use. At the highest level a user can request an image of a Deep Sky Object. If the imagine has been previously created, that will be returned with a link to a full resolution image available for download. If the image has not been taken it will be added to the queue. Each night the telescope will decide if it is safe to image an optimize the aquisition of the queue given the position of deep sky objects. After each session, intermediate images will be calibrated, and processed. Once a desired level of Signal to Noise ration has been achieved, the DSO will be removed from the queue and be available for download.

command -> image_command | status_command | control_command

image_command -> "image" dso_descriptor ["using" recipe]

dso_descriptor -> "m" number | "ngc" number


