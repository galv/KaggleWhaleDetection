Learning about audio signal processign and classification with the data of this
competition:

https://www.kaggle.com/c/whale-detection-challenge

The "data" folder (too large to upload) can be downloaded from above site.
To convert .aiff to .wav, do:

for f in *.aiff; do ffmpeg -i "$f" "${f%.aiff}.wav"; done

(Tip pulled from the competition's forums.)

Possible dependencies will be CARFAC, Slaney's Auditory Toolbox

"moby" was cloned from 

https://github.com/nmkridler/moby

Using it for learning purposes, as it won the competition.