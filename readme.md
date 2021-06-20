#Notes to make NVIDIA environments sonification compatible

You will need to open a console and some libraries manually:
    `conda install -c anaconda portaudio`
    `conda install -c anaconda pyaudio`
    `conda install -c conda-forge alsa-lib`

It looks like the NVIDIA environments can't see my local computer's speakers.
This isn't too surprising, but definitely makes "conversational data" much
more difficult.

I also cannot send network messages out which would be my other option. 
