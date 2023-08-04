# Merlot API

Here, `merlot.py` is a python script to extract image embedding for given video.

**Input**: a video represented by a list of frames stored in pickle (e.g. ``test.pkl``)
Each video is represented as a list of frames, each frame is a dictionary containing the following keys:

* ``start_time``: starting time of the video segment (``int``)
* ``end_time``: end time of the video segment (``int``)
* ``mid_time``: time of the frame (``int``)
* ``frame``: image (``numpy.ndarray``)
* ``spectrogram``: audio of the video segment (``numpy.ndarray``)
* ``idx``: index of the frame (``int``)


```bash
pip install youtube-dl
youtube-dl -f "best[height<=480,ext=mp4]" https://www.youtube.com/watch?v=pmjPjZZRhNQ -o "%(id)s.%(ext)s"
```
**Output**: image embedding for all the frames in the given video (``numpy.ndarray``)