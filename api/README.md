# Merlot API

Here, `api.py` is a python script to extract image embedding for given video.

**Input**: a MP4 video (e.g. ``test.mp4``)

```python
from api import get_image_embedding
embedding = get_image_embedding('pmjPjZZRhNQ.mp4', file_type='mp4')
```
**Output**: image embedding for all the frames in the given video (``numpy.ndarray``, ``[num_image, num_patch_per_img, embed_shape]``)