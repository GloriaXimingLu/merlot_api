"""
Demo for doing interesting things with a video
"""
import sys
sys.path.append('../')

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp
import pickle

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)


def get_image_embedding(video_file, file_type='mp4'):
    if file_type == 'mp4':
        video_segments = video_to_segments(video_file)
    else:
        video_segments = pickle.load(open(video_file, 'rb'))

    # Set up a fake classification task.
    video_segments[0]['text'] = 'in this video i\'ll be<|MASK|>'
    video_segments[0]['use_text_as_input'] = True
    for i in range(1,8):
        video_segments[i]['use_text_as_input'] = False

    video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=True)
    out_h = model.embed_image(**video_pre)
    # shape [num_image, num_patch_per_img, embed_shape]
    return out_h

