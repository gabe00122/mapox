The multi task wrapper should itself be confidered as a environment and as a vectorized wrapper. We should delete the multi task make function and the num_envs field off the regular make function. Because the multi task config is a env config that contains other env config use should use Box to break the self referential loop, preferable at the top enum level.

We should create a video writer wrapper that does a cpu render of a env at a fixed interval and writes it to a video file using a ffmpeg call to encode/compress the video
