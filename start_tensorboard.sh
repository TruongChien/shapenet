# this mainly for colab
docker run -it -p 8888:8888 -p 6006:6006 \
tensorflow/tensorflow:nightly-py3-jupyter

tensorboard --logdir=runs
