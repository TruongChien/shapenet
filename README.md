# Shapenet Generative models


This repository is the PyTorch implementation of graph generative models. 
- GraphRNN a graph generative model using auto-regressive
- Custom GraphRNN ongoing work on improvement.

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, [Rex Ying](https://cs.stanford.edu/people/rexy/)\*
, [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), [William L. Hamilton](https://stanford.edu/~wleif/)
, [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html)
, [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model](https://arxiv.org/abs/1802.08773) (ICML 2018)

## Installation

The code has been tested over PyTorch latest version 1.10
 - Install PyTorch following the instructions on the [official website](https://pytorch.org/).
 - Check requirement files.
 - 
```bash
conda install pytorch torchvision -c pytorch
```

```bash
abseil-cpp                20210324.2           h0e60522_0    conda-forge
absl-py                   1.0.0                    pypi_0    pypi
aiohttp                   3.8.0                    pypi_0    pypi
aiosignal                 1.2.0                    pypi_0    pypi
alabaster                 0.7.12                   pypi_0    pypi
anyio                     3.3.4                    pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
argon2-cffi               21.1.0                   pypi_0    pypi
astor                     0.8.1              pyh9f0ad1d_0    conda-forge
astunparse                1.6.3              pyhd8ed1ab_0    conda-forge
async-generator           1.10                     pypi_0    pypi
async-timeout             4.0.0                    pypi_0    pypi
attrs                     21.2.0             pyhd8ed1ab_0    conda-forge
babel                     2.9.1                    pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0
blas                      1.0                         mkl
bleach                    4.1.0                    pypi_0    pypi
blinker                   1.4                        py_1    conda-forge
bokeh                     2.4.2                    pypi_0    pypi
bottleneck                1.3.2            py39h7cc1a96_1
brotlipy                  0.7.0           py39hb82d6ee_1003    conda-forge
bzip2                     1.0.8                he774522_0    anaconda
ca-certificates           2021.10.26           haa95532_2
cached-property           1.5.2                hd8ed1ab_1    conda-forge
cached_property           1.5.2              pyha770c72_1    conda-forge
cachetools                4.2.4              pyhd8ed1ab_0    conda-forge
certifi                   2021.10.8        py39haa95532_0
cffi                      1.15.0           py39h0878f49_0    conda-forge
chardet                   3.0.4           py39h467e6f4_1008    conda-forge
charset-normalizer        2.0.7                    pypi_0    pypi
click                     8.0.3            py39hcbf5309_1    conda-forge
cloudpickle               2.0.0                    pypi_0    pypi
cmake                     3.19.6               h9ad04ae_0
colorama                  0.4.4              pyh9f0ad1d_0    conda-forge
commonmark                0.9.1                    pypi_0    pypi
cryptography              35.0.0           py39h7bc7c5c_2    conda-forge
cudatoolkit               11.3.1               h59b6b97_2
cycler                    0.11.0                   pypi_0    pypi
dask                      2.30.0                   pypi_0    pypi
dataclasses               0.8                pyhc8e2a94_3    conda-forge
debugpy                   1.5.1                    pypi_0    pypi
decorator                 5.1.0              pyhd3eb1b0_0
defusedxml                0.7.1                    pypi_0    pypi
dill                      0.3.4                    pypi_0    pypi
dm-tree                   0.1.6                    pypi_0    pypi
docutils                  0.17.1                   pypi_0    pypi
easydict                  1.9                      pypi_0    pypi
edward                    1.3.5                    pypi_0    pypi
entrypoints               0.3                      pypi_0    pypi
flatbuffers               2.0.0                h0e60522_0    conda-forge
freetype                  2.10.4               hd328e21_0
frozenlist                1.2.0                    pypi_0    pypi
fvcore                    0.1.5.post20210804    pyhd8ed1ab_0    conda-forge
gast                      0.4.0              pyh9f0ad1d_0    conda-forge
giflib                    5.2.1                h8d14728_2    conda-forge
google-auth               2.3.3                    pypi_0    pypi
google-auth-oauthlib      0.4.6              pyhd8ed1ab_0    conda-forge
google-pasta              0.2.0              pyh8c360ce_0    conda-forge
googledrivedownloader     0.4                      pypi_0    pypi
graphviz                  2.38                 hfd603c8_2    anaconda
grpcio                    1.41.1                   pypi_0    pypi
h11                       0.12.0                   pypi_0    pypi
h5py                      3.4.0           nompi_py39hd4deaf1_101    conda-forge
haversine                 2.5.1                    pypi_0    pypi
hdf5                      1.12.1          nompi_h2a0e4a3_100    conda-forge
heapdict                  1.0.1                    pypi_0    pypi
icc_rt                    2019.0.0             h0cc432a_1
icu                       68.2                 h0e60522_0    conda-forge
idna                      3.3                      pypi_0    pypi
igraph                    0.9.8                    pypi_0    pypi
imageio                   2.10.3                   pypi_0    pypi
imagesize                 1.2.0                    pypi_0    pypi
importlib-metadata        4.8.2            py39hcbf5309_0    conda-forge
intel-openmp              2021.4.0          haa95532_3556
iopath                    0.1.9                      py39    iopath
ipykernel                 5.5.6                    pypi_0    pypi
ipython                   7.1.1                    pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
ipywidgets                7.6.5                    pypi_0    pypi
isodate                   0.6.0                    pypi_0    pypi
jedi                      0.18.0           py39haa95532_1
jinja2                    3.0.2                    pypi_0    pypi
joblib                    1.1.0                    pypi_0    pypi
jpeg                      9d                   h2bbff1b_0
json5                     0.9.6                    pypi_0    pypi
jsonschema                4.2.1                    pypi_0    pypi
jupyter-client            7.0.6                    pypi_0    pypi
jupyter-core              4.9.1                    pypi_0    pypi
jupyter-server            1.11.2                   pypi_0    pypi
jupyterlab                3.2.3                    pypi_0    pypi
jupyterlab-pygments       0.1.2                    pypi_0    pypi
jupyterlab-server         2.8.2                    pypi_0    pypi
jupyterlab-widgets        1.0.2                    pypi_0    pypi
keras-preprocessing       1.1.2              pyhd8ed1ab_0    conda-forge
kiwisolver                1.3.2                    pypi_0    pypi
krb5                      1.19.2               h20d022d_3    conda-forge
libcurl                   7.79.1               h789b8ee_1    conda-forge
libpng                    1.6.37               h2a8f88b_0
libprotobuf               3.14.0               h7755175_0    conda-forge
libssh2                   1.10.0               h680486a_2    conda-forge
libtiff                   4.2.0                hd0e1b90_0
libuv                     1.40.0               he774522_0
libwebp                   1.2.0                h2bbff1b_0
littleutils               0.2.2                    pypi_0    pypi
llvmlite                  0.37.0                   pypi_0    pypi
locket                    0.2.1                    pypi_0    pypi
lz4-c                     1.9.3                h2bbff1b_1
markdown                  3.3.4              pyhd8ed1ab_0    conda-forge
markupsafe                2.0.1                    pypi_0    pypi
matplotlib                3.4.3                    pypi_0    pypi
matplotlib-inline         0.1.3                    pypi_0    pypi
meshio                    4.4.6                    pypi_0    pypi
mistune                   0.8.4                    pypi_0    pypi
mkl                       2021.4.0           haa95532_640
mkl-service               2.4.0            py39h2bbff1b_0
mkl_fft                   1.3.1            py39h277e83a_0
mkl_random                1.2.2            py39hf11a4ad_0
mpmath                    1.2.1                    pypi_0    pypi
msgpack                   1.0.3                    pypi_0    pypi
multidict                 5.2.0            py39hb82d6ee_1    conda-forge
multiprocess              0.70.12.2                pypi_0    pypi
nbclassic                 0.3.4                    pypi_0    pypi
nbclient                  0.5.5                    pypi_0    pypi
nbconvert                 6.2.0                    pypi_0    pypi
nbformat                  5.1.3                    pypi_0    pypi
nest-asyncio              1.5.1                    pypi_0    pypi
networkx                  2.6.3                    pypi_0    pypi
notebook                  6.4.5                    pypi_0    pypi
numba                     0.54.1                   pypi_0    pypi
numexpr                   2.7.3            py39hb80d3ca_1
numpy                     1.20.3                   pypi_0    pypi
numpy-base                1.21.2           py39h0829f74_0
oauthlib                  3.1.1              pyhd8ed1ab_0    conda-forge
observations              0.1.4                    pypi_0    pypi
ogb                       1.3.2                    pypi_0    pypi
olefile                   0.46               pyhd3eb1b0_0
opencv-python             4.5.4.58                 pypi_0    pypi
openssl                   1.1.1l               h2bbff1b_0
opt_einsum                3.3.0              pyhd8ed1ab_1    conda-forge
orca                      1.0                      pypi_0    pypi
outcome                   1.1.0                    pypi_0    pypi
outdated                  0.2.1                    pypi_0    pypi
packaging                 21.2                     pypi_0    pypi
pandas                    1.3.4            py39h6214cd6_0
pandocfilters             1.5.0                    pypi_0    pypi
panel                     0.12.5                   pypi_0    pypi
param                     1.12.0                   pypi_0    pypi
parso                     0.8.2              pyhd3eb1b0_0
partd                     1.2.0                    pypi_0    pypi
pathos                    0.2.8                    pypi_0    pypi
pickleshare               0.7.5           pyhd3eb1b0_1003
pillow                    8.4.0            py39hd45dc43_0
pip                       21.2.4           py39haa95532_0
portalocker               2.3.2            py39hcbf5309_0    conda-forge
powerlaw                  1.5                      pypi_0    pypi
pox                       0.3.0                    pypi_0    pypi
ppft                      1.6.6.4                  pypi_0    pypi
prometheus-client         0.12.0                   pypi_0    pypi
prompt-toolkit            2.0.10                   pypi_0    pypi
protobuf                  3.19.1                   pypi_0    pypi
psutil                    5.8.0                    pypi_0    pypi
pyasn1                    0.4.8                      py_0    conda-forge
pyasn1-modules            0.2.8                    pypi_0    pypi
pycparser                 2.21               pyhd8ed1ab_0    conda-forge
pyct                      0.4.8                    pypi_0    pypi
pydot                     1.4.2                    pypi_0    pypi
pyemd                     0.5.1           py39h2e25243_1003    conda-forge
pygments                  2.10.0             pyhd3eb1b0_0
pyjwt                     2.3.0              pyhd8ed1ab_0    conda-forge
pyopenssl                 21.0.0             pyhd8ed1ab_0    conda-forge
pyparsing                 2.4.7                    pypi_0    pypi
pyrsistent                0.18.0                   pypi_0    pypi
pysocks                   1.7.1            py39hcbf5309_4    conda-forge
python                    3.9.7                h6244533_1
python-dateutil           2.8.2              pyhd3eb1b0_0
python-flatbuffers        1.12               pyhd8ed1ab_1    conda-forge
python-louvain            0.15                     pypi_0    pypi
python_abi                3.9                      2_cp39    conda-forge
pytorch                   1.10.0          py3.9_cuda11.3_cudnn8_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytorch3d                 0.6.0                    pypi_0    pypi
pytz                      2021.3                   pypi_0    pypi
pyu2f                     0.1.5              pyhd8ed1ab_0    conda-forge
pyvista                   0.32.1                   pypi_0    pypi
pyviz-comms               2.1.0                    pypi_0    pypi
pywin32                   302              py39hb82d6ee_2    conda-forge
pywinpty                  1.1.5                    pypi_0    pypi
pyyaml                    6.0              py39hb82d6ee_3    conda-forge
pyzmq                     22.3.0                   pypi_0    pypi
rdflib                    6.0.2                    pypi_0    pypi
requests                  2.26.0                   pypi_0    pypi
requests-oauthlib         1.3.0              pyh9f0ad1d_0    conda-forge
rich                      10.15.0                  pypi_0    pypi
rsa                       4.7.2              pyh44b312d_0    conda-forge
scikit-learn              1.0.1                    pypi_0    pypi
scipy                     1.7.2                    pypi_0    pypi
scooby                    0.5.7                    pypi_0    pypi
seaborn                   0.11.2                   pypi_0    pypi
send2trash                1.8.0                    pypi_0    pypi
setuptools                58.0.4           py39haa95532_0
six                       1.16.0             pyhd3eb1b0_0
snappy                    1.1.8                ha925a31_3    conda-forge
sniffio                   1.2.0                    pypi_0    pypi
snowballstemmer           2.1.0                    pypi_0    pypi
sortedcontainers          2.4.0                    pypi_0    pypi
sphinx                    4.2.0                    pypi_0    pypi
sphinxcontrib-applehelp   1.0.2                    pypi_0    pypi
sphinxcontrib-devhelp     1.0.2                    pypi_0    pypi
sphinxcontrib-htmlhelp    2.0.0                    pypi_0    pypi
sphinxcontrib-jsmath      1.0.1                    pypi_0    pypi
sphinxcontrib-qthelp      1.0.3                    pypi_0    pypi
sphinxcontrib-serializinghtml 1.1.5                    pypi_0    pypi
sqlite                    3.36.0               h2bbff1b_0
tabulate                  0.8.9              pyhd8ed1ab_0    conda-forge
tblib                     1.7.0                    pypi_0    pypi
tensorboard               2.7.0                    pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-logger        0.1.0                    pypi_0    pypi
tensorboard-plugin-wit    1.8.0              pyh44b312d_0    conda-forge
tensorflow                2.6.0           mkl_py39h31650da_0
tensorflow-base           2.6.0           mkl_py39h9201259_0
tensorflow-estimator      2.6.0              pyh7b7c402_0
tensorflow-probability    0.14.1                   pypi_0    pypi
termcolor                 1.1.0                      py_2    conda-forge
terminado                 0.12.1                   pypi_0    pypi
testpath                  0.5.0                    pypi_0    pypi
texttable                 1.6.4                    pypi_0    pypi
threadpoolctl             3.0.0                    pypi_0    pypi
tk                        8.6.11               h2bbff1b_0
toolz                     0.11.2                   pypi_0    pypi
torch-cluster             1.5.9                    pypi_0    pypi
torch-geometric           2.0.2                    pypi_0    pypi
torch-scatter             2.0.9                    pypi_0    pypi
torch-sparse              0.6.12                   pypi_0    pypi
torch-spline-conv         1.2.1                    pypi_0    pypi
torchaudio                0.10.0               py39_cu113    pytorch
torchvision               0.11.1               py39_cu113    pytorch
tornado                   6.1                      pypi_0    pypi
tqdm                      4.62.3             pyhd8ed1ab_0    conda-forge
traitlets                 5.1.1              pyhd3eb1b0_0
trio                      0.19.0                   pypi_0    pypi
typing_extensions         3.10.0.2           pyh06a4308_0
tzdata                    2021e                hda174b7_0
urllib3                   1.26.7             pyhd8ed1ab_0    conda-forge
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
vtk                       9.1.0                    pypi_0    pypi
wcwidth                   0.2.5              pyhd3eb1b0_0
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.2.1                    pypi_0    pypi
werkzeug                  2.0.2                    pypi_0    pypi
wheel                     0.35.1             pyh9f0ad1d_0    conda-forge
widgetsnbextension        3.5.2                    pypi_0    pypi
win_inet_pton             1.1.0            py39hcbf5309_3    conda-forge
wincertstore              0.2              py39haa95532_2
wrapt                     1.13.3           py39hb82d6ee_1    conda-forge
wslink                    1.1.0                    pypi_0    pypi
wsproto                   1.0.0                    pypi_0    pypi
xz                        5.2.5                h62dcd97_0
yacs                      0.1.8                    pypi_0    pypi
yaml                      0.2.5                he774522_0    conda-forge
yarl                      1.7.2            py39hb82d6ee_1    conda-forge
zict                      2.0.0                    pypi_0    pypi
zipp                      3.6.0              pyhd8ed1ab_0    conda-forge
zlib                      1.2.11               h62dcd97_4
zstd                      1.4.9                h19a0ad4_0
```
Then install the other dependencies.

```bash
pip install -r requirements.txt
```

Create config.yaml file

By default, trainer will create directory indicated in config.yaml file, each model under result
Both configured in config.yaml Note not all variable syntactically checked. 
Validation for config.yaml still in TODO not all done so please use existing example.

Each experiment create set of directories.  Inside each directory we have generated graph 
that used for a train a model.  All model files serialized to pickle files.

## Test run

- Generative notebook mainly to run on colab


```bash
python main.py
```

## Code description

For the GraphRNN model:
`main.py` is the main executable file, and specific arguments are set in `args.py`.
`rnn_generator.py` includes training iterations and calls `model.py` and `data.py`
