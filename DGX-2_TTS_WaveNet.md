
# NV-WaveNet
check 
- [my fork](https://github.com/yhgon/nv-wavenet) 
- [official github](https://github.com/NVIDIA/nv-wavenet) site for issue tracking

More information for WaveNet, check 
- [WaveNet blog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) 
- [WaveNet paper](https://arxiv.org/abs/1609.03499)
- [Ryuich's WaveNet](https://github.com/r9y9/wavenet_vocoder)
- [Google's Magenta Nsyth WaveNet](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth/wavenet)
- [Parallel WaveNet paper](https://arxiv.org/abs/1711.10433)
- [Parallel WaveNet implementation](https://github.com/andabi/parallel-wavenet-vocoder) 

you could run nv-wavenet using same docker image when you build for tacotron2

```
docker pull hryu/pytorch:t2-1
nvidia-docker run –ti –v/mnt/git/nv-wavenet:/scratch hryu/pytorch:t2-1 bash
```

## step2. clone model
```
git clone https://github.com/yhgon/nv-wavenet.git

```

## step3. build 
nv-wavenet use cuda for optimized inferencing 
```
cd nv-wavenet && make
cd pytorch && make && python build.py
```
## prepare dataset
default configuration for nv-wavenet is 16Khz.
so use CMU artic dataset (16Khz)
```
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.tar.bz2
tar -xvjf cmu_us_clb_arctic-0.95-release.tar.bz2
cp -rf  cmu_us_clb_arctic/wav ./data
ls data/*.wav | tail -n+10 > train_files.txt
ls data/*.wav | head -n10 > test_files.txt
```

## train
using default configuration, you will run WaveNet training
```
$python train.py -c config.json 
```

multiGPU training
```
$python distribute.py -c config.json 
```



### log

```
batch size 8

3.76s/epoch 
Epoch: 28
Iter  224 : reduced loss : 	2.737933874 	0.47s/it 
Iter  225 : reduced loss : 	2.656390190 	0.44s/it 
Iter  226 : reduced loss : 	2.686673403 	0.45s/it 
Iter  227 : reduced loss : 	2.698653698 	0.44s/it 
Iter  228 : reduced loss : 	2.684102297 	0.44s/it 
Iter  229 : reduced loss : 	2.737958670 	0.44s/it 
Iter  230 : reduced loss : 	2.685201406 	0.44s/it 
Iter  231 : reduced loss : 	2.661728859 	0.44s/it 
3.77s/epoch 
Epoch: 29
Iter  232 : reduced loss : 	2.705214739 	0.44s/it 
Iter  233 : reduced loss : 	2.679220200 	0.45s/it 
Iter  234 : reduced loss : 	2.704427004 	0.44s/it 
Iter  235 : reduced loss : 	2.670950174 	0.44s/it 
Iter  236 : reduced loss : 	2.691043139 	0.44s/it 
Iter  237 : reduced loss : 	2.678122997 	0.44s/it 
Iter  238 : reduced loss : 	2.617443800 	0.44s/it 
Iter  239 : reduced loss : 	2.631478786 	0.45s/it 
3.80s/epoch 
Epoch: 30
Iter  240 : reduced loss : 	2.655239105 	0.44s/it 
Iter  241 : reduced loss : 	2.692391872 	0.45s/it 
Iter  242 : reduced loss : 	2.645486355 	0.45s/it 
Iter  243 : reduced loss : 	2.663440466 	0.45s/it 
Iter  244 : reduced loss : 	2.708695650 	0.44s/it 
Iter  245 : reduced loss : 	2.689195156 	0.45s/it 
Iter  246 : reduced loss : 	2.657229900 	0.44s/it 
```


```
 python distributed.py -c config.json
['train.py', '--config=config.json', '--rank=0', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=1', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=2', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=3', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=4', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=5', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=6', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=7', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=8', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=9', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=10', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=11', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=12', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=13', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=14', '--group_name=group_2018_11_04-184835']
['train.py', '--config=config.json', '--rank=15', '--group_name=group_2018_11_04-184835']


Iter  2 : reduced loss : 	5.616512775 	0.45s/it 
Iter  3 : reduced loss : 	5.512421131 	0.45s/it 
Iter  4 : reduced loss : 	5.451251984 	0.44s/it 
Iter  5 : reduced loss : 	5.416746616 	0.45s/it 
Iter  6 : reduced loss : 	5.374511242 	0.45s/it 
Iter  7 : reduced loss : 	5.339864254 	0.45s/it 
9.60s/epoch 
```


```
Sun Nov  4 18:49:23 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.33                 Driver Version: 410.33                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM3...  On   | 00000000:34:00.0 Off |                    0 |
| N/A   46C    P0   320W / 350W |   8432MiB / 32510MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM3...  On   | 00000000:36:00.0 Off |                    0 |
| N/A   42C    P0   253W / 350W |   8432MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM3...  On   | 00000000:39:00.0 Off |                    0 |
| N/A   45C    P0   282W / 350W |   8432MiB / 32510MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM3...  On   | 00000000:3B:00.0 Off |                    0 |
| N/A   48C    P0   240W / 350W |   8432MiB / 32510MiB |     93%      Default |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-SXM3...  On   | 00000000:57:00.0 Off |                    0 |
| N/A   41C    P0   224W / 350W |   8432MiB / 32510MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-SXM3...  On   | 00000000:59:00.0 Off |                    0 |
| N/A   44C    P0   222W / 350W |   8432MiB / 32510MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-SXM3...  On   | 00000000:5C:00.0 Off |                    0 |
| N/A   42C    P0   272W / 350W |   8432MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-SXM3...  On   | 00000000:5E:00.0 Off |                    0 |
| N/A   47C    P0   264W / 350W |   8432MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   8  Tesla V100-SXM3...  On   | 00000000:B7:00.0 Off |                    0 |
| N/A   43C    P0   295W / 350W |   8432MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   9  Tesla V100-SXM3...  On   | 00000000:B9:00.0 Off |                    0 |
| N/A   44C    P0   327W / 350W |   8434MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|  10  Tesla V100-SXM3...  On   | 00000000:BC:00.0 Off |                    0 |
| N/A   46C    P0   245W / 350W |   8432MiB / 32510MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+
|  11  Tesla V100-SXM3...  On   | 00000000:BE:00.0 Off |                    0 |
| N/A   46C    P0   223W / 350W |   8432MiB / 32510MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
|  12  Tesla V100-SXM3...  On   | 00000000:E0:00.0 Off |                    0 |
| N/A   44C    P0   272W / 350W |   8432MiB / 32510MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+
|  13  Tesla V100-SXM3...  On   | 00000000:E2:00.0 Off |                    0 |
| N/A   41C    P0   236W / 350W |   8432MiB / 32510MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|  14  Tesla V100-SXM3...  On   | 00000000:E5:00.0 Off |                    0 |
| N/A   46C    P0   247W / 350W |   8432MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|  15  Tesla V100-SXM3...  On   | 00000000:E7:00.0 Off |                    0 |
| N/A   45C    P0   235W / 350W |   8432MiB / 32510MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+

```

```
Iter  6163 : reduced loss : 	2.008907080 	1.71s/it 
3.72s/epoch 
Epoch: 3082
Iter  6164 : reduced loss : 	1.985289693 	1.75s/it 
Iter  6165 : reduced loss : 	2.004762173 	1.71s/it 
3.72s/epoch 
Epoch: 3083
Iter  6166 : reduced loss : 	2.017169952 	1.75s/it 
Iter  6167 : reduced loss : 	2.002875090 	1.71s/it 
3.72s/epoch 
Epoch: 3084
Iter  6168 : reduced loss : 	2.000064373 	2.56s/it 
Iter  6169 : reduced loss : 	1.992105246 	1.71s/it 
4.51s/epoch 
```
