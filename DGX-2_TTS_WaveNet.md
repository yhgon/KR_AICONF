
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
