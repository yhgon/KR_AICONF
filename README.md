# KR_AICONF_2018


# ResNet-50 

```
Unpack DGX-2

docker pull  nvcr.io/nvidia/tensorflow:18.10-py3

nvidia-docker run -it   nvcr.io/nvidia/tensorflow:18.10-py3 bash

python /workspace/nvidia-examples/cnn/resnet.py  --layers=50  --precision=fp32   --num_iter=400  --iter_unit=batch  --batch_size 128

mpirun --allow-run-as-root  --bind-to socket   -np 16 python /workspace/nvidia-examples/cnn/resnet.py --data_dir=/mnt/hryu/dataset-tf/train-val-tfrecord-480-subset --log_dir=/mnt/hryu/result/try3   --layers=50  --precision=fp16   --num_iter=400  --iter_unit=batch  --batch_size 256
```

# Pix2PixHD

```
$ python train.py --name label2city_512p --batchSize 2
 (epoch: 1, iters: 2900, time: 0.431) G_GAN: 0.859 G_GAN_Feat: 2.548 G_VGG: 2.765 D_real: 0.480 D_fake: 0.352 
End of epoch 1 / 200 	 Time Taken: 1325 sec

python train.py --name label2city_512p-8gpu --batchSize 16 --gpu_ids 0,1,2,3,4,5,6,7
(epoch: 116, iters: 2800, time: 0.090) G_GAN: 1.384 G_GAN_Feat: 2.265 G_VGG: 0.885 D_real: 0.143 D_fake: 0.117 
End of epoch 116 / 200 	 Time Taken: 283 sec

$ python train.py --name label2city_512p-16gpu --batchSize 128 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
End of epoch 1 / 200 	 Time Taken: 261 sec
(epoch: 2, iters: 1920, time: 0.042) G_GAN: 1.185 G_GAN_Feat: 5.136 G_VGG: 6.295 D_real: 0.913 D_fake: 0.875 
End of epoch 2 / 200 	 Time Taken: 184 sec
```

# Tacotron2

check [my fork](https://github.com/yhgon/tacotron2) and [official github](https://github.com/NVIDIA/tacotron2) for issue tracking

## step1.
Prepare NGC with pytorhc & utilities(librosa, tensorflow for hparam, tensorboardX for log, matplotlib/scikit-learn for chart

```
docker pull hryu/pytorch:t2-2
nvidia-docker run –ti –v/mnt/git/tacotron2:/scratch hryu/pytorch:t2-2 bash
```

## Step2.  clone 
I update bug patch for NGC (distributed bug) 
```
git clone https://github.com/yhgon/tacotron2.git
```

## Step3. prepare dataset
Keith Ito provide LJSpeech Dataset(22Khz audio) from libri speech. 
it's ok to use whole list. 
some wav file have problem so tacotron2 provide filelist.

```
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xvjf LJSpeech-1.1.tar.bz2
cp –rf LJSpeech-1.1/wavs ./DUMMY
default train_file_lists
```

## step4. run 

in single GPU
```
cd /scratch 
mkdir ch
$python train.py  --output_directory=ch/out_${NID}_${TID}  --log_directory=ch/log_${NID}_${TID}  --hparams=distributed_run=False,fp16_run=True,batch_size=48,epochs=20000,iters_per_checkpoint=500
```

in multiple GPU
```
mkdir logs
rm -rf distributed.dpt
$python -m multiproc train.py  --output_directory=ch/out_${NID}_${TID}  --log_directory=ch/log_${NID}_${TID}  --hparams=distributed_run=True,fp16_run=True,batch_size=192,epochs=20000,iters_per_checkpoint=500
```

###  Docker image and Dockerfile

check my [repository](https://hub.docker.com/r/hryu/pytorch/) and [tags](https://hub.docker.com/r/hryu/pytorch/tags/) 

you could use [simple version](https://github.com/yhgon/tacotron2/blob/master/Dockerfile)
```
FROM pytorch/pytorch:0.4_cuda9_cudnn7
RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow tensorboardX inflect==0.2.5 Unidecode==1.0.22 jupyter
```

you could use NGC version
```
FROM nvcr.io/nvidia/pytorch:18.08-py3

# APT update 
RUN apt-get update
RUN apt-get install -y --no-install-recommends  \
       pciutils  nano vim less ca-certificates unzip zip tar pv \
       build-essential  cmake  git  subversion  swig  \
       libjpeg-dev  libpng-dev   sox   libsox-dev  ffmpeg 	 
	
RUN pip install --upgrade  pip
RUN pip install setuptools tqdm toml prompt-toolkit  ipykernel jupyter  
RUN pip install numpy scipy sklearn               
RUN pip install Pillow matplotlib
RUN pip install librosa pysox                     
RUN pip install tensorflow tensorboardX           
RUN pip install Unicode inflect  six

EXPOSE 6006
EXPOSE 8888

```

# NV-WaveNet
you also check [my fork](https://github.com/yhgon/nv-wavenet) and [official github](https://github.com/NVIDIA/nv-wavenet) site for issue tracking

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

# WaveGlow
after NVIDIA release the source code, I'll update it. 



