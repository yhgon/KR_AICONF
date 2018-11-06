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

## Docker files
there are some  utilities CycleGan, Pix2PixHD and PGGAN

```
FROM nvcr.io/nvidia/pytorch:18.09-py3

# APT update 
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
	apt-utils

# utils
RUN apt-get install -y --no-install-recommends  \
	build-essential \
	pciutils \ 
	nano \
	vim \
	less \
	ca-certificates \ 
	unzip \
	zip \
	p7zip \
	tar \
	pv 

# related build 
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
	subversion \
	swig  

# realted internet 
RUN apt-get install -y --no-install-recommends \
        curl \
	wget \
	rsync

# related multimedia
RUN apt-get install -y --no-install-recommends \
        libjpeg-dev \
        libpng-dev \
	sox \
	libsox-dev \
	ffmpeg 	 
	
RUN pip install --upgrade  pip
RUN pip install setuptools tqdm toml prompt-toolkit  ipykernel jupyter # python general 
RUN pip install numpy scipy sklearn              # python numpy
RUN pip install Pillow matplotlib opencv-python moviepy   # python graphic
RUN pip install librosa pysox                    # python sound
RUN pip install tensorflow tensorboardX          # tensorflow
RUN pip install Unicode inflect  six sacred                                    
RUN pip install lws nnmnkwii nltk 
RUN pip install keras 
RUN pip install dominate  # for pix2pixHD

EXPOSE 6006


```

## prepare dataset

I'm using small citiscape dataset
```
drwxr-xr-x+ 14 25223 dip   15 Oct 30 00:46 gtFine
-rwxr-xr-x+  1 25223 dip 241M Oct 30 00:20 gtFine_trainvaltest.zip
-rw-r--r--+  1 25223 dip 1.7K Feb 17  2016 license.txt
```

use below bash script to make cityscale Fine Annotation dataset to fit in Pix2PixHD dataset format
```
mkdir test_inst
mv test/*gtFine_instanceIds.png test_inst/.
mkdir test_label
mv test/*gtFine_labelIds.png test_label/.

mkdir train_inst
mv train/*gtFine_instanceIds.png train_inst/.
mkdir train_label
mv train/*gtFine_labelIds.png train_label/.

mkdir val_inst
mv val/*gtFine_instanceIds.png val_inst/.
mkdir val_label
mv val/*gtFine_labelIds.png val_label/.

```

after run bash script, your data folder would be below 

```
I have no name!@5c5ab6c13517:/mnt/old/dataset/cityscape/gtFine$ du -h .
9.8M	./test_label
14M	./val_inst
9.8M	./test_img
76M	./train_inst
9.8M	./test_inst
12M	./val_label
16M	./val_img
88M	./train_json
91M	./train_img
11M	./test_json
65M	./train_label
18M	./val_json
414M	
```


## train
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


# PGGAN
I omit the detail in presentation chcek official [pggan site ](https://github.com/tkarras/progressive_growing_of_gans)

## prepare Dataset


CelebA share split 7zfiles but decompress with p7zip in cloude was  very slow. 
handle CelebA, CelebA HD dataset in cloud, I make some utilities.

- step1. download whole dataset in 7zflies.

- step2. decompress whole file in local storage

- step3. move files in sub folder

```
#!/bin/bash
c=1; 
d=1; 
mkdir -p SUB_${d}
for jpg_filelist in *.jpg
do
  if [ $c -eq 10000 ]
  then
    d=$(( d + 1 )); c=0; mkdir -p SUB_${d}
  fi
  mv "$jpg_filelist" SUB_${d}/
  c=$(( c + 1 ))
done
```

- step4. tar each files

```
for dir in `find . -maxdepth 1 -type d  | grep -v "^\.$" `; do tar -cvzf ${dir}.tar.gz ${dir}; done
```

- step5. upload each tar.gz file 


- step6. 
which is not split zip file so you could extract it in parallel. 

```
du -h .
768K	./SUB_2
768K	./SUB_8
768K	./SUB_10
771K	./SUB_17
768K	./SUB_5
768K	./SUB_1
768K	./SUB_19
768K	./SUB_13
210K	./SUB_21
768K	./SUB_14
768K	./SUB_6
768K	./SUB_4
771K	./SUB_16
771K	./SUB_11
768K	./SUB_9
768K	./SUB_3
768K	./SUB_7
768K	./SUB_15
768K	./SUB_12
771K	./SUB_18
768K	./SUB_20
16M	.
```

- Step7 . merge whole subfolder 

- Step8. using preprocessing utility in pggan





# Tacotron2

check [my fork](https://github.com/yhgon/tacotron2) and [official github](https://github.com/NVIDIA/tacotron2) for issue tracking

## step1. configure 
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

## step4. run train

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
check [official site](https://nv-adlr.github.io/WaveGlow) and [**Ryan Prenger, Rafael Valle, and Bryan Catanzaro, WaveGlow: a Flow-based Generative Network for Speech Synthesis** arxiv 1811.00002](https://arxiv.org/abs/1811.00002)

after NVIDIA release the source code, I'll update it. 


