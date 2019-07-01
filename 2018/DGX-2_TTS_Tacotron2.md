
# Tacotron2

check 
- [my fork](https://github.com/yhgon/tacotron2) and 
- [NVIDIA's official tacotron2](https://github.com/NVIDIA/tacotron2) for issue tracking

for more information, check 
- [google's tacotron blog](https://google.github.io/tacotron/)
- [tacotron paper ](https://arxiv.org/abs/1703.10135)
- [keithito's TF tacotron](https://github.com/keithito/tacotron), 
- [Ryuich's PYT Tacotron](https://github.com/r9y9/tacotron_pytorch)
- [tacotron2 paper](https://arxiv.org/abs/1712.05884) 
- [Rayhane-mamah's TF Tacotron2 implementation](https://github.com/Rayhane-mamah/Tacotron-2)

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

check 
-  [my repository](https://hub.docker.com/r/hryu/pytorch/) and [tags](https://hub.docker.com/r/hryu/pytorch/tags/) 

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

