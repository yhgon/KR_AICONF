
# Pix2PixHD

### DevOps Docker files
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

#### step1. download dataset 
I'm using small citiscape dataset
```
drwxr-xr-x+ 14 25223 dip   15 Oct 30 00:46 gtFine
-rwxr-xr-x+  1 25223 dip 241M Oct 30 00:20 gtFine_trainvaltest.zip
-rw-r--r--+  1 25223 dip 1.7K Feb 17  2016 license.txt
```

#### step2. modify  structure of dataset
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
#### step3. check dataset
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
train dataset is 91MB(raw imgs) and 76MB(segmentation info)


## train

#### train with single GPU
```
$ python train.py --name label2city_512p --batchSize 2
 (epoch: 1, iters: 2900, time: 0.431) G_GAN: 0.859 G_GAN_Feat: 2.548 G_VGG: 2.765 D_real: 0.480 D_fake: 0.352 
End of epoch 1 / 200 	 Time Taken: 1325 sec
```

#### train with 8 GPUs
```
python train.py --name label2city_512p-8gpu --batchSize 16 --gpu_ids 0,1,2,3,4,5,6,7
(epoch: 116, iters: 2800, time: 0.090) G_GAN: 1.384 G_GAN_Feat: 2.265 G_VGG: 0.885 D_real: 0.143 D_fake: 0.117 
End of epoch 116 / 200 	 Time Taken: 283 sec
```

#### train with 16 GPUs
```
$ python train.py --name label2city_512p-16gpu --batchSize 128 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
End of epoch 1 / 200 	 Time Taken: 261 sec
(epoch: 2, iters: 1920, time: 0.042) G_GAN: 1.185 G_GAN_Feat: 5.136 G_VGG: 6.295 D_real: 0.913 D_fake: 0.875 
End of epoch 2 / 200 	 Time Taken: 184 sec
```

## full log for 200 epoch using 8 GPUs

```
$ python train.py --name label2city_512p-8gpu --batchSize 16 --gpu_ids 0,1,2,3,4,5,6,7
------------ Options -------------
batchSize: 16
beta1: 0.5
checkpoints_dir: ./checkpoints
continue_train: False
data_type: 32
dataroot: ./datasets/cityscapes/
debug: False
display_freq: 100
display_winsize: 512
feat_num: 3
fineSize: 512
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
input_nc: 3
instance_feat: False
isTrain: True
label_feat: False
label_nc: 35
lambda_feat: 10.0
loadSize: 1024
load_features: False
load_pretrain: 
lr: 0.0002
max_dataset_size: inf
model: pix2pixHD
nThreads: 2
n_blocks_global: 9
n_blocks_local: 3
n_clusters: 10
n_downsample_E: 4
n_downsample_global: 4
n_layers_D: 3
n_local_enhancers: 1
name: label2city_512p-8gpu
ndf: 64
nef: 16
netG: global
ngf: 64
niter: 100
niter_decay: 100
niter_fix_global: 0
no_flip: False
no_ganFeat_loss: False
no_html: False
no_instance: False
no_lsgan: False
no_vgg_loss: False
norm: instance
num_D: 2
output_nc: 3
phase: train
pool_size: 0
print_freq: 100
resize_or_crop: scale_width
save_epoch_freq: 10
save_latest_freq: 1000
serial_batches: False
tf_log: False
use_dropout: False
verbose: False
which_epoch: latest
-------------- End ----------------
CustomDatasetDataLoader
dataset [AlignedDataset] was created
#training images = 2960
GlobalGenerator(
  (model): Sequential(
    (0): ReflectionPad2d((3, 3, 3, 3))
    (1): Conv2d(36, 64, kernel_size=(7, 7), stride=(1, 1))
    (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (3): ReLU(inplace)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (6): ReLU(inplace)
    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (9): ReLU(inplace)
    (10): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (11): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (12): ReLU(inplace)
    (13): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (14): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (15): ReLU(inplace)
    (16): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (17): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (18): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (19): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (20): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (21): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (22): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (23): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (24): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (25): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (26): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (27): ReLU(inplace)
    (28): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (29): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (30): ReLU(inplace)
    (31): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (32): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (33): ReLU(inplace)
    (34): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (35): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (36): ReLU(inplace)
    (37): ReflectionPad2d((3, 3, 3, 3))
    (38): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
    (39): Tanh()
  )
)
MultiscaleDiscriminator(
  (scale0_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (scale1_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (downsample): AvgPool2d(kernel_size=3, stride=2, padding=[1, 1])
)
create web directory ./checkpoints/label2city_512p-8gpu/web...


I have no name!@2232ba488464:/workspace$ nvidia-smi
Tue Oct 30 08:07:14 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:06:00.0 Off |                    0 |
| N/A   59C    P0   271W / 300W |  12527MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:07:00.0 Off |                    0 |
| N/A   62C    P0   279W / 300W |  10437MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:0A:00.0 Off |                    0 |
| N/A   65C    P0   279W / 300W |  10437MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:0B:00.0 Off |                    0 |
| N/A   57C    P0   269W / 300W |  10437MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-SXM2...  Off  | 00000000:85:00.0 Off |                    0 |
| N/A   59C    P0   278W / 300W |  10425MiB / 16160MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-SXM2...  Off  | 00000000:86:00.0 Off |                    0 |
| N/A   61C    P0   211W / 300W |  10431MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-SXM2...  Off  | 00000000:89:00.0 Off |                    0 |
| N/A   62C    P0    94W / 300W |  10177MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-SXM2...  Off  | 00000000:8A:00.0 Off |                    0 |
| N/A   56C    P0    75W / 300W |  10437MiB / 16160MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|


create web directory ./checkpoints/label2city_512p-8gpu/web...
/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:58: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
train.py:87: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
(epoch: 1, iters: 400, time: 0.101) G_GAN: 1.115 G_GAN_Feat: 7.521 G_VGG: 6.741 D_real: 0.682 D_fake: 0.726 
(epoch: 1, iters: 800, time: 0.097) G_GAN: 1.052 G_GAN_Feat: 5.360 G_VGG: 6.062 D_real: 0.725 D_fake: 0.808 
(epoch: 1, iters: 1200, time: 0.087) G_GAN: 0.835 G_GAN_Feat: 4.645 G_VGG: 7.222 D_real: 0.546 D_fake: 0.598 
(epoch: 1, iters: 1600, time: 0.080) G_GAN: 0.794 G_GAN_Feat: 3.223 G_VGG: 5.480 D_real: 0.607 D_fake: 0.624 
(epoch: 1, iters: 2000, time: 0.085) G_GAN: 0.912 G_GAN_Feat: 3.448 G_VGG: 5.407 D_real: 0.667 D_fake: 0.484 
saving the latest model (epoch 1, total_steps 2000)
(epoch: 1, iters: 2400, time: 0.090) G_GAN: 0.838 G_GAN_Feat: 3.623 G_VGG: 5.275 D_real: 0.584 D_fake: 0.527 
(epoch: 1, iters: 2800, time: 0.093) G_GAN: 1.059 G_GAN_Feat: 2.813 G_VGG: 5.146 D_real: 0.847 D_fake: 0.707 
End of epoch 1 / 200 	 Time Taken: 336 sec
(epoch: 2, iters: 240, time: 0.093) G_GAN: 0.842 G_GAN_Feat: 2.624 G_VGG: 5.295 D_real: 0.634 D_fake: 0.609 
(epoch: 2, iters: 640, time: 0.102) G_GAN: 0.688 G_GAN_Feat: 2.030 G_VGG: 4.858 D_real: 0.545 D_fake: 0.541 
(epoch: 2, iters: 1040, time: 0.106) G_GAN: 0.871 G_GAN_Feat: 2.196 G_VGG: 4.607 D_real: 0.648 D_fake: 0.604 
saving the latest model (epoch 2, total_steps 4000)
(epoch: 2, iters: 1440, time: 0.084) G_GAN: 0.631 G_GAN_Feat: 2.064 G_VGG: 4.437 D_real: 0.482 D_fake: 0.561 
(epoch: 2, iters: 1840, time: 0.102) G_GAN: 0.707 G_GAN_Feat: 2.107 G_VGG: 4.334 D_real: 0.503 D_fake: 0.447 
(epoch: 2, iters: 2240, time: 0.097) G_GAN: 0.758 G_GAN_Feat: 2.219 G_VGG: 4.556 D_real: 0.527 D_fake: 0.503 


(epoch: 94, iters: 2320, time: 0.090) G_GAN: 1.157 G_GAN_Feat: 1.905 G_VGG: 1.019 D_real: 0.200 D_fake: 0.196 
(epoch: 94, iters: 2720, time: 0.092) G_GAN: 0.763 G_GAN_Feat: 2.213 G_VGG: 0.965 D_real: 0.076 D_fake: 0.472 
saving the latest model (epoch 94, total_steps 278000)
End of epoch 94 / 200 	 Time Taken: 289 sec
(epoch: 95, iters: 160, time: 0.092) G_GAN: 1.393 G_GAN_Feat: 1.931 G_VGG: 0.990 D_real: 0.678 D_fake: 0.257 
(epoch: 95, iters: 560, time: 0.092) G_GAN: 1.881 G_GAN_Feat: 3.044 G_VGG: 1.017 D_real: 0.196 D_fake: 0.050 
(epoch: 95, iters: 960, time: 0.097) G_GAN: 1.341 G_GAN_Feat: 1.992 G_VGG: 1.033 D_real: 1.092 D_fake: 0.153 
(epoch: 95, iters: 1360, time: 0.085) G_GAN: 1.328 G_GAN_Feat: 2.096 G_VGG: 1.063 D_real: 0.300 D_fake: 0.156 
(epoch: 95, iters: 1760, time: 0.093) G_GAN: 1.594 G_GAN_Feat: 2.548 G_VGG: 0.908 D_real: 0.453 D_fake: 0.081 
saving the latest model (epoch 95, total_steps 280000)
(epoch: 95, iters: 2160, time: 0.097) G_GAN: 1.113 G_GAN_Feat: 1.887 G_VGG: 0.932 D_real: 0.344 D_fake: 0.215 
(epoch: 95, iters: 2560, time: 0.092) G_GAN: 0.884 G_GAN_Feat: 1.936 G_VGG: 0.892 D_real: 0.235 D_fake: 0.356 
(epoch: 95, iters: 2960, time: 0.096) G_GAN: 1.127 G_GAN_Feat: 1.793 G_VGG: 0.926 D_real: 0.274 D_fake: 0.215 
End of epoch 95 / 200 	 Time Taken: 285 sec
(epoch: 96, iters: 400, time: 0.096) G_GAN: 1.538 G_GAN_Feat: 2.287 G_VGG: 1.006 D_real: 0.669 D_fake: 0.136 
(epoch: 96, iters: 800, time: 0.093) G_GAN: 0.839 G_GAN_Feat: 2.098 G_VGG: 1.131 D_real: 0.406 D_fake: 0.446 
saving the latest model (epoch 96, total_steps 282000)
(epoch: 96, iters: 1200, time: 0.095) G_GAN: 1.266 G_GAN_Feat: 1.745 G_VGG: 0.992 D_real: 0.146 D_fake: 0.163 
(epoch: 96, iters: 1600, time: 0.084) G_GAN: 1.227 G_GAN_Feat: 2.553 G_VGG: 1.049 D_real: 0.319 D_fake: 0.155 
(epoch: 96, iters: 2000, time: 0.086) G_GAN: 1.064 G_GAN_Feat: 2.461 G_VGG: 1.012 D_real: 0.165 D_fake: 0.236 
(epoch: 96, iters: 2400, time: 0.093) G_GAN: 1.775 G_GAN_Feat: 2.613 G_VGG: 0.941 D_real: 0.346 D_fake: 0.049 
(epoch: 96, iters: 2800, time: 0.098) G_GAN: 0.780 G_GAN_Feat: 2.067 G_VGG: 0.977 D_real: 0.378 D_fake: 0.414 
saving the latest model (epoch 96, total_steps 284000)
End of epoch 96 / 200 	 Time Taken: 289 sec
(epoch: 97, iters: 240, time: 0.094) G_GAN: 1.223 G_GAN_Feat: 2.092 G_VGG: 0.949 D_real: 0.296 D_fake: 0.417 
(epoch: 97, iters: 640, time: 0.098) G_GAN: 1.411 G_GAN_Feat: 1.986 G_VGG: 0.873 D_real: 0.171 D_fake: 0.130 
(epoch: 97, iters: 1040, time: 0.100) G_GAN: 0.787 G_GAN_Feat: 1.404 G_VGG: 0.882 D_real: 0.460 D_fake: 0.338 
(epoch: 97, iters: 1440, time: 0.101) G_GAN: 1.240 G_GAN_Feat: 2.541 G_VGG: 0.931 D_real: 0.274 D_fake: 0.139 
(epoch: 97, iters: 1840, time: 0.090) G_GAN: 0.963 G_GAN_Feat: 1.858 G_VGG: 1.071 D_real: 0.201 D_fake: 0.288 
saving the latest model (epoch 97, total_steps 286000)
(epoch: 97, iters: 2240, time: 0.085) G_GAN: 1.090 G_GAN_Feat: 2.100 G_VGG: 0.951 D_real: 0.191 D_fake: 0.233 
(epoch: 97, iters: 2640, time: 0.096) G_GAN: 1.253 G_GAN_Feat: 1.908 G_VGG: 0.976 D_real: 0.264 D_fake: 0.211 
End of epoch 97 / 200 	 Time Taken: 285 sec
(epoch: 98, iters: 80, time: 0.093) G_GAN: 1.995 G_GAN_Feat: 2.353 G_VGG: 1.037 D_real: 0.521 D_fake: 0.069 
(epoch: 98, iters: 480, time: 0.093) G_GAN: 0.944 G_GAN_Feat: 2.211 G_VGG: 1.042 D_real: 0.225 D_fake: 0.334 
(epoch: 98, iters: 880, time: 0.091) G_GAN: 0.705 G_GAN_Feat: 1.520 G_VGG: 1.003 D_real: 0.378 D_fake: 0.407 
saving the latest model (epoch 98, total_steps 288000)
(epoch: 98, iters: 1280, time: 0.083) G_GAN: 1.171 G_GAN_Feat: 1.870 G_VGG: 0.865 D_real: 0.316 D_fake: 0.272 
(epoch: 98, iters: 1680, time: 0.086) G_GAN: 1.169 G_GAN_Feat: 2.017 G_VGG: 0.949 D_real: 0.278 D_fake: 0.237 
(epoch: 98, iters: 2080, time: 0.087) G_GAN: 0.921 G_GAN_Feat: 1.546 G_VGG: 1.108 D_real: 0.341 D_fake: 0.327 
(epoch: 98, iters: 2480, time: 0.091) G_GAN: 0.851 G_GAN_Feat: 1.241 G_VGG: 1.002 D_real: 0.375 D_fake: 0.358 
(epoch: 98, iters: 2880, time: 0.087) G_GAN: 0.488 G_GAN_Feat: 1.398 G_VGG: 0.990 D_real: 0.313 D_fake: 0.567 
saving the latest model (epoch 98, total_steps 290000)
End of epoch 98 / 200 	 Time Taken: 287 sec
(epoch: 99, iters: 320, time: 0.085) G_GAN: 1.046 G_GAN_Feat: 2.359 G_VGG: 1.070 D_real: 0.068 D_fake: 0.465 
(epoch: 99, iters: 720, time: 0.092) G_GAN: 0.886 G_GAN_Feat: 1.608 G_VGG: 0.942 D_real: 0.410 D_fake: 0.308 
(epoch: 99, iters: 1120, time: 0.092) G_GAN: 2.029 G_GAN_Feat: 2.642 G_VGG: 1.018 D_real: 0.263 D_fake: 0.053 
(epoch: 99, iters: 1520, time: 0.100) G_GAN: 1.546 G_GAN_Feat: 1.985 G_VGG: 0.974 D_real: 0.805 D_fake: 0.334 
(epoch: 99, iters: 1920, time: 0.090) G_GAN: 1.271 G_GAN_Feat: 2.097 G_VGG: 0.986 D_real: 0.100 D_fake: 0.223 
saving the latest model (epoch 99, total_steps 292000)
(epoch: 99, iters: 2320, time: 0.096) G_GAN: 0.463 G_GAN_Feat: 2.232 G_VGG: 0.944 D_real: 0.282 D_fake: 0.718 
(epoch: 99, iters: 2720, time: 0.087) G_GAN: 2.145 G_GAN_Feat: 1.927 G_VGG: 1.003 D_real: 0.854 D_fake: 0.107 
End of epoch 99 / 200 	 Time Taken: 286 sec
(epoch: 100, iters: 160, time: 0.085) G_GAN: 1.189 G_GAN_Feat: 1.774 G_VGG: 0.891 D_real: 0.647 D_fake: 0.280 
(epoch: 100, iters: 560, time: 0.091) G_GAN: 0.905 G_GAN_Feat: 1.636 G_VGG: 0.929 D_real: 0.527 D_fake: 0.282 
(epoch: 100, iters: 960, time: 0.096) G_GAN: 1.258 G_GAN_Feat: 2.116 G_VGG: 0.982 D_real: 0.115 D_fake: 0.226 
saving the latest model (epoch 100, total_steps 294000)
(epoch: 100, iters: 1360, time: 0.094) G_GAN: 1.674 G_GAN_Feat: 2.368 G_VGG: 0.907 D_real: 0.280 D_fake: 0.094 
(epoch: 100, iters: 1760, time: 0.092) G_GAN: 1.201 G_GAN_Feat: 2.054 G_VGG: 0.925 D_real: 0.503 D_fake: 0.346 
(epoch: 100, iters: 2160, time: 0.088) G_GAN: 1.403 G_GAN_Feat: 1.981 G_VGG: 1.052 D_real: 0.567 D_fake: 0.130 
(epoch: 100, iters: 2560, time: 0.093) G_GAN: 1.490 G_GAN_Feat: 1.857 G_VGG: 0.958 D_real: 0.285 D_fake: 0.168 
(epoch: 100, iters: 2960, time: 0.092) G_GAN: 1.174 G_GAN_Feat: 2.156 G_VGG: 0.886 D_real: 0.236 D_fake: 0.200 
saving the latest model (epoch 100, total_steps 296000)
End of epoch 100 / 200 	 Time Taken: 288 sec
saving the model at the end of epoch 100, iters 296000
(epoch: 101, iters: 400, time: 0.087) G_GAN: 0.646 G_GAN_Feat: 2.192 G_VGG: 1.011 D_real: 0.133 D_fake: 0.685 
(epoch: 101, iters: 800, time: 0.094) G_GAN: 0.924 G_GAN_Feat: 1.443 G_VGG: 0.973 D_real: 0.259 D_fake: 0.305 
(epoch: 101, iters: 1200, time: 0.089) G_GAN: 0.645 G_GAN_Feat: 1.831 G_VGG: 1.038 D_real: 0.254 D_fake: 0.429 
(epoch: 101, iters: 1600, time: 0.090) G_GAN: 0.487 G_GAN_Feat: 1.272 G_VGG: 0.867 D_real: 0.277 D_fake: 0.578 
(epoch: 101, iters: 2000, time: 0.093) G_GAN: 1.191 G_GAN_Feat: 1.265 G_VGG: 0.909 D_real: 0.331 D_fake: 0.298 
saving the latest model (epoch 101, total_steps 298000)
(epoch: 101, iters: 2400, time: 0.101) G_GAN: 0.663 G_GAN_Feat: 1.543 G_VGG: 0.997 D_real: 0.363 D_fake: 0.467 
(epoch: 101, iters: 2800, time: 0.098) G_GAN: 1.851 G_GAN_Feat: 2.125 G_VGG: 1.025 D_real: 0.328 D_fake: 0.182 
End of epoch 101 / 200 	 Time Taken: 282 sec
(epoch: 102, iters: 240, time: 0.091) G_GAN: 1.042 G_GAN_Feat: 2.178 G_VGG: 0.951 D_real: 0.098 D_fake: 0.394 
(epoch: 102, iters: 640, time: 0.090) G_GAN: 0.759 G_GAN_Feat: 2.256 G_VGG: 0.972 D_real: 0.100 D_fake: 0.430 
(epoch: 102, iters: 1040, time: 0.088) G_GAN: 1.040 G_GAN_Feat: 2.040 G_VGG: 0.995 D_real: 0.222 D_fake: 0.239 
saving the latest model (epoch 102, total_steps 300000)
(epoch: 102, iters: 1440, time: 0.092) G_GAN: 0.762 G_GAN_Feat: 2.334 G_VGG: 0.946 D_real: 0.037 D_fake: 0.449 
(epoch: 102, iters: 1840, time: 0.097) G_GAN: 0.814 G_GAN_Feat: 2.024 G_VGG: 0.931 D_real: 0.224 D_fake: 0.413 
(epoch: 102, iters: 2240, time: 0.096) G_GAN: 1.313 G_GAN_Feat: 2.007 G_VGG: 0.941 D_real: 0.262 D_fake: 0.194 
(epoch: 102, iters: 2640, time: 0.091) G_GAN: 1.025 G_GAN_Feat: 2.285 G_VGG: 1.002 D_real: 0.157 D_fake: 0.281 
End of epoch 102 / 200 	 Time Taken: 283 sec
(epoch: 103, iters: 80, time: 0.096) G_GAN: 0.837 G_GAN_Feat: 1.544 G_VGG: 0.969 D_real: 0.372 D_fake: 0.394 
saving the latest model (epoch 103, total_steps 302000)
(epoch: 103, iters: 480, time: 0.091) G_GAN: 1.291 G_GAN_Feat: 1.700 G_VGG: 1.005 D_real: 0.266 D_fake: 0.164 
(epoch: 103, iters: 880, time: 0.099) G_GAN: 0.666 G_GAN_Feat: 1.562 G_VGG: 0.991 D_real: 0.357 D_fake: 0.403 
(epoch: 103, iters: 1280, time: 0.084) G_GAN: 0.901 G_GAN_Feat: 2.002 G_VGG: 0.985 D_real: 0.360 D_fake: 0.320 
(epoch: 103, iters: 1680, time: 0.096) G_GAN: 0.962 G_GAN_Feat: 1.722 G_VGG: 1.012 D_real: 0.278 D_fake: 0.350 
(epoch: 103, iters: 2080, time: 0.087) G_GAN: 1.314 G_GAN_Feat: 2.050 G_VGG: 0.936 D_real: 0.431 D_fake: 0.122 
saving the latest model (epoch 103, total_steps 304000)
(epoch: 103, iters: 2480, time: 0.084) G_GAN: 1.639 G_GAN_Feat: 2.403 G_VGG: 1.013 D_real: 0.425 D_fake: 0.153 
(epoch: 103, iters: 2880, time: 0.094) G_GAN: 1.007 G_GAN_Feat: 2.012 G_VGG: 1.042 D_real: 0.651 D_fake: 0.269 
End of epoch 103 / 200 	 Time Taken: 288 sec
(epoch: 104, iters: 320, time: 0.093) G_GAN: 1.906 G_GAN_Feat: 2.679 G_VGG: 0.978 D_real: 0.452 D_fake: 0.042 
(epoch: 104, iters: 720, time: 0.084) G_GAN: 1.505 G_GAN_Feat: 2.409 G_VGG: 0.916 D_real: 0.245 D_fake: 0.141 
(epoch: 104, iters: 1120, time: 0.084) G_GAN: 0.891 G_GAN_Feat: 1.786 G_VGG: 0.970 D_real: 0.075 D_fake: 0.375 
saving the latest model (epoch 104, total_steps 306000)
(epoch: 104, iters: 1520, time: 0.084) G_GAN: 1.102 G_GAN_Feat: 2.655 G_VGG: 0.980 D_real: 0.096 D_fake: 0.180 
(epoch: 104, iters: 1920, time: 0.092) G_GAN: 0.782 G_GAN_Feat: 1.581 G_VGG: 0.946 D_real: 0.078 D_fake: 0.507 
(epoch: 104, iters: 2320, time: 0.091) G_GAN: 0.858 G_GAN_Feat: 2.382 G_VGG: 0.947 D_real: 0.160 D_fake: 0.490 
(epoch: 104, iters: 2720, time: 0.092) G_GAN: 0.376 G_GAN_Feat: 1.252 G_VGG: 0.926 D_real: 0.300 D_fake: 0.906 
End of epoch 104 / 200 	 Time Taken: 285 sec
(epoch: 105, iters: 160, time: 0.096) G_GAN: 0.811 G_GAN_Feat: 1.837 G_VGG: 0.930 D_real: 0.501 D_fake: 0.337 
saving the latest model (epoch 105, total_steps 308000)
(epoch: 105, iters: 560, time: 0.092) G_GAN: 1.100 G_GAN_Feat: 1.978 G_VGG: 0.993 D_real: 0.464 D_fake: 0.271 
(epoch: 105, iters: 960, time: 0.088) G_GAN: 0.467 G_GAN_Feat: 1.377 G_VGG: 0.893 D_real: 0.148 D_fake: 0.669 
(epoch: 105, iters: 1360, time: 0.090) G_GAN: 1.064 G_GAN_Feat: 1.725 G_VGG: 0.992 D_real: 0.299 D_fake: 0.223 
(epoch: 105, iters: 1760, time: 0.085) G_GAN: 1.266 G_GAN_Feat: 2.117 G_VGG: 0.987 D_real: 0.148 D_fake: 0.258 
(epoch: 105, iters: 2160, time: 0.087) G_GAN: 0.641 G_GAN_Feat: 1.860 G_VGG: 1.069 D_real: 0.141 D_fake: 0.479 
saving the latest model (epoch 105, total_steps 310000)
(epoch: 105, iters: 2560, time: 0.088) G_GAN: 0.609 G_GAN_Feat: 1.173 G_VGG: 0.941 D_real: 0.368 D_fake: 0.450 
(epoch: 105, iters: 2960, time: 0.091) G_GAN: 1.337 G_GAN_Feat: 1.816 G_VGG: 0.991 D_real: 0.387 D_fake: 0.240 
End of epoch 105 / 200 	 Time Taken: 289 sec
(epoch: 106, iters: 400, time: 0.090) G_GAN: 0.657 G_GAN_Feat: 1.982 G_VGG: 1.022 D_real: 0.194 D_fake: 0.537 
(epoch: 106, iters: 800, time: 0.091) G_GAN: 1.252 G_GAN_Feat: 2.335 G_VGG: 0.980 D_real: 0.258 D_fake: 0.174 
(epoch: 106, iters: 1200, time: 0.097) G_GAN: 1.372 G_GAN_Feat: 1.940 G_VGG: 0.956 D_real: 0.764 D_fake: 0.133 
saving the latest model (epoch 106, total_steps 312000)
(epoch: 106, iters: 1600, time: 0.088) G_GAN: 1.062 G_GAN_Feat: 1.690 G_VGG: 0.879 D_real: 0.436 D_fake: 0.252 
(epoch: 106, iters: 2000, time: 0.077) G_GAN: 1.322 G_GAN_Feat: 2.619 G_VGG: 1.064 D_real: 0.522 D_fake: 0.150 
(epoch: 106, iters: 2400, time: 0.093) G_GAN: 1.053 G_GAN_Feat: 1.762 G_VGG: 0.911 D_real: 0.153 D_fake: 0.316 
(epoch: 106, iters: 2800, time: 0.086) G_GAN: 1.197 G_GAN_Feat: 2.266 G_VGG: 0.978 D_real: 0.053 D_fake: 0.276 
End of epoch 106 / 200 	 Time Taken: 285 sec
(epoch: 107, iters: 240, time: 0.089) G_GAN: 0.862 G_GAN_Feat: 1.834 G_VGG: 0.894 D_real: 0.030 D_fake: 0.445 
saving the latest model (epoch 107, total_steps 314000)
(epoch: 107, iters: 640, time: 0.088) G_GAN: 0.895 G_GAN_Feat: 2.694 G_VGG: 0.915 D_real: 0.508 D_fake: 0.303 
(epoch: 107, iters: 1040, time: 0.086) G_GAN: 1.241 G_GAN_Feat: 1.952 G_VGG: 0.953 D_real: 0.079 D_fake: 0.220 
(epoch: 107, iters: 1440, time: 0.092) G_GAN: 1.810 G_GAN_Feat: 2.514 G_VGG: 0.846 D_real: 0.084 D_fake: 0.045 
(epoch: 107, iters: 1840, time: 0.091) G_GAN: 1.291 G_GAN_Feat: 2.318 G_VGG: 0.987 D_real: 0.116 D_fake: 0.406 
(epoch: 107, iters: 2240, time: 0.092) G_GAN: 0.502 G_GAN_Feat: 1.842 G_VGG: 0.989 D_real: 0.065 D_fake: 0.667 
saving the latest model (epoch 107, total_steps 316000)
(epoch: 107, iters: 2640, time: 0.100) G_GAN: 1.468 G_GAN_Feat: 1.743 G_VGG: 0.939 D_real: 0.499 D_fake: 0.114 
End of epoch 107 / 200 	 Time Taken: 286 sec
(epoch: 108, iters: 80, time: 0.096) G_GAN: 1.517 G_GAN_Feat: 2.661 G_VGG: 0.979 D_real: 0.165 D_fake: 0.068 
(epoch: 108, iters: 480, time: 0.099) G_GAN: 1.628 G_GAN_Feat: 2.165 G_VGG: 0.962 D_real: 0.494 D_fake: 0.217 
(epoch: 108, iters: 880, time: 0.093) G_GAN: 1.992 G_GAN_Feat: 2.634 G_VGG: 0.946 D_real: 0.230 D_fake: 0.034 
(epoch: 108, iters: 1280, time: 0.090) G_GAN: 1.723 G_GAN_Feat: 2.061 G_VGG: 0.915 D_real: 0.599 D_fake: 0.043 
saving the latest model (epoch 108, total_steps 318000)
(epoch: 108, iters: 1680, time: 0.100) G_GAN: 1.049 G_GAN_Feat: 1.961 G_VGG: 1.023 D_real: 0.328 D_fake: 0.250 
(epoch: 108, iters: 2080, time: 0.091) G_GAN: 0.990 G_GAN_Feat: 1.922 G_VGG: 1.014 D_real: 0.125 D_fake: 0.437 
(epoch: 108, iters: 2480, time: 0.095) G_GAN: 1.113 G_GAN_Feat: 1.874 G_VGG: 0.915 D_real: 0.519 D_fake: 0.303 
(epoch: 108, iters: 2880, time: 0.092) G_GAN: 2.437 G_GAN_Feat: 2.962 G_VGG: 0.846 D_real: 0.424 D_fake: 0.049 
End of epoch 108 / 200 	 Time Taken: 283 sec
(epoch: 109, iters: 320, time: 0.096) G_GAN: 1.684 G_GAN_Feat: 2.235 G_VGG: 0.914 D_real: 0.133 D_fake: 0.067 
saving the latest model (epoch 109, total_steps 320000)
(epoch: 109, iters: 720, time: 0.090) G_GAN: 1.287 G_GAN_Feat: 1.925 G_VGG: 0.871 D_real: 0.198 D_fake: 0.238 
(epoch: 109, iters: 1120, time: 0.090) G_GAN: 1.302 G_GAN_Feat: 1.928 G_VGG: 0.903 D_real: 0.079 D_fake: 0.183 
(epoch: 109, iters: 1520, time: 0.084) G_GAN: 0.733 G_GAN_Feat: 1.069 G_VGG: 0.940 D_real: 0.335 D_fake: 0.394 
(epoch: 109, iters: 1920, time: 0.097) G_GAN: 1.043 G_GAN_Feat: 1.064 G_VGG: 0.890 D_real: 0.437 D_fake: 0.286 
(epoch: 109, iters: 2320, time: 0.091) G_GAN: 1.042 G_GAN_Feat: 1.172 G_VGG: 0.932 D_real: 0.753 D_fake: 0.291 
saving the latest model (epoch 109, total_steps 322000)
(epoch: 109, iters: 2720, time: 0.084) G_GAN: 0.878 G_GAN_Feat: 1.534 G_VGG: 0.962 D_real: 0.209 D_fake: 0.336 
End of epoch 109 / 200 	 Time Taken: 287 sec
(epoch: 110, iters: 160, time: 0.089) G_GAN: 0.826 G_GAN_Feat: 1.614 G_VGG: 0.898 D_real: 0.253 D_fake: 0.348 
(epoch: 110, iters: 560, time: 0.090) G_GAN: 1.202 G_GAN_Feat: 2.330 G_VGG: 0.880 D_real: 0.191 D_fake: 0.424 
(epoch: 110, iters: 960, time: 0.089) G_GAN: 0.601 G_GAN_Feat: 2.155 G_VGG: 0.996 D_real: 0.417 D_fake: 0.495 
(epoch: 110, iters: 1360, time: 0.088) G_GAN: 1.055 G_GAN_Feat: 2.220 G_VGG: 0.968 D_real: 0.033 D_fake: 0.225 
saving the latest model (epoch 110, total_steps 324000)
(epoch: 110, iters: 1760, time: 0.095) G_GAN: 1.543 G_GAN_Feat: 2.182 G_VGG: 1.015 D_real: 0.135 D_fake: 0.095 
(epoch: 110, iters: 2160, time: 0.092) G_GAN: 1.275 G_GAN_Feat: 2.349 G_VGG: 0.935 D_real: 0.186 D_fake: 0.153 
(epoch: 110, iters: 2560, time: 0.092) G_GAN: 1.066 G_GAN_Feat: 1.868 G_VGG: 0.992 D_real: 0.261 D_fake: 0.316 
(epoch: 110, iters: 2960, time: 0.092) G_GAN: 1.227 G_GAN_Feat: 1.566 G_VGG: 0.947 D_real: 0.427 D_fake: 0.143 
End of epoch 110 / 200 	 Time Taken: 284 sec
saving the model at the end of epoch 110, iters 325600
(epoch: 111, iters: 400, time: 0.084) G_GAN: 1.025 G_GAN_Feat: 1.879 G_VGG: 0.837 D_real: 0.135 D_fake: 0.266 
saving the latest model (epoch 111, total_steps 326000)
(epoch: 111, iters: 800, time: 0.092) G_GAN: 0.905 G_GAN_Feat: 1.745 G_VGG: 0.943 D_real: 0.109 D_fake: 0.510 
(epoch: 111, iters: 1200, time: 0.095) G_GAN: 1.019 G_GAN_Feat: 2.086 G_VGG: 0.959 D_real: 0.834 D_fake: 0.290 
(epoch: 111, iters: 1600, time: 0.091) G_GAN: 1.042 G_GAN_Feat: 1.746 G_VGG: 0.882 D_real: 0.069 D_fake: 0.494 
(epoch: 111, iters: 2000, time: 0.091) G_GAN: 2.554 G_GAN_Feat: 2.316 G_VGG: 1.025 D_real: 0.582 D_fake: 0.062 
(epoch: 111, iters: 2400, time: 0.098) G_GAN: 1.381 G_GAN_Feat: 1.678 G_VGG: 1.062 D_real: 0.531 D_fake: 0.184 
saving the latest model (epoch 111, total_steps 328000)
(epoch: 111, iters: 2800, time: 0.096) G_GAN: 1.991 G_GAN_Feat: 2.251 G_VGG: 0.967 D_real: 0.323 D_fake: 0.070 
End of epoch 111 / 200 	 Time Taken: 286 sec
(epoch: 112, iters: 240, time: 0.093) G_GAN: 0.571 G_GAN_Feat: 1.382 G_VGG: 0.896 D_real: 0.251 D_fake: 0.511 
(epoch: 112, iters: 640, time: 0.089) G_GAN: 0.732 G_GAN_Feat: 1.825 G_VGG: 0.953 D_real: 0.237 D_fake: 0.444 
(epoch: 112, iters: 1040, time: 0.094) G_GAN: 1.004 G_GAN_Feat: 2.196 G_VGG: 0.934 D_real: 0.077 D_fake: 0.270 
(epoch: 112, iters: 1440, time: 0.093) G_GAN: 1.040 G_GAN_Feat: 1.833 G_VGG: 0.956 D_real: 0.034 D_fake: 0.364 
saving the latest model (epoch 112, total_steps 330000)
(epoch: 112, iters: 1840, time: 0.095) G_GAN: 1.634 G_GAN_Feat: 2.587 G_VGG: 0.969 D_real: 0.097 D_fake: 0.073 
(epoch: 112, iters: 2240, time: 0.085) G_GAN: 0.950 G_GAN_Feat: 2.197 G_VGG: 0.947 D_real: 0.527 D_fake: 0.327 
(epoch: 112, iters: 2640, time: 0.089) G_GAN: 0.810 G_GAN_Feat: 1.811 G_VGG: 0.913 D_real: 0.308 D_fake: 0.456 
End of epoch 112 / 200 	 Time Taken: 285 sec
(epoch: 113, iters: 80, time: 0.090) G_GAN: 0.818 G_GAN_Feat: 1.546 G_VGG: 0.818 D_real: 0.597 D_fake: 0.324 
(epoch: 113, iters: 480, time: 0.084) G_GAN: 0.577 G_GAN_Feat: 1.033 G_VGG: 0.829 D_real: 0.465 D_fake: 0.460 
saving the latest model (epoch 113, total_steps 332000)
(epoch: 113, iters: 880, time: 0.100) G_GAN: 0.566 G_GAN_Feat: 1.845 G_VGG: 0.825 D_real: 0.443 D_fake: 0.540 
(epoch: 113, iters: 1280, time: 0.092) G_GAN: 0.667 G_GAN_Feat: 2.071 G_VGG: 0.992 D_real: 0.049 D_fake: 0.488 
(epoch: 113, iters: 1680, time: 0.087) G_GAN: 1.289 G_GAN_Feat: 1.427 G_VGG: 0.911 D_real: 0.790 D_fake: 0.336 
(epoch: 113, iters: 2080, time: 0.084) G_GAN: 1.012 G_GAN_Feat: 1.609 G_VGG: 0.907 D_real: 0.517 D_fake: 0.369 
(epoch: 113, iters: 2480, time: 0.086) G_GAN: 1.482 G_GAN_Feat: 1.989 G_VGG: 0.906 D_real: 0.352 D_fake: 0.382 
saving the latest model (epoch 113, total_steps 334000)
(epoch: 113, iters: 2880, time: 0.085) G_GAN: 0.746 G_GAN_Feat: 1.893 G_VGG: 0.983 D_real: 0.074 D_fake: 0.677 
End of epoch 113 / 200 	 Time Taken: 286 sec
(epoch: 114, iters: 320, time: 0.095) G_GAN: 0.952 G_GAN_Feat: 2.318 G_VGG: 0.991 D_real: 0.075 D_fake: 0.308 
(epoch: 114, iters: 720, time: 0.088) G_GAN: 0.581 G_GAN_Feat: 1.880 G_VGG: 0.860 D_real: 0.117 D_fake: 0.600 
(epoch: 114, iters: 1120, time: 0.083) G_GAN: 1.627 G_GAN_Feat: 2.856 G_VGG: 0.951 D_real: 0.042 D_fake: 0.074 
(epoch: 114, iters: 1520, time: 0.093) G_GAN: 1.581 G_GAN_Feat: 2.331 G_VGG: 0.981 D_real: 0.415 D_fake: 0.173 
saving the latest model (epoch 114, total_steps 336000)
(epoch: 114, iters: 1920, time: 0.090) G_GAN: 1.081 G_GAN_Feat: 2.165 G_VGG: 0.987 D_real: 0.131 D_fake: 0.458 
(epoch: 114, iters: 2320, time: 0.084) G_GAN: 1.021 G_GAN_Feat: 2.093 G_VGG: 0.918 D_real: 0.361 D_fake: 0.302 
(epoch: 114, iters: 2720, time: 0.088) G_GAN: 0.620 G_GAN_Feat: 1.470 G_VGG: 0.881 D_real: 0.384 D_fake: 0.506 
End of epoch 114 / 200 	 Time Taken: 283 sec
(epoch: 115, iters: 160, time: 0.090) G_GAN: 1.207 G_GAN_Feat: 2.221 G_VGG: 0.907 D_real: 0.064 D_fake: 0.187 
(epoch: 115, iters: 560, time: 0.096) G_GAN: 1.300 G_GAN_Feat: 2.141 G_VGG: 0.877 D_real: 0.071 D_fake: 0.263 
saving the latest model (epoch 115, total_steps 338000)
(epoch: 115, iters: 960, time: 0.094) G_GAN: 1.543 G_GAN_Feat: 2.068 G_VGG: 0.960 D_real: 0.460 D_fake: 0.095 
(epoch: 115, iters: 1360, time: 0.091) G_GAN: 1.685 G_GAN_Feat: 2.121 G_VGG: 0.888 D_real: 0.777 D_fake: 0.082 
(epoch: 115, iters: 1760, time: 0.085) G_GAN: 1.999 G_GAN_Feat: 2.082 G_VGG: 0.896 D_real: 1.515 D_fake: 0.109 
(epoch: 115, iters: 2160, time: 0.077) G_GAN: 1.551 G_GAN_Feat: 1.879 G_VGG: 0.991 D_real: 0.626 D_fake: 0.216 
(epoch: 115, iters: 2560, time: 0.099) G_GAN: 1.164 G_GAN_Feat: 1.932 G_VGG: 0.888 D_real: 0.167 D_fake: 0.283 
saving the latest model (epoch 115, total_steps 340000)
(epoch: 115, iters: 2960, time: 0.089) G_GAN: 0.923 G_GAN_Feat: 1.947 G_VGG: 0.860 D_real: 0.055 D_fake: 0.321 
End of epoch 115 / 200 	 Time Taken: 288 sec
(epoch: 116, iters: 400, time: 0.097) G_GAN: 1.191 G_GAN_Feat: 1.987 G_VGG: 0.912 D_real: 0.364 D_fake: 0.291 
(epoch: 116, iters: 800, time: 0.084) G_GAN: 1.060 G_GAN_Feat: 2.349 G_VGG: 0.885 D_real: 0.121 D_fake: 0.278 
(epoch: 116, iters: 1200, time: 0.095) G_GAN: 0.962 G_GAN_Feat: 2.122 G_VGG: 0.963 D_real: 0.159 D_fake: 0.586 
(epoch: 116, iters: 1600, time: 0.083) G_GAN: 1.083 G_GAN_Feat: 1.664 G_VGG: 0.951 D_real: 0.147 D_fake: 0.351 
saving the latest model (epoch 116, total_steps 342000)
(epoch: 116, iters: 2000, time: 0.087) G_GAN: 1.286 G_GAN_Feat: 2.337 G_VGG: 1.108 D_real: 0.308 D_fake: 0.239 
(epoch: 116, iters: 2400, time: 0.085) G_GAN: 1.124 G_GAN_Feat: 2.153 G_VGG: 0.963 D_real: 0.319 D_fake: 0.281 
(epoch: 116, iters: 2800, time: 0.090) G_GAN: 1.384 G_GAN_Feat: 2.265 G_VGG: 0.885 D_real: 0.143 D_fake: 0.117 
End of epoch 116 / 200 	 Time Taken: 283 sec
(epoch: 117, iters: 240, time: 0.091) G_GAN: 1.039 G_GAN_Feat: 2.175 G_VGG: 0.941 D_real: 0.204 D_fake: 0.195 
(epoch: 117, iters: 640, time: 0.083) G_GAN: 0.806 G_GAN_Feat: 1.766 G_VGG: 0.928 D_real: 0.024 D_fake: 0.557 
saving the latest model (epoch 117, total_steps 344000)
(epoch: 117, iters: 1040, time: 0.084) G_GAN: 1.596 G_GAN_Feat: 2.263 G_VGG: 0.832 D_real: 0.763 D_fake: 0.110 
(epoch: 117, iters: 1440, time: 0.096) G_GAN: 0.961 G_GAN_Feat: 2.028 G_VGG: 0.789 D_real: 0.054 D_fake: 0.260 
(epoch: 117, iters: 1840, time: 0.094) G_GAN: 1.575 G_GAN_Feat: 1.955 G_VGG: 0.900 D_real: 0.225 D_fake: 0.217 
(epoch: 117, iters: 2240, time: 0.099) G_GAN: 1.302 G_GAN_Feat: 2.025 G_VGG: 0.886 D_real: 0.580 D_fake: 0.197 
(epoch: 117, iters: 2640, time: 0.085) G_GAN: 0.916 G_GAN_Feat: 2.100 G_VGG: 0.962 D_real: 0.270 D_fake: 0.279 
saving the latest model (epoch 117, total_steps 346000)
End of epoch 117 / 200 	 Time Taken: 286 sec
(epoch: 118, iters: 80, time: 0.093) G_GAN: 1.123 G_GAN_Feat: 1.637 G_VGG: 0.967 D_real: 0.236 D_fake: 0.242 
(epoch: 118, iters: 480, time: 0.095) G_GAN: 0.850 G_GAN_Feat: 1.876 G_VGG: 0.961 D_real: 0.066 D_fake: 0.534 
(epoch: 118, iters: 880, time: 0.093) G_GAN: 1.588 G_GAN_Feat: 2.251 G_VGG: 0.940 D_real: 0.362 D_fake: 0.084 
(epoch: 118, iters: 1280, time: 0.078) G_GAN: 1.219 G_GAN_Feat: 1.875 G_VGG: 0.785 D_real: 0.439 D_fake: 0.229 
(epoch: 118, iters: 1680, time: 0.085) G_GAN: 1.124 G_GAN_Feat: 2.027 G_VGG: 0.873 D_real: 0.093 D_fake: 0.344 
saving the latest model (epoch 118, total_steps 348000)
(epoch: 118, iters: 2080, time: 0.100) G_GAN: 1.054 G_GAN_Feat: 1.949 G_VGG: 0.927 D_real: 0.174 D_fake: 0.371 
(epoch: 118, iters: 2480, time: 0.093) G_GAN: 0.638 G_GAN_Feat: 1.818 G_VGG: 0.835 D_real: 0.043 D_fake: 0.526 
(epoch: 118, iters: 2880, time: 0.089) G_GAN: 1.810 G_GAN_Feat: 1.739 G_VGG: 0.937 D_real: 0.482 D_fake: 0.090 
End of epoch 118 / 200 	 Time Taken: 283 sec
(epoch: 119, iters: 320, time: 0.097) G_GAN: 0.861 G_GAN_Feat: 2.067 G_VGG: 0.949 D_real: 0.473 D_fake: 0.308 
(epoch: 119, iters: 720, time: 0.091) G_GAN: 1.238 G_GAN_Feat: 2.155 G_VGG: 0.852 D_real: 0.107 D_fake: 0.189 
saving the latest model (epoch 119, total_steps 350000)
(epoch: 119, iters: 1120, time: 0.096) G_GAN: 0.740 G_GAN_Feat: 1.605 G_VGG: 0.899 D_real: 0.234 D_fake: 0.410 
(epoch: 119, iters: 1520, time: 0.094) G_GAN: 0.439 G_GAN_Feat: 1.526 G_VGG: 0.817 D_real: 0.063 D_fake: 0.775 
(epoch: 119, iters: 1920, time: 0.099) G_GAN: 2.060 G_GAN_Feat: 2.334 G_VGG: 0.872 D_real: 0.804 D_fake: 0.045 
(epoch: 119, iters: 2320, time: 0.086) G_GAN: 1.052 G_GAN_Feat: 2.206 G_VGG: 1.005 D_real: 0.044 D_fake: 0.437 
(epoch: 119, iters: 2720, time: 0.093) G_GAN: 1.544 G_GAN_Feat: 2.195 G_VGG: 0.894 D_real: 0.178 D_fake: 0.152 
saving the latest model (epoch 119, total_steps 352000)
End of epoch 119 / 200 	 Time Taken: 286 sec
(epoch: 120, iters: 160, time: 0.095) G_GAN: 1.339 G_GAN_Feat: 2.150 G_VGG: 0.867 D_real: 0.241 D_fake: 0.157 
(epoch: 120, iters: 560, time: 0.094) G_GAN: 1.002 G_GAN_Feat: 1.995 G_VGG: 0.898 D_real: 0.050 D_fake: 0.428 
(epoch: 120, iters: 960, time: 0.085) G_GAN: 1.577 G_GAN_Feat: 2.286 G_VGG: 0.855 D_real: 0.239 D_fake: 0.144 
(epoch: 120, iters: 1360, time: 0.096) G_GAN: 1.137 G_GAN_Feat: 2.365 G_VGG: 0.927 D_real: 0.437 D_fake: 0.218 
(epoch: 120, iters: 1760, time: 0.097) G_GAN: 0.743 G_GAN_Feat: 2.410 G_VGG: 0.919 D_real: 0.037 D_fake: 0.510 
saving the latest model (epoch 120, total_steps 354000)
(epoch: 120, iters: 2160, time: 0.097) G_GAN: 0.940 G_GAN_Feat: 2.454 G_VGG: 0.924 D_real: 0.184 D_fake: 0.301 
(epoch: 120, iters: 2560, time: 0.091) G_GAN: 0.794 G_GAN_Feat: 1.338 G_VGG: 0.890 D_real: 0.299 D_fake: 0.347 
(epoch: 120, iters: 2960, time: 0.092) G_GAN: 1.299 G_GAN_Feat: 1.942 G_VGG: 0.971 D_real: 0.538 D_fake: 0.206 
End of epoch 120 / 200 	 Time Taken: 285 sec
saving the model at the end of epoch 120, iters 355200
(epoch: 121, iters: 400, time: 0.097) G_GAN: 1.325 G_GAN_Feat: 2.373 G_VGG: 1.030 D_real: 0.467 D_fake: 0.430 
(epoch: 121, iters: 800, time: 0.099) G_GAN: 1.272 G_GAN_Feat: 1.901 G_VGG: 0.975 D_real: 0.320 D_fake: 0.154 
saving the latest model (epoch 121, total_steps 356000)
(epoch: 121, iters: 1200, time: 0.094) G_GAN: 1.015 G_GAN_Feat: 2.006 G_VGG: 0.893 D_real: 0.163 D_fake: 0.366 
(epoch: 121, iters: 1600, time: 0.084) G_GAN: 1.666 G_GAN_Feat: 1.958 G_VGG: 0.981 D_real: 0.418 D_fake: 0.179 
(epoch: 121, iters: 2000, time: 0.086) G_GAN: 1.207 G_GAN_Feat: 2.219 G_VGG: 0.929 D_real: 0.500 D_fake: 0.158 
(epoch: 121, iters: 2400, time: 0.101) G_GAN: 0.908 G_GAN_Feat: 2.273 G_VGG: 0.852 D_real: 0.253 D_fake: 0.312 
(epoch: 121, iters: 2800, time: 0.089) G_GAN: 1.103 G_GAN_Feat: 2.052 G_VGG: 1.024 D_real: 0.083 D_fake: 0.427 
saving the latest model (epoch 121, total_steps 358000)
End of epoch 121 / 200 	 Time Taken: 287 sec
(epoch: 122, iters: 240, time: 0.091) G_GAN: 0.800 G_GAN_Feat: 1.913 G_VGG: 0.838 D_real: 0.026 D_fake: 0.414 
(epoch: 122, iters: 640, time: 0.097) G_GAN: 1.151 G_GAN_Feat: 1.897 G_VGG: 0.876 D_real: 0.352 D_fake: 0.244 
(epoch: 122, iters: 1040, time: 0.097) G_GAN: 1.066 G_GAN_Feat: 2.115 G_VGG: 0.964 D_real: 0.394 D_fake: 0.222 
(epoch: 122, iters: 1440, time: 0.096) G_GAN: 0.935 G_GAN_Feat: 2.159 G_VGG: 0.880 D_real: 0.818 D_fake: 0.266 
(epoch: 122, iters: 1840, time: 0.096) G_GAN: 1.217 G_GAN_Feat: 2.566 G_VGG: 0.937 D_real: 0.206 D_fake: 0.144 
saving the latest model (epoch 122, total_steps 360000)
(epoch: 122, iters: 2240, time: 0.086) G_GAN: 0.754 G_GAN_Feat: 2.195 G_VGG: 0.926 D_real: 0.293 D_fake: 0.452 
(epoch: 122, iters: 2640, time: 0.091) G_GAN: 1.357 G_GAN_Feat: 2.168 G_VGG: 0.922 D_real: 0.148 D_fake: 0.131 
End of epoch 122 / 200 	 Time Taken: 285 sec
(epoch: 123, iters: 80, time: 0.089) G_GAN: 0.807 G_GAN_Feat: 1.771 G_VGG: 0.869 D_real: 0.329 D_fake: 0.388 
(epoch: 123, iters: 480, time: 0.091) G_GAN: 1.634 G_GAN_Feat: 2.481 G_VGG: 0.912 D_real: 0.373 D_fake: 0.091 
(epoch: 123, iters: 880, time: 0.096) G_GAN: 0.724 G_GAN_Feat: 1.940 G_VGG: 1.034 D_real: 0.143 D_fake: 0.445 
saving the latest model (epoch 123, total_steps 362000)
(epoch: 123, iters: 1280, time: 0.093) G_GAN: 1.411 G_GAN_Feat: 2.749 G_VGG: 0.942 D_real: 0.090 D_fake: 0.221 
(epoch: 123, iters: 1680, time: 0.087) G_GAN: 1.568 G_GAN_Feat: 2.860 G_VGG: 0.916 D_real: 0.037 D_fake: 0.223 
(epoch: 123, iters: 2080, time: 0.096) G_GAN: 1.006 G_GAN_Feat: 2.009 G_VGG: 0.940 D_real: 0.082 D_fake: 0.286 
(epoch: 123, iters: 2480, time: 0.100) G_GAN: 1.132 G_GAN_Feat: 1.959 G_VGG: 0.992 D_real: 0.218 D_fake: 0.206 
(epoch: 123, iters: 2880, time: 0.097) G_GAN: 1.002 G_GAN_Feat: 2.163 G_VGG: 0.942 D_real: 0.441 D_fake: 0.245 
saving the latest model (epoch 123, total_steps 364000)
End of epoch 123 / 200 	 Time Taken: 290 sec
(epoch: 124, iters: 320, time: 0.098) G_GAN: 0.727 G_GAN_Feat: 2.077 G_VGG: 0.970 D_real: 0.213 D_fake: 0.430 
(epoch: 124, iters: 720, time: 0.092) G_GAN: 1.113 G_GAN_Feat: 1.729 G_VGG: 0.874 D_real: 0.134 D_fake: 0.396 
(epoch: 124, iters: 1120, time: 0.088) G_GAN: 1.163 G_GAN_Feat: 1.983 G_VGG: 0.958 D_real: 0.139 D_fake: 0.336 
(epoch: 124, iters: 1520, time: 0.088) G_GAN: 0.963 G_GAN_Feat: 2.234 G_VGG: 0.819 D_real: 0.052 D_fake: 0.356 
(epoch: 124, iters: 1920, time: 0.090) G_GAN: 0.921 G_GAN_Feat: 2.012 G_VGG: 0.915 D_real: 0.221 D_fake: 0.312 
saving the latest model (epoch 124, total_steps 366000)
(epoch: 124, iters: 2320, time: 0.091) G_GAN: 1.301 G_GAN_Feat: 2.110 G_VGG: 0.871 D_real: 0.048 D_fake: 0.221 
(epoch: 124, iters: 2720, time: 0.091) G_GAN: 1.494 G_GAN_Feat: 1.990 G_VGG: 0.818 D_real: 0.279 D_fake: 0.278 
End of epoch 124 / 200 	 Time Taken: 283 sec
(epoch: 125, iters: 160, time: 0.096) G_GAN: 1.326 G_GAN_Feat: 2.723 G_VGG: 1.036 D_real: 0.057 D_fake: 0.142 
(epoch: 125, iters: 560, time: 0.087) G_GAN: 0.795 G_GAN_Feat: 1.591 G_VGG: 0.862 D_real: 0.280 D_fake: 0.352 
(epoch: 125, iters: 960, time: 0.093) G_GAN: 1.132 G_GAN_Feat: 2.215 G_VGG: 0.929 D_real: 0.275 D_fake: 0.326 
saving the latest model (epoch 125, total_steps 368000)
(epoch: 125, iters: 1360, time: 0.092) G_GAN: 1.597 G_GAN_Feat: 2.377 G_VGG: 0.971 D_real: 0.171 D_fake: 0.144 
(epoch: 125, iters: 1760, time: 0.093) G_GAN: 0.983 G_GAN_Feat: 2.777 G_VGG: 0.965 D_real: 0.225 D_fake: 0.256 
(epoch: 125, iters: 2160, time: 0.094) G_GAN: 0.871 G_GAN_Feat: 2.048 G_VGG: 0.861 D_real: 0.030 D_fake: 0.856 
(epoch: 125, iters: 2560, time: 0.094) G_GAN: 1.371 G_GAN_Feat: 2.157 G_VGG: 0.897 D_real: 0.724 D_fake: 0.144 
(epoch: 125, iters: 2960, time: 0.091) G_GAN: 1.049 G_GAN_Feat: 1.868 G_VGG: 0.912 D_real: 0.327 D_fake: 0.255 
saving the latest model (epoch 125, total_steps 370000)
End of epoch 125 / 200 	 Time Taken: 287 sec
(epoch: 126, iters: 400, time: 0.094) G_GAN: 1.064 G_GAN_Feat: 2.158 G_VGG: 0.850 D_real: 0.083 D_fake: 0.370 
(epoch: 126, iters: 800, time: 0.093) G_GAN: 1.560 G_GAN_Feat: 2.449 G_VGG: 0.959 D_real: 0.206 D_fake: 0.139 
(epoch: 126, iters: 1200, time: 0.101) G_GAN: 0.973 G_GAN_Feat: 2.281 G_VGG: 0.914 D_real: 0.176 D_fake: 0.286 
(epoch: 126, iters: 1600, time: 0.085) G_GAN: 1.296 G_GAN_Feat: 2.721 G_VGG: 0.874 D_real: 0.031 D_fake: 0.201 
(epoch: 126, iters: 2000, time: 0.086) G_GAN: 1.205 G_GAN_Feat: 1.828 G_VGG: 0.940 D_real: 0.247 D_fake: 0.333 
saving the latest model (epoch 126, total_steps 372000)
(epoch: 126, iters: 2400, time: 0.094) G_GAN: 1.434 G_GAN_Feat: 2.198 G_VGG: 0.890 D_real: 0.144 D_fake: 0.333 
(epoch: 126, iters: 2800, time: 0.091) G_GAN: 0.874 G_GAN_Feat: 1.805 G_VGG: 0.828 D_real: 0.077 D_fake: 0.484 
End of epoch 126 / 200 	 Time Taken: 283 sec
(epoch: 127, iters: 240, time: 0.086) G_GAN: 0.944 G_GAN_Feat: 2.283 G_VGG: 0.827 D_real: 0.050 D_fake: 0.395 
(epoch: 127, iters: 640, time: 0.090) G_GAN: 1.143 G_GAN_Feat: 2.244 G_VGG: 0.980 D_real: 0.048 D_fake: 0.363 
(epoch: 127, iters: 1040, time: 0.086) G_GAN: 0.739 G_GAN_Feat: 2.524 G_VGG: 0.825 D_real: 0.056 D_fake: 0.421 
saving the latest model (epoch 127, total_steps 374000)
(epoch: 127, iters: 1440, time: 0.090) G_GAN: 1.486 G_GAN_Feat: 2.548 G_VGG: 0.920 D_real: 0.157 D_fake: 0.203 
(epoch: 127, iters: 1840, time: 0.087) G_GAN: 1.585 G_GAN_Feat: 2.294 G_VGG: 0.906 D_real: 0.188 D_fake: 0.324 
(epoch: 127, iters: 2240, time: 0.091) G_GAN: 1.489 G_GAN_Feat: 2.599 G_VGG: 0.901 D_real: 0.082 D_fake: 0.210 
(epoch: 127, iters: 2640, time: 0.093) G_GAN: 1.526 G_GAN_Feat: 2.220 G_VGG: 0.893 D_real: 0.206 D_fake: 0.287 
End of epoch 127 / 200 	 Time Taken: 285 sec
(epoch: 128, iters: 80, time: 0.090) G_GAN: 1.924 G_GAN_Feat: 2.360 G_VGG: 0.789 D_real: 0.452 D_fake: 0.071 
saving the latest model (epoch 128, total_steps 376000)
(epoch: 128, iters: 480, time: 0.094) G_GAN: 1.232 G_GAN_Feat: 2.147 G_VGG: 0.870 D_real: 0.083 D_fake: 0.245 
(epoch: 128, iters: 880, time: 0.094) G_GAN: 1.447 G_GAN_Feat: 2.374 G_VGG: 0.814 D_real: 0.316 D_fake: 0.148 
(epoch: 128, iters: 1280, time: 0.088) G_GAN: 1.375 G_GAN_Feat: 2.390 G_VGG: 0.915 D_real: 0.385 D_fake: 0.154 
(epoch: 128, iters: 1680, time: 0.094) G_GAN: 1.258 G_GAN_Feat: 2.707 G_VGG: 0.949 D_real: 0.089 D_fake: 0.194 
(epoch: 128, iters: 2080, time: 0.093) G_GAN: 1.397 G_GAN_Feat: 2.466 G_VGG: 0.895 D_real: 0.677 D_fake: 0.154 
saving the latest model (epoch 128, total_steps 378000)
(epoch: 128, iters: 2480, time: 0.088) G_GAN: 1.254 G_GAN_Feat: 2.487 G_VGG: 0.892 D_real: 0.050 D_fake: 0.189 
(epoch: 128, iters: 2880, time: 0.092) G_GAN: 1.040 G_GAN_Feat: 1.526 G_VGG: 0.756 D_real: 0.224 D_fake: 0.287 
End of epoch 128 / 200 	 Time Taken: 289 sec
(epoch: 129, iters: 320, time: 0.092) G_GAN: 0.995 G_GAN_Feat: 2.198 G_VGG: 0.947 D_real: 0.138 D_fake: 0.368 
(epoch: 129, iters: 720, time: 0.101) G_GAN: 1.246 G_GAN_Feat: 2.135 G_VGG: 0.879 D_real: 0.160 D_fake: 0.235 
(epoch: 129, iters: 1120, time: 0.085) G_GAN: 1.843 G_GAN_Feat: 2.282 G_VGG: 0.800 D_real: 0.282 D_fake: 0.248 
saving the latest model (epoch 129, total_steps 380000)
(epoch: 129, iters: 1520, time: 0.095) G_GAN: 0.712 G_GAN_Feat: 1.847 G_VGG: 0.870 D_real: 0.246 D_fake: 0.439 
(epoch: 129, iters: 1920, time: 0.085) G_GAN: 0.915 G_GAN_Feat: 2.097 G_VGG: 1.021 D_real: 0.105 D_fake: 0.448 
(epoch: 129, iters: 2320, time: 0.099) G_GAN: 1.137 G_GAN_Feat: 2.721 G_VGG: 0.991 D_real: 0.094 D_fake: 0.214 
(epoch: 129, iters: 2720, time: 0.090) G_GAN: 1.783 G_GAN_Feat: 2.559 G_VGG: 0.853 D_real: 0.298 D_fake: 0.044 
End of epoch 129 / 200 	 Time Taken: 283 sec
(epoch: 130, iters: 160, time: 0.096) G_GAN: 2.020 G_GAN_Feat: 3.279 G_VGG: 0.889 D_real: 0.152 D_fake: 0.090 
saving the latest model (epoch 130, total_steps 382000)
(epoch: 130, iters: 560, time: 0.099) G_GAN: 1.948 G_GAN_Feat: 3.309 G_VGG: 0.894 D_real: 0.064 D_fake: 0.017 
(epoch: 130, iters: 960, time: 0.092) G_GAN: 1.607 G_GAN_Feat: 2.737 G_VGG: 0.889 D_real: 0.192 D_fake: 0.170 
(epoch: 130, iters: 1360, time: 0.093) G_GAN: 1.686 G_GAN_Feat: 2.404 G_VGG: 0.886 D_real: 0.165 D_fake: 0.249 
(epoch: 130, iters: 1760, time: 0.093) G_GAN: 1.481 G_GAN_Feat: 2.203 G_VGG: 0.823 D_real: 0.275 D_fake: 0.395 
(epoch: 130, iters: 2160, time: 0.095) G_GAN: 2.044 G_GAN_Feat: 2.922 G_VGG: 0.959 D_real: 0.239 D_fake: 0.090 
saving the latest model (epoch 130, total_steps 384000)
(epoch: 130, iters: 2560, time: 0.097) G_GAN: 1.695 G_GAN_Feat: 2.897 G_VGG: 0.886 D_real: 0.027 D_fake: 0.102 
(epoch: 130, iters: 2960, time: 0.096) G_GAN: 1.518 G_GAN_Feat: 2.432 G_VGG: 0.825 D_real: 0.597 D_fake: 0.087 
End of epoch 130 / 200 	 Time Taken: 288 sec
saving the model at the end of epoch 130, iters 384800
(epoch: 131, iters: 400, time: 0.086) G_GAN: 1.067 G_GAN_Feat: 2.681 G_VGG: 0.860 D_real: 0.571 D_fake: 0.278 
(epoch: 131, iters: 800, time: 0.086) G_GAN: 0.886 G_GAN_Feat: 1.873 G_VGG: 0.761 D_real: 0.156 D_fake: 0.474 
(epoch: 131, iters: 1200, time: 0.085) G_GAN: 2.059 G_GAN_Feat: 2.925 G_VGG: 1.032 D_real: 0.152 D_fake: 0.045 
saving the latest model (epoch 131, total_steps 386000)
(epoch: 131, iters: 1600, time: 0.089) G_GAN: 1.032 G_GAN_Feat: 2.098 G_VGG: 0.916 D_real: 0.192 D_fake: 0.557 
(epoch: 131, iters: 2000, time: 0.099) G_GAN: 1.221 G_GAN_Feat: 2.832 G_VGG: 0.851 D_real: 0.165 D_fake: 0.203 
(epoch: 131, iters: 2400, time: 0.093) G_GAN: 1.069 G_GAN_Feat: 2.744 G_VGG: 0.921 D_real: 0.074 D_fake: 0.839 
(epoch: 131, iters: 2800, time: 0.095) G_GAN: 1.071 G_GAN_Feat: 2.150 G_VGG: 0.913 D_real: 0.045 D_fake: 0.309 
End of epoch 131 / 200 	 Time Taken: 284 sec
(epoch: 132, iters: 240, time: 0.090) G_GAN: 0.806 G_GAN_Feat: 2.127 G_VGG: 0.885 D_real: 0.193 D_fake: 0.465 
saving the latest model (epoch 132, total_steps 388000)
(epoch: 132, iters: 640, time: 0.092) G_GAN: 1.538 G_GAN_Feat: 2.577 G_VGG: 0.890 D_real: 0.100 D_fake: 0.321 
(epoch: 132, iters: 1040, time: 0.084) G_GAN: 1.022 G_GAN_Feat: 1.866 G_VGG: 0.847 D_real: 0.186 D_fake: 0.299 
(epoch: 132, iters: 1440, time: 0.091) G_GAN: 1.063 G_GAN_Feat: 2.335 G_VGG: 0.936 D_real: 0.053 D_fake: 0.475 
(epoch: 132, iters: 1840, time: 0.093) G_GAN: 1.376 G_GAN_Feat: 1.960 G_VGG: 0.895 D_real: 0.203 D_fake: 0.360 
(epoch: 132, iters: 2240, time: 0.099) G_GAN: 1.343 G_GAN_Feat: 2.645 G_VGG: 0.971 D_real: 0.207 D_fake: 0.157 
saving the latest model (epoch 132, total_steps 390000)
(epoch: 132, iters: 2640, time: 0.088) G_GAN: 1.315 G_GAN_Feat: 2.411 G_VGG: 0.848 D_real: 0.140 D_fake: 0.218 
End of epoch 132 / 200 	 Time Taken: 290 sec
(epoch: 133, iters: 80, time: 0.101) G_GAN: 1.466 G_GAN_Feat: 2.478 G_VGG: 0.874 D_real: 0.524 D_fake: 0.294 
(epoch: 133, iters: 480, time: 0.085) G_GAN: 1.401 G_GAN_Feat: 2.861 G_VGG: 0.865 D_real: 0.287 D_fake: 0.236 
(epoch: 133, iters: 880, time: 0.090) G_GAN: 2.081 G_GAN_Feat: 2.980 G_VGG: 0.908 D_real: 0.393 D_fake: 0.036 
(epoch: 133, iters: 1280, time: 0.090) G_GAN: 1.517 G_GAN_Feat: 2.462 G_VGG: 0.881 D_real: 0.103 D_fake: 0.097 
saving the latest model (epoch 133, total_steps 392000)
(epoch: 133, iters: 1680, time: 0.085) G_GAN: 1.922 G_GAN_Feat: 2.683 G_VGG: 0.857 D_real: 0.320 D_fake: 0.103 
(epoch: 133, iters: 2080, time: 0.092) G_GAN: 1.169 G_GAN_Feat: 2.166 G_VGG: 0.877 D_real: 0.241 D_fake: 0.375 
(epoch: 133, iters: 2480, time: 0.086) G_GAN: 1.504 G_GAN_Feat: 2.233 G_VGG: 0.809 D_real: 0.597 D_fake: 0.110 
(epoch: 133, iters: 2880, time: 0.087) G_GAN: 1.047 G_GAN_Feat: 2.091 G_VGG: 0.881 D_real: 0.326 D_fake: 0.270 
End of epoch 133 / 200 	 Time Taken: 285 sec
(epoch: 134, iters: 320, time: 0.097) G_GAN: 1.204 G_GAN_Feat: 2.565 G_VGG: 0.927 D_real: 0.158 D_fake: 0.455 
saving the latest model (epoch 134, total_steps 394000)
(epoch: 134, iters: 720, time: 0.086) G_GAN: 1.012 G_GAN_Feat: 2.581 G_VGG: 0.781 D_real: 0.342 D_fake: 0.331 
(epoch: 134, iters: 1120, time: 0.081) G_GAN: 1.066 G_GAN_Feat: 2.079 G_VGG: 0.780 D_real: 0.179 D_fake: 0.330 
(epoch: 134, iters: 1520, time: 0.095) G_GAN: 1.718 G_GAN_Feat: 2.662 G_VGG: 0.861 D_real: 0.248 D_fake: 0.049 
(epoch: 134, iters: 1920, time: 0.091) G_GAN: 1.314 G_GAN_Feat: 2.259 G_VGG: 0.926 D_real: 0.260 D_fake: 0.237 
(epoch: 134, iters: 2320, time: 0.093) G_GAN: 1.522 G_GAN_Feat: 3.116 G_VGG: 0.831 D_real: 0.035 D_fake: 0.084 
saving the latest model (epoch 134, total_steps 396000)
(epoch: 134, iters: 2720, time: 0.104) G_GAN: 1.683 G_GAN_Feat: 2.857 G_VGG: 0.885 D_real: 0.227 D_fake: 0.215 
End of epoch 134 / 200 	 Time Taken: 288 sec
(epoch: 135, iters: 160, time: 0.098) G_GAN: 0.650 G_GAN_Feat: 2.007 G_VGG: 0.869 D_real: 0.128 D_fake: 0.488 
(epoch: 135, iters: 560, time: 0.086) G_GAN: 1.510 G_GAN_Feat: 2.341 G_VGG: 0.895 D_real: 0.301 D_fake: 0.113 
(epoch: 135, iters: 960, time: 0.089) G_GAN: 1.378 G_GAN_Feat: 2.090 G_VGG: 0.802 D_real: 0.131 D_fake: 0.186 
(epoch: 135, iters: 1360, time: 0.092) G_GAN: 1.690 G_GAN_Feat: 2.974 G_VGG: 1.032 D_real: 0.774 D_fake: 0.061 
saving the latest model (epoch 135, total_steps 398000)
(epoch: 135, iters: 1760, time: 0.093) G_GAN: 1.189 G_GAN_Feat: 2.283 G_VGG: 0.848 D_real: 0.052 D_fake: 0.569 
(epoch: 135, iters: 2160, time: 0.086) G_GAN: 1.268 G_GAN_Feat: 2.308 G_VGG: 0.867 D_real: 0.620 D_fake: 0.210 
(epoch: 135, iters: 2560, time: 0.090) G_GAN: 0.950 G_GAN_Feat: 1.972 G_VGG: 0.886 D_real: 0.252 D_fake: 0.324 
(epoch: 135, iters: 2960, time: 0.089) G_GAN: 1.517 G_GAN_Feat: 2.429 G_VGG: 0.859 D_real: 0.314 D_fake: 0.220 
End of epoch 135 / 200 	 Time Taken: 286 sec
(epoch: 136, iters: 400, time: 0.088) G_GAN: 0.938 G_GAN_Feat: 1.865 G_VGG: 0.829 D_real: 0.069 D_fake: 0.454 
saving the latest model (epoch 136, total_steps 400000)
(epoch: 136, iters: 800, time: 0.092) G_GAN: 1.400 G_GAN_Feat: 2.198 G_VGG: 0.887 D_real: 0.326 D_fake: 0.181 
(epoch: 136, iters: 1200, time: 0.089) G_GAN: 1.478 G_GAN_Feat: 2.539 G_VGG: 0.947 D_real: 0.172 D_fake: 0.169 
(epoch: 136, iters: 1600, time: 0.085) G_GAN: 1.489 G_GAN_Feat: 2.640 G_VGG: 0.966 D_real: 0.212 D_fake: 0.110 
(epoch: 136, iters: 2000, time: 0.092) G_GAN: 1.206 G_GAN_Feat: 2.492 G_VGG: 0.856 D_real: 0.035 D_fake: 0.291 
(epoch: 136, iters: 2400, time: 0.096) G_GAN: 1.406 G_GAN_Feat: 2.589 G_VGG: 0.854 D_real: 0.201 D_fake: 0.174 
saving the latest model (epoch 136, total_steps 402000)
(epoch: 136, iters: 2800, time: 0.089) G_GAN: 2.019 G_GAN_Feat: 2.725 G_VGG: 0.795 D_real: 0.319 D_fake: 0.027 
End of epoch 136 / 200 	 Time Taken: 288 sec
(epoch: 137, iters: 240, time: 0.088) G_GAN: 1.042 G_GAN_Feat: 2.043 G_VGG: 0.837 D_real: 0.101 D_fake: 0.435 
(epoch: 137, iters: 640, time: 0.096) G_GAN: 1.146 G_GAN_Feat: 2.447 G_VGG: 0.822 D_real: 0.376 D_fake: 0.411 
(epoch: 137, iters: 1040, time: 0.093) G_GAN: 1.114 G_GAN_Feat: 1.741 G_VGG: 0.753 D_real: 0.778 D_fake: 0.509 
(epoch: 137, iters: 1440, time: 0.084) G_GAN: 0.998 G_GAN_Feat: 1.961 G_VGG: 0.824 D_real: 0.146 D_fake: 0.309 
saving the latest model (epoch 137, total_steps 404000)
(epoch: 137, iters: 1840, time: 0.096) G_GAN: 1.356 G_GAN_Feat: 2.354 G_VGG: 0.892 D_real: 0.373 D_fake: 0.121 
(epoch: 137, iters: 2240, time: 0.091) G_GAN: 1.752 G_GAN_Feat: 2.925 G_VGG: 0.873 D_real: 0.463 D_fake: 0.180 
(epoch: 137, iters: 2640, time: 0.096) G_GAN: 1.715 G_GAN_Feat: 2.479 G_VGG: 0.843 D_real: 0.140 D_fake: 0.078 
End of epoch 137 / 200 	 Time Taken: 284 sec
(epoch: 138, iters: 80, time: 0.087) G_GAN: 1.249 G_GAN_Feat: 2.243 G_VGG: 0.826 D_real: 0.273 D_fake: 0.227 
(epoch: 138, iters: 480, time: 0.088) G_GAN: 1.035 G_GAN_Feat: 1.556 G_VGG: 0.769 D_real: 0.314 D_fake: 0.256 
saving the latest model (epoch 138, total_steps 406000)
(epoch: 138, iters: 880, time: 0.090) G_GAN: 1.227 G_GAN_Feat: 2.240 G_VGG: 0.905 D_real: 0.187 D_fake: 0.167 
(epoch: 138, iters: 1280, time: 0.089) G_GAN: 1.050 G_GAN_Feat: 2.216 G_VGG: 0.899 D_real: 0.113 D_fake: 0.354 
(epoch: 138, iters: 1680, time: 0.086) G_GAN: 1.203 G_GAN_Feat: 2.221 G_VGG: 0.878 D_real: 0.107 D_fake: 0.251 
(epoch: 138, iters: 2080, time: 0.084) G_GAN: 1.431 G_GAN_Feat: 2.501 G_VGG: 0.837 D_real: 0.266 D_fake: 0.184 
(epoch: 138, iters: 2480, time: 0.083) G_GAN: 1.340 G_GAN_Feat: 2.454 G_VGG: 0.870 D_real: 0.328 D_fake: 0.309 
saving the latest model (epoch 138, total_steps 408000)
(epoch: 138, iters: 2880, time: 0.087) G_GAN: 1.611 G_GAN_Feat: 2.614 G_VGG: 0.843 D_real: 0.083 D_fake: 0.066 
End of epoch 138 / 200 	 Time Taken: 288 sec
(epoch: 139, iters: 320, time: 0.096) G_GAN: 2.001 G_GAN_Feat: 3.170 G_VGG: 0.854 D_real: 0.018 D_fake: 0.031 
(epoch: 139, iters: 720, time: 0.093) G_GAN: 1.154 G_GAN_Feat: 2.221 G_VGG: 0.914 D_real: 0.173 D_fake: 0.310 
(epoch: 139, iters: 1120, time: 0.087) G_GAN: 1.576 G_GAN_Feat: 2.263 G_VGG: 0.752 D_real: 0.503 D_fake: 0.330 
(epoch: 139, iters: 1520, time: 0.096) G_GAN: 1.240 G_GAN_Feat: 2.225 G_VGG: 0.865 D_real: 0.174 D_fake: 0.249 
saving the latest model (epoch 139, total_steps 410000)
(epoch: 139, iters: 1920, time: 0.095) G_GAN: 1.821 G_GAN_Feat: 2.321 G_VGG: 0.811 D_real: 0.246 D_fake: 0.089 
(epoch: 139, iters: 2320, time: 0.094) G_GAN: 1.919 G_GAN_Feat: 3.238 G_VGG: 0.865 D_real: 0.369 D_fake: 0.082 
(epoch: 139, iters: 2720, time: 0.095) G_GAN: 2.112 G_GAN_Feat: 2.632 G_VGG: 0.847 D_real: 0.401 D_fake: 0.080 
End of epoch 139 / 200 	 Time Taken: 282 sec
(epoch: 140, iters: 160, time: 0.101) G_GAN: 1.796 G_GAN_Feat: 2.923 G_VGG: 0.825 D_real: 0.055 D_fake: 0.058 
(epoch: 140, iters: 560, time: 0.088) G_GAN: 1.253 G_GAN_Feat: 2.185 G_VGG: 0.866 D_real: 0.109 D_fake: 0.368 
saving the latest model (epoch 140, total_steps 412000)
(epoch: 140, iters: 960, time: 0.086) G_GAN: 1.593 G_GAN_Feat: 2.680 G_VGG: 0.815 D_real: 0.040 D_fake: 0.108 
(epoch: 140, iters: 1360, time: 0.088) G_GAN: 1.525 G_GAN_Feat: 2.011 G_VGG: 0.827 D_real: 0.327 D_fake: 0.150 
(epoch: 140, iters: 1760, time: 0.089) G_GAN: 1.929 G_GAN_Feat: 3.112 G_VGG: 0.814 D_real: 0.027 D_fake: 0.026 
(epoch: 140, iters: 2160, time: 0.091) G_GAN: 1.233 G_GAN_Feat: 2.452 G_VGG: 0.894 D_real: 0.278 D_fake: 0.377 
(epoch: 140, iters: 2560, time: 0.092) G_GAN: 1.320 G_GAN_Feat: 2.411 G_VGG: 0.857 D_real: 0.088 D_fake: 0.462 
saving the latest model (epoch 140, total_steps 414000)
(epoch: 140, iters: 2960, time: 0.092) G_GAN: 1.497 G_GAN_Feat: 2.451 G_VGG: 0.855 D_real: 0.413 D_fake: 0.224 
End of epoch 140 / 200 	 Time Taken: 287 sec
saving the model at the end of epoch 140, iters 414400
(epoch: 141, iters: 400, time: 0.083) G_GAN: 1.274 G_GAN_Feat: 2.547 G_VGG: 0.880 D_real: 0.076 D_fake: 0.160 
(epoch: 141, iters: 800, time: 0.087) G_GAN: 2.608 G_GAN_Feat: 2.827 G_VGG: 0.811 D_real: 0.470 D_fake: 0.071 
(epoch: 141, iters: 1200, time: 0.087) G_GAN: 0.712 G_GAN_Feat: 1.866 G_VGG: 0.814 D_real: 0.124 D_fake: 0.579 
(epoch: 141, iters: 1600, time: 0.091) G_GAN: 1.212 G_GAN_Feat: 2.496 G_VGG: 0.872 D_real: 0.300 D_fake: 0.251 
saving the latest model (epoch 141, total_steps 416000)
(epoch: 141, iters: 2000, time: 0.092) G_GAN: 1.362 G_GAN_Feat: 2.371 G_VGG: 0.807 D_real: 0.188 D_fake: 0.406 
(epoch: 141, iters: 2400, time: 0.082) G_GAN: 1.196 G_GAN_Feat: 2.121 G_VGG: 0.900 D_real: 0.136 D_fake: 0.321 
(epoch: 141, iters: 2800, time: 0.098) G_GAN: 1.425 G_GAN_Feat: 2.285 G_VGG: 0.886 D_real: 0.403 D_fake: 0.229 
End of epoch 141 / 200 	 Time Taken: 286 sec
(epoch: 142, iters: 240, time: 0.085) G_GAN: 1.541 G_GAN_Feat: 2.139 G_VGG: 0.907 D_real: 0.293 D_fake: 0.201 
(epoch: 142, iters: 640, time: 0.086) G_GAN: 1.897 G_GAN_Feat: 2.943 G_VGG: 0.871 D_real: 0.150 D_fake: 0.094 
saving the latest model (epoch 142, total_steps 418000)
(epoch: 142, iters: 1040, time: 0.101) G_GAN: 1.080 G_GAN_Feat: 2.879 G_VGG: 0.918 D_real: 0.058 D_fake: 0.214 
(epoch: 142, iters: 1440, time: 0.096) G_GAN: 1.316 G_GAN_Feat: 2.153 G_VGG: 0.889 D_real: 0.099 D_fake: 0.334 
(epoch: 142, iters: 1840, time: 0.095) G_GAN: 1.691 G_GAN_Feat: 2.743 G_VGG: 0.841 D_real: 0.395 D_fake: 0.056 
(epoch: 142, iters: 2240, time: 0.096) G_GAN: 2.031 G_GAN_Feat: 3.010 G_VGG: 0.889 D_real: 0.161 D_fake: 0.020 
(epoch: 142, iters: 2640, time: 0.085) G_GAN: 1.389 G_GAN_Feat: 1.903 G_VGG: 0.816 D_real: 0.461 D_fake: 0.153 
saving the latest model (epoch 142, total_steps 420000)
End of epoch 142 / 200 	 Time Taken: 285 sec
(epoch: 143, iters: 80, time: 0.091) G_GAN: 1.960 G_GAN_Feat: 3.220 G_VGG: 0.857 D_real: 0.036 D_fake: 0.021 
(epoch: 143, iters: 480, time: 0.086) G_GAN: 0.943 G_GAN_Feat: 2.020 G_VGG: 0.786 D_real: 0.083 D_fake: 0.383 
(epoch: 143, iters: 880, time: 0.095) G_GAN: 1.503 G_GAN_Feat: 2.255 G_VGG: 0.866 D_real: 0.313 D_fake: 0.173 
(epoch: 143, iters: 1280, time: 0.094) G_GAN: 1.891 G_GAN_Feat: 2.811 G_VGG: 0.794 D_real: 0.230 D_fake: 0.085 
(epoch: 143, iters: 1680, time: 0.096) G_GAN: 1.458 G_GAN_Feat: 2.520 G_VGG: 0.798 D_real: 0.092 D_fake: 0.106 
saving the latest model (epoch 143, total_steps 422000)
(epoch: 143, iters: 2080, time: 0.089) G_GAN: 0.787 G_GAN_Feat: 1.925 G_VGG: 0.845 D_real: 0.143 D_fake: 0.398 
(epoch: 143, iters: 2480, time: 0.098) G_GAN: 2.086 G_GAN_Feat: 2.678 G_VGG: 0.869 D_real: 0.157 D_fake: 0.048 
(epoch: 143, iters: 2880, time: 0.091) G_GAN: 1.243 G_GAN_Feat: 3.141 G_VGG: 0.773 D_real: 0.326 D_fake: 0.215 
End of epoch 143 / 200 	 Time Taken: 287 sec
(epoch: 144, iters: 320, time: 0.097) G_GAN: 1.456 G_GAN_Feat: 2.227 G_VGG: 0.747 D_real: 0.150 D_fake: 0.116 
(epoch: 144, iters: 720, time: 0.087) G_GAN: 1.740 G_GAN_Feat: 2.253 G_VGG: 0.880 D_real: 0.059 D_fake: 0.108 
saving the latest model (epoch 144, total_steps 424000)
(epoch: 144, iters: 1120, time: 0.087) G_GAN: 1.022 G_GAN_Feat: 1.871 G_VGG: 0.947 D_real: 0.689 D_fake: 0.300 
(epoch: 144, iters: 1520, time: 0.090) G_GAN: 1.507 G_GAN_Feat: 2.094 G_VGG: 0.780 D_real: 0.249 D_fake: 0.172 
(epoch: 144, iters: 1920, time: 0.095) G_GAN: 1.035 G_GAN_Feat: 2.059 G_VGG: 0.857 D_real: 0.132 D_fake: 0.545 
(epoch: 144, iters: 2320, time: 0.094) G_GAN: 1.035 G_GAN_Feat: 2.046 G_VGG: 0.848 D_real: 0.043 D_fake: 0.469 
(epoch: 144, iters: 2720, time: 0.090) G_GAN: 1.892 G_GAN_Feat: 2.943 G_VGG: 0.848 D_real: 0.144 D_fake: 0.023 
saving the latest model (epoch 144, total_steps 426000)
End of epoch 144 / 200 	 Time Taken: 289 sec
(epoch: 145, iters: 160, time: 0.091) G_GAN: 1.201 G_GAN_Feat: 2.465 G_VGG: 0.827 D_real: 0.110 D_fake: 0.167 
(epoch: 145, iters: 560, time: 0.099) G_GAN: 0.923 G_GAN_Feat: 1.766 G_VGG: 0.770 D_real: 0.353 D_fake: 0.302 
(epoch: 145, iters: 960, time: 0.096) G_GAN: 1.139 G_GAN_Feat: 2.244 G_VGG: 0.774 D_real: 0.082 D_fake: 0.234 
(epoch: 145, iters: 1360, time: 0.088) G_GAN: 1.719 G_GAN_Feat: 2.375 G_VGG: 0.815 D_real: 0.087 D_fake: 0.120 
(epoch: 145, iters: 1760, time: 0.096) G_GAN: 1.329 G_GAN_Feat: 2.554 G_VGG: 0.869 D_real: 0.057 D_fake: 0.108 
saving the latest model (epoch 145, total_steps 428000)
(epoch: 145, iters: 2160, time: 0.092) G_GAN: 1.587 G_GAN_Feat: 2.330 G_VGG: 0.809 D_real: 0.106 D_fake: 0.101 
(epoch: 145, iters: 2560, time: 0.093) G_GAN: 1.699 G_GAN_Feat: 2.417 G_VGG: 0.852 D_real: 0.268 D_fake: 0.123 
(epoch: 145, iters: 2960, time: 0.094) G_GAN: 1.776 G_GAN_Feat: 2.634 G_VGG: 0.790 D_real: 0.365 D_fake: 0.040 
End of epoch 145 / 200 	 Time Taken: 286 sec
(epoch: 146, iters: 400, time: 0.094) G_GAN: 1.200 G_GAN_Feat: 1.861 G_VGG: 0.808 D_real: 0.251 D_fake: 0.277 
(epoch: 146, iters: 800, time: 0.095) G_GAN: 0.982 G_GAN_Feat: 1.774 G_VGG: 0.794 D_real: 0.252 D_fake: 0.280 
saving the latest model (epoch 146, total_steps 430000)
(epoch: 146, iters: 1200, time: 0.080) G_GAN: 1.384 G_GAN_Feat: 2.163 G_VGG: 0.847 D_real: 0.211 D_fake: 0.226 
(epoch: 146, iters: 1600, time: 0.090) G_GAN: 1.090 G_GAN_Feat: 1.854 G_VGG: 0.825 D_real: 0.212 D_fake: 0.327 
(epoch: 146, iters: 2000, time: 0.092) G_GAN: 1.214 G_GAN_Feat: 2.088 G_VGG: 0.835 D_real: 0.114 D_fake: 0.382 
(epoch: 146, iters: 2400, time: 0.096) G_GAN: 1.043 G_GAN_Feat: 2.061 G_VGG: 0.853 D_real: 0.080 D_fake: 0.374 
(epoch: 146, iters: 2800, time: 0.092) G_GAN: 1.293 G_GAN_Feat: 2.454 G_VGG: 0.830 D_real: 0.041 D_fake: 0.351 
saving the latest model (epoch 146, total_steps 432000)
End of epoch 146 / 200 	 Time Taken: 286 sec
(epoch: 147, iters: 240, time: 0.096) G_GAN: 1.529 G_GAN_Feat: 2.618 G_VGG: 0.808 D_real: 0.117 D_fake: 0.099 
(epoch: 147, iters: 640, time: 0.094) G_GAN: 1.780 G_GAN_Feat: 2.378 G_VGG: 0.873 D_real: 0.245 D_fake: 0.093 
(epoch: 147, iters: 1040, time: 0.091) G_GAN: 1.710 G_GAN_Feat: 2.350 G_VGG: 0.850 D_real: 0.154 D_fake: 0.067 
(epoch: 147, iters: 1440, time: 0.098) G_GAN: 1.176 G_GAN_Feat: 1.798 G_VGG: 0.813 D_real: 0.290 D_fake: 0.261 
(epoch: 147, iters: 1840, time: 0.085) G_GAN: 1.312 G_GAN_Feat: 2.027 G_VGG: 0.774 D_real: 0.159 D_fake: 0.194 
saving the latest model (epoch 147, total_steps 434000)
(epoch: 147, iters: 2240, time: 0.088) G_GAN: 1.382 G_GAN_Feat: 2.037 G_VGG: 0.774 D_real: 0.284 D_fake: 0.237 
(epoch: 147, iters: 2640, time: 0.092) G_GAN: 1.759 G_GAN_Feat: 2.723 G_VGG: 0.862 D_real: 0.070 D_fake: 0.065 
End of epoch 147 / 200 	 Time Taken: 284 sec
(epoch: 148, iters: 80, time: 0.095) G_GAN: 1.213 G_GAN_Feat: 2.444 G_VGG: 0.807 D_real: 0.176 D_fake: 0.394 
(epoch: 148, iters: 480, time: 0.086) G_GAN: 1.016 G_GAN_Feat: 1.996 G_VGG: 0.811 D_real: 0.250 D_fake: 0.557 
(epoch: 148, iters: 880, time: 0.084) G_GAN: 1.665 G_GAN_Feat: 2.567 G_VGG: 0.837 D_real: 0.254 D_fake: 0.052 
saving the latest model (epoch 148, total_steps 436000)
(epoch: 148, iters: 1280, time: 0.089) G_GAN: 1.217 G_GAN_Feat: 2.201 G_VGG: 0.955 D_real: 0.241 D_fake: 0.226 
(epoch: 148, iters: 1680, time: 0.085) G_GAN: 1.618 G_GAN_Feat: 2.404 G_VGG: 0.755 D_real: 0.158 D_fake: 0.059 
(epoch: 148, iters: 2080, time: 0.100) G_GAN: 1.187 G_GAN_Feat: 2.434 G_VGG: 0.819 D_real: 0.105 D_fake: 0.367 
(epoch: 148, iters: 2480, time: 0.088) G_GAN: 1.696 G_GAN_Feat: 2.201 G_VGG: 0.893 D_real: 0.271 D_fake: 0.078 
(epoch: 148, iters: 2880, time: 0.088) G_GAN: 1.080 G_GAN_Feat: 2.451 G_VGG: 0.832 D_real: 0.197 D_fake: 0.355 
saving the latest model (epoch 148, total_steps 438000)
End of epoch 148 / 200 	 Time Taken: 288 sec
(epoch: 149, iters: 320, time: 0.091) G_GAN: 1.215 G_GAN_Feat: 2.129 G_VGG: 0.759 D_real: 0.218 D_fake: 0.163 
(epoch: 149, iters: 720, time: 0.090) G_GAN: 1.214 G_GAN_Feat: 2.676 G_VGG: 0.831 D_real: 0.021 D_fake: 0.497 
(epoch: 149, iters: 1120, time: 0.095) G_GAN: 1.746 G_GAN_Feat: 2.562 G_VGG: 0.752 D_real: 0.071 D_fake: 0.089 
(epoch: 149, iters: 1520, time: 0.093) G_GAN: 1.352 G_GAN_Feat: 2.303 G_VGG: 0.863 D_real: 0.266 D_fake: 0.378 
(epoch: 149, iters: 1920, time: 0.091) G_GAN: 2.209 G_GAN_Feat: 2.618 G_VGG: 0.854 D_real: 0.150 D_fake: 0.032 
saving the latest model (epoch 149, total_steps 440000)
(epoch: 149, iters: 2320, time: 0.093) G_GAN: 1.734 G_GAN_Feat: 2.559 G_VGG: 0.852 D_real: 0.465 D_fake: 0.073 
(epoch: 149, iters: 2720, time: 0.090) G_GAN: 1.558 G_GAN_Feat: 2.890 G_VGG: 0.800 D_real: 0.029 D_fake: 0.069 
End of epoch 149 / 200 	 Time Taken: 285 sec
(epoch: 150, iters: 160, time: 0.092) G_GAN: 1.625 G_GAN_Feat: 2.503 G_VGG: 0.900 D_real: 0.201 D_fake: 0.122 
(epoch: 150, iters: 560, time: 0.088) G_GAN: 1.436 G_GAN_Feat: 2.137 G_VGG: 0.809 D_real: 0.327 D_fake: 0.204 
(epoch: 150, iters: 960, time: 0.093) G_GAN: 1.957 G_GAN_Feat: 2.497 G_VGG: 0.842 D_real: 0.416 D_fake: 0.052 
saving the latest model (epoch 150, total_steps 442000)
(epoch: 150, iters: 1360, time: 0.096) G_GAN: 1.411 G_GAN_Feat: 1.983 G_VGG: 0.891 D_real: 0.463 D_fake: 0.139 
(epoch: 150, iters: 1760, time: 0.095) G_GAN: 1.387 G_GAN_Feat: 2.156 G_VGG: 0.776 D_real: 0.204 D_fake: 0.130 
(epoch: 150, iters: 2160, time: 0.094) G_GAN: 1.241 G_GAN_Feat: 1.897 G_VGG: 0.775 D_real: 0.392 D_fake: 0.190 
(epoch: 150, iters: 2560, time: 0.100) G_GAN: 1.447 G_GAN_Feat: 2.254 G_VGG: 0.857 D_real: 0.118 D_fake: 0.451 
(epoch: 150, iters: 2960, time: 0.097) G_GAN: 1.967 G_GAN_Feat: 2.269 G_VGG: 0.873 D_real: 0.265 D_fake: 0.036 
saving the latest model (epoch 150, total_steps 444000)
End of epoch 150 / 200 	 Time Taken: 288 sec
saving the model at the end of epoch 150, iters 444000
(epoch: 151, iters: 400, time: 0.094) G_GAN: 1.259 G_GAN_Feat: 2.030 G_VGG: 0.790 D_real: 0.028 D_fake: 0.270 
(epoch: 151, iters: 800, time: 0.093) G_GAN: 1.173 G_GAN_Feat: 2.060 G_VGG: 0.823 D_real: 0.123 D_fake: 0.247 
(epoch: 151, iters: 1200, time: 0.086) G_GAN: 1.096 G_GAN_Feat: 2.412 G_VGG: 0.795 D_real: 0.169 D_fake: 0.351 
(epoch: 151, iters: 1600, time: 0.101) G_GAN: 0.922 G_GAN_Feat: 1.889 G_VGG: 0.771 D_real: 0.094 D_fake: 0.326 
(epoch: 151, iters: 2000, time: 0.086) G_GAN: 1.476 G_GAN_Feat: 2.269 G_VGG: 0.762 D_real: 0.050 D_fake: 0.108 
saving the latest model (epoch 151, total_steps 446000)
(epoch: 151, iters: 2400, time: 0.086) G_GAN: 0.924 G_GAN_Feat: 1.898 G_VGG: 0.835 D_real: 0.120 D_fake: 0.353 
(epoch: 151, iters: 2800, time: 0.098) G_GAN: 1.352 G_GAN_Feat: 2.016 G_VGG: 0.776 D_real: 0.226 D_fake: 0.232 
End of epoch 151 / 200 	 Time Taken: 284 sec
(epoch: 152, iters: 240, time: 0.098) G_GAN: 1.265 G_GAN_Feat: 2.248 G_VGG: 0.798 D_real: 0.047 D_fake: 0.283 
(epoch: 152, iters: 640, time: 0.095) G_GAN: 1.062 G_GAN_Feat: 2.057 G_VGG: 0.821 D_real: 0.081 D_fake: 0.468 
(epoch: 152, iters: 1040, time: 0.094) G_GAN: 1.134 G_GAN_Feat: 1.826 G_VGG: 0.777 D_real: 0.159 D_fake: 0.394 
saving the latest model (epoch 152, total_steps 448000)
(epoch: 152, iters: 1440, time: 0.087) G_GAN: 1.795 G_GAN_Feat: 2.391 G_VGG: 0.829 D_real: 0.124 D_fake: 0.038 
(epoch: 152, iters: 1840, time: 0.091) G_GAN: 1.257 G_GAN_Feat: 2.061 G_VGG: 0.758 D_real: 0.208 D_fake: 0.244 
(epoch: 152, iters: 2240, time: 0.091) G_GAN: 1.298 G_GAN_Feat: 1.928 G_VGG: 0.805 D_real: 0.131 D_fake: 0.250 
(epoch: 152, iters: 2640, time: 0.094) G_GAN: 1.435 G_GAN_Feat: 1.937 G_VGG: 0.811 D_real: 0.284 D_fake: 0.188 
End of epoch 152 / 200 	 Time Taken: 286 sec
(epoch: 153, iters: 80, time: 0.092) G_GAN: 1.633 G_GAN_Feat: 2.642 G_VGG: 0.870 D_real: 0.133 D_fake: 0.109 
saving the latest model (epoch 153, total_steps 450000)
(epoch: 153, iters: 480, time: 0.091) G_GAN: 1.051 G_GAN_Feat: 1.938 G_VGG: 0.806 D_real: 0.223 D_fake: 0.418 
(epoch: 153, iters: 880, time: 0.084) G_GAN: 2.030 G_GAN_Feat: 3.036 G_VGG: 0.853 D_real: 0.055 D_fake: 0.030 
(epoch: 153, iters: 1280, time: 0.092) G_GAN: 1.403 G_GAN_Feat: 2.330 G_VGG: 0.848 D_real: 0.201 D_fake: 0.192 
(epoch: 153, iters: 1680, time: 0.087) G_GAN: 1.127 G_GAN_Feat: 1.887 G_VGG: 0.817 D_real: 0.078 D_fake: 0.517 
(epoch: 153, iters: 2080, time: 0.086) G_GAN: 1.764 G_GAN_Feat: 2.733 G_VGG: 0.781 D_real: 0.103 D_fake: 0.042 
saving the latest model (epoch 153, total_steps 452000)
(epoch: 153, iters: 2480, time: 0.094) G_GAN: 1.069 G_GAN_Feat: 1.957 G_VGG: 0.809 D_real: 0.040 D_fake: 0.951 
(epoch: 153, iters: 2880, time: 0.087) G_GAN: 1.960 G_GAN_Feat: 2.727 G_VGG: 0.905 D_real: 0.384 D_fake: 0.026 
End of epoch 153 / 200 	 Time Taken: 288 sec
(epoch: 154, iters: 320, time: 0.094) G_GAN: 1.285 G_GAN_Feat: 2.172 G_VGG: 0.816 D_real: 0.114 D_fake: 0.448 
(epoch: 154, iters: 720, time: 0.097) G_GAN: 1.416 G_GAN_Feat: 2.101 G_VGG: 0.821 D_real: 0.294 D_fake: 0.181 
(epoch: 154, iters: 1120, time: 0.083) G_GAN: 1.906 G_GAN_Feat: 2.674 G_VGG: 0.799 D_real: 0.320 D_fake: 0.023 
saving the latest model (epoch 154, total_steps 454000)
(epoch: 154, iters: 1520, time: 0.095) G_GAN: 2.261 G_GAN_Feat: 2.778 G_VGG: 0.730 D_real: 0.141 D_fake: 0.027 
(epoch: 154, iters: 1920, time: 0.098) G_GAN: 1.736 G_GAN_Feat: 2.452 G_VGG: 0.776 D_real: 0.082 D_fake: 0.063 
(epoch: 154, iters: 2320, time: 0.092) G_GAN: 1.455 G_GAN_Feat: 2.227 G_VGG: 0.845 D_real: 0.163 D_fake: 0.188 
(epoch: 154, iters: 2720, time: 0.088) G_GAN: 1.563 G_GAN_Feat: 2.485 G_VGG: 0.951 D_real: 0.223 D_fake: 0.125 
End of epoch 154 / 200 	 Time Taken: 285 sec
(epoch: 155, iters: 160, time: 0.077) G_GAN: 2.161 G_GAN_Feat: 2.431 G_VGG: 0.807 D_real: 0.455 D_fake: 0.022 
saving the latest model (epoch 155, total_steps 456000)
(epoch: 155, iters: 560, time: 0.085) G_GAN: 1.363 G_GAN_Feat: 2.291 G_VGG: 0.760 D_real: 0.225 D_fake: 0.181 
(epoch: 155, iters: 960, time: 0.091) G_GAN: 0.989 G_GAN_Feat: 2.262 G_VGG: 0.773 D_real: 0.210 D_fake: 0.284 
(epoch: 155, iters: 1360, time: 0.099) G_GAN: 1.385 G_GAN_Feat: 2.114 G_VGG: 0.844 D_real: 0.135 D_fake: 0.349 
(epoch: 155, iters: 1760, time: 0.086) G_GAN: 1.210 G_GAN_Feat: 1.877 G_VGG: 0.884 D_real: 0.146 D_fake: 0.275 
(epoch: 155, iters: 2160, time: 0.096) G_GAN: 1.657 G_GAN_Feat: 2.400 G_VGG: 0.808 D_real: 0.060 D_fake: 0.101 
saving the latest model (epoch 155, total_steps 458000)
(epoch: 155, iters: 2560, time: 0.085) G_GAN: 1.376 G_GAN_Feat: 2.399 G_VGG: 0.854 D_real: 0.029 D_fake: 0.237 
(epoch: 155, iters: 2960, time: 0.094) G_GAN: 1.208 G_GAN_Feat: 2.131 G_VGG: 0.909 D_real: 0.072 D_fake: 0.286 
End of epoch 155 / 200 	 Time Taken: 286 sec
(epoch: 156, iters: 400, time: 0.090) G_GAN: 1.530 G_GAN_Feat: 2.199 G_VGG: 0.789 D_real: 0.251 D_fake: 0.218 
(epoch: 156, iters: 800, time: 0.090) G_GAN: 1.275 G_GAN_Feat: 2.065 G_VGG: 0.852 D_real: 0.082 D_fake: 0.228 
(epoch: 156, iters: 1200, time: 0.098) G_GAN: 1.939 G_GAN_Feat: 2.510 G_VGG: 0.777 D_real: 0.198 D_fake: 0.054 
saving the latest model (epoch 156, total_steps 460000)
(epoch: 156, iters: 1600, time: 0.095) G_GAN: 1.318 G_GAN_Feat: 2.332 G_VGG: 0.769 D_real: 0.073 D_fake: 0.194 
(epoch: 156, iters: 2000, time: 0.089) G_GAN: 1.420 G_GAN_Feat: 2.100 G_VGG: 0.813 D_real: 0.199 D_fake: 0.147 
(epoch: 156, iters: 2400, time: 0.086) G_GAN: 1.799 G_GAN_Feat: 2.456 G_VGG: 0.764 D_real: 0.045 D_fake: 0.059 
(epoch: 156, iters: 2800, time: 0.092) G_GAN: 1.575 G_GAN_Feat: 2.248 G_VGG: 0.832 D_real: 0.071 D_fake: 0.175 
End of epoch 156 / 200 	 Time Taken: 286 sec
(epoch: 157, iters: 240, time: 0.092) G_GAN: 1.775 G_GAN_Feat: 2.581 G_VGG: 0.761 D_real: 0.134 D_fake: 0.108 
saving the latest model (epoch 157, total_steps 462000)
(epoch: 157, iters: 640, time: 0.098) G_GAN: 1.192 G_GAN_Feat: 1.985 G_VGG: 0.779 D_real: 0.381 D_fake: 0.305 
(epoch: 157, iters: 1040, time: 0.092) G_GAN: 1.292 G_GAN_Feat: 2.216 G_VGG: 0.841 D_real: 0.124 D_fake: 0.208 
(epoch: 157, iters: 1440, time: 0.086) G_GAN: 1.433 G_GAN_Feat: 2.172 G_VGG: 0.884 D_real: 0.300 D_fake: 0.147 
(epoch: 157, iters: 1840, time: 0.094) G_GAN: 1.293 G_GAN_Feat: 1.914 G_VGG: 0.852 D_real: 0.207 D_fake: 0.201 
(epoch: 157, iters: 2240, time: 0.091) G_GAN: 1.460 G_GAN_Feat: 2.116 G_VGG: 0.851 D_real: 0.052 D_fake: 0.214 
saving the latest model (epoch 157, total_steps 464000)
(epoch: 157, iters: 2640, time: 0.094) G_GAN: 1.027 G_GAN_Feat: 2.117 G_VGG: 0.865 D_real: 0.080 D_fake: 0.405 
End of epoch 157 / 200 	 Time Taken: 285 sec
(epoch: 158, iters: 80, time: 0.088) G_GAN: 2.375 G_GAN_Feat: 2.500 G_VGG: 0.763 D_real: 0.316 D_fake: 0.041 
(epoch: 158, iters: 480, time: 0.087) G_GAN: 1.498 G_GAN_Feat: 2.069 G_VGG: 0.738 D_real: 0.063 D_fake: 0.140 
(epoch: 158, iters: 880, time: 0.099) G_GAN: 1.080 G_GAN_Feat: 2.006 G_VGG: 0.863 D_real: 0.119 D_fake: 0.345 
(epoch: 158, iters: 1280, time: 0.088) G_GAN: 1.709 G_GAN_Feat: 2.387 G_VGG: 0.852 D_real: 0.039 D_fake: 0.077 
saving the latest model (epoch 158, total_steps 466000)
(epoch: 158, iters: 1680, time: 0.093) G_GAN: 1.147 G_GAN_Feat: 1.727 G_VGG: 0.713 D_real: 0.399 D_fake: 0.283 
(epoch: 158, iters: 2080, time: 0.092) G_GAN: 1.656 G_GAN_Feat: 2.562 G_VGG: 0.777 D_real: 0.367 D_fake: 0.083 
(epoch: 158, iters: 2480, time: 0.093) G_GAN: 1.306 G_GAN_Feat: 1.986 G_VGG: 0.903 D_real: 0.223 D_fake: 0.199 
(epoch: 158, iters: 2880, time: 0.087) G_GAN: 1.576 G_GAN_Feat: 2.356 G_VGG: 0.815 D_real: 0.203 D_fake: 0.068 
End of epoch 158 / 200 	 Time Taken: 283 sec
(epoch: 159, iters: 320, time: 0.091) G_GAN: 1.474 G_GAN_Feat: 2.196 G_VGG: 0.761 D_real: 0.339 D_fake: 0.104 
saving the latest model (epoch 159, total_steps 468000)
(epoch: 159, iters: 720, time: 0.096) G_GAN: 1.060 G_GAN_Feat: 1.958 G_VGG: 0.837 D_real: 0.238 D_fake: 0.435 
(epoch: 159, iters: 1120, time: 0.089) G_GAN: 1.383 G_GAN_Feat: 1.966 G_VGG: 0.748 D_real: 0.413 D_fake: 0.146 
(epoch: 159, iters: 1520, time: 0.089) G_GAN: 0.821 G_GAN_Feat: 1.810 G_VGG: 0.857 D_real: 0.091 D_fake: 0.475 
(epoch: 159, iters: 1920, time: 0.092) G_GAN: 1.247 G_GAN_Feat: 2.125 G_VGG: 0.835 D_real: 0.168 D_fake: 0.321 
(epoch: 159, iters: 2320, time: 0.084) G_GAN: 1.105 G_GAN_Feat: 2.106 G_VGG: 0.804 D_real: 0.346 D_fake: 0.453 
saving the latest model (epoch 159, total_steps 470000)
(epoch: 159, iters: 2720, time: 0.092) G_GAN: 1.394 G_GAN_Feat: 2.145 G_VGG: 0.781 D_real: 0.116 D_fake: 0.189 
End of epoch 159 / 200 	 Time Taken: 290 sec
(epoch: 160, iters: 160, time: 0.093) G_GAN: 1.216 G_GAN_Feat: 2.349 G_VGG: 0.927 D_real: 0.205 D_fake: 0.291 
(epoch: 160, iters: 560, time: 0.087) G_GAN: 1.651 G_GAN_Feat: 2.781 G_VGG: 0.758 D_real: 0.024 D_fake: 0.046 
(epoch: 160, iters: 960, time: 0.097) G_GAN: 1.069 G_GAN_Feat: 1.897 G_VGG: 0.828 D_real: 0.402 D_fake: 0.277 
(epoch: 160, iters: 1360, time: 0.097) G_GAN: 1.199 G_GAN_Feat: 1.953 G_VGG: 0.891 D_real: 0.162 D_fake: 0.225 
saving the latest model (epoch 160, total_steps 472000)
(epoch: 160, iters: 1760, time: 0.092) G_GAN: 1.357 G_GAN_Feat: 2.163 G_VGG: 0.864 D_real: 0.096 D_fake: 0.317 
(epoch: 160, iters: 2160, time: 0.089) G_GAN: 1.073 G_GAN_Feat: 1.911 G_VGG: 0.803 D_real: 0.102 D_fake: 0.421 
(epoch: 160, iters: 2560, time: 0.093) G_GAN: 1.579 G_GAN_Feat: 2.094 G_VGG: 0.764 D_real: 0.212 D_fake: 0.130 
(epoch: 160, iters: 2960, time: 0.083) G_GAN: 1.714 G_GAN_Feat: 2.683 G_VGG: 0.813 D_real: 0.030 D_fake: 0.132 
End of epoch 160 / 200 	 Time Taken: 285 sec
saving the model at the end of epoch 160, iters 473600
(epoch: 161, iters: 400, time: 0.092) G_GAN: 1.135 G_GAN_Feat: 2.276 G_VGG: 0.784 D_real: 0.548 D_fake: 0.221 
saving the latest model (epoch 161, total_steps 474000)
(epoch: 161, iters: 800, time: 0.089) G_GAN: 1.222 G_GAN_Feat: 1.777 G_VGG: 0.761 D_real: 0.128 D_fake: 0.280 
(epoch: 161, iters: 1200, time: 0.092) G_GAN: 1.325 G_GAN_Feat: 1.919 G_VGG: 0.722 D_real: 0.167 D_fake: 0.250 
(epoch: 161, iters: 1600, time: 0.092) G_GAN: 1.348 G_GAN_Feat: 2.497 G_VGG: 0.895 D_real: 0.073 D_fake: 0.249 
(epoch: 161, iters: 2000, time: 0.091) G_GAN: 1.420 G_GAN_Feat: 2.437 G_VGG: 0.790 D_real: 0.059 D_fake: 0.107 
(epoch: 161, iters: 2400, time: 0.099) G_GAN: 1.497 G_GAN_Feat: 2.573 G_VGG: 0.706 D_real: 0.351 D_fake: 0.138 
saving the latest model (epoch 161, total_steps 476000)
(epoch: 161, iters: 2800, time: 0.091) G_GAN: 1.225 G_GAN_Feat: 2.127 G_VGG: 0.777 D_real: 0.164 D_fake: 0.190 
End of epoch 161 / 200 	 Time Taken: 289 sec
(epoch: 162, iters: 240, time: 0.091) G_GAN: 1.158 G_GAN_Feat: 2.075 G_VGG: 0.818 D_real: 0.078 D_fake: 0.345 
(epoch: 162, iters: 640, time: 0.091) G_GAN: 1.544 G_GAN_Feat: 2.421 G_VGG: 0.826 D_real: 0.024 D_fake: 0.198 
(epoch: 162, iters: 1040, time: 0.094) G_GAN: 1.051 G_GAN_Feat: 2.049 G_VGG: 0.876 D_real: 0.022 D_fake: 0.479 
(epoch: 162, iters: 1440, time: 0.090) G_GAN: 1.224 G_GAN_Feat: 1.775 G_VGG: 0.751 D_real: 0.211 D_fake: 0.292 
saving the latest model (epoch 162, total_steps 478000)
(epoch: 162, iters: 1840, time: 0.093) G_GAN: 1.349 G_GAN_Feat: 2.222 G_VGG: 0.868 D_real: 0.142 D_fake: 0.276 
(epoch: 162, iters: 2240, time: 0.088) G_GAN: 1.324 G_GAN_Feat: 2.043 G_VGG: 0.768 D_real: 0.153 D_fake: 0.254 
(epoch: 162, iters: 2640, time: 0.086) G_GAN: 0.975 G_GAN_Feat: 1.844 G_VGG: 0.752 D_real: 0.037 D_fake: 0.604 
End of epoch 162 / 200 	 Time Taken: 283 sec
(epoch: 163, iters: 80, time: 0.092) G_GAN: 1.023 G_GAN_Feat: 1.941 G_VGG: 0.830 D_real: 0.081 D_fake: 0.473 
(epoch: 163, iters: 480, time: 0.094) G_GAN: 1.493 G_GAN_Feat: 2.385 G_VGG: 0.832 D_real: 0.054 D_fake: 0.185 
saving the latest model (epoch 163, total_steps 480000)
(epoch: 163, iters: 880, time: 0.099) G_GAN: 1.454 G_GAN_Feat: 2.003 G_VGG: 0.827 D_real: 0.051 D_fake: 0.127 
(epoch: 163, iters: 1280, time: 0.092) G_GAN: 2.194 G_GAN_Feat: 2.647 G_VGG: 0.860 D_real: 0.287 D_fake: 0.027 
(epoch: 163, iters: 1680, time: 0.095) G_GAN: 1.105 G_GAN_Feat: 2.303 G_VGG: 0.859 D_real: 0.533 D_fake: 0.323 
(epoch: 163, iters: 2080, time: 0.090) G_GAN: 1.144 G_GAN_Feat: 2.074 G_VGG: 0.866 D_real: 0.164 D_fake: 0.233 
(epoch: 163, iters: 2480, time: 0.085) G_GAN: 1.396 G_GAN_Feat: 2.028 G_VGG: 0.766 D_real: 0.049 D_fake: 0.309 
saving the latest model (epoch 163, total_steps 482000)
(epoch: 163, iters: 2880, time: 0.092) G_GAN: 2.084 G_GAN_Feat: 2.842 G_VGG: 0.802 D_real: 0.148 D_fake: 0.035 
End of epoch 163 / 200 	 Time Taken: 288 sec
(epoch: 164, iters: 320, time: 0.094) G_GAN: 0.997 G_GAN_Feat: 2.109 G_VGG: 0.858 D_real: 0.155 D_fake: 0.396 
(epoch: 164, iters: 720, time: 0.086) G_GAN: 1.629 G_GAN_Feat: 1.947 G_VGG: 0.787 D_real: 0.212 D_fake: 0.111 
(epoch: 164, iters: 1120, time: 0.089) G_GAN: 1.248 G_GAN_Feat: 1.946 G_VGG: 0.817 D_real: 0.050 D_fake: 0.534 
(epoch: 164, iters: 1520, time: 0.103) G_GAN: 1.316 G_GAN_Feat: 2.047 G_VGG: 0.811 D_real: 0.048 D_fake: 0.311 
saving the latest model (epoch 164, total_steps 484000)
(epoch: 164, iters: 1920, time: 0.084) G_GAN: 1.381 G_GAN_Feat: 2.156 G_VGG: 0.892 D_real: 0.250 D_fake: 0.185 
(epoch: 164, iters: 2320, time: 0.093) G_GAN: 1.437 G_GAN_Feat: 2.056 G_VGG: 0.885 D_real: 0.105 D_fake: 0.145 
(epoch: 164, iters: 2720, time: 0.098) G_GAN: 1.469 G_GAN_Feat: 2.142 G_VGG: 0.853 D_real: 0.263 D_fake: 0.157 
End of epoch 164 / 200 	 Time Taken: 285 sec
(epoch: 165, iters: 160, time: 0.090) G_GAN: 1.174 G_GAN_Feat: 2.214 G_VGG: 0.889 D_real: 0.239 D_fake: 0.377 
(epoch: 165, iters: 560, time: 0.084) G_GAN: 0.559 G_GAN_Feat: 1.726 G_VGG: 0.832 D_real: 0.138 D_fake: 0.675 
saving the latest model (epoch 165, total_steps 486000)
(epoch: 165, iters: 960, time: 0.084) G_GAN: 1.374 G_GAN_Feat: 1.899 G_VGG: 0.797 D_real: 0.104 D_fake: 0.237 
(epoch: 165, iters: 1360, time: 0.088) G_GAN: 2.155 G_GAN_Feat: 2.353 G_VGG: 0.845 D_real: 0.056 D_fake: 0.065 
(epoch: 165, iters: 1760, time: 0.093) G_GAN: 1.370 G_GAN_Feat: 2.518 G_VGG: 0.807 D_real: 0.031 D_fake: 0.299 
(epoch: 165, iters: 2160, time: 0.086) G_GAN: 1.527 G_GAN_Feat: 2.517 G_VGG: 0.767 D_real: 0.098 D_fake: 0.085 
(epoch: 165, iters: 2560, time: 0.092) G_GAN: 1.830 G_GAN_Feat: 2.244 G_VGG: 0.759 D_real: 0.479 D_fake: 0.034 
saving the latest model (epoch 165, total_steps 488000)
(epoch: 165, iters: 2960, time: 0.085) G_GAN: 1.295 G_GAN_Feat: 2.175 G_VGG: 0.790 D_real: 0.056 D_fake: 0.195 
End of epoch 165 / 200 	 Time Taken: 288 sec
(epoch: 166, iters: 400, time: 0.085) G_GAN: 1.584 G_GAN_Feat: 2.106 G_VGG: 0.772 D_real: 0.155 D_fake: 0.081 
(epoch: 166, iters: 800, time: 0.095) G_GAN: 1.736 G_GAN_Feat: 2.443 G_VGG: 0.825 D_real: 0.385 D_fake: 0.078 
(epoch: 166, iters: 1200, time: 0.085) G_GAN: 1.376 G_GAN_Feat: 1.919 G_VGG: 0.772 D_real: 0.100 D_fake: 0.289 
(epoch: 166, iters: 1600, time: 0.094) G_GAN: 2.238 G_GAN_Feat: 2.824 G_VGG: 0.825 D_real: 0.228 D_fake: 0.035 
saving the latest model (epoch 166, total_steps 490000)
(epoch: 166, iters: 2000, time: 0.103) G_GAN: 1.908 G_GAN_Feat: 2.430 G_VGG: 0.764 D_real: 0.086 D_fake: 0.052 
(epoch: 166, iters: 2400, time: 0.095) G_GAN: 1.921 G_GAN_Feat: 2.381 G_VGG: 0.763 D_real: 0.159 D_fake: 0.064 
(epoch: 166, iters: 2800, time: 0.096) G_GAN: 0.696 G_GAN_Feat: 1.934 G_VGG: 0.849 D_real: 0.121 D_fake: 0.472 
End of epoch 166 / 200 	 Time Taken: 285 sec
(epoch: 167, iters: 240, time: 0.099) G_GAN: 2.150 G_GAN_Feat: 2.486 G_VGG: 0.779 D_real: 0.076 D_fake: 0.029 
(epoch: 167, iters: 640, time: 0.088) G_GAN: 1.288 G_GAN_Feat: 1.813 G_VGG: 0.729 D_real: 0.212 D_fake: 0.357 
saving the latest model (epoch 167, total_steps 492000)
(epoch: 167, iters: 1040, time: 0.092) G_GAN: 1.406 G_GAN_Feat: 2.013 G_VGG: 0.798 D_real: 0.071 D_fake: 0.189 
(epoch: 167, iters: 1440, time: 0.095) G_GAN: 1.449 G_GAN_Feat: 2.018 G_VGG: 0.827 D_real: 0.303 D_fake: 0.168 
(epoch: 167, iters: 1840, time: 0.086) G_GAN: 1.775 G_GAN_Feat: 2.658 G_VGG: 0.715 D_real: 0.079 D_fake: 0.039 
(epoch: 167, iters: 2240, time: 0.103) G_GAN: 1.391 G_GAN_Feat: 2.078 G_VGG: 0.784 D_real: 0.205 D_fake: 0.264 
(epoch: 167, iters: 2640, time: 0.098) G_GAN: 2.065 G_GAN_Feat: 2.821 G_VGG: 0.851 D_real: 0.077 D_fake: 0.018 
saving the latest model (epoch 167, total_steps 494000)
End of epoch 167 / 200 	 Time Taken: 289 sec
(epoch: 168, iters: 80, time: 0.087) G_GAN: 1.262 G_GAN_Feat: 2.117 G_VGG: 0.855 D_real: 0.028 D_fake: 0.474 
(epoch: 168, iters: 480, time: 0.091) G_GAN: 1.098 G_GAN_Feat: 1.897 G_VGG: 0.839 D_real: 0.118 D_fake: 0.346 
(epoch: 168, iters: 880, time: 0.096) G_GAN: 1.888 G_GAN_Feat: 2.598 G_VGG: 0.743 D_real: 0.104 D_fake: 0.071 
(epoch: 168, iters: 1280, time: 0.094) G_GAN: 1.755 G_GAN_Feat: 2.466 G_VGG: 0.821 D_real: 0.034 D_fake: 0.039 
(epoch: 168, iters: 1680, time: 0.095) G_GAN: 0.969 G_GAN_Feat: 1.942 G_VGG: 0.886 D_real: 0.120 D_fake: 0.346 
saving the latest model (epoch 168, total_steps 496000)
(epoch: 168, iters: 2080, time: 0.095) G_GAN: 1.202 G_GAN_Feat: 1.886 G_VGG: 0.814 D_real: 0.249 D_fake: 0.250 
(epoch: 168, iters: 2480, time: 0.095) G_GAN: 1.302 G_GAN_Feat: 1.774 G_VGG: 0.808 D_real: 0.107 D_fake: 0.279 
(epoch: 168, iters: 2880, time: 0.092) G_GAN: 1.820 G_GAN_Feat: 2.278 G_VGG: 0.769 D_real: 0.123 D_fake: 0.046 
End of epoch 168 / 200 	 Time Taken: 286 sec
(epoch: 169, iters: 320, time: 0.089) G_GAN: 1.526 G_GAN_Feat: 2.186 G_VGG: 0.743 D_real: 0.205 D_fake: 0.151 
(epoch: 169, iters: 720, time: 0.094) G_GAN: 1.079 G_GAN_Feat: 1.858 G_VGG: 0.855 D_real: 0.203 D_fake: 0.342 
saving the latest model (epoch 169, total_steps 498000)
(epoch: 169, iters: 1120, time: 0.086) G_GAN: 1.078 G_GAN_Feat: 2.169 G_VGG: 0.805 D_real: 0.038 D_fake: 0.378 
(epoch: 169, iters: 1520, time: 0.095) G_GAN: 1.380 G_GAN_Feat: 1.900 G_VGG: 0.798 D_real: 0.081 D_fake: 0.148 
(epoch: 169, iters: 1920, time: 0.086) G_GAN: 1.631 G_GAN_Feat: 2.352 G_VGG: 0.879 D_real: 0.112 D_fake: 0.099 
(epoch: 169, iters: 2320, time: 0.094) G_GAN: 1.306 G_GAN_Feat: 1.937 G_VGG: 0.826 D_real: 0.164 D_fake: 0.233 
(epoch: 169, iters: 2720, time: 0.087) G_GAN: 1.259 G_GAN_Feat: 1.908 G_VGG: 0.829 D_real: 0.214 D_fake: 0.284 
saving the latest model (epoch 169, total_steps 500000)
End of epoch 169 / 200 	 Time Taken: 289 sec
(epoch: 170, iters: 160, time: 0.094) G_GAN: 1.531 G_GAN_Feat: 2.131 G_VGG: 0.823 D_real: 0.191 D_fake: 0.161 
(epoch: 170, iters: 560, time: 0.092) G_GAN: 1.385 G_GAN_Feat: 2.160 G_VGG: 0.885 D_real: 0.168 D_fake: 0.255 
(epoch: 170, iters: 960, time: 0.091) G_GAN: 1.194 G_GAN_Feat: 2.236 G_VGG: 0.783 D_real: 0.035 D_fake: 0.401 
(epoch: 170, iters: 1360, time: 0.092) G_GAN: 1.464 G_GAN_Feat: 2.105 G_VGG: 0.816 D_real: 0.208 D_fake: 0.180 
(epoch: 170, iters: 1760, time: 0.079) G_GAN: 1.780 G_GAN_Feat: 2.319 G_VGG: 0.788 D_real: 0.204 D_fake: 0.056 
saving the latest model (epoch 170, total_steps 502000)
(epoch: 170, iters: 2160, time: 0.093) G_GAN: 1.038 G_GAN_Feat: 2.097 G_VGG: 0.817 D_real: 0.031 D_fake: 0.537 
(epoch: 170, iters: 2560, time: 0.091) G_GAN: 1.232 G_GAN_Feat: 2.106 G_VGG: 0.837 D_real: 0.178 D_fake: 0.215 
(epoch: 170, iters: 2960, time: 0.091) G_GAN: 1.469 G_GAN_Feat: 1.965 G_VGG: 0.830 D_real: 0.202 D_fake: 0.159 
End of epoch 170 / 200 	 Time Taken: 286 sec
saving the model at the end of epoch 170, iters 503200
(epoch: 171, iters: 400, time: 0.093) G_GAN: 1.083 G_GAN_Feat: 1.805 G_VGG: 0.804 D_real: 0.054 D_fake: 0.515 
(epoch: 171, iters: 800, time: 0.098) G_GAN: 2.183 G_GAN_Feat: 2.326 G_VGG: 0.806 D_real: 0.441 D_fake: 0.029 
saving the latest model (epoch 171, total_steps 504000)
(epoch: 171, iters: 1200, time: 0.089) G_GAN: 1.551 G_GAN_Feat: 2.207 G_VGG: 0.775 D_real: 0.436 D_fake: 0.115 
(epoch: 171, iters: 1600, time: 0.085) G_GAN: 1.861 G_GAN_Feat: 2.324 G_VGG: 0.860 D_real: 0.311 D_fake: 0.045 
(epoch: 171, iters: 2000, time: 0.093) G_GAN: 1.078 G_GAN_Feat: 1.906 G_VGG: 0.769 D_real: 0.053 D_fake: 0.385 
(epoch: 171, iters: 2400, time: 0.094) G_GAN: 1.568 G_GAN_Feat: 2.271 G_VGG: 0.761 D_real: 0.030 D_fake: 0.100 
(epoch: 171, iters: 2800, time: 0.093) G_GAN: 1.339 G_GAN_Feat: 1.950 G_VGG: 0.783 D_real: 0.165 D_fake: 0.317 
saving the latest model (epoch 171, total_steps 506000)
End of epoch 171 / 200 	 Time Taken: 287 sec
(epoch: 172, iters: 240, time: 0.097) G_GAN: 1.603 G_GAN_Feat: 2.207 G_VGG: 0.743 D_real: 0.071 D_fake: 0.137 
(epoch: 172, iters: 640, time: 0.094) G_GAN: 1.659 G_GAN_Feat: 2.275 G_VGG: 0.835 D_real: 0.069 D_fake: 0.180 
(epoch: 172, iters: 1040, time: 0.098) G_GAN: 1.616 G_GAN_Feat: 2.107 G_VGG: 0.783 D_real: 0.377 D_fake: 0.111 
(epoch: 172, iters: 1440, time: 0.091) G_GAN: 1.531 G_GAN_Feat: 2.145 G_VGG: 0.796 D_real: 0.039 D_fake: 0.164 
(epoch: 172, iters: 1840, time: 0.087) G_GAN: 1.326 G_GAN_Feat: 1.785 G_VGG: 0.753 D_real: 0.854 D_fake: 0.222 
saving the latest model (epoch 172, total_steps 508000)
(epoch: 172, iters: 2240, time: 0.087) G_GAN: 2.130 G_GAN_Feat: 2.275 G_VGG: 0.843 D_real: 0.170 D_fake: 0.030 
(epoch: 172, iters: 2640, time: 0.092) G_GAN: 1.544 G_GAN_Feat: 2.044 G_VGG: 0.834 D_real: 0.188 D_fake: 0.150 
End of epoch 172 / 200 	 Time Taken: 283 sec
(epoch: 173, iters: 80, time: 0.092) G_GAN: 1.347 G_GAN_Feat: 1.910 G_VGG: 0.807 D_real: 0.156 D_fake: 0.211 
(epoch: 173, iters: 480, time: 0.089) G_GAN: 1.437 G_GAN_Feat: 1.955 G_VGG: 0.791 D_real: 0.239 D_fake: 0.153 
(epoch: 173, iters: 880, time: 0.094) G_GAN: 1.527 G_GAN_Feat: 2.243 G_VGG: 0.808 D_real: 0.052 D_fake: 0.092 
saving the latest model (epoch 173, total_steps 510000)
(epoch: 173, iters: 1280, time: 0.090) G_GAN: 1.341 G_GAN_Feat: 1.816 G_VGG: 0.717 D_real: 0.230 D_fake: 0.213 
(epoch: 173, iters: 1680, time: 0.096) G_GAN: 1.292 G_GAN_Feat: 1.917 G_VGG: 0.749 D_real: 0.037 D_fake: 0.357 
(epoch: 173, iters: 2080, time: 0.091) G_GAN: 1.172 G_GAN_Feat: 2.018 G_VGG: 0.796 D_real: 0.026 D_fake: 0.310 
(epoch: 173, iters: 2480, time: 0.094) G_GAN: 1.309 G_GAN_Feat: 2.167 G_VGG: 0.857 D_real: 0.152 D_fake: 0.206 
(epoch: 173, iters: 2880, time: 0.090) G_GAN: 1.247 G_GAN_Feat: 2.094 G_VGG: 0.823 D_real: 0.077 D_fake: 0.321 
saving the latest model (epoch 173, total_steps 512000)
End of epoch 173 / 200 	 Time Taken: 289 sec
(epoch: 174, iters: 320, time: 0.084) G_GAN: 1.898 G_GAN_Feat: 2.311 G_VGG: 0.823 D_real: 0.166 D_fake: 0.033 
(epoch: 174, iters: 720, time: 0.094) G_GAN: 1.174 G_GAN_Feat: 1.773 G_VGG: 0.856 D_real: 0.784 D_fake: 0.305 
(epoch: 174, iters: 1120, time: 0.091) G_GAN: 0.670 G_GAN_Feat: 1.669 G_VGG: 0.736 D_real: 0.317 D_fake: 0.484 
(epoch: 174, iters: 1520, time: 0.088) G_GAN: 1.372 G_GAN_Feat: 2.228 G_VGG: 0.860 D_real: 0.076 D_fake: 0.221 
(epoch: 174, iters: 1920, time: 0.098) G_GAN: 1.350 G_GAN_Feat: 1.967 G_VGG: 0.812 D_real: 0.141 D_fake: 0.203 
saving the latest model (epoch 174, total_steps 514000)
(epoch: 174, iters: 2320, time: 0.088) G_GAN: 1.460 G_GAN_Feat: 2.118 G_VGG: 0.830 D_real: 0.058 D_fake: 0.215 
(epoch: 174, iters: 2720, time: 0.088) G_GAN: 0.839 G_GAN_Feat: 2.020 G_VGG: 0.784 D_real: 0.071 D_fake: 0.435 
End of epoch 174 / 200 	 Time Taken: 288 sec
(epoch: 175, iters: 160, time: 0.091) G_GAN: 1.605 G_GAN_Feat: 2.269 G_VGG: 0.888 D_real: 0.083 D_fake: 0.107 
(epoch: 175, iters: 560, time: 0.084) G_GAN: 1.489 G_GAN_Feat: 1.987 G_VGG: 0.753 D_real: 0.146 D_fake: 0.145 
(epoch: 175, iters: 960, time: 0.089) G_GAN: 1.583 G_GAN_Feat: 2.084 G_VGG: 0.809 D_real: 0.212 D_fake: 0.104 
saving the latest model (epoch 175, total_steps 516000)
(epoch: 175, iters: 1360, time: 0.085) G_GAN: 1.196 G_GAN_Feat: 1.671 G_VGG: 0.694 D_real: 0.056 D_fake: 0.312 
(epoch: 175, iters: 1760, time: 0.090) G_GAN: 1.250 G_GAN_Feat: 2.101 G_VGG: 0.887 D_real: 0.255 D_fake: 0.247 
(epoch: 175, iters: 2160, time: 0.091) G_GAN: 1.596 G_GAN_Feat: 1.922 G_VGG: 0.737 D_real: 0.333 D_fake: 0.138 
(epoch: 175, iters: 2560, time: 0.090) G_GAN: 1.356 G_GAN_Feat: 1.928 G_VGG: 0.773 D_real: 0.210 D_fake: 0.168 
(epoch: 175, iters: 2960, time: 0.092) G_GAN: 1.119 G_GAN_Feat: 1.686 G_VGG: 0.788 D_real: 0.331 D_fake: 0.271 
saving the latest model (epoch 175, total_steps 518000)
End of epoch 175 / 200 	 Time Taken: 287 sec
(epoch: 176, iters: 400, time: 0.092) G_GAN: 1.543 G_GAN_Feat: 2.264 G_VGG: 0.754 D_real: 0.055 D_fake: 0.096 
(epoch: 176, iters: 800, time: 0.096) G_GAN: 1.186 G_GAN_Feat: 1.850 G_VGG: 0.806 D_real: 0.063 D_fake: 0.424 
(epoch: 176, iters: 1200, time: 0.103) G_GAN: 1.513 G_GAN_Feat: 2.416 G_VGG: 0.815 D_real: 0.029 D_fake: 0.094 
(epoch: 176, iters: 1600, time: 0.085) G_GAN: 1.129 G_GAN_Feat: 1.939 G_VGG: 0.857 D_real: 0.095 D_fake: 0.487 
(epoch: 176, iters: 2000, time: 0.091) G_GAN: 1.417 G_GAN_Feat: 2.143 G_VGG: 0.784 D_real: 0.169 D_fake: 0.257 
saving the latest model (epoch 176, total_steps 520000)
(epoch: 176, iters: 2400, time: 0.094) G_GAN: 1.249 G_GAN_Feat: 1.621 G_VGG: 0.755 D_real: 0.246 D_fake: 0.231 
(epoch: 176, iters: 2800, time: 0.091) G_GAN: 1.371 G_GAN_Feat: 1.801 G_VGG: 0.804 D_real: 0.159 D_fake: 0.200 
End of epoch 176 / 200 	 Time Taken: 286 sec
(epoch: 177, iters: 240, time: 0.087) G_GAN: 1.240 G_GAN_Feat: 1.815 G_VGG: 0.824 D_real: 0.252 D_fake: 0.223 
(epoch: 177, iters: 640, time: 0.095) G_GAN: 1.788 G_GAN_Feat: 2.137 G_VGG: 0.775 D_real: 0.130 D_fake: 0.061 
(epoch: 177, iters: 1040, time: 0.098) G_GAN: 1.255 G_GAN_Feat: 1.997 G_VGG: 0.750 D_real: 0.124 D_fake: 0.246 
saving the latest model (epoch 177, total_steps 522000)
(epoch: 177, iters: 1440, time: 0.093) G_GAN: 1.827 G_GAN_Feat: 2.013 G_VGG: 0.727 D_real: 0.146 D_fake: 0.043 
(epoch: 177, iters: 1840, time: 0.090) G_GAN: 1.458 G_GAN_Feat: 2.052 G_VGG: 0.851 D_real: 0.206 D_fake: 0.155 
(epoch: 177, iters: 2240, time: 0.090) G_GAN: 1.444 G_GAN_Feat: 2.117 G_VGG: 0.767 D_real: 0.310 D_fake: 0.177 
(epoch: 177, iters: 2640, time: 0.092) G_GAN: 1.208 G_GAN_Feat: 2.076 G_VGG: 0.812 D_real: 0.165 D_fake: 0.356 
End of epoch 177 / 200 	 Time Taken: 284 sec
(epoch: 178, iters: 80, time: 0.094) G_GAN: 1.255 G_GAN_Feat: 2.066 G_VGG: 0.816 D_real: 0.064 D_fake: 0.329 
saving the latest model (epoch 178, total_steps 524000)
(epoch: 178, iters: 480, time: 0.095) G_GAN: 1.510 G_GAN_Feat: 2.323 G_VGG: 0.834 D_real: 0.192 D_fake: 0.200 
(epoch: 178, iters: 880, time: 0.096) G_GAN: 1.316 G_GAN_Feat: 1.916 G_VGG: 0.787 D_real: 0.157 D_fake: 0.296 
(epoch: 178, iters: 1280, time: 0.099) G_GAN: 1.045 G_GAN_Feat: 1.907 G_VGG: 0.832 D_real: 0.263 D_fake: 0.229 
(epoch: 178, iters: 1680, time: 0.091) G_GAN: 1.502 G_GAN_Feat: 1.915 G_VGG: 0.799 D_real: 0.211 D_fake: 0.126 
(epoch: 178, iters: 2080, time: 0.100) G_GAN: 1.754 G_GAN_Feat: 2.220 G_VGG: 0.768 D_real: 0.101 D_fake: 0.057 
saving the latest model (epoch 178, total_steps 526000)
(epoch: 178, iters: 2480, time: 0.095) G_GAN: 2.354 G_GAN_Feat: 2.455 G_VGG: 0.850 D_real: 0.307 D_fake: 0.040 
(epoch: 178, iters: 2880, time: 0.091) G_GAN: 2.034 G_GAN_Feat: 2.087 G_VGG: 0.771 D_real: 0.405 D_fake: 0.027 
End of epoch 178 / 200 	 Time Taken: 294 sec
(epoch: 179, iters: 320, time: 0.099) G_GAN: 1.676 G_GAN_Feat: 1.985 G_VGG: 0.707 D_real: 0.384 D_fake: 0.077 
(epoch: 179, iters: 720, time: 0.090) G_GAN: 1.271 G_GAN_Feat: 1.960 G_VGG: 0.806 D_real: 0.057 D_fake: 0.247 
(epoch: 179, iters: 1120, time: 0.086) G_GAN: 1.564 G_GAN_Feat: 1.943 G_VGG: 0.782 D_real: 0.175 D_fake: 0.098 
saving the latest model (epoch 179, total_steps 528000)
(epoch: 179, iters: 1520, time: 0.086) G_GAN: 1.701 G_GAN_Feat: 2.278 G_VGG: 0.826 D_real: 0.215 D_fake: 0.066 
(epoch: 179, iters: 1920, time: 0.095) G_GAN: 1.332 G_GAN_Feat: 1.903 G_VGG: 0.795 D_real: 0.056 D_fake: 0.237 
(epoch: 179, iters: 2320, time: 0.077) G_GAN: 1.296 G_GAN_Feat: 2.019 G_VGG: 0.868 D_real: 0.043 D_fake: 0.247 
(epoch: 179, iters: 2720, time: 0.094) G_GAN: 1.496 G_GAN_Feat: 2.174 G_VGG: 0.839 D_real: 0.130 D_fake: 0.201 
End of epoch 179 / 200 	 Time Taken: 283 sec
(epoch: 180, iters: 160, time: 0.094) G_GAN: 0.852 G_GAN_Feat: 1.726 G_VGG: 0.767 D_real: 0.469 D_fake: 0.362 
saving the latest model (epoch 180, total_steps 530000)
(epoch: 180, iters: 560, time: 0.094) G_GAN: 1.772 G_GAN_Feat: 2.227 G_VGG: 0.841 D_real: 0.486 D_fake: 0.057 
(epoch: 180, iters: 960, time: 0.089) G_GAN: 1.322 G_GAN_Feat: 1.918 G_VGG: 0.782 D_real: 0.136 D_fake: 0.200 
(epoch: 180, iters: 1360, time: 0.099) G_GAN: 1.422 G_GAN_Feat: 2.099 G_VGG: 0.858 D_real: 0.059 D_fake: 0.162 
(epoch: 180, iters: 1760, time: 0.085) G_GAN: 1.514 G_GAN_Feat: 2.057 G_VGG: 0.746 D_real: 0.204 D_fake: 0.126 
(epoch: 180, iters: 2160, time: 0.094) G_GAN: 1.317 G_GAN_Feat: 2.128 G_VGG: 0.875 D_real: 0.140 D_fake: 0.219 
saving the latest model (epoch 180, total_steps 532000)
(epoch: 180, iters: 2560, time: 0.089) G_GAN: 1.171 G_GAN_Feat: 2.068 G_VGG: 0.824 D_real: 0.044 D_fake: 0.494 
(epoch: 180, iters: 2960, time: 0.087) G_GAN: 1.171 G_GAN_Feat: 1.686 G_VGG: 0.755 D_real: 0.051 D_fake: 0.451 
End of epoch 180 / 200 	 Time Taken: 286 sec
saving the model at the end of epoch 180, iters 532800
(epoch: 181, iters: 400, time: 0.085) G_GAN: 1.586 G_GAN_Feat: 2.297 G_VGG: 0.905 D_real: 0.143 D_fake: 0.121 
(epoch: 181, iters: 800, time: 0.088) G_GAN: 1.892 G_GAN_Feat: 2.411 G_VGG: 0.859 D_real: 0.085 D_fake: 0.044 
(epoch: 181, iters: 1200, time: 0.091) G_GAN: 1.440 G_GAN_Feat: 2.077 G_VGG: 0.759 D_real: 0.093 D_fake: 0.150 
saving the latest model (epoch 181, total_steps 534000)
(epoch: 181, iters: 1600, time: 0.091) G_GAN: 1.495 G_GAN_Feat: 2.029 G_VGG: 0.737 D_real: 0.089 D_fake: 0.165 
(epoch: 181, iters: 2000, time: 0.092) G_GAN: 1.236 G_GAN_Feat: 1.962 G_VGG: 0.831 D_real: 0.077 D_fake: 0.352 
(epoch: 181, iters: 2400, time: 0.088) G_GAN: 1.345 G_GAN_Feat: 1.831 G_VGG: 0.810 D_real: 0.247 D_fake: 0.145 
(epoch: 181, iters: 2800, time: 0.086) G_GAN: 1.306 G_GAN_Feat: 1.903 G_VGG: 0.790 D_real: 0.169 D_fake: 0.228 
End of epoch 181 / 200 	 Time Taken: 288 sec
(epoch: 182, iters: 240, time: 0.089) G_GAN: 2.214 G_GAN_Feat: 2.230 G_VGG: 0.721 D_real: 0.170 D_fake: 0.054 
saving the latest model (epoch 182, total_steps 536000)
(epoch: 182, iters: 640, time: 0.097) G_GAN: 1.166 G_GAN_Feat: 1.809 G_VGG: 0.844 D_real: 0.131 D_fake: 0.284 
(epoch: 182, iters: 1040, time: 0.090) G_GAN: 0.844 G_GAN_Feat: 1.673 G_VGG: 0.762 D_real: 0.183 D_fake: 0.356 
(epoch: 182, iters: 1440, time: 0.091) G_GAN: 1.234 G_GAN_Feat: 1.908 G_VGG: 0.879 D_real: 0.198 D_fake: 0.301 
(epoch: 182, iters: 1840, time: 0.098) G_GAN: 1.421 G_GAN_Feat: 2.118 G_VGG: 0.773 D_real: 0.125 D_fake: 0.241 
(epoch: 182, iters: 2240, time: 0.092) G_GAN: 1.511 G_GAN_Feat: 2.189 G_VGG: 0.827 D_real: 0.075 D_fake: 0.156 
saving the latest model (epoch 182, total_steps 538000)
(epoch: 182, iters: 2640, time: 0.076) G_GAN: 1.302 G_GAN_Feat: 1.779 G_VGG: 0.711 D_real: 0.235 D_fake: 0.262 
End of epoch 182 / 200 	 Time Taken: 287 sec
(epoch: 183, iters: 80, time: 0.098) G_GAN: 1.320 G_GAN_Feat: 1.735 G_VGG: 0.790 D_real: 0.101 D_fake: 0.214 
(epoch: 183, iters: 480, time: 0.086) G_GAN: 1.683 G_GAN_Feat: 2.178 G_VGG: 0.794 D_real: 0.061 D_fake: 0.103 
(epoch: 183, iters: 880, time: 0.088) G_GAN: 1.286 G_GAN_Feat: 1.982 G_VGG: 0.862 D_real: 0.193 D_fake: 0.282 
(epoch: 183, iters: 1280, time: 0.094) G_GAN: 1.100 G_GAN_Feat: 1.899 G_VGG: 0.862 D_real: 0.104 D_fake: 0.366 
saving the latest model (epoch 183, total_steps 540000)
(epoch: 183, iters: 1680, time: 0.092) G_GAN: 1.303 G_GAN_Feat: 1.983 G_VGG: 0.844 D_real: 0.266 D_fake: 0.232 
(epoch: 183, iters: 2080, time: 0.097) G_GAN: 1.281 G_GAN_Feat: 1.858 G_VGG: 0.783 D_real: 0.144 D_fake: 0.253 
(epoch: 183, iters: 2480, time: 0.087) G_GAN: 1.567 G_GAN_Feat: 2.020 G_VGG: 0.841 D_real: 0.274 D_fake: 0.148 
(epoch: 183, iters: 2880, time: 0.092) G_GAN: 1.850 G_GAN_Feat: 2.314 G_VGG: 0.751 D_real: 0.066 D_fake: 0.053 
End of epoch 183 / 200 	 Time Taken: 285 sec
(epoch: 184, iters: 320, time: 0.084) G_GAN: 1.591 G_GAN_Feat: 2.017 G_VGG: 0.817 D_real: 0.095 D_fake: 0.189 
saving the latest model (epoch 184, total_steps 542000)
(epoch: 184, iters: 720, time: 0.092) G_GAN: 1.241 G_GAN_Feat: 1.887 G_VGG: 0.749 D_real: 0.109 D_fake: 0.262 
(epoch: 184, iters: 1120, time: 0.099) G_GAN: 1.724 G_GAN_Feat: 1.993 G_VGG: 0.796 D_real: 0.045 D_fake: 0.088 
(epoch: 184, iters: 1520, time: 0.088) G_GAN: 1.355 G_GAN_Feat: 1.857 G_VGG: 0.801 D_real: 0.107 D_fake: 0.185 
(epoch: 184, iters: 1920, time: 0.090) G_GAN: 1.279 G_GAN_Feat: 1.753 G_VGG: 0.717 D_real: 0.081 D_fake: 0.284 
(epoch: 184, iters: 2320, time: 0.093) G_GAN: 1.594 G_GAN_Feat: 2.085 G_VGG: 0.850 D_real: 0.187 D_fake: 0.131 
saving the latest model (epoch 184, total_steps 544000)
(epoch: 184, iters: 2720, time: 0.083) G_GAN: 1.297 G_GAN_Feat: 1.757 G_VGG: 0.719 D_real: 0.139 D_fake: 0.337 
End of epoch 184 / 200 	 Time Taken: 288 sec
(epoch: 185, iters: 160, time: 0.084) G_GAN: 1.911 G_GAN_Feat: 2.216 G_VGG: 0.726 D_real: 0.412 D_fake: 0.078 
(epoch: 185, iters: 560, time: 0.091) G_GAN: 1.307 G_GAN_Feat: 1.917 G_VGG: 0.801 D_real: 0.080 D_fake: 0.243 
(epoch: 185, iters: 960, time: 0.094) G_GAN: 1.583 G_GAN_Feat: 1.815 G_VGG: 0.770 D_real: 0.212 D_fake: 0.096 
(epoch: 185, iters: 1360, time: 0.086) G_GAN: 1.360 G_GAN_Feat: 1.837 G_VGG: 0.802 D_real: 0.133 D_fake: 0.206 
saving the latest model (epoch 185, total_steps 546000)
(epoch: 185, iters: 1760, time: 0.092) G_GAN: 1.480 G_GAN_Feat: 1.820 G_VGG: 0.738 D_real: 0.106 D_fake: 0.135 
(epoch: 185, iters: 2160, time: 0.092) G_GAN: 1.038 G_GAN_Feat: 1.657 G_VGG: 0.820 D_real: 0.211 D_fake: 0.329 
(epoch: 185, iters: 2560, time: 0.087) G_GAN: 1.176 G_GAN_Feat: 1.761 G_VGG: 0.793 D_real: 0.206 D_fake: 0.302 
(epoch: 185, iters: 2960, time: 0.097) G_GAN: 1.324 G_GAN_Feat: 1.830 G_VGG: 0.812 D_real: 0.092 D_fake: 0.224 
End of epoch 185 / 200 	 Time Taken: 285 sec
(epoch: 186, iters: 400, time: 0.085) G_GAN: 1.300 G_GAN_Feat: 1.697 G_VGG: 0.796 D_real: 0.188 D_fake: 0.220 
saving the latest model (epoch 186, total_steps 548000)
(epoch: 186, iters: 800, time: 0.086) G_GAN: 1.610 G_GAN_Feat: 2.153 G_VGG: 0.919 D_real: 0.211 D_fake: 0.095 
(epoch: 186, iters: 1200, time: 0.088) G_GAN: 1.139 G_GAN_Feat: 1.806 G_VGG: 0.773 D_real: 0.072 D_fake: 0.390 
(epoch: 186, iters: 1600, time: 0.091) G_GAN: 1.598 G_GAN_Feat: 1.854 G_VGG: 0.798 D_real: 0.083 D_fake: 0.119 
(epoch: 186, iters: 2000, time: 0.089) G_GAN: 1.435 G_GAN_Feat: 1.915 G_VGG: 0.768 D_real: 0.091 D_fake: 0.185 
(epoch: 186, iters: 2400, time: 0.091) G_GAN: 1.815 G_GAN_Feat: 2.250 G_VGG: 0.784 D_real: 0.042 D_fake: 0.068 
saving the latest model (epoch 186, total_steps 550000)
(epoch: 186, iters: 2800, time: 0.097) G_GAN: 1.411 G_GAN_Feat: 1.985 G_VGG: 0.893 D_real: 0.098 D_fake: 0.153 
End of epoch 186 / 200 	 Time Taken: 289 sec
(epoch: 187, iters: 240, time: 0.096) G_GAN: 1.637 G_GAN_Feat: 2.036 G_VGG: 0.758 D_real: 0.137 D_fake: 0.129 
(epoch: 187, iters: 640, time: 0.093) G_GAN: 1.267 G_GAN_Feat: 1.900 G_VGG: 0.818 D_real: 0.162 D_fake: 0.242 
(epoch: 187, iters: 1040, time: 0.098) G_GAN: 2.013 G_GAN_Feat: 2.227 G_VGG: 0.859 D_real: 0.134 D_fake: 0.036 
(epoch: 187, iters: 1440, time: 0.091) G_GAN: 0.976 G_GAN_Feat: 1.503 G_VGG: 0.702 D_real: 0.172 D_fake: 0.421 
saving the latest model (epoch 187, total_steps 552000)
(epoch: 187, iters: 1840, time: 0.098) G_GAN: 1.336 G_GAN_Feat: 1.730 G_VGG: 0.729 D_real: 0.113 D_fake: 0.244 
(epoch: 187, iters: 2240, time: 0.092) G_GAN: 1.555 G_GAN_Feat: 2.119 G_VGG: 0.823 D_real: 0.120 D_fake: 0.158 
(epoch: 187, iters: 2640, time: 0.092) G_GAN: 1.878 G_GAN_Feat: 2.244 G_VGG: 0.803 D_real: 0.114 D_fake: 0.061 
End of epoch 187 / 200 	 Time Taken: 289 sec
(epoch: 188, iters: 80, time: 0.093) G_GAN: 1.410 G_GAN_Feat: 2.138 G_VGG: 0.894 D_real: 0.171 D_fake: 0.214 
(epoch: 188, iters: 480, time: 0.096) G_GAN: 1.914 G_GAN_Feat: 1.984 G_VGG: 0.803 D_real: 0.157 D_fake: 0.046 
saving the latest model (epoch 188, total_steps 554000)
(epoch: 188, iters: 880, time: 0.086) G_GAN: 1.275 G_GAN_Feat: 1.811 G_VGG: 0.854 D_real: 0.148 D_fake: 0.269 
(epoch: 188, iters: 1280, time: 0.085) G_GAN: 1.345 G_GAN_Feat: 1.650 G_VGG: 0.716 D_real: 0.186 D_fake: 0.216 
(epoch: 188, iters: 1680, time: 0.099) G_GAN: 1.388 G_GAN_Feat: 2.024 G_VGG: 0.914 D_real: 0.220 D_fake: 0.198 
(epoch: 188, iters: 2080, time: 0.103) G_GAN: 1.371 G_GAN_Feat: 1.626 G_VGG: 0.730 D_real: 0.249 D_fake: 0.225 
(epoch: 188, iters: 2480, time: 0.090) G_GAN: 1.551 G_GAN_Feat: 1.981 G_VGG: 0.780 D_real: 0.128 D_fake: 0.097 
saving the latest model (epoch 188, total_steps 556000)
(epoch: 188, iters: 2880, time: 0.101) G_GAN: 1.512 G_GAN_Feat: 2.066 G_VGG: 0.765 D_real: 0.512 D_fake: 0.191 
End of epoch 188 / 200 	 Time Taken: 288 sec
(epoch: 189, iters: 320, time: 0.084) G_GAN: 1.360 G_GAN_Feat: 1.934 G_VGG: 0.866 D_real: 0.110 D_fake: 0.230 
(epoch: 189, iters: 720, time: 0.092) G_GAN: 1.454 G_GAN_Feat: 1.695 G_VGG: 0.745 D_real: 0.101 D_fake: 0.195 
(epoch: 189, iters: 1120, time: 0.094) G_GAN: 1.345 G_GAN_Feat: 1.913 G_VGG: 0.840 D_real: 0.134 D_fake: 0.199 
(epoch: 189, iters: 1520, time: 0.094) G_GAN: 1.170 G_GAN_Feat: 1.654 G_VGG: 0.780 D_real: 0.232 D_fake: 0.302 
saving the latest model (epoch 189, total_steps 558000)
(epoch: 189, iters: 1920, time: 0.092) G_GAN: 1.456 G_GAN_Feat: 1.892 G_VGG: 0.777 D_real: 0.092 D_fake: 0.172 
(epoch: 189, iters: 2320, time: 0.095) G_GAN: 1.305 G_GAN_Feat: 1.652 G_VGG: 0.741 D_real: 0.187 D_fake: 0.218 
(epoch: 189, iters: 2720, time: 0.086) G_GAN: 1.238 G_GAN_Feat: 1.527 G_VGG: 0.703 D_real: 0.238 D_fake: 0.241 
End of epoch 189 / 200 	 Time Taken: 284 sec
(epoch: 190, iters: 160, time: 0.093) G_GAN: 1.917 G_GAN_Feat: 1.910 G_VGG: 0.744 D_real: 0.114 D_fake: 0.039 
(epoch: 190, iters: 560, time: 0.095) G_GAN: 1.296 G_GAN_Feat: 1.757 G_VGG: 0.779 D_real: 0.071 D_fake: 0.291 
saving the latest model (epoch 190, total_steps 560000)
(epoch: 190, iters: 960, time: 0.095) G_GAN: 1.393 G_GAN_Feat: 1.785 G_VGG: 0.786 D_real: 0.254 D_fake: 0.188 
(epoch: 190, iters: 1360, time: 0.092) G_GAN: 1.721 G_GAN_Feat: 1.980 G_VGG: 0.810 D_real: 0.080 D_fake: 0.108 
(epoch: 190, iters: 1760, time: 0.089) G_GAN: 1.208 G_GAN_Feat: 1.765 G_VGG: 0.758 D_real: 0.108 D_fake: 0.293 
(epoch: 190, iters: 2160, time: 0.097) G_GAN: 1.926 G_GAN_Feat: 1.900 G_VGG: 0.760 D_real: 0.068 D_fake: 0.057 
(epoch: 190, iters: 2560, time: 0.083) G_GAN: 1.311 G_GAN_Feat: 1.774 G_VGG: 0.776 D_real: 0.047 D_fake: 0.257 
saving the latest model (epoch 190, total_steps 562000)
(epoch: 190, iters: 2960, time: 0.096) G_GAN: 1.351 G_GAN_Feat: 2.178 G_VGG: 0.853 D_real: 0.485 D_fake: 0.226 
End of epoch 190 / 200 	 Time Taken: 289 sec
saving the model at the end of epoch 190, iters 562400
(epoch: 191, iters: 400, time: 0.084) G_GAN: 1.308 G_GAN_Feat: 1.953 G_VGG: 0.874 D_real: 0.164 D_fake: 0.311 
(epoch: 191, iters: 800, time: 0.078) G_GAN: 1.417 G_GAN_Feat: 1.890 G_VGG: 0.831 D_real: 0.125 D_fake: 0.241 
(epoch: 191, iters: 1200, time: 0.102) G_GAN: 1.652 G_GAN_Feat: 1.939 G_VGG: 0.864 D_real: 0.186 D_fake: 0.111 
(epoch: 191, iters: 1600, time: 0.092) G_GAN: 1.651 G_GAN_Feat: 1.810 G_VGG: 0.819 D_real: 0.279 D_fake: 0.070 
saving the latest model (epoch 191, total_steps 564000)
(epoch: 191, iters: 2000, time: 0.084) G_GAN: 1.458 G_GAN_Feat: 1.911 G_VGG: 0.875 D_real: 0.134 D_fake: 0.213 
(epoch: 191, iters: 2400, time: 0.096) G_GAN: 1.195 G_GAN_Feat: 1.748 G_VGG: 0.803 D_real: 0.128 D_fake: 0.394 
(epoch: 191, iters: 2800, time: 0.098) G_GAN: 1.310 G_GAN_Feat: 1.688 G_VGG: 0.791 D_real: 0.128 D_fake: 0.289 
End of epoch 191 / 200 	 Time Taken: 285 sec
(epoch: 192, iters: 240, time: 0.087) G_GAN: 1.518 G_GAN_Feat: 1.843 G_VGG: 0.763 D_real: 0.164 D_fake: 0.141 
(epoch: 192, iters: 640, time: 0.091) G_GAN: 1.313 G_GAN_Feat: 1.733 G_VGG: 0.753 D_real: 0.084 D_fake: 0.207 
saving the latest model (epoch 192, total_steps 566000)
(epoch: 192, iters: 1040, time: 0.091) G_GAN: 1.607 G_GAN_Feat: 1.838 G_VGG: 0.825 D_real: 0.154 D_fake: 0.079 
(epoch: 192, iters: 1440, time: 0.085) G_GAN: 1.481 G_GAN_Feat: 1.728 G_VGG: 0.753 D_real: 0.125 D_fake: 0.186 
(epoch: 192, iters: 1840, time: 0.088) G_GAN: 1.503 G_GAN_Feat: 1.866 G_VGG: 0.750 D_real: 0.215 D_fake: 0.169 
(epoch: 192, iters: 2240, time: 0.092) G_GAN: 1.215 G_GAN_Feat: 1.991 G_VGG: 0.805 D_real: 0.093 D_fake: 0.272 
(epoch: 192, iters: 2640, time: 0.088) G_GAN: 1.663 G_GAN_Feat: 1.929 G_VGG: 0.790 D_real: 0.299 D_fake: 0.067 
saving the latest model (epoch 192, total_steps 568000)
End of epoch 192 / 200 	 Time Taken: 285 sec
(epoch: 193, iters: 80, time: 0.085) G_GAN: 1.296 G_GAN_Feat: 1.611 G_VGG: 0.714 D_real: 0.134 D_fake: 0.205 
(epoch: 193, iters: 480, time: 0.095) G_GAN: 2.011 G_GAN_Feat: 1.953 G_VGG: 0.786 D_real: 0.059 D_fake: 0.070 
(epoch: 193, iters: 880, time: 0.092) G_GAN: 1.365 G_GAN_Feat: 1.706 G_VGG: 0.759 D_real: 0.136 D_fake: 0.171 
(epoch: 193, iters: 1280, time: 0.087) G_GAN: 1.315 G_GAN_Feat: 1.816 G_VGG: 0.893 D_real: 0.196 D_fake: 0.233 
(epoch: 193, iters: 1680, time: 0.093) G_GAN: 1.421 G_GAN_Feat: 1.777 G_VGG: 0.746 D_real: 0.220 D_fake: 0.207 
saving the latest model (epoch 193, total_steps 570000)
(epoch: 193, iters: 2080, time: 0.084) G_GAN: 1.234 G_GAN_Feat: 1.662 G_VGG: 0.785 D_real: 0.098 D_fake: 0.325 
(epoch: 193, iters: 2480, time: 0.095) G_GAN: 1.418 G_GAN_Feat: 1.780 G_VGG: 0.828 D_real: 0.109 D_fake: 0.183 
(epoch: 193, iters: 2880, time: 0.098) G_GAN: 1.455 G_GAN_Feat: 2.081 G_VGG: 0.805 D_real: 0.142 D_fake: 0.180 
End of epoch 193 / 200 	 Time Taken: 284 sec
(epoch: 194, iters: 320, time: 0.086) G_GAN: 1.533 G_GAN_Feat: 1.848 G_VGG: 0.843 D_real: 0.145 D_fake: 0.170 
(epoch: 194, iters: 720, time: 0.101) G_GAN: 1.446 G_GAN_Feat: 1.863 G_VGG: 0.805 D_real: 0.182 D_fake: 0.150 
saving the latest model (epoch 194, total_steps 572000)
(epoch: 194, iters: 1120, time: 0.093) G_GAN: 1.633 G_GAN_Feat: 1.778 G_VGG: 0.787 D_real: 0.188 D_fake: 0.088 
(epoch: 194, iters: 1520, time: 0.092) G_GAN: 1.331 G_GAN_Feat: 1.922 G_VGG: 0.879 D_real: 0.172 D_fake: 0.221 
(epoch: 194, iters: 1920, time: 0.101) G_GAN: 1.197 G_GAN_Feat: 1.845 G_VGG: 0.841 D_real: 0.120 D_fake: 0.317 
(epoch: 194, iters: 2320, time: 0.092) G_GAN: 1.350 G_GAN_Feat: 1.720 G_VGG: 0.809 D_real: 0.152 D_fake: 0.223 
(epoch: 194, iters: 2720, time: 0.091) G_GAN: 1.250 G_GAN_Feat: 1.722 G_VGG: 0.764 D_real: 0.260 D_fake: 0.205 
saving the latest model (epoch 194, total_steps 574000)
End of epoch 194 / 200 	 Time Taken: 288 sec
(epoch: 195, iters: 160, time: 0.096) G_GAN: 1.448 G_GAN_Feat: 1.771 G_VGG: 0.799 D_real: 0.154 D_fake: 0.183 
(epoch: 195, iters: 560, time: 0.090) G_GAN: 1.505 G_GAN_Feat: 1.756 G_VGG: 0.766 D_real: 0.210 D_fake: 0.121 
(epoch: 195, iters: 960, time: 0.093) G_GAN: 1.340 G_GAN_Feat: 1.558 G_VGG: 0.740 D_real: 0.201 D_fake: 0.216 
(epoch: 195, iters: 1360, time: 0.089) G_GAN: 1.280 G_GAN_Feat: 1.732 G_VGG: 0.779 D_real: 0.118 D_fake: 0.250 
(epoch: 195, iters: 1760, time: 0.084) G_GAN: 1.492 G_GAN_Feat: 1.847 G_VGG: 0.778 D_real: 0.119 D_fake: 0.177 
saving the latest model (epoch 195, total_steps 576000)
(epoch: 195, iters: 2160, time: 0.085) G_GAN: 1.348 G_GAN_Feat: 1.769 G_VGG: 0.739 D_real: 0.222 D_fake: 0.195 
(epoch: 195, iters: 2560, time: 0.089) G_GAN: 1.421 G_GAN_Feat: 1.960 G_VGG: 0.798 D_real: 0.195 D_fake: 0.220 
(epoch: 195, iters: 2960, time: 0.084) G_GAN: 1.311 G_GAN_Feat: 1.690 G_VGG: 0.802 D_real: 0.198 D_fake: 0.249 
End of epoch 195 / 200 	 Time Taken: 284 sec
(epoch: 196, iters: 400, time: 0.088) G_GAN: 1.322 G_GAN_Feat: 1.690 G_VGG: 0.828 D_real: 0.192 D_fake: 0.233 
(epoch: 196, iters: 800, time: 0.085) G_GAN: 1.471 G_GAN_Feat: 1.779 G_VGG: 0.817 D_real: 0.210 D_fake: 0.188 
saving the latest model (epoch 196, total_steps 578000)
(epoch: 196, iters: 1200, time: 0.101) G_GAN: 1.460 G_GAN_Feat: 1.835 G_VGG: 0.871 D_real: 0.193 D_fake: 0.178 
(epoch: 196, iters: 1600, time: 0.088) G_GAN: 1.373 G_GAN_Feat: 1.698 G_VGG: 0.768 D_real: 0.130 D_fake: 0.228 
(epoch: 196, iters: 2000, time: 0.092) G_GAN: 1.333 G_GAN_Feat: 1.759 G_VGG: 0.817 D_real: 0.140 D_fake: 0.224 
(epoch: 196, iters: 2400, time: 0.101) G_GAN: 1.457 G_GAN_Feat: 1.861 G_VGG: 0.838 D_real: 0.194 D_fake: 0.187 
(epoch: 196, iters: 2800, time: 0.089) G_GAN: 1.475 G_GAN_Feat: 1.686 G_VGG: 0.788 D_real: 0.175 D_fake: 0.166 
saving the latest model (epoch 196, total_steps 580000)
End of epoch 196 / 200 	 Time Taken: 289 sec
(epoch: 197, iters: 240, time: 0.084) G_GAN: 1.393 G_GAN_Feat: 1.720 G_VGG: 0.776 D_real: 0.176 D_fake: 0.192 
(epoch: 197, iters: 640, time: 0.092) G_GAN: 1.389 G_GAN_Feat: 1.726 G_VGG: 0.807 D_real: 0.198 D_fake: 0.178 
(epoch: 197, iters: 1040, time: 0.085) G_GAN: 1.263 G_GAN_Feat: 1.758 G_VGG: 0.860 D_real: 0.163 D_fake: 0.274 
(epoch: 197, iters: 1440, time: 0.095) G_GAN: 1.320 G_GAN_Feat: 1.713 G_VGG: 0.831 D_real: 0.175 D_fake: 0.221 
(epoch: 197, iters: 1840, time: 0.096) G_GAN: 1.310 G_GAN_Feat: 1.929 G_VGG: 0.830 D_real: 0.180 D_fake: 0.207 
saving the latest model (epoch 197, total_steps 582000)
(epoch: 197, iters: 2240, time: 0.084) G_GAN: 1.357 G_GAN_Feat: 1.796 G_VGG: 0.800 D_real: 0.198 D_fake: 0.169 
(epoch: 197, iters: 2640, time: 0.095) G_GAN: 1.274 G_GAN_Feat: 1.638 G_VGG: 0.734 D_real: 0.220 D_fake: 0.207 
End of epoch 197 / 200 	 Time Taken: 285 sec
(epoch: 198, iters: 80, time: 0.086) G_GAN: 1.282 G_GAN_Feat: 1.650 G_VGG: 0.789 D_real: 0.179 D_fake: 0.257 
(epoch: 198, iters: 480, time: 0.095) G_GAN: 1.345 G_GAN_Feat: 1.734 G_VGG: 0.824 D_real: 0.195 D_fake: 0.213 
(epoch: 198, iters: 880, time: 0.097) G_GAN: 1.256 G_GAN_Feat: 1.565 G_VGG: 0.744 D_real: 0.220 D_fake: 0.253 
saving the latest model (epoch 198, total_steps 584000)
(epoch: 198, iters: 1280, time: 0.082) G_GAN: 1.369 G_GAN_Feat: 1.671 G_VGG: 0.795 D_real: 0.203 D_fake: 0.206 
(epoch: 198, iters: 1680, time: 0.088) G_GAN: 1.341 G_GAN_Feat: 1.582 G_VGG: 0.729 D_real: 0.190 D_fake: 0.205 
(epoch: 198, iters: 2080, time: 0.094) G_GAN: 1.370 G_GAN_Feat: 1.711 G_VGG: 0.794 D_real: 0.208 D_fake: 0.184 
(epoch: 198, iters: 2480, time: 0.093) G_GAN: 1.388 G_GAN_Feat: 1.711 G_VGG: 0.789 D_real: 0.199 D_fake: 0.178 
(epoch: 198, iters: 2880, time: 0.084) G_GAN: 1.492 G_GAN_Feat: 1.844 G_VGG: 0.812 D_real: 0.178 D_fake: 0.136 
saving the latest model (epoch 198, total_steps 586000)
End of epoch 198 / 200 	 Time Taken: 290 sec
(epoch: 199, iters: 320, time: 0.089) G_GAN: 1.315 G_GAN_Feat: 1.655 G_VGG: 0.816 D_real: 0.209 D_fake: 0.216 
(epoch: 199, iters: 720, time: 0.093) G_GAN: 1.308 G_GAN_Feat: 1.688 G_VGG: 0.845 D_real: 0.236 D_fake: 0.213 
(epoch: 199, iters: 1120, time: 0.088) G_GAN: 1.416 G_GAN_Feat: 1.781 G_VGG: 0.848 D_real: 0.171 D_fake: 0.170 
(epoch: 199, iters: 1520, time: 0.094) G_GAN: 1.335 G_GAN_Feat: 1.615 G_VGG: 0.769 D_real: 0.184 D_fake: 0.199 
(epoch: 199, iters: 1920, time: 0.092) G_GAN: 1.291 G_GAN_Feat: 1.735 G_VGG: 0.854 D_real: 0.225 D_fake: 0.253 
saving the latest model (epoch 199, total_steps 588000)
(epoch: 199, iters: 2320, time: 0.092) G_GAN: 1.272 G_GAN_Feat: 1.724 G_VGG: 0.808 D_real: 0.190 D_fake: 0.212 
(epoch: 199, iters: 2720, time: 0.088) G_GAN: 1.274 G_GAN_Feat: 1.613 G_VGG: 0.766 D_real: 0.206 D_fake: 0.232 
End of epoch 199 / 200 	 Time Taken: 286 sec
(epoch: 200, iters: 160, time: 0.087) G_GAN: 1.425 G_GAN_Feat: 1.657 G_VGG: 0.801 D_real: 0.220 D_fake: 0.179 
(epoch: 200, iters: 560, time: 0.093) G_GAN: 1.322 G_GAN_Feat: 1.610 G_VGG: 0.742 D_real: 0.193 D_fake: 0.202 
(epoch: 200, iters: 960, time: 0.094) G_GAN: 1.335 G_GAN_Feat: 1.607 G_VGG: 0.799 D_real: 0.215 D_fake: 0.211 
saving the latest model (epoch 200, total_steps 590000)
(epoch: 200, iters: 1360, time: 0.098) G_GAN: 1.325 G_GAN_Feat: 1.638 G_VGG: 0.801 D_real: 0.230 D_fake: 0.224 
(epoch: 200, iters: 1760, time: 0.091) G_GAN: 1.336 G_GAN_Feat: 1.669 G_VGG: 0.728 D_real: 0.175 D_fake: 0.220 
(epoch: 200, iters: 2160, time: 0.089) G_GAN: 1.371 G_GAN_Feat: 1.961 G_VGG: 0.906 D_real: 0.223 D_fake: 0.208 
(epoch: 200, iters: 2560, time: 0.084) G_GAN: 1.298 G_GAN_Feat: 1.458 G_VGG: 0.724 D_real: 0.235 D_fake: 0.211 
(epoch: 200, iters: 2960, time: 0.089) G_GAN: 1.325 G_GAN_Feat: 1.905 G_VGG: 0.867 D_real: 0.189 D_fake: 0.227 



=======================================================================================================================


I have no name!@2232ba488464:/mnt/old/git/pix2pixHD$ python train.py --name label2city_512p-8gpu-8batch --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
------------ Options -------------
batchSize: 8
beta1: 0.5
checkpoints_dir: ./checkpoints
continue_train: False
data_type: 32
dataroot: ./datasets/cityscapes/
debug: False
display_freq: 100
display_winsize: 512
feat_num: 3
fineSize: 512
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
input_nc: 3
instance_feat: False
isTrain: True
label_feat: False
label_nc: 35
lambda_feat: 10.0
loadSize: 1024
load_features: False
load_pretrain: 
lr: 0.0002
max_dataset_size: inf
model: pix2pixHD
nThreads: 2
n_blocks_global: 9
n_blocks_local: 3
n_clusters: 10
n_downsample_E: 4
n_downsample_global: 4
n_layers_D: 3
n_local_enhancers: 1
name: label2city_512p-8gpu-8batch
ndf: 64
nef: 16
netG: global
ngf: 64
niter: 100
niter_decay: 100
niter_fix_global: 0
no_flip: False
no_ganFeat_loss: False
no_html: False
no_instance: False
no_lsgan: False
no_vgg_loss: False
norm: instance
num_D: 2
output_nc: 3
phase: train
pool_size: 0
print_freq: 100
resize_or_crop: scale_width
save_epoch_freq: 10
save_latest_freq: 1000
serial_batches: False
tf_log: False
use_dropout: False
verbose: False
which_epoch: latest
-------------- End ----------------
CustomDatasetDataLoader
dataset [AlignedDataset] was created
#training images = 2968
GlobalGenerator(
  (model): Sequential(
    (0): ReflectionPad2d((3, 3, 3, 3))
    (1): Conv2d(36, 64, kernel_size=(7, 7), stride=(1, 1))
    (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (3): ReLU(inplace)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (6): ReLU(inplace)
    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (9): ReLU(inplace)
    (10): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (11): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (12): ReLU(inplace)
    (13): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (14): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (15): ReLU(inplace)
    (16): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (17): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (18): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (19): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (20): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (21): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (22): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (23): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (24): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (25): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (26): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (27): ReLU(inplace)
    (28): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (29): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (30): ReLU(inplace)
    (31): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (32): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (33): ReLU(inplace)
    (34): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (35): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (36): ReLU(inplace)
    (37): ReflectionPad2d((3, 3, 3, 3))
    (38): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
    (39): Tanh()
  )
)
MultiscaleDiscriminator(
  (scale0_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (scale1_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (downsample): AvgPool2d(kernel_size=3, stride=2, padding=[1, 1])
)
create web directory ./checkpoints/label2city_512p-8gpu-8batch/web...
/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:58: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
train.py:87: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
(epoch: 1, iters: 200, time: 0.106) G_GAN: 1.329 G_GAN_Feat: 7.856 G_VGG: 6.187 D_real: 0.771 D_fake: 0.756 
(epoch: 1, iters: 400, time: 0.110) G_GAN: 1.876 G_GAN_Feat: 5.968 G_VGG: 5.816 D_real: 1.226 D_fake: 1.124 
(epoch: 1, iters: 600, time: 0.103) G_GAN: 0.838 G_GAN_Feat: 4.795 G_VGG: 6.215 D_real: 0.555 D_fake: 0.530 
(epoch: 1, iters: 800, time: 0.090) G_GAN: 1.085 G_GAN_Feat: 4.377 G_VGG: 5.948 D_real: 0.843 D_fake: 0.756 
(epoch: 1, iters: 1000, time: 0.107) G_GAN: 0.781 G_GAN_Feat: 3.971 G_VGG: 5.231 D_real: 0.575 D_fake: 0.516 
saving the latest model (epoch 1, total_steps 1000)
(epoch: 1, iters: 1200, time: 0.101) G_GAN: 0.769 G_GAN_Feat: 3.429 G_VGG: 5.951 D_real: 0.432 D_fake: 0.453 
(epoch: 1, iters: 1400, time: 0.102) G_GAN: 1.457 G_GAN_Feat: 3.768 G_VGG: 5.219 D_real: 1.211 D_fake: 0.895 
(epoch: 1, iters: 1600, time: 0.103) G_GAN: 1.007 G_GAN_Feat: 2.746 G_VGG: 5.113 D_real: 0.803 D_fake: 0.456 
(epoch: 1, iters: 1800, time: 0.103) G_GAN: 0.680 G_GAN_Feat: 3.158 G_VGG: 5.310 D_real: 0.443 D_fake: 0.496 
(epoch: 1, iters: 2000, time: 0.104) G_GAN: 0.832 G_GAN_Feat: 2.346 G_VGG: 5.120 D_real: 0.593 D_fake: 0.428 
saving the latest model (epoch 1, total_steps 2000)


(epoch: 145, iters: 2208, time: 0.099) G_GAN: 2.019 G_GAN_Feat: 2.527 G_VGG: 0.868 D_real: 0.061 D_fake: 0.028 
(epoch: 145, iters: 2408, time: 0.095) G_GAN: 1.556 G_GAN_Feat: 2.695 G_VGG: 0.814 D_real: 0.081 D_fake: 0.094 
(epoch: 145, iters: 2608, time: 0.095) G_GAN: 1.457 G_GAN_Feat: 2.611 G_VGG: 0.825 D_real: 0.028 D_fake: 0.155 
saving the latest model (epoch 145, total_steps 430000)
(epoch: 145, iters: 2808, time: 0.100) G_GAN: 1.079 G_GAN_Feat: 2.055 G_VGG: 0.817 D_real: 0.401 D_fake: 0.236 
End of epoch 145 / 200 	 Time Taken: 324 sec
(epoch: 146, iters: 40, time: 0.099) G_GAN: 1.313 G_GAN_Feat: 2.354 G_VGG: 0.767 D_real: 0.043 D_fake: 0.155 
(epoch: 146, iters: 240, time: 0.101) G_GAN: 1.506 G_GAN_Feat: 2.763 G_VGG: 0.703 D_real: 0.032 D_fake: 0.182 
(epoch: 146, iters: 440, time: 0.109) G_GAN: 1.633 G_GAN_Feat: 2.695 G_VGG: 0.823 D_real: 0.194 D_fake: 0.050 
(epoch: 146, iters: 640, time: 0.102) G_GAN: 1.530 G_GAN_Feat: 2.188 G_VGG: 0.763 D_real: 0.068 D_fake: 0.148 
saving the latest model (epoch 146, total_steps 431000)
(epoch: 146, iters: 840, time: 0.106) G_GAN: 1.866 G_GAN_Feat: 3.215 G_VGG: 0.779 D_real: 0.040 D_fake: 0.022 
(epoch: 146, iters: 1040, time: 0.102) G_GAN: 2.090 G_GAN_Feat: 3.091 G_VGG: 0.751 D_real: 0.112 D_fake: 0.019 
(epoch: 146, iters: 1240, time: 0.108) G_GAN: 0.909 G_GAN_Feat: 2.246 G_VGG: 0.730 D_real: 0.116 D_fake: 0.569 
(epoch: 146, iters: 1440, time: 0.090) G_GAN: 1.528 G_GAN_Feat: 2.459 G_VGG: 0.812 D_real: 0.157 D_fake: 0.171 
(epoch: 146, iters: 1640, time: 0.100) G_GAN: 1.770 G_GAN_Feat: 2.686 G_VGG: 0.792 D_real: 0.074 D_fake: 0.038 
saving the latest model (epoch 146, total_steps 432000)
(epoch: 146, iters: 1840, time: 0.101) G_GAN: 1.638 G_GAN_Feat: 2.176 G_VGG: 0.780 D_real: 0.173 D_fake: 0.170 
(epoch: 146, iters: 2040, time: 0.099) G_GAN: 1.906 G_GAN_Feat: 3.058 G_VGG: 0.832 D_real: 0.038 D_fake: 0.038 
(epoch: 146, iters: 2240, time: 0.103) G_GAN: 1.244 G_GAN_Feat: 2.539 G_VGG: 0.791 D_real: 0.055 D_fake: 0.246 
(epoch: 146, iters: 2440, time: 0.094) G_GAN: 2.005 G_GAN_Feat: 2.933 G_VGG: 0.782 D_real: 0.059 D_fake: 0.018 
(epoch: 146, iters: 2640, time: 0.105) G_GAN: 2.129 G_GAN_Feat: 3.208 G_VGG: 0.813 D_real: 0.087 D_fake: 0.015 
saving the latest model (epoch 146, total_steps 433000)
(epoch: 146, iters: 2840, time: 0.108) G_GAN: 1.767 G_GAN_Feat: 2.932 G_VGG: 0.741 D_real: 0.017 D_fake: 0.053 
End of epoch 146 / 200 	 Time Taken: 326 sec
(epoch: 147, iters: 72, time: 0.095) G_GAN: 1.879 G_GAN_Feat: 3.007 G_VGG: 0.821 D_real: 0.049 D_fake: 0.054 
(epoch: 147, iters: 272, time: 0.093) G_GAN: 2.040 G_GAN_Feat: 2.603 G_VGG: 0.742 D_real: 0.184 D_fake: 0.019 
(epoch: 147, iters: 472, time: 0.101) G_GAN: 1.147 G_GAN_Feat: 2.019 G_VGG: 0.698 D_real: 0.021 D_fake: 0.603 
(epoch: 147, iters: 672, time: 0.104) G_GAN: 1.707 G_GAN_Feat: 2.872 G_VGG: 0.827 D_real: 0.040 D_fake: 0.060 
saving the latest model (epoch 147, total_steps 434000)
(epoch: 147, iters: 872, time: 0.102) G_GAN: 1.436 G_GAN_Feat: 2.739 G_VGG: 0.734 D_real: 0.113 D_fake: 0.306 
(epoch: 147, iters: 1072, time: 0.100) G_GAN: 2.124 G_GAN_Feat: 2.787 G_VGG: 0.766 D_real: 0.282 D_fake: 0.031 
(epoch: 147, iters: 1272, time: 0.107) G_GAN: 1.952 G_GAN_Feat: 2.869 G_VGG: 0.885 D_real: 0.119 D_fake: 0.105 
(epoch: 147, iters: 1472, time: 0.097) G_GAN: 1.439 G_GAN_Feat: 2.791 G_VGG: 0.697 D_real: 0.083 D_fake: 0.119 
(epoch: 147, iters: 1672, time: 0.100) G_GAN: 1.796 G_GAN_Feat: 3.036 G_VGG: 0.791 D_real: 0.019 D_fake: 0.021 
saving the latest model (epoch 147, total_steps 435000)
(epoch: 147, iters: 1872, time: 0.107) G_GAN: 1.477 G_GAN_Feat: 2.413 G_VGG: 0.803 D_real: 0.040 D_fake: 0.165 
(epoch: 147, iters: 2072, time: 0.104) G_GAN: 1.840 G_GAN_Feat: 3.116 G_VGG: 0.860 D_real: 0.064 D_fake: 0.021 
(epoch: 147, iters: 2272, time: 0.098) G_GAN: 1.585 G_GAN_Feat: 2.602 G_VGG: 0.792 D_real: 0.223 D_fake: 0.066 
(epoch: 147, iters: 2472, time: 0.095) G_GAN: 1.124 G_GAN_Feat: 1.773 G_VGG: 0.699 D_real: 0.684 D_fake: 0.233 
(epoch: 147, iters: 2672, time: 0.094) G_GAN: 1.291 G_GAN_Feat: 2.319 G_VGG: 0.783 D_real: 0.102 D_fake: 0.155 
saving the latest model (epoch 147, total_steps 436000)
(epoch: 147, iters: 2872, time: 0.102) G_GAN: 1.625 G_GAN_Feat: 2.506 G_VGG: 0.746 D_real: 0.072 D_fake: 0.068 
End of epoch 147 / 200 	 Time Taken: 322 sec
(epoch: 148, iters: 104, time: 0.098) G_GAN: 1.566 G_GAN_Feat: 2.185 G_VGG: 0.727 D_real: 0.105 D_fake: 0.099 
(epoch: 148, iters: 304, time: 0.102) G_GAN: 1.755 G_GAN_Feat: 2.321 G_VGG: 0.620 D_real: 0.143 D_fake: 0.050 
(epoch: 148, iters: 504, time: 0.104) G_GAN: 0.843 G_GAN_Feat: 2.088 G_VGG: 0.781 D_real: 0.173 D_fake: 0.361 
(epoch: 148, iters: 704, time: 0.101) G_GAN: 1.895 G_GAN_Feat: 2.754 G_VGG: 0.769 D_real: 0.042 D_fake: 0.020 
saving the latest model (epoch 148, total_steps 437000)
(epoch: 148, iters: 904, time: 0.095) G_GAN: 1.508 G_GAN_Feat: 2.219 G_VGG: 0.727 D_real: 0.042 D_fake: 0.131 
(epoch: 148, iters: 1104, time: 0.103) G_GAN: 1.404 G_GAN_Feat: 1.687 G_VGG: 0.719 D_real: 0.368 D_fake: 0.353 
(epoch: 148, iters: 1304, time: 0.107) G_GAN: 2.166 G_GAN_Feat: 3.381 G_VGG: 0.864 D_real: 0.054 D_fake: 0.026 
(epoch: 148, iters: 1504, time: 0.099) G_GAN: 1.575 G_GAN_Feat: 2.598 G_VGG: 0.657 D_real: 0.056 D_fake: 0.128 
(epoch: 148, iters: 1704, time: 0.096) G_GAN: 1.370 G_GAN_Feat: 2.165 G_VGG: 0.835 D_real: 0.536 D_fake: 0.269 
saving the latest model (epoch 148, total_steps 438000)
(epoch: 148, iters: 1904, time: 0.104) G_GAN: 1.492 G_GAN_Feat: 2.043 G_VGG: 0.797 D_real: 0.074 D_fake: 0.170 
(epoch: 148, iters: 2104, time: 0.103) G_GAN: 1.440 G_GAN_Feat: 2.520 G_VGG: 0.787 D_real: 0.082 D_fake: 0.167 
(epoch: 148, iters: 2304, time: 0.095) G_GAN: 2.412 G_GAN_Feat: 3.184 G_VGG: 0.919 D_real: 0.040 D_fake: 0.043 
(epoch: 148, iters: 2504, time: 0.094) G_GAN: 2.124 G_GAN_Feat: 2.638 G_VGG: 0.716 D_real: 0.017 D_fake: 0.025 
(epoch: 148, iters: 2704, time: 0.104) G_GAN: 1.682 G_GAN_Feat: 2.470 G_VGG: 0.759 D_real: 0.057 D_fake: 0.083 
saving the latest model (epoch 148, total_steps 439000)
(epoch: 148, iters: 2904, time: 0.096) G_GAN: 1.283 G_GAN_Feat: 2.293 G_VGG: 0.820 D_real: 0.112 D_fake: 0.277 
End of epoch 148 / 200 	 Time Taken: 323 sec
(epoch: 149, iters: 136, time: 0.100) G_GAN: 1.264 G_GAN_Feat: 1.772 G_VGG: 0.766 D_real: 0.279 D_fake: 0.257 
(epoch: 149, iters: 336, time: 0.102) G_GAN: 1.546 G_GAN_Feat: 2.227 G_VGG: 0.691 D_real: 0.201 D_fake: 0.122 
(epoch: 149, iters: 536, time: 0.100) G_GAN: 1.379 G_GAN_Feat: 2.399 G_VGG: 0.758 D_real: 0.329 D_fake: 0.122 
(epoch: 149, iters: 736, time: 0.097) G_GAN: 1.982 G_GAN_Feat: 2.872 G_VGG: 0.865 D_real: 0.049 D_fake: 0.106 
saving the latest model (epoch 149, total_steps 440000)
(epoch: 149, iters: 936, time: 0.098) G_GAN: 1.820 G_GAN_Feat: 2.303 G_VGG: 0.730 D_real: 0.018 D_fake: 0.067 
(epoch: 149, iters: 1136, time: 0.096) G_GAN: 1.180 G_GAN_Feat: 1.505 G_VGG: 0.705 D_real: 0.270 D_fake: 0.254 
(epoch: 149, iters: 1336, time: 0.095) G_GAN: 0.855 G_GAN_Feat: 2.158 G_VGG: 0.683 D_real: 0.240 D_fake: 0.326 
(epoch: 149, iters: 1536, time: 0.103) G_GAN: 1.267 G_GAN_Feat: 2.265 G_VGG: 0.796 D_real: 0.047 D_fake: 0.181 
(epoch: 149, iters: 1736, time: 0.099) G_GAN: 1.151 G_GAN_Feat: 2.444 G_VGG: 0.875 D_real: 0.167 D_fake: 0.298 
saving the latest model (epoch 149, total_steps 441000)
(epoch: 149, iters: 1936, time: 0.099) G_GAN: 1.515 G_GAN_Feat: 2.243 G_VGG: 0.787 D_real: 0.035 D_fake: 0.126 
(epoch: 149, iters: 2136, time: 0.089) G_GAN: 1.443 G_GAN_Feat: 2.627 G_VGG: 0.802 D_real: 0.043 D_fake: 0.222 
(epoch: 149, iters: 2336, time: 0.106) G_GAN: 1.155 G_GAN_Feat: 2.103 G_VGG: 0.702 D_real: 0.200 D_fake: 0.287 
(epoch: 149, iters: 2536, time: 0.099) G_GAN: 1.665 G_GAN_Feat: 2.516 G_VGG: 0.838 D_real: 0.224 D_fake: 0.149 
(epoch: 149, iters: 2736, time: 0.093) G_GAN: 1.484 G_GAN_Feat: 2.269 G_VGG: 0.806 D_real: 0.073 D_fake: 0.201 
saving the latest model (epoch 149, total_steps 442000)
(epoch: 149, iters: 2936, time: 0.102) G_GAN: 2.075 G_GAN_Feat: 2.540 G_VGG: 0.745 D_real: 0.132 D_fake: 0.025 
End of epoch 149 / 200 	 Time Taken: 323 sec
(epoch: 150, iters: 168, time: 0.104) G_GAN: 1.529 G_GAN_Feat: 2.640 G_VGG: 0.777 D_real: 0.033 D_fake: 0.082 
(epoch: 150, iters: 368, time: 0.106) G_GAN: 1.856 G_GAN_Feat: 2.940 G_VGG: 0.871 D_real: 0.020 D_fake: 0.033 
(epoch: 150, iters: 568, time: 0.094) G_GAN: 1.931 G_GAN_Feat: 3.445 G_VGG: 0.839 D_real: 0.443 D_fake: 0.025 
(epoch: 150, iters: 768, time: 0.100) G_GAN: 1.589 G_GAN_Feat: 2.579 G_VGG: 0.757 D_real: 0.019 D_fake: 0.082 
saving the latest model (epoch 150, total_steps 443000)
(epoch: 150, iters: 968, time: 0.102) G_GAN: 1.395 G_GAN_Feat: 2.162 G_VGG: 0.781 D_real: 0.222 D_fake: 0.253 
(epoch: 150, iters: 1168, time: 0.097) G_GAN: 1.599 G_GAN_Feat: 2.923 G_VGG: 0.765 D_real: 0.193 D_fake: 0.270 
(epoch: 150, iters: 1368, time: 0.097) G_GAN: 1.327 G_GAN_Feat: 2.456 G_VGG: 0.771 D_real: 0.412 D_fake: 0.284 
(epoch: 150, iters: 1568, time: 0.100) G_GAN: 1.281 G_GAN_Feat: 2.395 G_VGG: 0.731 D_real: 0.059 D_fake: 0.290 
(epoch: 150, iters: 1768, time: 0.096) G_GAN: 0.626 G_GAN_Feat: 1.895 G_VGG: 0.748 D_real: 0.043 D_fake: 0.748 
saving the latest model (epoch 150, total_steps 444000)
(epoch: 150, iters: 1968, time: 0.094) G_GAN: 1.722 G_GAN_Feat: 2.659 G_VGG: 0.896 D_real: 0.086 D_fake: 0.112 
(epoch: 150, iters: 2168, time: 0.096) G_GAN: 1.080 G_GAN_Feat: 1.921 G_VGG: 0.792 D_real: 0.207 D_fake: 0.335 
(epoch: 150, iters: 2368, time: 0.098) G_GAN: 1.243 G_GAN_Feat: 2.069 G_VGG: 0.797 D_real: 0.050 D_fake: 0.273 
(epoch: 150, iters: 2568, time: 0.109) G_GAN: 1.324 G_GAN_Feat: 2.396 G_VGG: 0.804 D_real: 0.253 D_fake: 0.248 
(epoch: 150, iters: 2768, time: 0.094) G_GAN: 1.570 G_GAN_Feat: 2.443 G_VGG: 0.887 D_real: 0.267 D_fake: 0.145 
saving the latest model (epoch 150, total_steps 445000)
(epoch: 150, iters: 2968, time: 0.109) G_GAN: 2.092 G_GAN_Feat: 2.955 G_VGG: 0.731 D_real: 0.044 D_fake: 0.065 
End of epoch 150 / 200 	 Time Taken: 325 sec
saving the model at the end of epoch 150, iters 445200
(epoch: 151, iters: 200, time: 0.103) G_GAN: 1.376 G_GAN_Feat: 2.154 G_VGG: 0.759 D_real: 0.122 D_fake: 0.181 
(epoch: 151, iters: 400, time: 0.095) G_GAN: 1.650 G_GAN_Feat: 2.736 G_VGG: 0.796 D_real: 0.049 D_fake: 0.069 
(epoch: 151, iters: 600, time: 0.099) G_GAN: 1.321 G_GAN_Feat: 2.122 G_VGG: 0.780 D_real: 0.030 D_fake: 0.271 
(epoch: 151, iters: 800, time: 0.098) G_GAN: 2.081 G_GAN_Feat: 2.919 G_VGG: 0.768 D_real: 0.129 D_fake: 0.016 
saving the latest model (epoch 151, total_steps 446000)
(epoch: 151, iters: 1000, time: 0.106) G_GAN: 1.374 G_GAN_Feat: 1.927 G_VGG: 0.577 D_real: 0.248 D_fake: 0.222 
(epoch: 151, iters: 1200, time: 0.095) G_GAN: 1.712 G_GAN_Feat: 2.217 G_VGG: 0.679 D_real: 0.021 D_fake: 0.100 
(epoch: 151, iters: 1400, time: 0.106) G_GAN: 1.669 G_GAN_Feat: 2.202 G_VGG: 0.734 D_real: 0.034 D_fake: 0.182 
(epoch: 151, iters: 1600, time: 0.095) G_GAN: 1.298 G_GAN_Feat: 2.516 G_VGG: 0.780 D_real: 0.147 D_fake: 0.374 
(epoch: 151, iters: 1800, time: 0.106) G_GAN: 1.850 G_GAN_Feat: 2.863 G_VGG: 0.888 D_real: 0.035 D_fake: 0.067 
saving the latest model (epoch 151, total_steps 447000)
(epoch: 151, iters: 2000, time: 0.103) G_GAN: 2.109 G_GAN_Feat: 2.603 G_VGG: 0.783 D_real: 0.083 D_fake: 0.058 
(epoch: 151, iters: 2200, time: 0.100) G_GAN: 1.725 G_GAN_Feat: 2.150 G_VGG: 0.673 D_real: 0.059 D_fake: 0.062 
(epoch: 151, iters: 2400, time: 0.094) G_GAN: 1.332 G_GAN_Feat: 2.764 G_VGG: 0.835 D_real: 0.034 D_fake: 0.331 
(epoch: 151, iters: 2600, time: 0.104) G_GAN: 1.964 G_GAN_Feat: 2.297 G_VGG: 0.690 D_real: 0.229 D_fake: 0.024 
(epoch: 151, iters: 2800, time: 0.106) G_GAN: 1.539 G_GAN_Feat: 2.349 G_VGG: 0.779 D_real: 0.042 D_fake: 0.153 
saving the latest model (epoch 151, total_steps 448000)
End of epoch 151 / 200 	 Time Taken: 326 sec
(epoch: 152, iters: 32, time: 0.102) G_GAN: 1.772 G_GAN_Feat: 2.791 G_VGG: 0.677 D_real: 0.039 D_fake: 0.032 
(epoch: 152, iters: 232, time: 0.098) G_GAN: 1.834 G_GAN_Feat: 2.368 G_VGG: 0.769 D_real: 0.021 D_fake: 0.048 
(epoch: 152, iters: 432, time: 0.097) G_GAN: 1.331 G_GAN_Feat: 2.341 G_VGG: 0.700 D_real: 0.033 D_fake: 0.268 
(epoch: 152, iters: 632, time: 0.102) G_GAN: 2.033 G_GAN_Feat: 2.652 G_VGG: 0.823 D_real: 0.045 D_fake: 0.020 
(epoch: 152, iters: 832, time: 0.103) G_GAN: 0.814 G_GAN_Feat: 1.739 G_VGG: 0.739 D_real: 0.307 D_fake: 0.353 
saving the latest model (epoch 152, total_steps 449000)
(epoch: 152, iters: 1032, time: 0.104) G_GAN: 0.929 G_GAN_Feat: 1.916 G_VGG: 0.753 D_real: 0.048 D_fake: 0.300 
(epoch: 152, iters: 1232, time: 0.099) G_GAN: 1.276 G_GAN_Feat: 1.911 G_VGG: 0.678 D_real: 0.045 D_fake: 0.194 
(epoch: 152, iters: 1432, time: 0.099) G_GAN: 1.634 G_GAN_Feat: 2.320 G_VGG: 0.746 D_real: 0.035 D_fake: 0.066 
(epoch: 152, iters: 1632, time: 0.103) G_GAN: 2.037 G_GAN_Feat: 2.664 G_VGG: 0.855 D_real: 0.048 D_fake: 0.055 
(epoch: 152, iters: 1832, time: 0.103) G_GAN: 1.413 G_GAN_Feat: 2.397 G_VGG: 0.909 D_real: 0.164 D_fake: 0.137 
saving the latest model (epoch 152, total_steps 450000)
(epoch: 152, iters: 2032, time: 0.105) G_GAN: 1.061 G_GAN_Feat: 2.077 G_VGG: 0.730 D_real: 0.262 D_fake: 0.276 
(epoch: 152, iters: 2232, time: 0.100) G_GAN: 1.623 G_GAN_Feat: 2.551 G_VGG: 0.788 D_real: 0.036 D_fake: 0.058 
(epoch: 152, iters: 2432, time: 0.100) G_GAN: 1.828 G_GAN_Feat: 2.955 G_VGG: 0.779 D_real: 0.043 D_fake: 0.030 
(epoch: 152, iters: 2632, time: 0.102) G_GAN: 1.814 G_GAN_Feat: 2.935 G_VGG: 0.788 D_real: 0.068 D_fake: 0.081 
(epoch: 152, iters: 2832, time: 0.094) G_GAN: 1.805 G_GAN_Feat: 2.509 G_VGG: 0.742 D_real: 0.479 D_fake: 0.066 
saving the latest model (epoch 152, total_steps 451000)
End of epoch 152 / 200 	 Time Taken: 325 sec
(epoch: 153, iters: 64, time: 0.102) G_GAN: 1.557 G_GAN_Feat: 2.102 G_VGG: 0.691 D_real: 0.156 D_fake: 0.140 
(epoch: 153, iters: 264, time: 0.102) G_GAN: 2.199 G_GAN_Feat: 2.558 G_VGG: 0.657 D_real: 0.076 D_fake: 0.024 
(epoch: 153, iters: 464, time: 0.092) G_GAN: 1.135 G_GAN_Feat: 1.691 G_VGG: 0.738 D_real: 0.270 D_fake: 0.307 
(epoch: 153, iters: 664, time: 0.096) G_GAN: 1.513 G_GAN_Feat: 2.357 G_VGG: 0.852 D_real: 0.188 D_fake: 0.083 
(epoch: 153, iters: 864, time: 0.104) G_GAN: 1.283 G_GAN_Feat: 2.151 G_VGG: 0.746 D_real: 0.223 D_fake: 0.218 
saving the latest model (epoch 153, total_steps 452000)
(epoch: 153, iters: 1064, time: 0.101) G_GAN: 1.662 G_GAN_Feat: 2.596 G_VGG: 0.844 D_real: 0.043 D_fake: 0.100 
(epoch: 153, iters: 1264, time: 0.094) G_GAN: 1.411 G_GAN_Feat: 2.496 G_VGG: 0.823 D_real: 0.093 D_fake: 0.274 
(epoch: 153, iters: 1464, time: 0.108) G_GAN: 1.385 G_GAN_Feat: 2.774 G_VGG: 0.762 D_real: 0.052 D_fake: 0.190 
(epoch: 153, iters: 1664, time: 0.095) G_GAN: 1.610 G_GAN_Feat: 2.317 G_VGG: 0.694 D_real: 0.016 D_fake: 0.159 
(epoch: 153, iters: 1864, time: 0.095) G_GAN: 1.687 G_GAN_Feat: 2.669 G_VGG: 0.801 D_real: 0.093 D_fake: 0.040 
saving the latest model (epoch 153, total_steps 453000)
(epoch: 153, iters: 2064, time: 0.104) G_GAN: 1.468 G_GAN_Feat: 2.292 G_VGG: 0.692 D_real: 0.021 D_fake: 0.136 
(epoch: 153, iters: 2264, time: 0.104) G_GAN: 1.031 G_GAN_Feat: 2.250 G_VGG: 0.812 D_real: 0.367 D_fake: 0.499 
(epoch: 153, iters: 2464, time: 0.100) G_GAN: 1.055 G_GAN_Feat: 2.352 G_VGG: 0.738 D_real: 0.076 D_fake: 0.261 
(epoch: 153, iters: 2664, time: 0.111) G_GAN: 2.066 G_GAN_Feat: 3.124 G_VGG: 0.822 D_real: 0.175 D_fake: 0.022 
(epoch: 153, iters: 2864, time: 0.096) G_GAN: 2.374 G_GAN_Feat: 2.840 G_VGG: 0.719 D_real: 0.177 D_fake: 0.054 
saving the latest model (epoch 153, total_steps 454000)
End of epoch 153 / 200 	 Time Taken: 324 sec
(epoch: 154, iters: 96, time: 0.096) G_GAN: 1.991 G_GAN_Feat: 2.259 G_VGG: 0.758 D_real: 0.249 D_fake: 0.017 
(epoch: 154, iters: 296, time: 0.096) G_GAN: 1.058 G_GAN_Feat: 2.233 G_VGG: 0.860 D_real: 0.600 D_fake: 0.446 
(epoch: 154, iters: 496, time: 0.107) G_GAN: 1.926 G_GAN_Feat: 2.679 G_VGG: 0.905 D_real: 0.026 D_fake: 0.047 
(epoch: 154, iters: 696, time: 0.106) G_GAN: 1.165 G_GAN_Feat: 2.213 G_VGG: 0.765 D_real: 0.027 D_fake: 0.413 
(epoch: 154, iters: 896, time: 0.094) G_GAN: 1.591 G_GAN_Feat: 2.065 G_VGG: 0.754 D_real: 0.155 D_fake: 0.132 
saving the latest model (epoch 154, total_steps 455000)
(epoch: 154, iters: 1096, time: 0.098) G_GAN: 1.940 G_GAN_Feat: 3.539 G_VGG: 0.763 D_real: 0.019 D_fake: 0.014 
(epoch: 154, iters: 1296, time: 0.098) G_GAN: 1.518 G_GAN_Feat: 2.613 G_VGG: 0.909 D_real: 0.195 D_fake: 0.179 
(epoch: 154, iters: 1496, time: 0.095) G_GAN: 1.685 G_GAN_Feat: 2.915 G_VGG: 0.747 D_real: 0.029 D_fake: 0.085 
(epoch: 154, iters: 1696, time: 0.105) G_GAN: 1.281 G_GAN_Feat: 2.123 G_VGG: 0.738 D_real: 0.136 D_fake: 0.201 
(epoch: 154, iters: 1896, time: 0.109) G_GAN: 0.831 G_GAN_Feat: 2.022 G_VGG: 0.743 D_real: 0.253 D_fake: 0.478 
saving the latest model (epoch 154, total_steps 456000)
(epoch: 154, iters: 2096, time: 0.088) G_GAN: 1.889 G_GAN_Feat: 2.973 G_VGG: 0.806 D_real: 0.055 D_fake: 0.030 
(epoch: 154, iters: 2296, time: 0.108) G_GAN: 1.254 G_GAN_Feat: 1.854 G_VGG: 0.707 D_real: 0.343 D_fake: 0.320 
(epoch: 154, iters: 2496, time: 0.109) G_GAN: 1.997 G_GAN_Feat: 2.460 G_VGG: 0.656 D_real: 0.029 D_fake: 0.030 
(epoch: 154, iters: 2696, time: 0.088) G_GAN: 1.116 G_GAN_Feat: 2.000 G_VGG: 0.735 D_real: 0.127 D_fake: 0.630 
(epoch: 154, iters: 2896, time: 0.105) G_GAN: 1.895 G_GAN_Feat: 2.436 G_VGG: 0.825 D_real: 0.035 D_fake: 0.054 
saving the latest model (epoch 154, total_steps 457000)
End of epoch 154 / 200 	 Time Taken: 325 sec
(epoch: 155, iters: 128, time: 0.095) G_GAN: 1.657 G_GAN_Feat: 2.252 G_VGG: 0.758 D_real: 0.037 D_fake: 0.049 
(epoch: 155, iters: 328, time: 0.110) G_GAN: 1.489 G_GAN_Feat: 1.929 G_VGG: 0.714 D_real: 0.088 D_fake: 0.171 
(epoch: 155, iters: 528, time: 0.092) G_GAN: 2.254 G_GAN_Feat: 3.406 G_VGG: 0.730 D_real: 0.061 D_fake: 0.023 
(epoch: 155, iters: 728, time: 0.102) G_GAN: 1.321 G_GAN_Feat: 2.111 G_VGG: 0.678 D_real: 0.031 D_fake: 0.213 
(epoch: 155, iters: 928, time: 0.109) G_GAN: 1.288 G_GAN_Feat: 2.511 G_VGG: 0.803 D_real: 0.587 D_fake: 0.179 
saving the latest model (epoch 155, total_steps 458000)
(epoch: 155, iters: 1128, time: 0.094) G_GAN: 2.057 G_GAN_Feat: 2.491 G_VGG: 0.659 D_real: 0.092 D_fake: 0.020 
(epoch: 155, iters: 1328, time: 0.101) G_GAN: 1.459 G_GAN_Feat: 2.341 G_VGG: 0.709 D_real: 0.147 D_fake: 0.170 
(epoch: 155, iters: 1528, time: 0.102) G_GAN: 1.885 G_GAN_Feat: 2.668 G_VGG: 0.785 D_real: 0.049 D_fake: 0.048 
(epoch: 155, iters: 1728, time: 0.099) G_GAN: 1.803 G_GAN_Feat: 2.742 G_VGG: 0.730 D_real: 0.037 D_fake: 0.049 
(epoch: 155, iters: 1928, time: 0.106) G_GAN: 2.062 G_GAN_Feat: 2.813 G_VGG: 0.873 D_real: 0.103 D_fake: 0.067 
saving the latest model (epoch 155, total_steps 459000)
(epoch: 155, iters: 2128, time: 0.097) G_GAN: 2.444 G_GAN_Feat: 2.822 G_VGG: 0.808 D_real: 0.064 D_fake: 0.050 
(epoch: 155, iters: 2328, time: 0.101) G_GAN: 1.176 G_GAN_Feat: 2.963 G_VGG: 0.788 D_real: 0.064 D_fake: 0.211 
(epoch: 155, iters: 2528, time: 0.099) G_GAN: 1.320 G_GAN_Feat: 2.421 G_VGG: 0.741 D_real: 0.110 D_fake: 0.161 
(epoch: 155, iters: 2728, time: 0.102) G_GAN: 1.732 G_GAN_Feat: 2.406 G_VGG: 0.774 D_real: 0.072 D_fake: 0.084 
(epoch: 155, iters: 2928, time: 0.098) G_GAN: 1.889 G_GAN_Feat: 2.390 G_VGG: 0.749 D_real: 0.017 D_fake: 0.018 
saving the latest model (epoch 155, total_steps 460000)
End of epoch 155 / 200 	 Time Taken: 323 sec
(epoch: 156, iters: 160, time: 0.099) G_GAN: 1.484 G_GAN_Feat: 2.522 G_VGG: 0.830 D_real: 0.066 D_fake: 0.266 
(epoch: 156, iters: 360, time: 0.097) G_GAN: 1.696 G_GAN_Feat: 2.565 G_VGG: 0.812 D_real: 0.115 D_fake: 0.062 
(epoch: 156, iters: 560, time: 0.096) G_GAN: 1.259 G_GAN_Feat: 1.842 G_VGG: 0.743 D_real: 0.333 D_fake: 0.267 
(epoch: 156, iters: 760, time: 0.101) G_GAN: 2.057 G_GAN_Feat: 2.845 G_VGG: 0.772 D_real: 0.215 D_fake: 0.021 
(epoch: 156, iters: 960, time: 0.105) G_GAN: 0.888 G_GAN_Feat: 2.004 G_VGG: 0.672 D_real: 0.058 D_fake: 0.440 
saving the latest model (epoch 156, total_steps 461000)
(epoch: 156, iters: 1160, time: 0.100) G_GAN: 1.689 G_GAN_Feat: 2.729 G_VGG: 0.759 D_real: 0.043 D_fake: 0.135 
(epoch: 156, iters: 1360, time: 0.094) G_GAN: 2.062 G_GAN_Feat: 2.925 G_VGG: 0.801 D_real: 0.043 D_fake: 0.024 
(epoch: 156, iters: 1560, time: 0.103) G_GAN: 1.374 G_GAN_Feat: 2.694 G_VGG: 0.741 D_real: 0.039 D_fake: 0.193 
(epoch: 156, iters: 1760, time: 0.098) G_GAN: 2.290 G_GAN_Feat: 2.602 G_VGG: 0.825 D_real: 0.248 D_fake: 0.032 
(epoch: 156, iters: 1960, time: 0.099) G_GAN: 1.200 G_GAN_Feat: 2.251 G_VGG: 0.695 D_real: 0.385 D_fake: 0.314 
saving the latest model (epoch 156, total_steps 462000)
(epoch: 156, iters: 2160, time: 0.098) G_GAN: 1.060 G_GAN_Feat: 2.129 G_VGG: 0.786 D_real: 0.137 D_fake: 0.301 
(epoch: 156, iters: 2360, time: 0.109) G_GAN: 1.524 G_GAN_Feat: 2.071 G_VGG: 0.742 D_real: 0.303 D_fake: 0.122 
(epoch: 156, iters: 2560, time: 0.098) G_GAN: 1.642 G_GAN_Feat: 2.326 G_VGG: 0.780 D_real: 0.112 D_fake: 0.099 
(epoch: 156, iters: 2760, time: 0.093) G_GAN: 2.113 G_GAN_Feat: 3.058 G_VGG: 0.804 D_real: 0.062 D_fake: 0.020 
(epoch: 156, iters: 2960, time: 0.103) G_GAN: 1.289 G_GAN_Feat: 2.142 G_VGG: 0.802 D_real: 0.088 D_fake: 0.323 
saving the latest model (epoch 156, total_steps 463000)
End of epoch 156 / 200 	 Time Taken: 323 sec
(epoch: 157, iters: 192, time: 0.104) G_GAN: 1.466 G_GAN_Feat: 2.685 G_VGG: 0.851 D_real: 0.025 D_fake: 0.124 
(epoch: 157, iters: 392, time: 0.099) G_GAN: 1.174 G_GAN_Feat: 2.157 G_VGG: 0.770 D_real: 0.033 D_fake: 0.229 
(epoch: 157, iters: 592, time: 0.107) G_GAN: 1.739 G_GAN_Feat: 2.354 G_VGG: 0.758 D_real: 0.159 D_fake: 0.077 
(epoch: 157, iters: 792, time: 0.103) G_GAN: 2.118 G_GAN_Feat: 2.490 G_VGG: 0.782 D_real: 0.309 D_fake: 0.033 
(epoch: 157, iters: 992, time: 0.105) G_GAN: 1.580 G_GAN_Feat: 2.265 G_VGG: 0.716 D_real: 0.036 D_fake: 0.116 
saving the latest model (epoch 157, total_steps 464000)
(epoch: 157, iters: 1192, time: 0.105) G_GAN: 1.852 G_GAN_Feat: 2.711 G_VGG: 0.820 D_real: 0.139 D_fake: 0.029 
(epoch: 157, iters: 1392, time: 0.100) G_GAN: 1.862 G_GAN_Feat: 2.380 G_VGG: 0.789 D_real: 0.049 D_fake: 0.042 
(epoch: 157, iters: 1592, time: 0.100) G_GAN: 1.692 G_GAN_Feat: 2.545 G_VGG: 0.860 D_real: 0.181 D_fake: 0.074 
(epoch: 157, iters: 1792, time: 0.105) G_GAN: 1.277 G_GAN_Feat: 2.116 G_VGG: 0.697 D_real: 0.273 D_fake: 0.222 
(epoch: 157, iters: 1992, time: 0.101) G_GAN: 1.890 G_GAN_Feat: 2.331 G_VGG: 0.753 D_real: 0.196 D_fake: 0.065 
saving the latest model (epoch 157, total_steps 465000)
(epoch: 157, iters: 2192, time: 0.099) G_GAN: 1.074 G_GAN_Feat: 1.657 G_VGG: 0.735 D_real: 0.256 D_fake: 0.277 
(epoch: 157, iters: 2392, time: 0.100) G_GAN: 1.348 G_GAN_Feat: 1.856 G_VGG: 0.773 D_real: 0.236 D_fake: 0.214 
(epoch: 157, iters: 2592, time: 0.103) G_GAN: 1.521 G_GAN_Feat: 2.354 G_VGG: 0.752 D_real: 0.076 D_fake: 0.089 
(epoch: 157, iters: 2792, time: 0.098) G_GAN: 1.955 G_GAN_Feat: 2.474 G_VGG: 0.689 D_real: 0.058 D_fake: 0.073 
End of epoch 157 / 200 	 Time Taken: 321 sec
(epoch: 158, iters: 24, time: 0.105) G_GAN: 1.731 G_GAN_Feat: 2.132 G_VGG: 0.696 D_real: 0.134 D_fake: 0.085 
saving the latest model (epoch 158, total_steps 466000)
(epoch: 158, iters: 224, time: 0.101) G_GAN: 1.748 G_GAN_Feat: 3.260 G_VGG: 0.732 D_real: 0.049 D_fake: 0.066 
(epoch: 158, iters: 424, time: 0.103) G_GAN: 1.303 G_GAN_Feat: 2.349 G_VGG: 0.842 D_real: 0.022 D_fake: 0.219 
(epoch: 158, iters: 624, time: 0.099) G_GAN: 1.598 G_GAN_Feat: 2.822 G_VGG: 0.749 D_real: 0.089 D_fake: 0.138 
(epoch: 158, iters: 824, time: 0.097) G_GAN: 1.603 G_GAN_Feat: 2.521 G_VGG: 0.745 D_real: 0.033 D_fake: 0.068 
(epoch: 158, iters: 1024, time: 0.100) G_GAN: 1.329 G_GAN_Feat: 2.505 G_VGG: 0.774 D_real: 0.107 D_fake: 0.178 
saving the latest model (epoch 158, total_steps 467000)
(epoch: 158, iters: 1224, time: 0.101) G_GAN: 1.612 G_GAN_Feat: 2.433 G_VGG: 0.790 D_real: 0.024 D_fake: 0.056 
(epoch: 158, iters: 1424, time: 0.101) G_GAN: 2.043 G_GAN_Feat: 2.941 G_VGG: 0.731 D_real: 0.035 D_fake: 0.019 
(epoch: 158, iters: 1624, time: 0.096) G_GAN: 1.404 G_GAN_Feat: 2.397 G_VGG: 0.748 D_real: 0.084 D_fake: 0.224 
(epoch: 158, iters: 1824, time: 0.097) G_GAN: 2.161 G_GAN_Feat: 2.958 G_VGG: 0.740 D_real: 0.084 D_fake: 0.014 
(epoch: 158, iters: 2024, time: 0.110) G_GAN: 1.019 G_GAN_Feat: 2.607 G_VGG: 0.866 D_real: 0.110 D_fake: 0.485 
saving the latest model (epoch 158, total_steps 468000)
(epoch: 158, iters: 2224, time: 0.106) G_GAN: 1.004 G_GAN_Feat: 2.387 G_VGG: 0.828 D_real: 0.093 D_fake: 0.305 
(epoch: 158, iters: 2424, time: 0.094) G_GAN: 1.518 G_GAN_Feat: 2.261 G_VGG: 0.825 D_real: 0.270 D_fake: 0.230 
(epoch: 158, iters: 2624, time: 0.099) G_GAN: 2.097 G_GAN_Feat: 2.692 G_VGG: 0.780 D_real: 0.197 D_fake: 0.026 
(epoch: 158, iters: 2824, time: 0.097) G_GAN: 1.140 G_GAN_Feat: 1.889 G_VGG: 0.798 D_real: 0.177 D_fake: 0.305 
End of epoch 158 / 200 	 Time Taken: 324 sec
(epoch: 159, iters: 56, time: 0.110) G_GAN: 1.834 G_GAN_Feat: 2.573 G_VGG: 0.830 D_real: 0.157 D_fake: 0.051 
saving the latest model (epoch 159, total_steps 469000)
(epoch: 159, iters: 256, time: 0.099) G_GAN: 1.876 G_GAN_Feat: 2.737 G_VGG: 0.693 D_real: 0.043 D_fake: 0.026 
(epoch: 159, iters: 456, time: 0.103) G_GAN: 1.188 G_GAN_Feat: 2.112 G_VGG: 0.728 D_real: 0.038 D_fake: 0.265 
(epoch: 159, iters: 656, time: 0.106) G_GAN: 1.616 G_GAN_Feat: 2.476 G_VGG: 0.796 D_real: 0.432 D_fake: 0.101 
(epoch: 159, iters: 856, time: 0.095) G_GAN: 1.919 G_GAN_Feat: 2.970 G_VGG: 0.860 D_real: 0.059 D_fake: 0.035 
(epoch: 159, iters: 1056, time: 0.105) G_GAN: 1.520 G_GAN_Feat: 2.231 G_VGG: 0.761 D_real: 0.110 D_fake: 0.121 
saving the latest model (epoch 159, total_steps 470000)
(epoch: 159, iters: 1256, time: 0.098) G_GAN: 1.826 G_GAN_Feat: 2.367 G_VGG: 0.712 D_real: 0.134 D_fake: 0.020 
(epoch: 159, iters: 1456, time: 0.104) G_GAN: 2.158 G_GAN_Feat: 3.014 G_VGG: 0.756 D_real: 0.047 D_fake: 0.024 
(epoch: 159, iters: 1656, time: 0.104) G_GAN: 2.133 G_GAN_Feat: 2.649 G_VGG: 0.730 D_real: 0.590 D_fake: 0.028 
(epoch: 159, iters: 1856, time: 0.096) G_GAN: 1.385 G_GAN_Feat: 2.513 G_VGG: 0.819 D_real: 0.252 D_fake: 0.211 
(epoch: 159, iters: 2056, time: 0.105) G_GAN: 1.494 G_GAN_Feat: 2.756 G_VGG: 0.837 D_real: 0.284 D_fake: 0.137 
saving the latest model (epoch 159, total_steps 471000)
(epoch: 159, iters: 2256, time: 0.104) G_GAN: 1.319 G_GAN_Feat: 1.903 G_VGG: 0.682 D_real: 0.087 D_fake: 0.204 
(epoch: 159, iters: 2456, time: 0.106) G_GAN: 0.950 G_GAN_Feat: 1.933 G_VGG: 0.697 D_real: 0.100 D_fake: 0.417 
(epoch: 159, iters: 2656, time: 0.106) G_GAN: 2.069 G_GAN_Feat: 2.203 G_VGG: 0.688 D_real: 0.083 D_fake: 0.035 
(epoch: 159, iters: 2856, time: 0.103) G_GAN: 1.682 G_GAN_Feat: 2.554 G_VGG: 0.771 D_real: 0.071 D_fake: 0.157 
End of epoch 159 / 200 	 Time Taken: 323 sec
(epoch: 160, iters: 88, time: 0.109) G_GAN: 1.791 G_GAN_Feat: 2.052 G_VGG: 0.652 D_real: 0.151 D_fake: 0.069 
saving the latest model (epoch 160, total_steps 472000)
(epoch: 160, iters: 288, time: 0.097) G_GAN: 1.442 G_GAN_Feat: 2.634 G_VGG: 0.752 D_real: 0.285 D_fake: 0.198 
(epoch: 160, iters: 488, time: 0.099) G_GAN: 1.179 G_GAN_Feat: 2.034 G_VGG: 0.767 D_real: 0.119 D_fake: 0.195 
(epoch: 160, iters: 688, time: 0.101) G_GAN: 1.599 G_GAN_Feat: 2.213 G_VGG: 0.767 D_real: 0.040 D_fake: 0.077 
(epoch: 160, iters: 888, time: 0.098) G_GAN: 1.255 G_GAN_Feat: 1.883 G_VGG: 0.688 D_real: 0.263 D_fake: 0.206 
(epoch: 160, iters: 1088, time: 0.108) G_GAN: 1.844 G_GAN_Feat: 2.681 G_VGG: 0.815 D_real: 0.144 D_fake: 0.040 
saving the latest model (epoch 160, total_steps 473000)
(epoch: 160, iters: 1288, time: 0.109) G_GAN: 1.774 G_GAN_Feat: 2.483 G_VGG: 0.691 D_real: 0.075 D_fake: 0.051 
(epoch: 160, iters: 1488, time: 0.107) G_GAN: 1.290 G_GAN_Feat: 1.937 G_VGG: 0.658 D_real: 0.028 D_fake: 0.243 
(epoch: 160, iters: 1688, time: 0.102) G_GAN: 1.610 G_GAN_Feat: 1.879 G_VGG: 0.690 D_real: 0.063 D_fake: 0.093 
(epoch: 160, iters: 1888, time: 0.102) G_GAN: 1.522 G_GAN_Feat: 2.360 G_VGG: 0.861 D_real: 0.041 D_fake: 0.112 
(epoch: 160, iters: 2088, time: 0.104) G_GAN: 0.953 G_GAN_Feat: 2.165 G_VGG: 0.718 D_real: 0.090 D_fake: 0.382 
saving the latest model (epoch 160, total_steps 474000)
(epoch: 160, iters: 2288, time: 0.105) G_GAN: 0.989 G_GAN_Feat: 2.340 G_VGG: 0.764 D_real: 0.115 D_fake: 0.328 
(epoch: 160, iters: 2488, time: 0.098) G_GAN: 1.462 G_GAN_Feat: 2.174 G_VGG: 0.727 D_real: 0.225 D_fake: 0.150 
(epoch: 160, iters: 2688, time: 0.099) G_GAN: 1.241 G_GAN_Feat: 2.008 G_VGG: 0.748 D_real: 0.060 D_fake: 0.306 
(epoch: 160, iters: 2888, time: 0.102) G_GAN: 1.565 G_GAN_Feat: 2.378 G_VGG: 0.744 D_real: 0.018 D_fake: 0.113 
End of epoch 160 / 200 	 Time Taken: 326 sec
saving the model at the end of epoch 160, iters 474880
(epoch: 161, iters: 120, time: 0.101) G_GAN: 1.550 G_GAN_Feat: 2.039 G_VGG: 0.732 D_real: 0.055 D_fake: 0.159 
saving the latest model (epoch 161, total_steps 475000)
(epoch: 161, iters: 320, time: 0.098) G_GAN: 1.872 G_GAN_Feat: 2.851 G_VGG: 0.774 D_real: 0.039 D_fake: 0.027 
(epoch: 161, iters: 520, time: 0.105) G_GAN: 1.184 G_GAN_Feat: 2.106 G_VGG: 0.817 D_real: 0.028 D_fake: 0.447 
(epoch: 161, iters: 720, time: 0.106) G_GAN: 1.797 G_GAN_Feat: 2.356 G_VGG: 0.709 D_real: 0.062 D_fake: 0.059 
(epoch: 161, iters: 920, time: 0.102) G_GAN: 1.642 G_GAN_Feat: 2.463 G_VGG: 0.792 D_real: 0.036 D_fake: 0.152 
(epoch: 161, iters: 1120, time: 0.102) G_GAN: 1.244 G_GAN_Feat: 2.335 G_VGG: 0.726 D_real: 0.548 D_fake: 0.296 
saving the latest model (epoch 161, total_steps 476000)
(epoch: 161, iters: 1320, time: 0.102) G_GAN: 1.427 G_GAN_Feat: 2.086 G_VGG: 0.698 D_real: 0.121 D_fake: 0.207 
(epoch: 161, iters: 1520, time: 0.103) G_GAN: 1.556 G_GAN_Feat: 2.232 G_VGG: 0.770 D_real: 0.134 D_fake: 0.108 
(epoch: 161, iters: 1720, time: 0.101) G_GAN: 1.130 G_GAN_Feat: 1.994 G_VGG: 0.687 D_real: 0.025 D_fake: 0.409 
(epoch: 161, iters: 1920, time: 0.103) G_GAN: 1.293 G_GAN_Feat: 2.127 G_VGG: 0.805 D_real: 0.152 D_fake: 0.242 
(epoch: 161, iters: 2120, time: 0.105) G_GAN: 1.515 G_GAN_Feat: 2.354 G_VGG: 0.819 D_real: 0.020 D_fake: 0.206 
saving the latest model (epoch 161, total_steps 477000)
(epoch: 161, iters: 2320, time: 0.099) G_GAN: 1.188 G_GAN_Feat: 2.174 G_VGG: 0.746 D_real: 0.025 D_fake: 0.409 
(epoch: 161, iters: 2520, time: 0.107) G_GAN: 1.770 G_GAN_Feat: 2.584 G_VGG: 0.789 D_real: 0.016 D_fake: 0.071 
(epoch: 161, iters: 2720, time: 0.107) G_GAN: 1.895 G_GAN_Feat: 2.480 G_VGG: 0.771 D_real: 0.549 D_fake: 0.110 
(epoch: 161, iters: 2920, time: 0.106) G_GAN: 1.750 G_GAN_Feat: 2.343 G_VGG: 0.746 D_real: 0.427 D_fake: 0.088 
End of epoch 161 / 200 	 Time Taken: 324 sec
(epoch: 162, iters: 152, time: 0.108) G_GAN: 1.416 G_GAN_Feat: 2.061 G_VGG: 0.803 D_real: 0.192 D_fake: 0.214 
saving the latest model (epoch 162, total_steps 478000)
(epoch: 162, iters: 352, time: 0.102) G_GAN: 1.852 G_GAN_Feat: 2.492 G_VGG: 0.806 D_real: 0.021 D_fake: 0.033 
(epoch: 162, iters: 552, time: 0.101) G_GAN: 2.128 G_GAN_Feat: 2.987 G_VGG: 0.737 D_real: 0.228 D_fake: 0.103 
(epoch: 162, iters: 752, time: 0.095) G_GAN: 1.737 G_GAN_Feat: 2.277 G_VGG: 0.794 D_real: 0.096 D_fake: 0.054 
(epoch: 162, iters: 952, time: 0.105) G_GAN: 2.382 G_GAN_Feat: 2.720 G_VGG: 0.718 D_real: 0.029 D_fake: 0.032 
(epoch: 162, iters: 1152, time: 0.101) G_GAN: 1.641 G_GAN_Feat: 2.142 G_VGG: 0.690 D_real: 0.515 D_fake: 0.081 
saving the latest model (epoch 162, total_steps 479000)
(epoch: 162, iters: 1352, time: 0.096) G_GAN: 1.861 G_GAN_Feat: 2.590 G_VGG: 0.885 D_real: 0.049 D_fake: 0.028 
(epoch: 162, iters: 1552, time: 0.104) G_GAN: 1.813 G_GAN_Feat: 2.725 G_VGG: 0.769 D_real: 0.036 D_fake: 0.051 
(epoch: 162, iters: 1752, time: 0.103) G_GAN: 1.223 G_GAN_Feat: 1.946 G_VGG: 0.664 D_real: 0.349 D_fake: 0.290 
(epoch: 162, iters: 1952, time: 0.106) G_GAN: 1.626 G_GAN_Feat: 2.201 G_VGG: 0.743 D_real: 0.210 D_fake: 0.127 
(epoch: 162, iters: 2152, time: 0.099) G_GAN: 1.729 G_GAN_Feat: 2.525 G_VGG: 0.707 D_real: 0.067 D_fake: 0.053 
saving the latest model (epoch 162, total_steps 480000)
(epoch: 162, iters: 2352, time: 0.098) G_GAN: 1.853 G_GAN_Feat: 2.313 G_VGG: 0.707 D_real: 0.089 D_fake: 0.126 
(epoch: 162, iters: 2552, time: 0.101) G_GAN: 1.772 G_GAN_Feat: 2.588 G_VGG: 0.655 D_real: 0.032 D_fake: 0.041 
(epoch: 162, iters: 2752, time: 0.104) G_GAN: 1.513 G_GAN_Feat: 2.304 G_VGG: 0.672 D_real: 0.204 D_fake: 0.127 
(epoch: 162, iters: 2952, time: 0.099) G_GAN: 1.478 G_GAN_Feat: 2.301 G_VGG: 0.800 D_real: 0.026 D_fake: 0.172 
End of epoch 162 / 200 	 Time Taken: 323 sec
(epoch: 163, iters: 184, time: 0.098) G_GAN: 1.852 G_GAN_Feat: 2.160 G_VGG: 0.726 D_real: 0.198 D_fake: 0.034 
saving the latest model (epoch 163, total_steps 481000)
(epoch: 163, iters: 384, time: 0.107) G_GAN: 1.417 G_GAN_Feat: 2.769 G_VGG: 0.873 D_real: 0.067 D_fake: 0.251 
(epoch: 163, iters: 584, time: 0.105) G_GAN: 0.975 G_GAN_Feat: 2.166 G_VGG: 0.746 D_real: 0.106 D_fake: 0.329 
(epoch: 163, iters: 784, time: 0.103) G_GAN: 1.700 G_GAN_Feat: 2.553 G_VGG: 0.811 D_real: 0.029 D_fake: 0.038 
(epoch: 163, iters: 984, time: 0.096) G_GAN: 1.866 G_GAN_Feat: 2.840 G_VGG: 0.870 D_real: 0.350 D_fake: 0.029 
(epoch: 163, iters: 1184, time: 0.096) G_GAN: 1.677 G_GAN_Feat: 2.095 G_VGG: 0.729 D_real: 0.155 D_fake: 0.152 
saving the latest model (epoch 163, total_steps 482000)
(epoch: 163, iters: 1384, time: 0.095) G_GAN: 1.515 G_GAN_Feat: 2.328 G_VGG: 0.689 D_real: 0.022 D_fake: 0.213 
(epoch: 163, iters: 1584, time: 0.100) G_GAN: 1.325 G_GAN_Feat: 2.093 G_VGG: 0.667 D_real: 0.034 D_fake: 0.199 
(epoch: 163, iters: 1784, time: 0.097) G_GAN: 1.315 G_GAN_Feat: 1.907 G_VGG: 0.696 D_real: 0.626 D_fake: 0.255 
(epoch: 163, iters: 1984, time: 0.102) G_GAN: 1.411 G_GAN_Feat: 2.326 G_VGG: 0.791 D_real: 0.020 D_fake: 0.142 
(epoch: 163, iters: 2184, time: 0.107) G_GAN: 1.283 G_GAN_Feat: 2.270 G_VGG: 0.812 D_real: 0.135 D_fake: 0.313 
saving the latest model (epoch 163, total_steps 483000)
(epoch: 163, iters: 2384, time: 0.103) G_GAN: 1.810 G_GAN_Feat: 2.771 G_VGG: 0.767 D_real: 0.019 D_fake: 0.032 
(epoch: 163, iters: 2584, time: 0.094) G_GAN: 1.089 G_GAN_Feat: 1.707 G_VGG: 0.652 D_real: 0.036 D_fake: 0.400 
(epoch: 163, iters: 2784, time: 0.100) G_GAN: 1.418 G_GAN_Feat: 2.487 G_VGG: 0.774 D_real: 0.027 D_fake: 0.156 
End of epoch 163 / 200 	 Time Taken: 326 sec
(epoch: 164, iters: 16, time: 0.107) G_GAN: 1.514 G_GAN_Feat: 2.819 G_VGG: 0.594 D_real: 0.078 D_fake: 0.216 
(epoch: 164, iters: 216, time: 0.092) G_GAN: 1.686 G_GAN_Feat: 2.128 G_VGG: 0.679 D_real: 0.183 D_fake: 0.082 
saving the latest model (epoch 164, total_steps 484000)
(epoch: 164, iters: 416, time: 0.101) G_GAN: 1.692 G_GAN_Feat: 2.622 G_VGG: 0.877 D_real: 0.219 D_fake: 0.113 
(epoch: 164, iters: 616, time: 0.107) G_GAN: 1.882 G_GAN_Feat: 2.685 G_VGG: 0.777 D_real: 0.044 D_fake: 0.066 
(epoch: 164, iters: 816, time: 0.099) G_GAN: 1.693 G_GAN_Feat: 2.738 G_VGG: 0.738 D_real: 0.030 D_fake: 0.041 
(epoch: 164, iters: 1016, time: 0.102) G_GAN: 2.048 G_GAN_Feat: 2.763 G_VGG: 0.776 D_real: 0.131 D_fake: 0.026 
(epoch: 164, iters: 1216, time: 0.110) G_GAN: 1.898 G_GAN_Feat: 2.669 G_VGG: 0.668 D_real: 0.019 D_fake: 0.044 
saving the latest model (epoch 164, total_steps 485000)
(epoch: 164, iters: 1416, time: 0.101) G_GAN: 1.774 G_GAN_Feat: 2.342 G_VGG: 0.691 D_real: 0.116 D_fake: 0.072 
(epoch: 164, iters: 1616, time: 0.100) G_GAN: 1.506 G_GAN_Feat: 2.141 G_VGG: 0.701 D_real: 0.115 D_fake: 0.131 
(epoch: 164, iters: 1816, time: 0.105) G_GAN: 1.382 G_GAN_Feat: 2.164 G_VGG: 0.675 D_real: 0.092 D_fake: 0.280 
(epoch: 164, iters: 2016, time: 0.108) G_GAN: 2.252 G_GAN_Feat: 2.849 G_VGG: 0.766 D_real: 0.200 D_fake: 0.028 
(epoch: 164, iters: 2216, time: 0.098) G_GAN: 1.895 G_GAN_Feat: 3.497 G_VGG: 0.923 D_real: 0.021 D_fake: 0.016 
saving the latest model (epoch 164, total_steps 486000)
(epoch: 164, iters: 2416, time: 0.100) G_GAN: 1.844 G_GAN_Feat: 2.636 G_VGG: 0.716 D_real: 0.045 D_fake: 0.080 
(epoch: 164, iters: 2616, time: 0.104) G_GAN: 1.740 G_GAN_Feat: 2.314 G_VGG: 0.738 D_real: 0.153 D_fake: 0.074 
(epoch: 164, iters: 2816, time: 0.109) G_GAN: 1.680 G_GAN_Feat: 2.552 G_VGG: 0.708 D_real: 0.113 D_fake: 0.092 
End of epoch 164 / 200 	 Time Taken: 326 sec
(epoch: 165, iters: 48, time: 0.105) G_GAN: 1.633 G_GAN_Feat: 2.048 G_VGG: 0.707 D_real: 0.323 D_fake: 0.092 
(epoch: 165, iters: 248, time: 0.102) G_GAN: 1.178 G_GAN_Feat: 2.196 G_VGG: 0.708 D_real: 0.065 D_fake: 0.273 
saving the latest model (epoch 165, total_steps 487000)
(epoch: 165, iters: 448, time: 0.094) G_GAN: 1.596 G_GAN_Feat: 2.174 G_VGG: 0.660 D_real: 0.031 D_fake: 0.080 
(epoch: 165, iters: 648, time: 0.105) G_GAN: 2.491 G_GAN_Feat: 2.754 G_VGG: 0.785 D_real: 0.199 D_fake: 0.043 
(epoch: 165, iters: 848, time: 0.091) G_GAN: 1.829 G_GAN_Feat: 2.483 G_VGG: 0.590 D_real: 0.157 D_fake: 0.058 
(epoch: 165, iters: 1048, time: 0.105) G_GAN: 1.591 G_GAN_Feat: 2.199 G_VGG: 0.724 D_real: 0.164 D_fake: 0.119 
(epoch: 165, iters: 1248, time: 0.098) G_GAN: 1.267 G_GAN_Feat: 2.306 G_VGG: 0.770 D_real: 0.036 D_fake: 0.196 
saving the latest model (epoch 165, total_steps 488000)
(epoch: 165, iters: 1448, time: 0.102) G_GAN: 1.328 G_GAN_Feat: 1.944 G_VGG: 0.709 D_real: 0.129 D_fake: 0.176 
(epoch: 165, iters: 1648, time: 0.101) G_GAN: 1.898 G_GAN_Feat: 2.361 G_VGG: 0.732 D_real: 0.092 D_fake: 0.020 
(epoch: 165, iters: 1848, time: 0.104) G_GAN: 1.891 G_GAN_Feat: 2.106 G_VGG: 0.681 D_real: 0.066 D_fake: 0.035 
(epoch: 165, iters: 2048, time: 0.101) G_GAN: 1.541 G_GAN_Feat: 2.215 G_VGG: 0.820 D_real: 0.015 D_fake: 0.131 
(epoch: 165, iters: 2248, time: 0.111) G_GAN: 1.956 G_GAN_Feat: 2.600 G_VGG: 0.754 D_real: 0.219 D_fake: 0.025 
saving the latest model (epoch 165, total_steps 489000)
(epoch: 165, iters: 2448, time: 0.093) G_GAN: 1.775 G_GAN_Feat: 2.531 G_VGG: 0.775 D_real: 0.016 D_fake: 0.091 
(epoch: 165, iters: 2648, time: 0.097) G_GAN: 1.387 G_GAN_Feat: 2.188 G_VGG: 0.808 D_real: 0.057 D_fake: 0.231 
(epoch: 165, iters: 2848, time: 0.105) G_GAN: 2.218 G_GAN_Feat: 2.416 G_VGG: 0.735 D_real: 0.025 D_fake: 0.027 
End of epoch 165 / 200 	 Time Taken: 327 sec
(epoch: 166, iters: 80, time: 0.106) G_GAN: 1.429 G_GAN_Feat: 2.120 G_VGG: 0.802 D_real: 0.130 D_fake: 0.135 
(epoch: 166, iters: 280, time: 0.102) G_GAN: 2.175 G_GAN_Feat: 2.759 G_VGG: 0.614 D_real: 0.048 D_fake: 0.022 
saving the latest model (epoch 166, total_steps 490000)
(epoch: 166, iters: 480, time: 0.102) G_GAN: 1.749 G_GAN_Feat: 2.163 G_VGG: 0.759 D_real: 0.302 D_fake: 0.065 
(epoch: 166, iters: 680, time: 0.104) G_GAN: 1.558 G_GAN_Feat: 2.512 G_VGG: 0.813 D_real: 0.025 D_fake: 0.140 
(epoch: 166, iters: 880, time: 0.100) G_GAN: 1.955 G_GAN_Feat: 2.378 G_VGG: 0.746 D_real: 0.036 D_fake: 0.040 
(epoch: 166, iters: 1080, time: 0.096) G_GAN: 2.009 G_GAN_Feat: 2.361 G_VGG: 0.742 D_real: 0.089 D_fake: 0.036 
(epoch: 166, iters: 1280, time: 0.095) G_GAN: 1.138 G_GAN_Feat: 2.128 G_VGG: 0.753 D_real: 0.058 D_fake: 0.278 
saving the latest model (epoch 166, total_steps 491000)
(epoch: 166, iters: 1480, time: 0.101) G_GAN: 1.429 G_GAN_Feat: 2.993 G_VGG: 0.819 D_real: 0.024 D_fake: 0.234 
(epoch: 166, iters: 1680, time: 0.094) G_GAN: 1.675 G_GAN_Feat: 2.387 G_VGG: 0.661 D_real: 0.063 D_fake: 0.081 
(epoch: 166, iters: 1880, time: 0.093) G_GAN: 1.755 G_GAN_Feat: 2.471 G_VGG: 0.660 D_real: 0.031 D_fake: 0.050 
(epoch: 166, iters: 2080, time: 0.102) G_GAN: 1.813 G_GAN_Feat: 2.403 G_VGG: 0.779 D_real: 0.178 D_fake: 0.076 
(epoch: 166, iters: 2280, time: 0.102) G_GAN: 2.045 G_GAN_Feat: 2.332 G_VGG: 0.707 D_real: 0.152 D_fake: 0.056 
saving the latest model (epoch 166, total_steps 492000)
(epoch: 166, iters: 2480, time: 0.101) G_GAN: 1.419 G_GAN_Feat: 2.005 G_VGG: 0.847 D_real: 0.500 D_fake: 0.103 
(epoch: 166, iters: 2680, time: 0.104) G_GAN: 1.970 G_GAN_Feat: 2.707 G_VGG: 0.796 D_real: 0.042 D_fake: 0.027 
(epoch: 166, iters: 2880, time: 0.102) G_GAN: 2.011 G_GAN_Feat: 2.099 G_VGG: 0.699 D_real: 0.190 D_fake: 0.023 
End of epoch 166 / 200 	 Time Taken: 323 sec
(epoch: 167, iters: 112, time: 0.094) G_GAN: 1.702 G_GAN_Feat: 2.433 G_VGG: 0.726 D_real: 0.078 D_fake: 0.145 
(epoch: 167, iters: 312, time: 0.094) G_GAN: 1.622 G_GAN_Feat: 2.676 G_VGG: 0.778 D_real: 0.020 D_fake: 0.070 
saving the latest model (epoch 167, total_steps 493000)
(epoch: 167, iters: 512, time: 0.090) G_GAN: 1.347 G_GAN_Feat: 1.881 G_VGG: 0.711 D_real: 0.139 D_fake: 0.249 
(epoch: 167, iters: 712, time: 0.099) G_GAN: 2.020 G_GAN_Feat: 3.044 G_VGG: 0.801 D_real: 0.037 D_fake: 0.028 
(epoch: 167, iters: 912, time: 0.104) G_GAN: 1.522 G_GAN_Feat: 2.503 G_VGG: 0.750 D_real: 0.239 D_fake: 0.210 
(epoch: 167, iters: 1112, time: 0.094) G_GAN: 1.526 G_GAN_Feat: 2.148 G_VGG: 0.603 D_real: 0.098 D_fake: 0.174 
(epoch: 167, iters: 1312, time: 0.094) G_GAN: 1.811 G_GAN_Feat: 2.425 G_VGG: 0.760 D_real: 0.036 D_fake: 0.054 
saving the latest model (epoch 167, total_steps 494000)
(epoch: 167, iters: 1512, time: 0.103) G_GAN: 1.731 G_GAN_Feat: 2.429 G_VGG: 0.737 D_real: 0.036 D_fake: 0.069 
(epoch: 167, iters: 1712, time: 0.099) G_GAN: 1.861 G_GAN_Feat: 2.869 G_VGG: 0.809 D_real: 0.103 D_fake: 0.027 
(epoch: 167, iters: 1912, time: 0.098) G_GAN: 1.742 G_GAN_Feat: 2.490 G_VGG: 0.789 D_real: 0.365 D_fake: 0.098 
(epoch: 167, iters: 2112, time: 0.098) G_GAN: 1.616 G_GAN_Feat: 2.511 G_VGG: 0.797 D_real: 0.022 D_fake: 0.087 
(epoch: 167, iters: 2312, time: 0.098) G_GAN: 1.613 G_GAN_Feat: 2.703 G_VGG: 0.923 D_real: 0.207 D_fake: 0.094 
saving the latest model (epoch 167, total_steps 495000)
(epoch: 167, iters: 2512, time: 0.095) G_GAN: 1.648 G_GAN_Feat: 2.323 G_VGG: 0.704 D_real: 0.014 D_fake: 0.065 
(epoch: 167, iters: 2712, time: 0.102) G_GAN: 1.975 G_GAN_Feat: 3.029 G_VGG: 0.822 D_real: 0.026 D_fake: 0.018 
(epoch: 167, iters: 2912, time: 0.107) G_GAN: 1.280 G_GAN_Feat: 2.266 G_VGG: 0.799 D_real: 0.032 D_fake: 0.282 
End of epoch 167 / 200 	 Time Taken: 323 sec
(epoch: 168, iters: 144, time: 0.110) G_GAN: 1.526 G_GAN_Feat: 2.262 G_VGG: 0.737 D_real: 0.033 D_fake: 0.231 
(epoch: 168, iters: 344, time: 0.106) G_GAN: 1.595 G_GAN_Feat: 2.808 G_VGG: 0.762 D_real: 0.028 D_fake: 0.075 
saving the latest model (epoch 168, total_steps 496000)
(epoch: 168, iters: 544, time: 0.095) G_GAN: 1.861 G_GAN_Feat: 2.323 G_VGG: 0.741 D_real: 0.252 D_fake: 0.031 
(epoch: 168, iters: 744, time: 0.100) G_GAN: 1.351 G_GAN_Feat: 2.657 G_VGG: 0.801 D_real: 0.033 D_fake: 0.261 
(epoch: 168, iters: 944, time: 0.100) G_GAN: 2.142 G_GAN_Feat: 2.765 G_VGG: 0.795 D_real: 0.280 D_fake: 0.020 
(epoch: 168, iters: 1144, time: 0.094) G_GAN: 1.156 G_GAN_Feat: 2.022 G_VGG: 0.813 D_real: 0.280 D_fake: 0.379 
(epoch: 168, iters: 1344, time: 0.100) G_GAN: 1.727 G_GAN_Feat: 2.106 G_VGG: 0.613 D_real: 0.409 D_fake: 0.060 
saving the latest model (epoch 168, total_steps 497000)
(epoch: 168, iters: 1544, time: 0.101) G_GAN: 1.641 G_GAN_Feat: 2.319 G_VGG: 0.651 D_real: 0.041 D_fake: 0.102 
(epoch: 168, iters: 1744, time: 0.104) G_GAN: 1.721 G_GAN_Feat: 2.325 G_VGG: 0.781 D_real: 0.036 D_fake: 0.084 
(epoch: 168, iters: 1944, time: 0.095) G_GAN: 1.808 G_GAN_Feat: 2.341 G_VGG: 0.760 D_real: 0.142 D_fake: 0.136 
(epoch: 168, iters: 2144, time: 0.102) G_GAN: 1.595 G_GAN_Feat: 2.476 G_VGG: 0.687 D_real: 0.023 D_fake: 0.077 
(epoch: 168, iters: 2344, time: 0.099) G_GAN: 2.016 G_GAN_Feat: 2.188 G_VGG: 0.707 D_real: 0.074 D_fake: 0.022 
saving the latest model (epoch 168, total_steps 498000)
(epoch: 168, iters: 2544, time: 0.105) G_GAN: 1.424 G_GAN_Feat: 2.305 G_VGG: 0.696 D_real: 0.149 D_fake: 0.179 
(epoch: 168, iters: 2744, time: 0.089) G_GAN: 1.736 G_GAN_Feat: 2.477 G_VGG: 0.717 D_real: 0.031 D_fake: 0.046 
(epoch: 168, iters: 2944, time: 0.103) G_GAN: 1.394 G_GAN_Feat: 2.078 G_VGG: 0.680 D_real: 0.117 D_fake: 0.165 
End of epoch 168 / 200 	 Time Taken: 324 sec
(epoch: 169, iters: 176, time: 0.088) G_GAN: 1.180 G_GAN_Feat: 2.154 G_VGG: 0.681 D_real: 0.185 D_fake: 0.319 
(epoch: 169, iters: 376, time: 0.100) G_GAN: 2.088 G_GAN_Feat: 2.492 G_VGG: 0.716 D_real: 0.247 D_fake: 0.040 
saving the latest model (epoch 169, total_steps 499000)
(epoch: 169, iters: 576, time: 0.089) G_GAN: 1.244 G_GAN_Feat: 1.805 G_VGG: 0.689 D_real: 0.029 D_fake: 0.439 
(epoch: 169, iters: 776, time: 0.093) G_GAN: 1.482 G_GAN_Feat: 2.521 G_VGG: 0.762 D_real: 0.100 D_fake: 0.184 
(epoch: 169, iters: 976, time: 0.102) G_GAN: 2.082 G_GAN_Feat: 2.811 G_VGG: 0.775 D_real: 0.672 D_fake: 0.015 
(epoch: 169, iters: 1176, time: 0.098) G_GAN: 1.382 G_GAN_Feat: 2.180 G_VGG: 0.748 D_real: 0.087 D_fake: 0.223 
(epoch: 169, iters: 1376, time: 0.101) G_GAN: 1.534 G_GAN_Feat: 2.396 G_VGG: 0.699 D_real: 0.040 D_fake: 0.099 
saving the latest model (epoch 169, total_steps 500000)
(epoch: 169, iters: 1576, time: 0.100) G_GAN: 1.722 G_GAN_Feat: 2.339 G_VGG: 0.698 D_real: 0.074 D_fake: 0.057 
(epoch: 169, iters: 1776, time: 0.095) G_GAN: 1.827 G_GAN_Feat: 2.569 G_VGG: 0.796 D_real: 0.197 D_fake: 0.063 
(epoch: 169, iters: 1976, time: 0.094) G_GAN: 1.286 G_GAN_Feat: 1.799 G_VGG: 0.734 D_real: 0.258 D_fake: 0.311 
(epoch: 169, iters: 2176, time: 0.103) G_GAN: 1.867 G_GAN_Feat: 2.040 G_VGG: 0.687 D_real: 0.315 D_fake: 0.111 
(epoch: 169, iters: 2376, time: 0.098) G_GAN: 1.761 G_GAN_Feat: 2.544 G_VGG: 0.760 D_real: 0.068 D_fake: 0.103 
saving the latest model (epoch 169, total_steps 501000)
(epoch: 169, iters: 2576, time: 0.096) G_GAN: 1.871 G_GAN_Feat: 2.339 G_VGG: 0.736 D_real: 0.183 D_fake: 0.149 
(epoch: 169, iters: 2776, time: 0.103) G_GAN: 1.359 G_GAN_Feat: 2.430 G_VGG: 0.657 D_real: 0.314 D_fake: 0.184 
End of epoch 169 / 200 	 Time Taken: 325 sec
(epoch: 170, iters: 8, time: 0.113) G_GAN: 1.567 G_GAN_Feat: 2.292 G_VGG: 0.754 D_real: 0.146 D_fake: 0.134 
(epoch: 170, iters: 208, time: 0.102) G_GAN: 1.885 G_GAN_Feat: 2.737 G_VGG: 0.649 D_real: 0.093 D_fake: 0.032 
(epoch: 170, iters: 408, time: 0.103) G_GAN: 1.239 G_GAN_Feat: 2.344 G_VGG: 0.801 D_real: 0.057 D_fake: 0.293 
saving the latest model (epoch 170, total_steps 502000)
(epoch: 170, iters: 608, time: 0.098) G_GAN: 1.746 G_GAN_Feat: 2.940 G_VGG: 0.877 D_real: 0.034 D_fake: 0.099 
(epoch: 170, iters: 808, time: 0.109) G_GAN: 1.517 G_GAN_Feat: 2.379 G_VGG: 0.729 D_real: 0.050 D_fake: 0.128 
(epoch: 170, iters: 1008, time: 0.096) G_GAN: 1.554 G_GAN_Feat: 2.487 G_VGG: 0.752 D_real: 0.038 D_fake: 0.117 
(epoch: 170, iters: 1208, time: 0.100) G_GAN: 1.292 G_GAN_Feat: 2.496 G_VGG: 0.813 D_real: 0.026 D_fake: 0.234 
(epoch: 170, iters: 1408, time: 0.105) G_GAN: 2.106 G_GAN_Feat: 2.658 G_VGG: 0.769 D_real: 0.191 D_fake: 0.018 
saving the latest model (epoch 170, total_steps 503000)
(epoch: 170, iters: 1608, time: 0.105) G_GAN: 1.197 G_GAN_Feat: 2.193 G_VGG: 0.717 D_real: 0.050 D_fake: 0.253 
(epoch: 170, iters: 1808, time: 0.105) G_GAN: 1.446 G_GAN_Feat: 2.256 G_VGG: 0.690 D_real: 0.071 D_fake: 0.209 
(epoch: 170, iters: 2008, time: 0.098) G_GAN: 1.520 G_GAN_Feat: 1.738 G_VGG: 0.585 D_real: 0.361 D_fake: 0.146 
(epoch: 170, iters: 2208, time: 0.105) G_GAN: 2.346 G_GAN_Feat: 3.129 G_VGG: 0.648 D_real: 0.436 D_fake: 0.028 
(epoch: 170, iters: 2408, time: 0.104) G_GAN: 2.466 G_GAN_Feat: 2.864 G_VGG: 0.692 D_real: 0.064 D_fake: 0.056 
saving the latest model (epoch 170, total_steps 504000)
(epoch: 170, iters: 2608, time: 0.103) G_GAN: 1.248 G_GAN_Feat: 1.919 G_VGG: 0.660 D_real: 0.026 D_fake: 0.418 
(epoch: 170, iters: 2808, time: 0.109) G_GAN: 1.769 G_GAN_Feat: 2.416 G_VGG: 0.745 D_real: 0.064 D_fake: 0.048 
End of epoch 170 / 200 	 Time Taken: 324 sec
saving the model at the end of epoch 170, iters 504560
(epoch: 171, iters: 40, time: 0.104) G_GAN: 1.468 G_GAN_Feat: 2.348 G_VGG: 0.685 D_real: 0.221 D_fake: 0.162 
(epoch: 171, iters: 240, time: 0.103) G_GAN: 1.840 G_GAN_Feat: 2.352 G_VGG: 0.750 D_real: 0.064 D_fake: 0.077 
(epoch: 171, iters: 440, time: 0.106) G_GAN: 2.076 G_GAN_Feat: 2.542 G_VGG: 0.700 D_real: 0.171 D_fake: 0.031 
saving the latest model (epoch 171, total_steps 505000)
(epoch: 171, iters: 640, time: 0.096) G_GAN: 1.324 G_GAN_Feat: 2.294 G_VGG: 0.805 D_real: 0.100 D_fake: 0.269 
(epoch: 171, iters: 840, time: 0.101) G_GAN: 1.551 G_GAN_Feat: 2.381 G_VGG: 0.737 D_real: 0.247 D_fake: 0.130 
(epoch: 171, iters: 1040, time: 0.101) G_GAN: 1.907 G_GAN_Feat: 2.410 G_VGG: 0.803 D_real: 0.052 D_fake: 0.050 
(epoch: 171, iters: 1240, time: 0.097) G_GAN: 1.741 G_GAN_Feat: 2.495 G_VGG: 0.781 D_real: 0.197 D_fake: 0.042 
(epoch: 171, iters: 1440, time: 0.102) G_GAN: 1.597 G_GAN_Feat: 2.216 G_VGG: 0.758 D_real: 0.050 D_fake: 0.223 
saving the latest model (epoch 171, total_steps 506000)
(epoch: 171, iters: 1640, time: 0.100) G_GAN: 1.557 G_GAN_Feat: 2.781 G_VGG: 0.707 D_real: 0.083 D_fake: 0.084 
(epoch: 171, iters: 1840, time: 0.103) G_GAN: 1.524 G_GAN_Feat: 2.441 G_VGG: 0.745 D_real: 0.066 D_fake: 0.117 
(epoch: 171, iters: 2040, time: 0.095) G_GAN: 1.754 G_GAN_Feat: 3.520 G_VGG: 0.815 D_real: 0.031 D_fake: 0.025 
(epoch: 171, iters: 2240, time: 0.099) G_GAN: 1.500 G_GAN_Feat: 2.177 G_VGG: 0.737 D_real: 0.018 D_fake: 0.102 
(epoch: 171, iters: 2440, time: 0.097) G_GAN: 1.557 G_GAN_Feat: 2.397 G_VGG: 0.728 D_real: 0.399 D_fake: 0.178 
saving the latest model (epoch 171, total_steps 507000)
(epoch: 171, iters: 2640, time: 0.101) G_GAN: 1.264 G_GAN_Feat: 1.953 G_VGG: 0.827 D_real: 0.185 D_fake: 0.232 
(epoch: 171, iters: 2840, time: 0.103) G_GAN: 1.289 G_GAN_Feat: 1.994 G_VGG: 0.772 D_real: 0.162 D_fake: 0.220 
End of epoch 171 / 200 	 Time Taken: 325 sec
(epoch: 172, iters: 72, time: 0.105) G_GAN: 1.981 G_GAN_Feat: 2.814 G_VGG: 0.809 D_real: 0.032 D_fake: 0.015 
(epoch: 172, iters: 272, time: 0.106) G_GAN: 2.298 G_GAN_Feat: 2.785 G_VGG: 0.704 D_real: 0.770 D_fake: 0.038 
(epoch: 172, iters: 472, time: 0.097) G_GAN: 1.369 G_GAN_Feat: 2.621 G_VGG: 0.807 D_real: 0.026 D_fake: 0.291 
saving the latest model (epoch 172, total_steps 508000)
(epoch: 172, iters: 672, time: 0.103) G_GAN: 1.781 G_GAN_Feat: 2.332 G_VGG: 0.752 D_real: 0.027 D_fake: 0.049 
(epoch: 172, iters: 872, time: 0.102) G_GAN: 1.949 G_GAN_Feat: 2.666 G_VGG: 0.723 D_real: 0.066 D_fake: 0.015 
(epoch: 172, iters: 1072, time: 0.104) G_GAN: 1.867 G_GAN_Feat: 2.144 G_VGG: 0.653 D_real: 0.161 D_fake: 0.055 
(epoch: 172, iters: 1272, time: 0.097) G_GAN: 2.004 G_GAN_Feat: 2.732 G_VGG: 0.788 D_real: 0.383 D_fake: 0.042 
(epoch: 172, iters: 1472, time: 0.088) G_GAN: 1.911 G_GAN_Feat: 2.890 G_VGG: 0.788 D_real: 0.073 D_fake: 0.066 
saving the latest model (epoch 172, total_steps 509000)
(epoch: 172, iters: 1672, time: 0.106) G_GAN: 1.399 G_GAN_Feat: 2.384 G_VGG: 0.718 D_real: 0.049 D_fake: 0.165 
(epoch: 172, iters: 1872, time: 0.094) G_GAN: 1.926 G_GAN_Feat: 2.326 G_VGG: 0.716 D_real: 0.029 D_fake: 0.041 
(epoch: 172, iters: 2072, time: 0.100) G_GAN: 1.622 G_GAN_Feat: 2.408 G_VGG: 0.766 D_real: 0.056 D_fake: 0.078 
(epoch: 172, iters: 2272, time: 0.096) G_GAN: 1.251 G_GAN_Feat: 2.237 G_VGG: 0.645 D_real: 0.139 D_fake: 0.282 
(epoch: 172, iters: 2472, time: 0.104) G_GAN: 1.629 G_GAN_Feat: 2.189 G_VGG: 0.682 D_real: 0.092 D_fake: 0.072 
saving the latest model (epoch 172, total_steps 510000)
(epoch: 172, iters: 2672, time: 0.101) G_GAN: 1.423 G_GAN_Feat: 1.866 G_VGG: 0.678 D_real: 0.682 D_fake: 0.132 
(epoch: 172, iters: 2872, time: 0.097) G_GAN: 1.941 G_GAN_Feat: 2.743 G_VGG: 0.794 D_real: 0.058 D_fake: 0.133 
End of epoch 172 / 200 	 Time Taken: 324 sec
(epoch: 173, iters: 104, time: 0.093) G_GAN: 1.141 G_GAN_Feat: 2.167 G_VGG: 0.601 D_real: 0.106 D_fake: 0.258 
(epoch: 173, iters: 304, time: 0.107) G_GAN: 1.656 G_GAN_Feat: 2.430 G_VGG: 0.743 D_real: 0.423 D_fake: 0.105 
(epoch: 173, iters: 504, time: 0.103) G_GAN: 1.322 G_GAN_Feat: 2.184 G_VGG: 0.883 D_real: 0.166 D_fake: 0.241 
saving the latest model (epoch 173, total_steps 511000)
(epoch: 173, iters: 704, time: 0.095) G_GAN: 2.019 G_GAN_Feat: 2.587 G_VGG: 0.709 D_real: 0.019 D_fake: 0.016 
(epoch: 173, iters: 904, time: 0.103) G_GAN: 1.630 G_GAN_Feat: 2.282 G_VGG: 0.733 D_real: 0.034 D_fake: 0.128 
(epoch: 173, iters: 1104, time: 0.096) G_GAN: 1.606 G_GAN_Feat: 2.340 G_VGG: 0.823 D_real: 0.178 D_fake: 0.099 
(epoch: 173, iters: 1304, time: 0.102) G_GAN: 2.313 G_GAN_Feat: 2.665 G_VGG: 0.796 D_real: 0.062 D_fake: 0.032 
(epoch: 173, iters: 1504, time: 0.102) G_GAN: 1.705 G_GAN_Feat: 2.721 G_VGG: 0.837 D_real: 0.034 D_fake: 0.068 
saving the latest model (epoch 173, total_steps 512000)
(epoch: 173, iters: 1704, time: 0.103) G_GAN: 1.795 G_GAN_Feat: 2.602 G_VGG: 0.696 D_real: 0.240 D_fake: 0.056 
(epoch: 173, iters: 1904, time: 0.102) G_GAN: 1.996 G_GAN_Feat: 2.357 G_VGG: 0.787 D_real: 0.080 D_fake: 0.032 
(epoch: 173, iters: 2104, time: 0.100) G_GAN: 2.351 G_GAN_Feat: 2.959 G_VGG: 0.786 D_real: 0.265 D_fake: 0.034 
(epoch: 173, iters: 2304, time: 0.098) G_GAN: 1.368 G_GAN_Feat: 2.044 G_VGG: 0.623 D_real: 0.106 D_fake: 0.139 
(epoch: 173, iters: 2504, time: 0.108) G_GAN: 1.695 G_GAN_Feat: 2.254 G_VGG: 0.750 D_real: 0.055 D_fake: 0.057 
saving the latest model (epoch 173, total_steps 513000)
(epoch: 173, iters: 2704, time: 0.093) G_GAN: 1.297 G_GAN_Feat: 2.180 G_VGG: 0.684 D_real: 0.118 D_fake: 0.266 
(epoch: 173, iters: 2904, time: 0.109) G_GAN: 2.111 G_GAN_Feat: 2.903 G_VGG: 0.786 D_real: 0.073 D_fake: 0.036 
End of epoch 173 / 200 	 Time Taken: 325 sec
(epoch: 174, iters: 136, time: 0.097) G_GAN: 1.733 G_GAN_Feat: 2.662 G_VGG: 0.846 D_real: 0.063 D_fake: 0.060 
(epoch: 174, iters: 336, time: 0.100) G_GAN: 2.070 G_GAN_Feat: 2.653 G_VGG: 0.756 D_real: 0.021 D_fake: 0.022 
(epoch: 174, iters: 536, time: 0.106) G_GAN: 1.742 G_GAN_Feat: 2.386 G_VGG: 0.760 D_real: 0.051 D_fake: 0.030 
saving the latest model (epoch 174, total_steps 514000)
(epoch: 174, iters: 736, time: 0.102) G_GAN: 1.924 G_GAN_Feat: 2.917 G_VGG: 0.719 D_real: 0.086 D_fake: 0.036 
(epoch: 174, iters: 936, time: 0.094) G_GAN: 1.841 G_GAN_Feat: 3.414 G_VGG: 0.956 D_real: 0.030 D_fake: 0.048 
(epoch: 174, iters: 1136, time: 0.103) G_GAN: 1.961 G_GAN_Feat: 3.247 G_VGG: 0.744 D_real: 0.025 D_fake: 0.036 
(epoch: 174, iters: 1336, time: 0.105) G_GAN: 1.480 G_GAN_Feat: 2.346 G_VGG: 0.844 D_real: 0.132 D_fake: 0.130 
(epoch: 174, iters: 1536, time: 0.098) G_GAN: 1.535 G_GAN_Feat: 2.132 G_VGG: 0.691 D_real: 0.452 D_fake: 0.257 
saving the latest model (epoch 174, total_steps 515000)
(epoch: 174, iters: 1736, time: 0.093) G_GAN: 1.364 G_GAN_Feat: 2.143 G_VGG: 0.744 D_real: 0.404 D_fake: 0.268 
(epoch: 174, iters: 1936, time: 0.100) G_GAN: 1.755 G_GAN_Feat: 2.125 G_VGG: 0.729 D_real: 0.058 D_fake: 0.124 
(epoch: 174, iters: 2136, time: 0.107) G_GAN: 1.959 G_GAN_Feat: 2.716 G_VGG: 0.670 D_real: 0.115 D_fake: 0.019 
(epoch: 174, iters: 2336, time: 0.101) G_GAN: 1.584 G_GAN_Feat: 2.740 G_VGG: 0.804 D_real: 0.097 D_fake: 0.225 
(epoch: 174, iters: 2536, time: 0.107) G_GAN: 1.136 G_GAN_Feat: 1.917 G_VGG: 0.737 D_real: 0.020 D_fake: 0.635 
saving the latest model (epoch 174, total_steps 516000)
(epoch: 174, iters: 2736, time: 0.096) G_GAN: 1.310 G_GAN_Feat: 2.130 G_VGG: 0.685 D_real: 0.081 D_fake: 0.252 
(epoch: 174, iters: 2936, time: 0.104) G_GAN: 1.144 G_GAN_Feat: 2.031 G_VGG: 0.867 D_real: 0.071 D_fake: 0.441 
End of epoch 174 / 200 	 Time Taken: 325 sec
(epoch: 175, iters: 168, time: 0.099) G_GAN: 2.272 G_GAN_Feat: 2.579 G_VGG: 0.778 D_real: 0.030 D_fake: 0.032 
(epoch: 175, iters: 368, time: 0.106) G_GAN: 1.612 G_GAN_Feat: 1.975 G_VGG: 0.674 D_real: 0.054 D_fake: 0.126 
(epoch: 175, iters: 568, time: 0.102) G_GAN: 1.904 G_GAN_Feat: 2.399 G_VGG: 0.730 D_real: 0.079 D_fake: 0.042 
saving the latest model (epoch 175, total_steps 517000)
(epoch: 175, iters: 768, time: 0.104) G_GAN: 2.256 G_GAN_Feat: 3.004 G_VGG: 0.719 D_real: 0.065 D_fake: 0.028 
(epoch: 175, iters: 968, time: 0.106) G_GAN: 1.237 G_GAN_Feat: 1.995 G_VGG: 0.667 D_real: 0.022 D_fake: 0.382 
(epoch: 175, iters: 1168, time: 0.110) G_GAN: 1.842 G_GAN_Feat: 2.681 G_VGG: 0.807 D_real: 0.030 D_fake: 0.059 
(epoch: 175, iters: 1368, time: 0.096) G_GAN: 1.969 G_GAN_Feat: 2.202 G_VGG: 0.731 D_real: 0.199 D_fake: 0.038 
(epoch: 175, iters: 1568, time: 0.102) G_GAN: 1.985 G_GAN_Feat: 2.353 G_VGG: 0.670 D_real: 0.289 D_fake: 0.039 
saving the latest model (epoch 175, total_steps 518000)
(epoch: 175, iters: 1768, time: 0.108) G_GAN: 1.496 G_GAN_Feat: 2.241 G_VGG: 0.696 D_real: 0.060 D_fake: 0.131 
(epoch: 175, iters: 1968, time: 0.100) G_GAN: 1.372 G_GAN_Feat: 2.042 G_VGG: 0.763 D_real: 0.124 D_fake: 0.271 
(epoch: 175, iters: 2168, time: 0.100) G_GAN: 1.732 G_GAN_Feat: 2.461 G_VGG: 0.846 D_real: 0.188 D_fake: 0.060 
(epoch: 175, iters: 2368, time: 0.105) G_GAN: 1.714 G_GAN_Feat: 2.413 G_VGG: 0.708 D_real: 0.236 D_fake: 0.089 
(epoch: 175, iters: 2568, time: 0.105) G_GAN: 1.321 G_GAN_Feat: 2.024 G_VGG: 0.684 D_real: 0.166 D_fake: 0.243 
saving the latest model (epoch 175, total_steps 519000)
(epoch: 175, iters: 2768, time: 0.104) G_GAN: 1.661 G_GAN_Feat: 2.093 G_VGG: 0.752 D_real: 0.013 D_fake: 0.065 
(epoch: 175, iters: 2968, time: 0.109) G_GAN: 1.809 G_GAN_Feat: 2.645 G_VGG: 0.733 D_real: 0.146 D_fake: 0.035 
End of epoch 175 / 200 	 Time Taken: 324 sec
(epoch: 176, iters: 200, time: 0.104) G_GAN: 1.180 G_GAN_Feat: 2.277 G_VGG: 0.827 D_real: 0.071 D_fake: 0.313 
(epoch: 176, iters: 400, time: 0.096) G_GAN: 2.238 G_GAN_Feat: 2.805 G_VGG: 0.756 D_real: 0.380 D_fake: 0.035 
(epoch: 176, iters: 600, time: 0.103) G_GAN: 1.912 G_GAN_Feat: 2.583 G_VGG: 0.698 D_real: 0.096 D_fake: 0.032 
saving the latest model (epoch 176, total_steps 520000)
(epoch: 176, iters: 800, time: 0.103) G_GAN: 1.444 G_GAN_Feat: 2.390 G_VGG: 0.806 D_real: 0.324 D_fake: 0.175 
(epoch: 176, iters: 1000, time: 0.094) G_GAN: 1.797 G_GAN_Feat: 2.281 G_VGG: 0.735 D_real: 0.018 D_fake: 0.161 
(epoch: 176, iters: 1200, time: 0.094) G_GAN: 2.041 G_GAN_Feat: 2.655 G_VGG: 0.728 D_real: 0.115 D_fake: 0.019 
(epoch: 176, iters: 1400, time: 0.101) G_GAN: 1.456 G_GAN_Feat: 1.802 G_VGG: 0.601 D_real: 0.051 D_fake: 0.168 
(epoch: 176, iters: 1600, time: 0.098) G_GAN: 1.461 G_GAN_Feat: 2.198 G_VGG: 0.757 D_real: 0.073 D_fake: 0.184 
saving the latest model (epoch 176, total_steps 521000)
(epoch: 176, iters: 1800, time: 0.094) G_GAN: 1.752 G_GAN_Feat: 2.535 G_VGG: 0.891 D_real: 0.072 D_fake: 0.115 
(epoch: 176, iters: 2000, time: 0.096) G_GAN: 1.607 G_GAN_Feat: 2.340 G_VGG: 0.744 D_real: 0.027 D_fake: 0.098 
(epoch: 176, iters: 2200, time: 0.102) G_GAN: 2.264 G_GAN_Feat: 2.424 G_VGG: 0.603 D_real: 0.216 D_fake: 0.030 
(epoch: 176, iters: 2400, time: 0.098) G_GAN: 1.743 G_GAN_Feat: 2.625 G_VGG: 0.698 D_real: 0.216 D_fake: 0.099 
(epoch: 176, iters: 2600, time: 0.093) G_GAN: 1.696 G_GAN_Feat: 2.485 G_VGG: 0.829 D_real: 0.020 D_fake: 0.068 
saving the latest model (epoch 176, total_steps 522000)
(epoch: 176, iters: 2800, time: 0.102) G_GAN: 1.405 G_GAN_Feat: 2.097 G_VGG: 0.670 D_real: 0.438 D_fake: 0.256 
End of epoch 176 / 200 	 Time Taken: 323 sec
(epoch: 177, iters: 32, time: 0.099) G_GAN: 1.410 G_GAN_Feat: 1.930 G_VGG: 0.663 D_real: 0.070 D_fake: 0.165 
(epoch: 177, iters: 232, time: 0.102) G_GAN: 1.790 G_GAN_Feat: 2.527 G_VGG: 0.727 D_real: 0.065 D_fake: 0.072 
(epoch: 177, iters: 432, time: 0.100) G_GAN: 1.122 G_GAN_Feat: 1.877 G_VGG: 0.721 D_real: 0.099 D_fake: 0.463 
(epoch: 177, iters: 632, time: 0.108) G_GAN: 1.784 G_GAN_Feat: 2.141 G_VGG: 0.630 D_real: 0.066 D_fake: 0.040 
saving the latest model (epoch 177, total_steps 523000)
(epoch: 177, iters: 832, time: 0.098) G_GAN: 1.427 G_GAN_Feat: 2.270 G_VGG: 0.779 D_real: 0.076 D_fake: 0.170 
(epoch: 177, iters: 1032, time: 0.097) G_GAN: 1.822 G_GAN_Feat: 2.599 G_VGG: 0.785 D_real: 0.033 D_fake: 0.034 
(epoch: 177, iters: 1232, time: 0.105) G_GAN: 1.723 G_GAN_Feat: 2.522 G_VGG: 0.827 D_real: 0.019 D_fake: 0.066 
(epoch: 177, iters: 1432, time: 0.096) G_GAN: 1.738 G_GAN_Feat: 2.651 G_VGG: 0.818 D_real: 0.035 D_fake: 0.045 
(epoch: 177, iters: 1632, time: 0.094) G_GAN: 1.245 G_GAN_Feat: 2.030 G_VGG: 0.681 D_real: 0.017 D_fake: 0.345 
saving the latest model (epoch 177, total_steps 524000)
(epoch: 177, iters: 1832, time: 0.094) G_GAN: 2.091 G_GAN_Feat: 3.303 G_VGG: 0.782 D_real: 0.026 D_fake: 0.012 
(epoch: 177, iters: 2032, time: 0.105) G_GAN: 1.405 G_GAN_Feat: 1.974 G_VGG: 0.704 D_real: 0.197 D_fake: 0.198 
(epoch: 177, iters: 2232, time: 0.095) G_GAN: 1.706 G_GAN_Feat: 2.423 G_VGG: 0.773 D_real: 0.025 D_fake: 0.062 
(epoch: 177, iters: 2432, time: 0.097) G_GAN: 2.213 G_GAN_Feat: 2.663 G_VGG: 0.712 D_real: 0.119 D_fake: 0.029 
(epoch: 177, iters: 2632, time: 0.100) G_GAN: 1.023 G_GAN_Feat: 2.101 G_VGG: 0.764 D_real: 0.054 D_fake: 0.343 
saving the latest model (epoch 177, total_steps 525000)
(epoch: 177, iters: 2832, time: 0.104) G_GAN: 0.886 G_GAN_Feat: 1.674 G_VGG: 0.711 D_real: 0.056 D_fake: 0.484 
End of epoch 177 / 200 	 Time Taken: 325 sec
(epoch: 178, iters: 64, time: 0.095) G_GAN: 1.532 G_GAN_Feat: 2.287 G_VGG: 0.776 D_real: 0.058 D_fake: 0.125 
(epoch: 178, iters: 264, time: 0.099) G_GAN: 1.548 G_GAN_Feat: 2.283 G_VGG: 0.706 D_real: 0.021 D_fake: 0.102 
(epoch: 178, iters: 464, time: 0.097) G_GAN: 1.974 G_GAN_Feat: 2.603 G_VGG: 0.755 D_real: 0.036 D_fake: 0.041 
(epoch: 178, iters: 664, time: 0.102) G_GAN: 1.176 G_GAN_Feat: 2.261 G_VGG: 0.825 D_real: 0.245 D_fake: 0.337 
saving the latest model (epoch 178, total_steps 526000)
(epoch: 178, iters: 864, time: 0.101) G_GAN: 2.055 G_GAN_Feat: 2.927 G_VGG: 0.792 D_real: 0.122 D_fake: 0.017 
(epoch: 178, iters: 1064, time: 0.102) G_GAN: 1.761 G_GAN_Feat: 2.527 G_VGG: 0.753 D_real: 0.175 D_fake: 0.077 
(epoch: 178, iters: 1264, time: 0.103) G_GAN: 1.971 G_GAN_Feat: 2.601 G_VGG: 0.796 D_real: 0.323 D_fake: 0.075 
(epoch: 178, iters: 1464, time: 0.102) G_GAN: 1.913 G_GAN_Feat: 2.656 G_VGG: 0.765 D_real: 0.106 D_fake: 0.026 
(epoch: 178, iters: 1664, time: 0.102) G_GAN: 1.479 G_GAN_Feat: 2.108 G_VGG: 0.781 D_real: 0.056 D_fake: 0.248 
saving the latest model (epoch 178, total_steps 527000)
(epoch: 178, iters: 1864, time: 0.101) G_GAN: 1.189 G_GAN_Feat: 1.906 G_VGG: 0.669 D_real: 0.029 D_fake: 0.298 
(epoch: 178, iters: 2064, time: 0.103) G_GAN: 1.983 G_GAN_Feat: 2.657 G_VGG: 0.745 D_real: 0.156 D_fake: 0.026 
(epoch: 178, iters: 2264, time: 0.098) G_GAN: 1.811 G_GAN_Feat: 2.204 G_VGG: 0.693 D_real: 0.164 D_fake: 0.047 
(epoch: 178, iters: 2464, time: 0.104) G_GAN: 1.736 G_GAN_Feat: 2.098 G_VGG: 0.728 D_real: 0.091 D_fake: 0.109 
(epoch: 178, iters: 2664, time: 0.097) G_GAN: 1.687 G_GAN_Feat: 2.369 G_VGG: 0.700 D_real: 0.382 D_fake: 0.094 
saving the latest model (epoch 178, total_steps 528000)
(epoch: 178, iters: 2864, time: 0.096) G_GAN: 1.795 G_GAN_Feat: 2.218 G_VGG: 0.671 D_real: 0.053 D_fake: 0.059 
End of epoch 178 / 200 	 Time Taken: 325 sec
(epoch: 179, iters: 96, time: 0.092) G_GAN: 2.066 G_GAN_Feat: 2.972 G_VGG: 0.729 D_real: 0.036 D_fake: 0.014 
(epoch: 179, iters: 296, time: 0.099) G_GAN: 2.161 G_GAN_Feat: 2.654 G_VGG: 0.710 D_real: 0.050 D_fake: 0.037 
(epoch: 179, iters: 496, time: 0.099) G_GAN: 1.632 G_GAN_Feat: 2.463 G_VGG: 0.749 D_real: 0.049 D_fake: 0.130 
(epoch: 179, iters: 696, time: 0.102) G_GAN: 1.587 G_GAN_Feat: 2.279 G_VGG: 0.805 D_real: 0.060 D_fake: 0.175 
saving the latest model (epoch 179, total_steps 529000)
(epoch: 179, iters: 896, time: 0.103) G_GAN: 1.738 G_GAN_Feat: 2.602 G_VGG: 0.732 D_real: 0.038 D_fake: 0.032 
(epoch: 179, iters: 1096, time: 0.086) G_GAN: 1.656 G_GAN_Feat: 2.248 G_VGG: 0.682 D_real: 0.156 D_fake: 0.109 
(epoch: 179, iters: 1296, time: 0.099) G_GAN: 1.695 G_GAN_Feat: 2.531 G_VGG: 0.684 D_real: 0.049 D_fake: 0.113 
(epoch: 179, iters: 1496, time: 0.102) G_GAN: 1.738 G_GAN_Feat: 2.250 G_VGG: 0.717 D_real: 0.165 D_fake: 0.080 
(epoch: 179, iters: 1696, time: 0.103) G_GAN: 1.394 G_GAN_Feat: 2.233 G_VGG: 0.746 D_real: 0.334 D_fake: 0.265 
saving the latest model (epoch 179, total_steps 530000)
(epoch: 179, iters: 1896, time: 0.110) G_GAN: 1.533 G_GAN_Feat: 2.275 G_VGG: 0.730 D_real: 0.038 D_fake: 0.173 
(epoch: 179, iters: 2096, time: 0.096) G_GAN: 1.425 G_GAN_Feat: 2.238 G_VGG: 0.728 D_real: 0.267 D_fake: 0.198 
(epoch: 179, iters: 2296, time: 0.098) G_GAN: 2.307 G_GAN_Feat: 2.690 G_VGG: 0.702 D_real: 0.014 D_fake: 0.038 
(epoch: 179, iters: 2496, time: 0.100) G_GAN: 1.293 G_GAN_Feat: 2.155 G_VGG: 0.736 D_real: 0.237 D_fake: 0.216 
(epoch: 179, iters: 2696, time: 0.099) G_GAN: 1.347 G_GAN_Feat: 2.084 G_VGG: 0.726 D_real: 0.083 D_fake: 0.270 
saving the latest model (epoch 179, total_steps 531000)
(epoch: 179, iters: 2896, time: 0.092) G_GAN: 1.464 G_GAN_Feat: 2.111 G_VGG: 0.684 D_real: 0.136 D_fake: 0.149 
End of epoch 179 / 200 	 Time Taken: 325 sec
(epoch: 180, iters: 128, time: 0.102) G_GAN: 2.106 G_GAN_Feat: 2.714 G_VGG: 0.747 D_real: 0.018 D_fake: 0.021 
(epoch: 180, iters: 328, time: 0.099) G_GAN: 1.972 G_GAN_Feat: 3.093 G_VGG: 0.679 D_real: 0.094 D_fake: 0.028 
(epoch: 180, iters: 528, time: 0.099) G_GAN: 2.040 G_GAN_Feat: 2.264 G_VGG: 0.632 D_real: 0.018 D_fake: 0.036 
(epoch: 180, iters: 728, time: 0.111) G_GAN: 2.108 G_GAN_Feat: 2.920 G_VGG: 0.788 D_real: 0.050 D_fake: 0.013 
saving the latest model (epoch 180, total_steps 532000)
(epoch: 180, iters: 928, time: 0.102) G_GAN: 1.840 G_GAN_Feat: 2.313 G_VGG: 0.644 D_real: 0.094 D_fake: 0.027 
(epoch: 180, iters: 1128, time: 0.105) G_GAN: 1.590 G_GAN_Feat: 2.239 G_VGG: 0.736 D_real: 0.209 D_fake: 0.165 
(epoch: 180, iters: 1328, time: 0.102) G_GAN: 1.743 G_GAN_Feat: 2.054 G_VGG: 0.658 D_real: 0.252 D_fake: 0.101 
(epoch: 180, iters: 1528, time: 0.103) G_GAN: 1.810 G_GAN_Feat: 2.359 G_VGG: 0.595 D_real: 0.047 D_fake: 0.042 
(epoch: 180, iters: 1728, time: 0.102) G_GAN: 1.485 G_GAN_Feat: 2.238 G_VGG: 0.741 D_real: 0.053 D_fake: 0.123 
saving the latest model (epoch 180, total_steps 533000)
(epoch: 180, iters: 1928, time: 0.099) G_GAN: 1.687 G_GAN_Feat: 2.469 G_VGG: 0.762 D_real: 0.028 D_fake: 0.091 
(epoch: 180, iters: 2128, time: 0.103) G_GAN: 1.886 G_GAN_Feat: 2.559 G_VGG: 0.753 D_real: 0.183 D_fake: 0.032 
(epoch: 180, iters: 2328, time: 0.100) G_GAN: 2.115 G_GAN_Feat: 2.628 G_VGG: 0.790 D_real: 0.079 D_fake: 0.021 
(epoch: 180, iters: 2528, time: 0.094) G_GAN: 1.841 G_GAN_Feat: 2.496 G_VGG: 0.738 D_real: 0.031 D_fake: 0.040 
(epoch: 180, iters: 2728, time: 0.103) G_GAN: 1.958 G_GAN_Feat: 2.552 G_VGG: 0.770 D_real: 0.034 D_fake: 0.032 
saving the latest model (epoch 180, total_steps 534000)
(epoch: 180, iters: 2928, time: 0.104) G_GAN: 1.286 G_GAN_Feat: 2.178 G_VGG: 0.732 D_real: 0.025 D_fake: 0.332 
End of epoch 180 / 200 	 Time Taken: 328 sec
saving the model at the end of epoch 180, iters 534240
(epoch: 181, iters: 160, time: 0.100) G_GAN: 1.890 G_GAN_Feat: 2.368 G_VGG: 0.671 D_real: 0.052 D_fake: 0.019 
(epoch: 181, iters: 360, time: 0.100) G_GAN: 1.379 G_GAN_Feat: 2.052 G_VGG: 0.774 D_real: 0.075 D_fake: 0.248 
(epoch: 181, iters: 560, time: 0.093) G_GAN: 1.804 G_GAN_Feat: 2.301 G_VGG: 0.742 D_real: 0.038 D_fake: 0.038 
(epoch: 181, iters: 760, time: 0.106) G_GAN: 1.195 G_GAN_Feat: 2.153 G_VGG: 0.753 D_real: 0.126 D_fake: 0.410 
saving the latest model (epoch 181, total_steps 535000)
(epoch: 181, iters: 960, time: 0.098) G_GAN: 1.530 G_GAN_Feat: 1.978 G_VGG: 0.639 D_real: 0.046 D_fake: 0.165 
(epoch: 181, iters: 1160, time: 0.093) G_GAN: 1.692 G_GAN_Feat: 2.419 G_VGG: 0.780 D_real: 0.031 D_fake: 0.139 
(epoch: 181, iters: 1360, time: 0.099) G_GAN: 1.668 G_GAN_Feat: 2.651 G_VGG: 0.816 D_real: 0.299 D_fake: 0.052 
(epoch: 181, iters: 1560, time: 0.103) G_GAN: 2.264 G_GAN_Feat: 2.956 G_VGG: 0.731 D_real: 0.069 D_fake: 0.028 
(epoch: 181, iters: 1760, time: 0.099) G_GAN: 1.463 G_GAN_Feat: 2.251 G_VGG: 0.776 D_real: 0.024 D_fake: 0.196 
saving the latest model (epoch 181, total_steps 536000)
(epoch: 181, iters: 1960, time: 0.103) G_GAN: 1.648 G_GAN_Feat: 2.149 G_VGG: 0.720 D_real: 0.041 D_fake: 0.059 
(epoch: 181, iters: 2160, time: 0.102) G_GAN: 1.337 G_GAN_Feat: 2.382 G_VGG: 0.855 D_real: 0.035 D_fake: 0.269 
(epoch: 181, iters: 2360, time: 0.105) G_GAN: 1.364 G_GAN_Feat: 2.129 G_VGG: 0.682 D_real: 0.314 D_fake: 0.208 
(epoch: 181, iters: 2560, time: 0.111) G_GAN: 1.628 G_GAN_Feat: 2.487 G_VGG: 0.734 D_real: 0.147 D_fake: 0.063 
(epoch: 181, iters: 2760, time: 0.096) G_GAN: 1.560 G_GAN_Feat: 2.265 G_VGG: 0.687 D_real: 0.103 D_fake: 0.122 
saving the latest model (epoch 181, total_steps 537000)
(epoch: 181, iters: 2960, time: 0.103) G_GAN: 1.723 G_GAN_Feat: 2.284 G_VGG: 0.752 D_real: 0.027 D_fake: 0.075 
End of epoch 181 / 200 	 Time Taken: 325 sec
(epoch: 182, iters: 192, time: 0.103) G_GAN: 1.245 G_GAN_Feat: 2.228 G_VGG: 0.812 D_real: 0.204 D_fake: 0.286 
(epoch: 182, iters: 392, time: 0.103) G_GAN: 1.727 G_GAN_Feat: 2.419 G_VGG: 0.751 D_real: 0.019 D_fake: 0.158 
(epoch: 182, iters: 592, time: 0.095) G_GAN: 1.951 G_GAN_Feat: 2.463 G_VGG: 0.766 D_real: 0.039 D_fake: 0.044 
(epoch: 182, iters: 792, time: 0.095) G_GAN: 1.706 G_GAN_Feat: 2.499 G_VGG: 0.853 D_real: 0.317 D_fake: 0.101 
saving the latest model (epoch 182, total_steps 538000)
(epoch: 182, iters: 992, time: 0.098) G_GAN: 1.855 G_GAN_Feat: 2.168 G_VGG: 0.653 D_real: 0.045 D_fake: 0.055 
(epoch: 182, iters: 1192, time: 0.101) G_GAN: 1.931 G_GAN_Feat: 2.418 G_VGG: 0.706 D_real: 0.050 D_fake: 0.021 
(epoch: 182, iters: 1392, time: 0.100) G_GAN: 2.044 G_GAN_Feat: 2.710 G_VGG: 0.770 D_real: 0.037 D_fake: 0.032 
(epoch: 182, iters: 1592, time: 0.104) G_GAN: 1.837 G_GAN_Feat: 2.353 G_VGG: 0.694 D_real: 0.204 D_fake: 0.039 
(epoch: 182, iters: 1792, time: 0.101) G_GAN: 1.869 G_GAN_Feat: 2.540 G_VGG: 0.791 D_real: 0.075 D_fake: 0.043 
saving the latest model (epoch 182, total_steps 539000)
(epoch: 182, iters: 1992, time: 0.106) G_GAN: 1.807 G_GAN_Feat: 2.322 G_VGG: 0.769 D_real: 0.015 D_fake: 0.069 
(epoch: 182, iters: 2192, time: 0.100) G_GAN: 1.227 G_GAN_Feat: 1.755 G_VGG: 0.589 D_real: 0.044 D_fake: 0.362 
(epoch: 182, iters: 2392, time: 0.096) G_GAN: 1.741 G_GAN_Feat: 2.145 G_VGG: 0.610 D_real: 0.263 D_fake: 0.059 
(epoch: 182, iters: 2592, time: 0.101) G_GAN: 1.818 G_GAN_Feat: 2.753 G_VGG: 0.835 D_real: 0.035 D_fake: 0.034 
(epoch: 182, iters: 2792, time: 0.103) G_GAN: 0.811 G_GAN_Feat: 1.856 G_VGG: 0.738 D_real: 0.205 D_fake: 0.363 
saving the latest model (epoch 182, total_steps 540000)
End of epoch 182 / 200 	 Time Taken: 322 sec
(epoch: 183, iters: 24, time: 0.105) G_GAN: 1.721 G_GAN_Feat: 2.000 G_VGG: 0.739 D_real: 0.356 D_fake: 0.097 
(epoch: 183, iters: 224, time: 0.105) G_GAN: 1.533 G_GAN_Feat: 2.013 G_VGG: 0.741 D_real: 0.095 D_fake: 0.097 
(epoch: 183, iters: 424, time: 0.100) G_GAN: 2.018 G_GAN_Feat: 2.456 G_VGG: 0.691 D_real: 0.069 D_fake: 0.070 
(epoch: 183, iters: 624, time: 0.094) G_GAN: 1.831 G_GAN_Feat: 2.332 G_VGG: 0.734 D_real: 0.045 D_fake: 0.075 
(epoch: 183, iters: 824, time: 0.094) G_GAN: 1.458 G_GAN_Feat: 2.157 G_VGG: 0.672 D_real: 0.104 D_fake: 0.177 
saving the latest model (epoch 183, total_steps 541000)
(epoch: 183, iters: 1024, time: 0.104) G_GAN: 1.459 G_GAN_Feat: 1.824 G_VGG: 0.607 D_real: 0.056 D_fake: 0.197 
(epoch: 183, iters: 1224, time: 0.104) G_GAN: 1.353 G_GAN_Feat: 1.861 G_VGG: 0.595 D_real: 0.040 D_fake: 0.245 
(epoch: 183, iters: 1424, time: 0.106) G_GAN: 1.623 G_GAN_Feat: 2.137 G_VGG: 0.725 D_real: 0.224 D_fake: 0.090 
(epoch: 183, iters: 1624, time: 0.101) G_GAN: 1.580 G_GAN_Feat: 2.317 G_VGG: 0.773 D_real: 0.065 D_fake: 0.128 
(epoch: 183, iters: 1824, time: 0.089) G_GAN: 1.619 G_GAN_Feat: 1.860 G_VGG: 0.651 D_real: 0.088 D_fake: 0.164 
saving the latest model (epoch 183, total_steps 542000)
(epoch: 183, iters: 2024, time: 0.102) G_GAN: 1.873 G_GAN_Feat: 2.315 G_VGG: 0.761 D_real: 0.082 D_fake: 0.046 
(epoch: 183, iters: 2224, time: 0.105) G_GAN: 1.386 G_GAN_Feat: 2.071 G_VGG: 0.700 D_real: 0.050 D_fake: 0.245 
(epoch: 183, iters: 2424, time: 0.109) G_GAN: 1.423 G_GAN_Feat: 2.235 G_VGG: 0.773 D_real: 0.068 D_fake: 0.223 
(epoch: 183, iters: 2624, time: 0.102) G_GAN: 1.230 G_GAN_Feat: 2.193 G_VGG: 0.811 D_real: 0.123 D_fake: 0.377 
(epoch: 183, iters: 2824, time: 0.105) G_GAN: 1.887 G_GAN_Feat: 2.566 G_VGG: 0.787 D_real: 0.048 D_fake: 0.062 
saving the latest model (epoch 183, total_steps 543000)
End of epoch 183 / 200 	 Time Taken: 324 sec
(epoch: 184, iters: 56, time: 0.104) G_GAN: 2.003 G_GAN_Feat: 2.853 G_VGG: 0.705 D_real: 0.023 D_fake: 0.014 
(epoch: 184, iters: 256, time: 0.095) G_GAN: 1.681 G_GAN_Feat: 2.062 G_VGG: 0.670 D_real: 0.014 D_fake: 0.086 
(epoch: 184, iters: 456, time: 0.102) G_GAN: 1.452 G_GAN_Feat: 2.095 G_VGG: 0.658 D_real: 0.052 D_fake: 0.207 
(epoch: 184, iters: 656, time: 0.096) G_GAN: 1.899 G_GAN_Feat: 2.496 G_VGG: 0.760 D_real: 0.426 D_fake: 0.040 
(epoch: 184, iters: 856, time: 0.102) G_GAN: 1.604 G_GAN_Feat: 2.125 G_VGG: 0.750 D_real: 0.063 D_fake: 0.091 
saving the latest model (epoch 184, total_steps 544000)
(epoch: 184, iters: 1056, time: 0.101) G_GAN: 1.707 G_GAN_Feat: 2.320 G_VGG: 0.714 D_real: 0.082 D_fake: 0.060 
(epoch: 184, iters: 1256, time: 0.099) G_GAN: 1.444 G_GAN_Feat: 2.067 G_VGG: 0.783 D_real: 0.052 D_fake: 0.329 
(epoch: 184, iters: 1456, time: 0.094) G_GAN: 1.662 G_GAN_Feat: 2.519 G_VGG: 0.814 D_real: 0.296 D_fake: 0.064 
(epoch: 184, iters: 1656, time: 0.092) G_GAN: 1.092 G_GAN_Feat: 2.259 G_VGG: 0.792 D_real: 0.093 D_fake: 0.502 
(epoch: 184, iters: 1856, time: 0.098) G_GAN: 1.837 G_GAN_Feat: 2.035 G_VGG: 0.607 D_real: 0.217 D_fake: 0.047 
saving the latest model (epoch 184, total_steps 545000)
(epoch: 184, iters: 2056, time: 0.104) G_GAN: 1.926 G_GAN_Feat: 2.294 G_VGG: 0.767 D_real: 0.066 D_fake: 0.035 
(epoch: 184, iters: 2256, time: 0.101) G_GAN: 2.035 G_GAN_Feat: 2.446 G_VGG: 0.709 D_real: 0.253 D_fake: 0.020 
(epoch: 184, iters: 2456, time: 0.100) G_GAN: 1.190 G_GAN_Feat: 2.188 G_VGG: 0.713 D_real: 0.028 D_fake: 0.244 
(epoch: 184, iters: 2656, time: 0.101) G_GAN: 1.492 G_GAN_Feat: 2.309 G_VGG: 0.722 D_real: 0.050 D_fake: 0.154 
(epoch: 184, iters: 2856, time: 0.104) G_GAN: 1.228 G_GAN_Feat: 1.923 G_VGG: 0.648 D_real: 0.044 D_fake: 0.329 
saving the latest model (epoch 184, total_steps 546000)
End of epoch 184 / 200 	 Time Taken: 325 sec
(epoch: 185, iters: 88, time: 0.106) G_GAN: 1.249 G_GAN_Feat: 2.018 G_VGG: 0.715 D_real: 0.105 D_fake: 0.289 
(epoch: 185, iters: 288, time: 0.106) G_GAN: 1.741 G_GAN_Feat: 2.081 G_VGG: 0.719 D_real: 0.056 D_fake: 0.053 
(epoch: 185, iters: 488, time: 0.100) G_GAN: 1.721 G_GAN_Feat: 2.496 G_VGG: 0.823 D_real: 0.041 D_fake: 0.062 
(epoch: 185, iters: 688, time: 0.103) G_GAN: 1.588 G_GAN_Feat: 2.257 G_VGG: 0.838 D_real: 0.080 D_fake: 0.129 
(epoch: 185, iters: 888, time: 0.098) G_GAN: 2.005 G_GAN_Feat: 2.457 G_VGG: 0.663 D_real: 0.114 D_fake: 0.018 
saving the latest model (epoch 185, total_steps 547000)
(epoch: 185, iters: 1088, time: 0.094) G_GAN: 1.937 G_GAN_Feat: 2.818 G_VGG: 0.764 D_real: 0.057 D_fake: 0.017 
(epoch: 185, iters: 1288, time: 0.098) G_GAN: 1.641 G_GAN_Feat: 2.312 G_VGG: 0.740 D_real: 0.047 D_fake: 0.118 
(epoch: 185, iters: 1488, time: 0.107) G_GAN: 1.625 G_GAN_Feat: 2.238 G_VGG: 0.750 D_real: 0.166 D_fake: 0.093 
(epoch: 185, iters: 1688, time: 0.101) G_GAN: 1.655 G_GAN_Feat: 2.102 G_VGG: 0.729 D_real: 0.052 D_fake: 0.090 
(epoch: 185, iters: 1888, time: 0.104) G_GAN: 1.629 G_GAN_Feat: 2.002 G_VGG: 0.727 D_real: 0.309 D_fake: 0.067 
saving the latest model (epoch 185, total_steps 548000)
(epoch: 185, iters: 2088, time: 0.101) G_GAN: 2.349 G_GAN_Feat: 2.509 G_VGG: 0.798 D_real: 0.153 D_fake: 0.036 
(epoch: 185, iters: 2288, time: 0.100) G_GAN: 1.444 G_GAN_Feat: 1.981 G_VGG: 0.638 D_real: 0.164 D_fake: 0.173 
(epoch: 185, iters: 2488, time: 0.102) G_GAN: 1.580 G_GAN_Feat: 1.884 G_VGG: 0.637 D_real: 0.161 D_fake: 0.163 
(epoch: 185, iters: 2688, time: 0.101) G_GAN: 2.021 G_GAN_Feat: 2.535 G_VGG: 0.787 D_real: 0.038 D_fake: 0.052 
(epoch: 185, iters: 2888, time: 0.100) G_GAN: 1.659 G_GAN_Feat: 2.020 G_VGG: 0.621 D_real: 0.174 D_fake: 0.124 
saving the latest model (epoch 185, total_steps 549000)
End of epoch 185 / 200 	 Time Taken: 324 sec
(epoch: 186, iters: 120, time: 0.096) G_GAN: 1.545 G_GAN_Feat: 2.075 G_VGG: 0.683 D_real: 0.042 D_fake: 0.077 
(epoch: 186, iters: 320, time: 0.105) G_GAN: 1.655 G_GAN_Feat: 2.141 G_VGG: 0.718 D_real: 0.057 D_fake: 0.071 
(epoch: 186, iters: 520, time: 0.097) G_GAN: 1.576 G_GAN_Feat: 2.155 G_VGG: 0.639 D_real: 0.021 D_fake: 0.085 
(epoch: 186, iters: 720, time: 0.106) G_GAN: 1.474 G_GAN_Feat: 1.944 G_VGG: 0.635 D_real: 0.142 D_fake: 0.131 
(epoch: 186, iters: 920, time: 0.109) G_GAN: 1.667 G_GAN_Feat: 2.175 G_VGG: 0.784 D_real: 0.069 D_fake: 0.087 
saving the latest model (epoch 186, total_steps 550000)
(epoch: 186, iters: 1120, time: 0.104) G_GAN: 1.528 G_GAN_Feat: 2.123 G_VGG: 0.740 D_real: 0.042 D_fake: 0.166 
(epoch: 186, iters: 1320, time: 0.098) G_GAN: 1.770 G_GAN_Feat: 2.459 G_VGG: 0.760 D_real: 0.261 D_fake: 0.059 
(epoch: 186, iters: 1520, time: 0.096) G_GAN: 1.738 G_GAN_Feat: 2.369 G_VGG: 0.800 D_real: 0.063 D_fake: 0.079 
(epoch: 186, iters: 1720, time: 0.093) G_GAN: 1.341 G_GAN_Feat: 2.318 G_VGG: 0.703 D_real: 0.035 D_fake: 0.340 
(epoch: 186, iters: 1920, time: 0.100) G_GAN: 2.060 G_GAN_Feat: 2.639 G_VGG: 0.714 D_real: 0.163 D_fake: 0.030 
saving the latest model (epoch 186, total_steps 551000)
(epoch: 186, iters: 2120, time: 0.107) G_GAN: 1.519 G_GAN_Feat: 2.203 G_VGG: 0.697 D_real: 0.146 D_fake: 0.165 
(epoch: 186, iters: 2320, time: 0.097) G_GAN: 1.506 G_GAN_Feat: 2.351 G_VGG: 0.644 D_real: 0.064 D_fake: 0.164 
(epoch: 186, iters: 2520, time: 0.097) G_GAN: 2.167 G_GAN_Feat: 2.722 G_VGG: 0.768 D_real: 0.201 D_fake: 0.023 
(epoch: 186, iters: 2720, time: 0.102) G_GAN: 1.952 G_GAN_Feat: 2.211 G_VGG: 0.697 D_real: 0.210 D_fake: 0.029 
(epoch: 186, iters: 2920, time: 0.107) G_GAN: 1.955 G_GAN_Feat: 2.703 G_VGG: 0.662 D_real: 0.046 D_fake: 0.019 
saving the latest model (epoch 186, total_steps 552000)
End of epoch 186 / 200 	 Time Taken: 322 sec
(epoch: 187, iters: 152, time: 0.103) G_GAN: 1.915 G_GAN_Feat: 2.295 G_VGG: 0.821 D_real: 0.185 D_fake: 0.032 
(epoch: 187, iters: 352, time: 0.096) G_GAN: 1.711 G_GAN_Feat: 2.455 G_VGG: 0.794 D_real: 0.034 D_fake: 0.072 
(epoch: 187, iters: 552, time: 0.093) G_GAN: 2.085 G_GAN_Feat: 2.223 G_VGG: 0.679 D_real: 0.065 D_fake: 0.024 
(epoch: 187, iters: 752, time: 0.101) G_GAN: 1.727 G_GAN_Feat: 2.436 G_VGG: 0.765 D_real: 0.045 D_fake: 0.050 
(epoch: 187, iters: 952, time: 0.100) G_GAN: 1.710 G_GAN_Feat: 2.222 G_VGG: 0.747 D_real: 0.074 D_fake: 0.093 
saving the latest model (epoch 187, total_steps 553000)
(epoch: 187, iters: 1152, time: 0.098) G_GAN: 1.628 G_GAN_Feat: 2.129 G_VGG: 0.664 D_real: 0.417 D_fake: 0.109 
(epoch: 187, iters: 1352, time: 0.098) G_GAN: 1.775 G_GAN_Feat: 2.241 G_VGG: 0.782 D_real: 0.027 D_fake: 0.041 
(epoch: 187, iters: 1552, time: 0.093) G_GAN: 1.604 G_GAN_Feat: 2.116 G_VGG: 0.737 D_real: 0.139 D_fake: 0.130 
(epoch: 187, iters: 1752, time: 0.100) G_GAN: 1.603 G_GAN_Feat: 2.256 G_VGG: 0.710 D_real: 0.020 D_fake: 0.083 
(epoch: 187, iters: 1952, time: 0.108) G_GAN: 1.866 G_GAN_Feat: 2.403 G_VGG: 0.655 D_real: 0.081 D_fake: 0.056 
saving the latest model (epoch 187, total_steps 554000)
(epoch: 187, iters: 2152, time: 0.108) G_GAN: 1.613 G_GAN_Feat: 2.098 G_VGG: 0.703 D_real: 0.190 D_fake: 0.100 
(epoch: 187, iters: 2352, time: 0.103) G_GAN: 1.766 G_GAN_Feat: 2.231 G_VGG: 0.726 D_real: 0.082 D_fake: 0.048 
(epoch: 187, iters: 2552, time: 0.102) G_GAN: 1.829 G_GAN_Feat: 2.227 G_VGG: 0.720 D_real: 0.072 D_fake: 0.034 
(epoch: 187, iters: 2752, time: 0.101) G_GAN: 1.581 G_GAN_Feat: 2.268 G_VGG: 0.754 D_real: 0.085 D_fake: 0.119 
(epoch: 187, iters: 2952, time: 0.108) G_GAN: 1.528 G_GAN_Feat: 2.206 G_VGG: 0.787 D_real: 0.108 D_fake: 0.116 
saving the latest model (epoch 187, total_steps 555000)
End of epoch 187 / 200 	 Time Taken: 324 sec
(epoch: 188, iters: 184, time: 0.097) G_GAN: 1.993 G_GAN_Feat: 2.191 G_VGG: 0.643 D_real: 0.200 D_fake: 0.068 
(epoch: 188, iters: 384, time: 0.101) G_GAN: 2.092 G_GAN_Feat: 2.686 G_VGG: 0.755 D_real: 0.092 D_fake: 0.022 
(epoch: 188, iters: 584, time: 0.105) G_GAN: 1.666 G_GAN_Feat: 2.560 G_VGG: 0.701 D_real: 0.078 D_fake: 0.097 
(epoch: 188, iters: 784, time: 0.102) G_GAN: 1.179 G_GAN_Feat: 1.809 G_VGG: 0.723 D_real: 0.070 D_fake: 0.240 
(epoch: 188, iters: 984, time: 0.108) G_GAN: 1.865 G_GAN_Feat: 2.654 G_VGG: 0.793 D_real: 0.017 D_fake: 0.024 
saving the latest model (epoch 188, total_steps 556000)
(epoch: 188, iters: 1184, time: 0.111) G_GAN: 1.264 G_GAN_Feat: 1.752 G_VGG: 0.684 D_real: 0.196 D_fake: 0.329 
(epoch: 188, iters: 1384, time: 0.106) G_GAN: 1.744 G_GAN_Feat: 2.275 G_VGG: 0.670 D_real: 0.201 D_fake: 0.058 
(epoch: 188, iters: 1584, time: 0.104) G_GAN: 1.922 G_GAN_Feat: 2.393 G_VGG: 0.705 D_real: 0.065 D_fake: 0.029 
(epoch: 188, iters: 1784, time: 0.095) G_GAN: 1.704 G_GAN_Feat: 2.033 G_VGG: 0.720 D_real: 0.027 D_fake: 0.139 
(epoch: 188, iters: 1984, time: 0.103) G_GAN: 1.849 G_GAN_Feat: 1.936 G_VGG: 0.613 D_real: 0.102 D_fake: 0.062 
saving the latest model (epoch 188, total_steps 557000)
(epoch: 188, iters: 2184, time: 0.105) G_GAN: 1.551 G_GAN_Feat: 2.159 G_VGG: 0.658 D_real: 0.114 D_fake: 0.115 
(epoch: 188, iters: 2384, time: 0.105) G_GAN: 1.947 G_GAN_Feat: 2.425 G_VGG: 0.667 D_real: 0.112 D_fake: 0.039 
(epoch: 188, iters: 2584, time: 0.087) G_GAN: 1.402 G_GAN_Feat: 1.951 G_VGG: 0.716 D_real: 0.052 D_fake: 0.188 
(epoch: 188, iters: 2784, time: 0.096) G_GAN: 1.626 G_GAN_Feat: 2.216 G_VGG: 0.791 D_real: 0.091 D_fake: 0.211 
End of epoch 188 / 200 	 Time Taken: 320 sec
(epoch: 189, iters: 16, time: 0.097) G_GAN: 1.764 G_GAN_Feat: 2.162 G_VGG: 0.724 D_real: 0.030 D_fake: 0.055 
saving the latest model (epoch 189, total_steps 558000)
(epoch: 189, iters: 216, time: 0.105) G_GAN: 1.354 G_GAN_Feat: 1.852 G_VGG: 0.695 D_real: 0.053 D_fake: 0.161 
(epoch: 189, iters: 416, time: 0.095) G_GAN: 1.272 G_GAN_Feat: 2.211 G_VGG: 0.727 D_real: 0.088 D_fake: 0.251 
(epoch: 189, iters: 616, time: 0.102) G_GAN: 1.422 G_GAN_Feat: 1.979 G_VGG: 0.732 D_real: 0.029 D_fake: 0.194 
(epoch: 189, iters: 816, time: 0.095) G_GAN: 1.425 G_GAN_Feat: 2.107 G_VGG: 0.758 D_real: 0.159 D_fake: 0.293 
(epoch: 189, iters: 1016, time: 0.100) G_GAN: 1.879 G_GAN_Feat: 2.354 G_VGG: 0.806 D_real: 0.059 D_fake: 0.054 
saving the latest model (epoch 189, total_steps 559000)
(epoch: 189, iters: 1216, time: 0.098) G_GAN: 1.719 G_GAN_Feat: 2.165 G_VGG: 0.741 D_real: 0.080 D_fake: 0.068 
(epoch: 189, iters: 1416, time: 0.102) G_GAN: 2.106 G_GAN_Feat: 2.275 G_VGG: 0.716 D_real: 0.058 D_fake: 0.018 
(epoch: 189, iters: 1616, time: 0.104) G_GAN: 1.711 G_GAN_Feat: 2.363 G_VGG: 0.786 D_real: 0.026 D_fake: 0.109 
(epoch: 189, iters: 1816, time: 0.093) G_GAN: 1.485 G_GAN_Feat: 2.099 G_VGG: 0.677 D_real: 0.029 D_fake: 0.138 
(epoch: 189, iters: 2016, time: 0.106) G_GAN: 2.183 G_GAN_Feat: 2.351 G_VGG: 0.753 D_real: 0.091 D_fake: 0.025 
saving the latest model (epoch 189, total_steps 560000)
(epoch: 189, iters: 2216, time: 0.094) G_GAN: 1.276 G_GAN_Feat: 2.085 G_VGG: 0.736 D_real: 0.018 D_fake: 0.315 
(epoch: 189, iters: 2416, time: 0.102) G_GAN: 1.247 G_GAN_Feat: 1.992 G_VGG: 0.689 D_real: 0.066 D_fake: 0.322 
(epoch: 189, iters: 2616, time: 0.105) G_GAN: 1.666 G_GAN_Feat: 2.068 G_VGG: 0.676 D_real: 0.030 D_fake: 0.116 
(epoch: 189, iters: 2816, time: 0.099) G_GAN: 1.507 G_GAN_Feat: 2.268 G_VGG: 0.762 D_real: 0.036 D_fake: 0.119 
End of epoch 189 / 200 	 Time Taken: 325 sec
(epoch: 190, iters: 48, time: 0.097) G_GAN: 1.329 G_GAN_Feat: 2.075 G_VGG: 0.707 D_real: 0.064 D_fake: 0.310 
saving the latest model (epoch 190, total_steps 561000)
(epoch: 190, iters: 248, time: 0.089) G_GAN: 1.633 G_GAN_Feat: 2.058 G_VGG: 0.717 D_real: 0.049 D_fake: 0.076 
(epoch: 190, iters: 448, time: 0.108) G_GAN: 1.479 G_GAN_Feat: 2.172 G_VGG: 0.834 D_real: 0.048 D_fake: 0.239 
(epoch: 190, iters: 648, time: 0.096) G_GAN: 1.522 G_GAN_Feat: 1.928 G_VGG: 0.664 D_real: 0.184 D_fake: 0.129 
(epoch: 190, iters: 848, time: 0.094) G_GAN: 1.889 G_GAN_Feat: 2.309 G_VGG: 0.758 D_real: 0.281 D_fake: 0.035 
(epoch: 190, iters: 1048, time: 0.094) G_GAN: 1.316 G_GAN_Feat: 1.811 G_VGG: 0.648 D_real: 0.054 D_fake: 0.320 
saving the latest model (epoch 190, total_steps 562000)
(epoch: 190, iters: 1248, time: 0.097) G_GAN: 1.689 G_GAN_Feat: 2.122 G_VGG: 0.738 D_real: 0.043 D_fake: 0.075 
(epoch: 190, iters: 1448, time: 0.098) G_GAN: 1.417 G_GAN_Feat: 1.866 G_VGG: 0.701 D_real: 0.131 D_fake: 0.204 
(epoch: 190, iters: 1648, time: 0.097) G_GAN: 1.390 G_GAN_Feat: 1.868 G_VGG: 0.628 D_real: 0.025 D_fake: 0.173 
(epoch: 190, iters: 1848, time: 0.095) G_GAN: 1.372 G_GAN_Feat: 2.121 G_VGG: 0.724 D_real: 0.037 D_fake: 0.214 
(epoch: 190, iters: 2048, time: 0.101) G_GAN: 1.593 G_GAN_Feat: 2.131 G_VGG: 0.793 D_real: 0.118 D_fake: 0.109 
saving the latest model (epoch 190, total_steps 563000)
(epoch: 190, iters: 2248, time: 0.096) G_GAN: 1.859 G_GAN_Feat: 2.246 G_VGG: 0.612 D_real: 0.098 D_fake: 0.029 
(epoch: 190, iters: 2448, time: 0.099) G_GAN: 1.563 G_GAN_Feat: 2.064 G_VGG: 0.709 D_real: 0.166 D_fake: 0.124 
(epoch: 190, iters: 2648, time: 0.098) G_GAN: 1.446 G_GAN_Feat: 2.346 G_VGG: 0.650 D_real: 0.048 D_fake: 0.184 
(epoch: 190, iters: 2848, time: 0.103) G_GAN: 1.880 G_GAN_Feat: 2.387 G_VGG: 0.789 D_real: 0.039 D_fake: 0.089 
End of epoch 190 / 200 	 Time Taken: 324 sec
saving the model at the end of epoch 190, iters 563920
(epoch: 191, iters: 80, time: 0.096) G_GAN: 1.869 G_GAN_Feat: 1.942 G_VGG: 0.686 D_real: 0.083 D_fake: 0.112 
saving the latest model (epoch 191, total_steps 564000)
(epoch: 191, iters: 280, time: 0.099) G_GAN: 1.158 G_GAN_Feat: 1.819 G_VGG: 0.732 D_real: 0.061 D_fake: 0.399 
(epoch: 191, iters: 480, time: 0.101) G_GAN: 1.545 G_GAN_Feat: 1.989 G_VGG: 0.707 D_real: 0.034 D_fake: 0.104 
(epoch: 191, iters: 680, time: 0.103) G_GAN: 1.934 G_GAN_Feat: 2.719 G_VGG: 0.832 D_real: 0.192 D_fake: 0.034 
(epoch: 191, iters: 880, time: 0.095) G_GAN: 1.310 G_GAN_Feat: 1.997 G_VGG: 0.760 D_real: 0.080 D_fake: 0.225 
(epoch: 191, iters: 1080, time: 0.105) G_GAN: 1.158 G_GAN_Feat: 1.716 G_VGG: 0.638 D_real: 0.189 D_fake: 0.432 
saving the latest model (epoch 191, total_steps 565000)
(epoch: 191, iters: 1280, time: 0.106) G_GAN: 1.637 G_GAN_Feat: 2.280 G_VGG: 0.750 D_real: 0.075 D_fake: 0.122 
(epoch: 191, iters: 1480, time: 0.101) G_GAN: 1.709 G_GAN_Feat: 2.102 G_VGG: 0.715 D_real: 0.041 D_fake: 0.082 
(epoch: 191, iters: 1680, time: 0.095) G_GAN: 1.351 G_GAN_Feat: 2.071 G_VGG: 0.783 D_real: 0.181 D_fake: 0.280 
(epoch: 191, iters: 1880, time: 0.101) G_GAN: 1.795 G_GAN_Feat: 1.935 G_VGG: 0.621 D_real: 0.087 D_fake: 0.075 
(epoch: 191, iters: 2080, time: 0.098) G_GAN: 2.220 G_GAN_Feat: 2.309 G_VGG: 0.667 D_real: 0.028 D_fake: 0.027 
saving the latest model (epoch 191, total_steps 566000)
(epoch: 191, iters: 2280, time: 0.103) G_GAN: 1.712 G_GAN_Feat: 2.752 G_VGG: 0.877 D_real: 0.413 D_fake: 0.075 
(epoch: 191, iters: 2480, time: 0.106) G_GAN: 1.940 G_GAN_Feat: 2.137 G_VGG: 0.710 D_real: 0.391 D_fake: 0.056 
(epoch: 191, iters: 2680, time: 0.094) G_GAN: 1.442 G_GAN_Feat: 1.922 G_VGG: 0.674 D_real: 0.070 D_fake: 0.184 
(epoch: 191, iters: 2880, time: 0.107) G_GAN: 1.412 G_GAN_Feat: 2.057 G_VGG: 0.754 D_real: 0.091 D_fake: 0.252 
End of epoch 191 / 200 	 Time Taken: 327 sec
(epoch: 192, iters: 112, time: 0.100) G_GAN: 1.632 G_GAN_Feat: 2.267 G_VGG: 0.733 D_real: 0.095 D_fake: 0.121 
saving the latest model (epoch 192, total_steps 567000)
(epoch: 192, iters: 312, time: 0.101) G_GAN: 1.362 G_GAN_Feat: 1.876 G_VGG: 0.730 D_real: 0.066 D_fake: 0.243 
(epoch: 192, iters: 512, time: 0.101) G_GAN: 1.826 G_GAN_Feat: 2.178 G_VGG: 0.727 D_real: 0.076 D_fake: 0.065 
(epoch: 192, iters: 712, time: 0.096) G_GAN: 1.920 G_GAN_Feat: 2.290 G_VGG: 0.760 D_real: 0.081 D_fake: 0.030 
(epoch: 192, iters: 912, time: 0.104) G_GAN: 1.985 G_GAN_Feat: 1.971 G_VGG: 0.648 D_real: 0.036 D_fake: 0.030 
(epoch: 192, iters: 1112, time: 0.105) G_GAN: 1.345 G_GAN_Feat: 2.092 G_VGG: 0.752 D_real: 0.047 D_fake: 0.237 
saving the latest model (epoch 192, total_steps 568000)
(epoch: 192, iters: 1312, time: 0.100) G_GAN: 1.521 G_GAN_Feat: 2.522 G_VGG: 0.798 D_real: 0.141 D_fake: 0.174 
(epoch: 192, iters: 1512, time: 0.103) G_GAN: 1.599 G_GAN_Feat: 2.051 G_VGG: 0.720 D_real: 0.117 D_fake: 0.127 
(epoch: 192, iters: 1712, time: 0.102) G_GAN: 1.351 G_GAN_Feat: 2.138 G_VGG: 0.705 D_real: 0.082 D_fake: 0.220 
(epoch: 192, iters: 1912, time: 0.097) G_GAN: 1.434 G_GAN_Feat: 2.277 G_VGG: 0.807 D_real: 0.141 D_fake: 0.180 
(epoch: 192, iters: 2112, time: 0.101) G_GAN: 1.409 G_GAN_Feat: 1.919 G_VGG: 0.777 D_real: 0.081 D_fake: 0.243 
saving the latest model (epoch 192, total_steps 569000)
(epoch: 192, iters: 2312, time: 0.093) G_GAN: 1.470 G_GAN_Feat: 2.063 G_VGG: 0.665 D_real: 0.135 D_fake: 0.180 
(epoch: 192, iters: 2512, time: 0.107) G_GAN: 1.707 G_GAN_Feat: 2.183 G_VGG: 0.700 D_real: 0.151 D_fake: 0.064 
(epoch: 192, iters: 2712, time: 0.107) G_GAN: 1.663 G_GAN_Feat: 2.710 G_VGG: 0.900 D_real: 0.074 D_fake: 0.143 
(epoch: 192, iters: 2912, time: 0.103) G_GAN: 1.254 G_GAN_Feat: 1.870 G_VGG: 0.731 D_real: 0.089 D_fake: 0.389 
End of epoch 192 / 200 	 Time Taken: 323 sec
(epoch: 193, iters: 144, time: 0.101) G_GAN: 1.436 G_GAN_Feat: 1.895 G_VGG: 0.698 D_real: 0.128 D_fake: 0.163 
saving the latest model (epoch 193, total_steps 570000)
(epoch: 193, iters: 344, time: 0.106) G_GAN: 1.610 G_GAN_Feat: 2.138 G_VGG: 0.711 D_real: 0.119 D_fake: 0.157 
(epoch: 193, iters: 544, time: 0.102) G_GAN: 1.571 G_GAN_Feat: 1.932 G_VGG: 0.643 D_real: 0.105 D_fake: 0.117 
(epoch: 193, iters: 744, time: 0.103) G_GAN: 1.557 G_GAN_Feat: 2.262 G_VGG: 0.790 D_real: 0.057 D_fake: 0.180 
(epoch: 193, iters: 944, time: 0.109) G_GAN: 1.599 G_GAN_Feat: 2.172 G_VGG: 0.763 D_real: 0.303 D_fake: 0.097 
(epoch: 193, iters: 1144, time: 0.105) G_GAN: 1.739 G_GAN_Feat: 1.912 G_VGG: 0.623 D_real: 0.456 D_fake: 0.121 
saving the latest model (epoch 193, total_steps 571000)
(epoch: 193, iters: 1344, time: 0.105) G_GAN: 1.877 G_GAN_Feat: 2.127 G_VGG: 0.694 D_real: 0.088 D_fake: 0.050 
(epoch: 193, iters: 1544, time: 0.097) G_GAN: 1.868 G_GAN_Feat: 2.293 G_VGG: 0.754 D_real: 0.113 D_fake: 0.040 
(epoch: 193, iters: 1744, time: 0.102) G_GAN: 1.638 G_GAN_Feat: 2.116 G_VGG: 0.768 D_real: 0.150 D_fake: 0.106 
(epoch: 193, iters: 1944, time: 0.101) G_GAN: 1.628 G_GAN_Feat: 2.060 G_VGG: 0.724 D_real: 0.242 D_fake: 0.085 
(epoch: 193, iters: 2144, time: 0.103) G_GAN: 1.546 G_GAN_Feat: 2.163 G_VGG: 0.779 D_real: 0.044 D_fake: 0.155 
saving the latest model (epoch 193, total_steps 572000)
(epoch: 193, iters: 2344, time: 0.107) G_GAN: 1.244 G_GAN_Feat: 1.987 G_VGG: 0.738 D_real: 0.065 D_fake: 0.344 
(epoch: 193, iters: 2544, time: 0.101) G_GAN: 1.733 G_GAN_Feat: 2.077 G_VGG: 0.668 D_real: 0.022 D_fake: 0.056 
(epoch: 193, iters: 2744, time: 0.098) G_GAN: 2.041 G_GAN_Feat: 2.402 G_VGG: 0.848 D_real: 0.123 D_fake: 0.051 
(epoch: 193, iters: 2944, time: 0.102) G_GAN: 1.470 G_GAN_Feat: 1.934 G_VGG: 0.663 D_real: 0.047 D_fake: 0.225 
End of epoch 193 / 200 	 Time Taken: 323 sec
(epoch: 194, iters: 176, time: 0.101) G_GAN: 1.433 G_GAN_Feat: 2.102 G_VGG: 0.762 D_real: 0.108 D_fake: 0.207 
saving the latest model (epoch 194, total_steps 573000)
(epoch: 194, iters: 376, time: 0.102) G_GAN: 1.466 G_GAN_Feat: 2.075 G_VGG: 0.738 D_real: 0.363 D_fake: 0.132 
(epoch: 194, iters: 576, time: 0.109) G_GAN: 1.265 G_GAN_Feat: 2.086 G_VGG: 0.795 D_real: 0.060 D_fake: 0.315 
(epoch: 194, iters: 776, time: 0.104) G_GAN: 1.660 G_GAN_Feat: 2.262 G_VGG: 0.771 D_real: 0.135 D_fake: 0.079 
(epoch: 194, iters: 976, time: 0.101) G_GAN: 1.965 G_GAN_Feat: 2.121 G_VGG: 0.749 D_real: 0.117 D_fake: 0.038 
(epoch: 194, iters: 1176, time: 0.097) G_GAN: 1.662 G_GAN_Feat: 1.818 G_VGG: 0.619 D_real: 0.064 D_fake: 0.161 
saving the latest model (epoch 194, total_steps 574000)
(epoch: 194, iters: 1376, time: 0.101) G_GAN: 1.625 G_GAN_Feat: 2.106 G_VGG: 0.680 D_real: 0.104 D_fake: 0.095 
(epoch: 194, iters: 1576, time: 0.101) G_GAN: 1.357 G_GAN_Feat: 1.992 G_VGG: 0.742 D_real: 0.195 D_fake: 0.277 
(epoch: 194, iters: 1776, time: 0.100) G_GAN: 1.640 G_GAN_Feat: 2.206 G_VGG: 0.756 D_real: 0.057 D_fake: 0.107 
(epoch: 194, iters: 1976, time: 0.106) G_GAN: 1.284 G_GAN_Feat: 1.829 G_VGG: 0.624 D_real: 0.066 D_fake: 0.260 
(epoch: 194, iters: 2176, time: 0.109) G_GAN: 1.616 G_GAN_Feat: 2.190 G_VGG: 0.702 D_real: 0.179 D_fake: 0.083 
saving the latest model (epoch 194, total_steps 575000)
(epoch: 194, iters: 2376, time: 0.107) G_GAN: 1.253 G_GAN_Feat: 2.059 G_VGG: 0.839 D_real: 0.092 D_fake: 0.411 
(epoch: 194, iters: 2576, time: 0.101) G_GAN: 1.339 G_GAN_Feat: 2.014 G_VGG: 0.725 D_real: 0.247 D_fake: 0.200 
(epoch: 194, iters: 2776, time: 0.100) G_GAN: 1.835 G_GAN_Feat: 2.142 G_VGG: 0.651 D_real: 0.074 D_fake: 0.068 
End of epoch 194 / 200 	 Time Taken: 324 sec
(epoch: 195, iters: 8, time: 0.110) G_GAN: 1.610 G_GAN_Feat: 2.352 G_VGG: 0.714 D_real: 0.110 D_fake: 0.170 
(epoch: 195, iters: 208, time: 0.100) G_GAN: 1.297 G_GAN_Feat: 1.765 G_VGG: 0.649 D_real: 0.088 D_fake: 0.368 
saving the latest model (epoch 195, total_steps 576000)
(epoch: 195, iters: 408, time: 0.087) G_GAN: 1.481 G_GAN_Feat: 1.895 G_VGG: 0.695 D_real: 0.168 D_fake: 0.217 
(epoch: 195, iters: 608, time: 0.104) G_GAN: 1.535 G_GAN_Feat: 2.151 G_VGG: 0.839 D_real: 0.147 D_fake: 0.149 
(epoch: 195, iters: 808, time: 0.112) G_GAN: 1.656 G_GAN_Feat: 1.922 G_VGG: 0.684 D_real: 0.079 D_fake: 0.073 
(epoch: 195, iters: 1008, time: 0.099) G_GAN: 1.414 G_GAN_Feat: 2.013 G_VGG: 0.722 D_real: 0.192 D_fake: 0.224 
(epoch: 195, iters: 1208, time: 0.094) G_GAN: 1.396 G_GAN_Feat: 2.082 G_VGG: 0.760 D_real: 0.074 D_fake: 0.225 
saving the latest model (epoch 195, total_steps 577000)
(epoch: 195, iters: 1408, time: 0.107) G_GAN: 1.262 G_GAN_Feat: 1.867 G_VGG: 0.688 D_real: 0.253 D_fake: 0.268 
(epoch: 195, iters: 1608, time: 0.099) G_GAN: 1.520 G_GAN_Feat: 2.015 G_VGG: 0.747 D_real: 0.140 D_fake: 0.157 
(epoch: 195, iters: 1808, time: 0.106) G_GAN: 1.873 G_GAN_Feat: 2.104 G_VGG: 0.732 D_real: 0.155 D_fake: 0.053 
(epoch: 195, iters: 2008, time: 0.098) G_GAN: 1.346 G_GAN_Feat: 1.990 G_VGG: 0.748 D_real: 0.093 D_fake: 0.299 
(epoch: 195, iters: 2208, time: 0.099) G_GAN: 1.498 G_GAN_Feat: 1.893 G_VGG: 0.640 D_real: 0.096 D_fake: 0.189 
saving the latest model (epoch 195, total_steps 578000)
(epoch: 195, iters: 2408, time: 0.095) G_GAN: 1.363 G_GAN_Feat: 1.845 G_VGG: 0.669 D_real: 0.092 D_fake: 0.306 
(epoch: 195, iters: 2608, time: 0.097) G_GAN: 1.660 G_GAN_Feat: 2.284 G_VGG: 0.699 D_real: 0.115 D_fake: 0.087 
(epoch: 195, iters: 2808, time: 0.099) G_GAN: 1.441 G_GAN_Feat: 1.930 G_VGG: 0.734 D_real: 0.201 D_fake: 0.153 
End of epoch 195 / 200 	 Time Taken: 325 sec
(epoch: 196, iters: 40, time: 0.093) G_GAN: 1.481 G_GAN_Feat: 1.962 G_VGG: 0.713 D_real: 0.124 D_fake: 0.154 
(epoch: 196, iters: 240, time: 0.099) G_GAN: 1.482 G_GAN_Feat: 2.118 G_VGG: 0.765 D_real: 0.111 D_fake: 0.193 
saving the latest model (epoch 196, total_steps 579000)
(epoch: 196, iters: 440, time: 0.103) G_GAN: 1.393 G_GAN_Feat: 1.902 G_VGG: 0.721 D_real: 0.160 D_fake: 0.222 
(epoch: 196, iters: 640, time: 0.092) G_GAN: 1.785 G_GAN_Feat: 2.135 G_VGG: 0.733 D_real: 0.069 D_fake: 0.068 
(epoch: 196, iters: 840, time: 0.103) G_GAN: 1.353 G_GAN_Feat: 2.033 G_VGG: 0.795 D_real: 0.078 D_fake: 0.251 
(epoch: 196, iters: 1040, time: 0.106) G_GAN: 1.915 G_GAN_Feat: 2.141 G_VGG: 0.673 D_real: 0.101 D_fake: 0.113 
(epoch: 196, iters: 1240, time: 0.103) G_GAN: 1.519 G_GAN_Feat: 1.996 G_VGG: 0.687 D_real: 0.081 D_fake: 0.152 
saving the latest model (epoch 196, total_steps 580000)
(epoch: 196, iters: 1440, time: 0.104) G_GAN: 1.452 G_GAN_Feat: 1.960 G_VGG: 0.738 D_real: 0.183 D_fake: 0.134 
(epoch: 196, iters: 1640, time: 0.106) G_GAN: 1.221 G_GAN_Feat: 1.757 G_VGG: 0.656 D_real: 0.109 D_fake: 0.369 
(epoch: 196, iters: 1840, time: 0.091) G_GAN: 1.806 G_GAN_Feat: 2.240 G_VGG: 0.685 D_real: 0.083 D_fake: 0.054 
(epoch: 196, iters: 2040, time: 0.106) G_GAN: 1.328 G_GAN_Feat: 1.842 G_VGG: 0.690 D_real: 0.059 D_fake: 0.231 
(epoch: 196, iters: 2240, time: 0.094) G_GAN: 1.418 G_GAN_Feat: 1.914 G_VGG: 0.615 D_real: 0.069 D_fake: 0.173 
saving the latest model (epoch 196, total_steps 581000)
(epoch: 196, iters: 2440, time: 0.095) G_GAN: 1.513 G_GAN_Feat: 1.914 G_VGG: 0.681 D_real: 0.145 D_fake: 0.192 
(epoch: 196, iters: 2640, time: 0.106) G_GAN: 1.193 G_GAN_Feat: 1.931 G_VGG: 0.769 D_real: 0.057 D_fake: 0.355 
(epoch: 196, iters: 2840, time: 0.102) G_GAN: 1.626 G_GAN_Feat: 1.959 G_VGG: 0.731 D_real: 0.158 D_fake: 0.082 
End of epoch 196 / 200 	 Time Taken: 324 sec
(epoch: 197, iters: 72, time: 0.099) G_GAN: 1.480 G_GAN_Feat: 2.041 G_VGG: 0.755 D_real: 0.144 D_fake: 0.179 
(epoch: 197, iters: 272, time: 0.087) G_GAN: 1.880 G_GAN_Feat: 1.847 G_VGG: 0.636 D_real: 0.259 D_fake: 0.029 
saving the latest model (epoch 197, total_steps 582000)
(epoch: 197, iters: 472, time: 0.087) G_GAN: 1.296 G_GAN_Feat: 1.864 G_VGG: 0.657 D_real: 0.160 D_fake: 0.287 
(epoch: 197, iters: 672, time: 0.107) G_GAN: 1.439 G_GAN_Feat: 2.311 G_VGG: 0.867 D_real: 0.137 D_fake: 0.219 
(epoch: 197, iters: 872, time: 0.105) G_GAN: 1.310 G_GAN_Feat: 1.877 G_VGG: 0.716 D_real: 0.094 D_fake: 0.251 
(epoch: 197, iters: 1072, time: 0.105) G_GAN: 1.394 G_GAN_Feat: 1.970 G_VGG: 0.759 D_real: 0.138 D_fake: 0.207 
(epoch: 197, iters: 1272, time: 0.089) G_GAN: 1.435 G_GAN_Feat: 1.920 G_VGG: 0.749 D_real: 0.168 D_fake: 0.160 
saving the latest model (epoch 197, total_steps 583000)
(epoch: 197, iters: 1472, time: 0.097) G_GAN: 1.408 G_GAN_Feat: 1.876 G_VGG: 0.727 D_real: 0.235 D_fake: 0.151 
(epoch: 197, iters: 1672, time: 0.113) G_GAN: 1.415 G_GAN_Feat: 1.966 G_VGG: 0.729 D_real: 0.090 D_fake: 0.188 
(epoch: 197, iters: 1872, time: 0.099) G_GAN: 1.640 G_GAN_Feat: 1.906 G_VGG: 0.665 D_real: 0.178 D_fake: 0.113 
(epoch: 197, iters: 2072, time: 0.098) G_GAN: 1.485 G_GAN_Feat: 2.203 G_VGG: 0.823 D_real: 0.212 D_fake: 0.133 
(epoch: 197, iters: 2272, time: 0.097) G_GAN: 1.404 G_GAN_Feat: 2.048 G_VGG: 0.805 D_real: 0.204 D_fake: 0.185 
saving the latest model (epoch 197, total_steps 584000)
(epoch: 197, iters: 2472, time: 0.097) G_GAN: 1.517 G_GAN_Feat: 1.880 G_VGG: 0.677 D_real: 0.128 D_fake: 0.143 
(epoch: 197, iters: 2672, time: 0.102) G_GAN: 1.415 G_GAN_Feat: 1.957 G_VGG: 0.775 D_real: 0.157 D_fake: 0.168 
(epoch: 197, iters: 2872, time: 0.102) G_GAN: 1.484 G_GAN_Feat: 2.196 G_VGG: 0.734 D_real: 0.090 D_fake: 0.198 
End of epoch 197 / 200 	 Time Taken: 323 sec
(epoch: 198, iters: 104, time: 0.102) G_GAN: 1.319 G_GAN_Feat: 1.910 G_VGG: 0.730 D_real: 0.173 D_fake: 0.234 
(epoch: 198, iters: 304, time: 0.106) G_GAN: 1.378 G_GAN_Feat: 1.725 G_VGG: 0.644 D_real: 0.141 D_fake: 0.223 
saving the latest model (epoch 198, total_steps 585000)
(epoch: 198, iters: 504, time: 0.104) G_GAN: 1.352 G_GAN_Feat: 1.936 G_VGG: 0.702 D_real: 0.130 D_fake: 0.186 
(epoch: 198, iters: 704, time: 0.106) G_GAN: 1.552 G_GAN_Feat: 2.146 G_VGG: 0.715 D_real: 0.117 D_fake: 0.160 
(epoch: 198, iters: 904, time: 0.087) G_GAN: 1.356 G_GAN_Feat: 1.729 G_VGG: 0.694 D_real: 0.150 D_fake: 0.246 
(epoch: 198, iters: 1104, time: 0.101) G_GAN: 1.469 G_GAN_Feat: 2.068 G_VGG: 0.796 D_real: 0.177 D_fake: 0.182 
(epoch: 198, iters: 1304, time: 0.101) G_GAN: 1.339 G_GAN_Feat: 1.966 G_VGG: 0.758 D_real: 0.161 D_fake: 0.190 
saving the latest model (epoch 198, total_steps 586000)
(epoch: 198, iters: 1504, time: 0.092) G_GAN: 1.406 G_GAN_Feat: 1.863 G_VGG: 0.696 D_real: 0.139 D_fake: 0.174 
(epoch: 198, iters: 1704, time: 0.109) G_GAN: 1.552 G_GAN_Feat: 1.895 G_VGG: 0.681 D_real: 0.127 D_fake: 0.151 
(epoch: 198, iters: 1904, time: 0.103) G_GAN: 1.353 G_GAN_Feat: 1.992 G_VGG: 0.738 D_real: 0.157 D_fake: 0.218 
(epoch: 198, iters: 2104, time: 0.102) G_GAN: 1.434 G_GAN_Feat: 1.834 G_VGG: 0.686 D_real: 0.214 D_fake: 0.152 
(epoch: 198, iters: 2304, time: 0.094) G_GAN: 1.591 G_GAN_Feat: 1.840 G_VGG: 0.639 D_real: 0.170 D_fake: 0.159 
saving the latest model (epoch 198, total_steps 587000)
(epoch: 198, iters: 2504, time: 0.114) G_GAN: 1.315 G_GAN_Feat: 1.753 G_VGG: 0.691 D_real: 0.093 D_fake: 0.240 
(epoch: 198, iters: 2704, time: 0.111) G_GAN: 1.357 G_GAN_Feat: 1.872 G_VGG: 0.712 D_real: 0.131 D_fake: 0.211 
(epoch: 198, iters: 2904, time: 0.099) G_GAN: 1.556 G_GAN_Feat: 2.042 G_VGG: 0.760 D_real: 0.166 D_fake: 0.175 
End of epoch 198 / 200 	 Time Taken: 324 sec
(epoch: 199, iters: 136, time: 0.109) G_GAN: 1.289 G_GAN_Feat: 1.874 G_VGG: 0.759 D_real: 0.243 D_fake: 0.226 
(epoch: 199, iters: 336, time: 0.096) G_GAN: 1.557 G_GAN_Feat: 2.385 G_VGG: 0.804 D_real: 0.186 D_fake: 0.153 
saving the latest model (epoch 199, total_steps 588000)
(epoch: 199, iters: 536, time: 0.100) G_GAN: 1.607 G_GAN_Feat: 2.637 G_VGG: 0.918 D_real: 0.122 D_fake: 0.119 
(epoch: 199, iters: 736, time: 0.098) G_GAN: 1.421 G_GAN_Feat: 1.984 G_VGG: 0.724 D_real: 0.134 D_fake: 0.236 
(epoch: 199, iters: 936, time: 0.100) G_GAN: 1.419 G_GAN_Feat: 2.063 G_VGG: 0.737 D_real: 0.120 D_fake: 0.186 
(epoch: 199, iters: 1136, time: 0.101) G_GAN: 1.381 G_GAN_Feat: 1.942 G_VGG: 0.770 D_real: 0.151 D_fake: 0.181 
(epoch: 199, iters: 1336, time: 0.102) G_GAN: 1.434 G_GAN_Feat: 1.933 G_VGG: 0.696 D_real: 0.164 D_fake: 0.205 
saving the latest model (epoch 199, total_steps 589000)
(epoch: 199, iters: 1536, time: 0.105) G_GAN: 1.353 G_GAN_Feat: 2.132 G_VGG: 0.759 D_real: 0.172 D_fake: 0.215 
(epoch: 199, iters: 1736, time: 0.104) G_GAN: 1.491 G_GAN_Feat: 1.963 G_VGG: 0.716 D_real: 0.178 D_fake: 0.190 
(epoch: 199, iters: 1936, time: 0.099) G_GAN: 1.478 G_GAN_Feat: 2.131 G_VGG: 0.714 D_real: 0.162 D_fake: 0.168 
(epoch: 199, iters: 2136, time: 0.097) G_GAN: 1.456 G_GAN_Feat: 2.065 G_VGG: 0.795 D_real: 0.183 D_fake: 0.154 
(epoch: 199, iters: 2336, time: 0.093) G_GAN: 1.381 G_GAN_Feat: 1.893 G_VGG: 0.713 D_real: 0.221 D_fake: 0.163 
saving the latest model (epoch 199, total_steps 590000)
(epoch: 199, iters: 2536, time: 0.103) G_GAN: 1.640 G_GAN_Feat: 1.586 G_VGG: 0.591 D_real: 0.157 D_fake: 0.112 
(epoch: 199, iters: 2736, time: 0.103) G_GAN: 1.480 G_GAN_Feat: 1.970 G_VGG: 0.726 D_real: 0.165 D_fake: 0.167 
(epoch: 199, iters: 2936, time: 0.102) G_GAN: 1.265 G_GAN_Feat: 1.779 G_VGG: 0.703 D_real: 0.149 D_fake: 0.256 
End of epoch 199 / 200 	 Time Taken: 325 sec
(epoch: 200, iters: 168, time: 0.113) G_GAN: 1.195 G_GAN_Feat: 1.805 G_VGG: 0.720 D_real: 0.193 D_fake: 0.278 
(epoch: 200, iters: 368, time: 0.101) G_GAN: 1.398 G_GAN_Feat: 1.936 G_VGG: 0.744 D_real: 0.153 D_fake: 0.175 
saving the latest model (epoch 200, total_steps 591000)
(epoch: 200, iters: 568, time: 0.098) G_GAN: 1.500 G_GAN_Feat: 1.912 G_VGG: 0.708 D_real: 0.199 D_fake: 0.154 
(epoch: 200, iters: 768, time: 0.096) G_GAN: 1.320 G_GAN_Feat: 1.828 G_VGG: 0.722 D_real: 0.167 D_fake: 0.216 
(epoch: 200, iters: 968, time: 0.107) G_GAN: 1.535 G_GAN_Feat: 2.094 G_VGG: 0.778 D_real: 0.176 D_fake: 0.125 
(epoch: 200, iters: 1168, time: 0.102) G_GAN: 1.481 G_GAN_Feat: 1.922 G_VGG: 0.763 D_real: 0.282 D_fake: 0.133 
(epoch: 200, iters: 1368, time: 0.105) G_GAN: 1.507 G_GAN_Feat: 1.887 G_VGG: 0.692 D_real: 0.194 D_fake: 0.172 
saving the latest model (epoch 200, total_steps 592000)
(epoch: 200, iters: 1568, time: 0.098) G_GAN: 1.509 G_GAN_Feat: 2.126 G_VGG: 0.796 D_real: 0.180 D_fake: 0.142 
(epoch: 200, iters: 1768, time: 0.107) G_GAN: 1.272 G_GAN_Feat: 1.653 G_VGG: 0.599 D_real: 0.123 D_fake: 0.324 
(epoch: 200, iters: 1968, time: 0.105) G_GAN: 1.387 G_GAN_Feat: 2.042 G_VGG: 0.799 D_real: 0.224 D_fake: 0.169 
(epoch: 200, iters: 2168, time: 0.106) G_GAN: 1.559 G_GAN_Feat: 1.915 G_VGG: 0.713 D_real: 0.161 D_fake: 0.110 
(epoch: 200, iters: 2368, time: 0.102) G_GAN: 1.238 G_GAN_Feat: 1.847 G_VGG: 0.749 D_real: 0.164 D_fake: 0.260 
saving the latest model (epoch 200, total_steps 593000)
(epoch: 200, iters: 2568, time: 0.104) G_GAN: 1.307 G_GAN_Feat: 1.765 G_VGG: 0.708 D_real: 0.220 D_fake: 0.245 
(epoch: 200, iters: 2768, time: 0.097) G_GAN: 1.467 G_GAN_Feat: 2.296 G_VGG: 0.809 D_real: 0.194 D_fake: 0.151 
(epoch: 200, iters: 2968, time: 0.107) G_GAN: 1.340 G_GAN_Feat: 2.025 G_VGG: 0.792 D_real: 0.206 D_fake: 0.240 
End of epoch 200 / 200 	 Time Taken: 323 sec
saving the model at the end of epoch 200, iters 593600
```

## log for 16 GPUs
$ python train.py --name label2city_512p-16gpu --batchSize 128 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
------------ Options -------------
batchSize: 128
beta1: 0.5
checkpoints_dir: ./checkpoints
continue_train: False
data_type: 32
dataroot: ./datasets/cityscapes/
debug: False
display_freq: 100
display_winsize: 512
feat_num: 3
fineSize: 512
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
input_nc: 3
instance_feat: False
isTrain: True
label_feat: False
label_nc: 35
lambda_feat: 10.0
loadSize: 1024
load_features: False
load_pretrain: 
lr: 0.0002
max_dataset_size: inf
model: pix2pixHD
nThreads: 2
n_blocks_global: 9
n_blocks_local: 3
n_clusters: 10
n_downsample_E: 4
n_downsample_global: 4
n_layers_D: 3
n_local_enhancers: 1
name: label2city_512p-16gpu
ndf: 64
nef: 16
netG: global
ngf: 64
niter: 100
niter_decay: 100
niter_fix_global: 0
no_flip: False
no_ganFeat_loss: False
no_html: False
no_instance: False
no_lsgan: False
no_vgg_loss: False
norm: instance
num_D: 2
output_nc: 3
phase: train
pool_size: 0
print_freq: 100
resize_or_crop: scale_width
save_epoch_freq: 10
save_latest_freq: 1000
serial_batches: False
tf_log: False
use_dropout: False
verbose: False
which_epoch: latest
-------------- End ----------------
CustomDatasetDataLoader
dataset [AlignedDataset] was created
#training images = 2944
GlobalGenerator(
  (model): Sequential(
    (0): ReflectionPad2d((3, 3, 3, 3))
    (1): Conv2d(36, 64, kernel_size=(7, 7), stride=(1, 1))
    (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (3): ReLU(inplace)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (6): ReLU(inplace)
    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (9): ReLU(inplace)
    (10): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (11): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (12): ReLU(inplace)
    (13): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (14): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (15): ReLU(inplace)
    (16): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (17): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (18): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (19): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (20): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (21): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (22): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (23): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (24): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (25): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (26): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (27): ReLU(inplace)
    (28): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (29): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (30): ReLU(inplace)
    (31): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (32): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (33): ReLU(inplace)
    (34): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (35): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (36): ReLU(inplace)
    (37): ReflectionPad2d((3, 3, 3, 3))
    (38): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
    (39): Tanh()
  )
)
MultiscaleDiscriminator(
  (scale0_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (scale1_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (downsample): AvgPool2d(kernel_size=3, stride=2, padding=[1, 1])
)
create web directory ./checkpoints/label2city_512p-16gpu/web...


End of epoch 1 / 200 	 Time Taken: 261 sec
(epoch: 2, iters: 1920, time: 0.042) G_GAN: 1.185 G_GAN_Feat: 5.136 G_VGG: 6.295 D_real: 0.913 D_fake: 0.875 
End of epoch 2 / 200 	 Time Taken: 184 sec
(epoch: 3, iters: 1440, time: 0.039) G_GAN: 1.258 G_GAN_Feat: 4.284 G_VGG: 5.800 D_real: 1.026 D_fake: 1.183 
End of epoch 3 / 200 	 Time Taken: 181 sec
(epoch: 4, iters: 960, time: 0.041) G_GAN: 0.809 G_GAN_Feat: 3.365 G_VGG: 5.828 D_real: 0.622 D_fake: 0.574 
End of epoch 4 / 200 	 Time Taken: 185 sec
(epoch: 5, iters: 480, time: 0.044) G_GAN: 0.849 G_GAN_Feat: 3.103 G_VGG: 5.321 D_real: 0.693 D_fake: 0.658 
saving the latest model (epoch 5, total_steps 12000)
(epoch: 5, iters: 2880, time: 0.040) G_GAN: 0.802 G_GAN_Feat: 3.162 G_VGG: 4.956 D_real: 0.602 D_fake: 0.672 
End of epoch 5 / 200 	 Time Taken: 185 sec
(epoch: 6, iters: 2400, time: 0.037) G_GAN: 0.691 G_GAN_Feat: 2.528 G_VGG: 4.695 D_real: 0.556 D_fake: 0.756 
End of epoch 6 / 200 	 Time Taken: 184 sec
(epoch: 7, iters: 1920, time: 0.040) G_GAN: 0.637 G_GAN_Feat: 2.269 G_VGG: 4.614 D_real: 0.482 D_fake: 0.576 
End of epoch 7 / 200 	 Time Taken: 185 sec
(epoch: 8, iters: 1440, time: 0.036) G_GAN: 0.792 G_GAN_Feat: 2.659 G_VGG: 4.692 D_real: 0.567 D_fake: 0.512 
End of epoch 8 / 200 	 Time Taken: 186 sec
(epoch: 9, iters: 960, time: 0.039) G_GAN: 0.712 G_GAN_Feat: 1.939 G_VGG: 4.146 D_real: 0.567 D_fake: 0.542 
saving the latest model (epoch 9, total_steps 24000)
End of epoch 9 / 200 	 Time Taken: 186 sec
(epoch: 10, iters: 480, time: 0.040) G_GAN: 0.726 G_GAN_Feat: 2.291 G_VGG: 4.413 D_real: 0.558 D_fake: 0.754 
(epoch: 10, iters: 2880, time: 0.042) G_GAN: 0.817 G_GAN_Feat: 2.275 G_VGG: 4.183 D_real: 0.523 D_fake: 0.435 
End of epoch 10 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 10, iters 28800
(epoch: 11, iters: 2400, time: 0.041) G_GAN: 2.388 G_GAN_Feat: 1.448 G_VGG: 3.916 D_real: 2.362 D_fake: 2.010 
End of epoch 11 / 200 	 Time Taken: 186 sec
(epoch: 12, iters: 1920, time: 0.039) G_GAN: 0.543 G_GAN_Feat: 1.259 G_VGG: 3.647 D_real: 0.498 D_fake: 0.548 
End of epoch 12 / 200 	 Time Taken: 186 sec
(epoch: 13, iters: 1440, time: 0.044) G_GAN: 0.576 G_GAN_Feat: 1.198 G_VGG: 3.632 D_real: 0.529 D_fake: 0.524 
saving the latest model (epoch 13, total_steps 36000)
End of epoch 13 / 200 	 Time Taken: 186 sec
(epoch: 14, iters: 960, time: 0.038) G_GAN: 0.572 G_GAN_Feat: 1.218 G_VGG: 3.603 D_real: 0.535 D_fake: 0.487 
End of epoch 14 / 200 	 Time Taken: 185 sec
(epoch: 15, iters: 480, time: 0.042) G_GAN: 0.464 G_GAN_Feat: 1.192 G_VGG: 3.342 D_real: 0.422 D_fake: 0.599 
(epoch: 15, iters: 2880, time: 0.043) G_GAN: 0.636 G_GAN_Feat: 1.066 G_VGG: 3.367 D_real: 0.578 D_fake: 0.443 
End of epoch 15 / 200 	 Time Taken: 186 sec
^[[C^[[C(epoch: 16, iters: 2400, time: 0.043) G_GAN: 0.594 G_GAN_Feat: 0.984 G_VGG: 3.284 D_real: 0.550 D_fake: 0.550 
End of epoch 16 / 200 	 Time Taken: 186 sec
(epoch: 17, iters: 1920, time: 0.039) G_GAN: 0.481 G_GAN_Feat: 1.041 G_VGG: 3.173 D_real: 0.445 D_fake: 0.803 
saving the latest model (epoch 17, total_steps 48000)
End of epoch 17 / 200 	 Time Taken: 186 sec
(epoch: 18, iters: 1440, time: 0.038) G_GAN: 0.727 G_GAN_Feat: 0.934 G_VGG: 3.047 D_real: 0.675 D_fake: 0.390 
End of epoch 18 / 200 	 Time Taken: 184 sec
(epoch: 19, iters: 960, time: 0.037) G_GAN: 0.903 G_GAN_Feat: 1.633 G_VGG: 3.862 D_real: 0.797 D_fake: 0.347 
End of epoch 19 / 200 	 Time Taken: 186 sec
(epoch: 20, iters: 480, time: 0.037) G_GAN: 0.430 G_GAN_Feat: 1.144 G_VGG: 3.212 D_real: 0.407 D_fake: 0.785 
(epoch: 20, iters: 2880, time: 0.040) G_GAN: 0.420 G_GAN_Feat: 1.468 G_VGG: 3.204 D_real: 0.368 D_fake: 0.651 
End of epoch 20 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 20, iters 57600
(epoch: 21, iters: 2400, time: 0.039) G_GAN: 0.729 G_GAN_Feat: 1.007 G_VGG: 2.963 D_real: 0.627 D_fake: 0.399 
saving the latest model (epoch 21, total_steps 60000)
End of epoch 21 / 200 	 Time Taken: 188 sec
(epoch: 22, iters: 1920, time: 0.036) G_GAN: 0.540 G_GAN_Feat: 0.942 G_VGG: 2.932 D_real: 0.479 D_fake: 0.496 
End of epoch 22 / 200 	 Time Taken: 187 sec
(epoch: 23, iters: 1440, time: 0.045) G_GAN: 0.527 G_GAN_Feat: 1.062 G_VGG: 3.012 D_real: 0.488 D_fake: 0.517 
End of epoch 23 / 200 	 Time Taken: 185 sec
(epoch: 24, iters: 960, time: 0.039) G_GAN: 0.568 G_GAN_Feat: 0.925 G_VGG: 2.859 D_real: 0.527 D_fake: 0.616 
End of epoch 24 / 200 	 Time Taken: 184 sec
(epoch: 25, iters: 480, time: 0.041) G_GAN: 0.522 G_GAN_Feat: 1.199 G_VGG: 2.675 D_real: 0.460 D_fake: 0.525 
(epoch: 25, iters: 2880, time: 0.039) G_GAN: 0.596 G_GAN_Feat: 0.852 G_VGG: 2.645 D_real: 0.548 D_fake: 0.458 
saving the latest model (epoch 25, total_steps 72000)
End of epoch 25 / 200 	 Time Taken: 191 sec
(epoch: 26, iters: 2400, time: 0.036) G_GAN: 0.721 G_GAN_Feat: 0.812 G_VGG: 2.695 D_real: 0.681 D_fake: 0.527 
End of epoch 26 / 200 	 Time Taken: 185 sec
(epoch: 27, iters: 1920, time: 0.036) G_GAN: 0.469 G_GAN_Feat: 0.933 G_VGG: 2.806 D_real: 0.422 D_fake: 0.568 
End of epoch 27 / 200 	 Time Taken: 187 sec
(epoch: 28, iters: 1440, time: 0.046) G_GAN: 0.563 G_GAN_Feat: 0.890 G_VGG: 2.556 D_real: 0.523 D_fake: 0.484 
End of epoch 28 / 200 	 Time Taken: 185 sec
(epoch: 29, iters: 960, time: 0.036) G_GAN: 0.632 G_GAN_Feat: 0.854 G_VGG: 2.469 D_real: 0.576 D_fake: 0.442 
End of epoch 29 / 200 	 Time Taken: 187 sec
(epoch: 30, iters: 480, time: 0.041) G_GAN: 0.663 G_GAN_Feat: 0.974 G_VGG: 2.572 D_real: 0.582 D_fake: 0.406 
saving the latest model (epoch 30, total_steps 84000)
(epoch: 30, iters: 2880, time: 0.041) G_GAN: 0.715 G_GAN_Feat: 0.773 G_VGG: 2.495 D_real: 0.660 D_fake: 0.412 
End of epoch 30 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 30, iters 86400
(epoch: 31, iters: 2400, time: 0.041) G_GAN: 0.879 G_GAN_Feat: 0.962 G_VGG: 2.444 D_real: 0.824 D_fake: 0.332 
End of epoch 31 / 200 	 Time Taken: 185 sec
(epoch: 32, iters: 1920, time: 0.043) G_GAN: 0.451 G_GAN_Feat: 0.886 G_VGG: 2.410 D_real: 0.397 D_fake: 0.602 
End of epoch 32 / 200 	 Time Taken: 186 sec
(epoch: 33, iters: 1440, time: 0.040) G_GAN: 0.559 G_GAN_Feat: 0.823 G_VGG: 2.304 D_real: 0.516 D_fake: 0.509 
End of epoch 33 / 200 	 Time Taken: 186 sec
(epoch: 34, iters: 960, time: 0.040) G_GAN: 0.612 G_GAN_Feat: 0.974 G_VGG: 2.407 D_real: 0.560 D_fake: 0.455 
saving the latest model (epoch 34, total_steps 96000)
End of epoch 34 / 200 	 Time Taken: 188 sec
(epoch: 35, iters: 480, time: 0.037) G_GAN: 1.081 G_GAN_Feat: 1.143 G_VGG: 2.385 D_real: 0.999 D_fake: 0.240 
(epoch: 35, iters: 2880, time: 0.038) G_GAN: 0.567 G_GAN_Feat: 1.219 G_VGG: 2.287 D_real: 0.404 D_fake: 0.485 
End of epoch 35 / 200 	 Time Taken: 186 sec
(epoch: 36, iters: 2400, time: 0.039) G_GAN: 0.474 G_GAN_Feat: 0.791 G_VGG: 2.215 D_real: 0.427 D_fake: 0.686 
End of epoch 36 / 200 	 Time Taken: 186 sec
(epoch: 37, iters: 1920, time: 0.040) G_GAN: 0.649 G_GAN_Feat: 0.974 G_VGG: 2.312 D_real: 0.596 D_fake: 0.411 
End of epoch 37 / 200 	 Time Taken: 186 sec
(epoch: 38, iters: 1440, time: 0.039) G_GAN: 0.532 G_GAN_Feat: 0.918 G_VGG: 2.252 D_real: 0.414 D_fake: 0.502 
saving the latest model (epoch 38, total_steps 108000)
End of epoch 38 / 200 	 Time Taken: 187 sec
(epoch: 39, iters: 960, time: 0.045) G_GAN: 0.600 G_GAN_Feat: 1.166 G_VGG: 2.237 D_real: 0.439 D_fake: 0.438 
End of epoch 39 / 200 	 Time Taken: 185 sec
(epoch: 40, iters: 480, time: 0.044) G_GAN: 0.673 G_GAN_Feat: 0.807 G_VGG: 2.206 D_real: 0.604 D_fake: 0.406 
(epoch: 40, iters: 2880, time: 0.041) G_GAN: 0.855 G_GAN_Feat: 1.221 G_VGG: 2.168 D_real: 0.637 D_fake: 0.324 
End of epoch 40 / 200 	 Time Taken: 185 sec
saving the model at the end of epoch 40, iters 115200
(epoch: 41, iters: 2400, time: 0.044) G_GAN: 0.384 G_GAN_Feat: 0.871 G_VGG: 2.214 D_real: 0.358 D_fake: 0.847 
End of epoch 41 / 200 	 Time Taken: 184 sec
(epoch: 42, iters: 1920, time: 0.039) G_GAN: 0.680 G_GAN_Feat: 0.824 G_VGG: 2.170 D_real: 0.669 D_fake: 0.394 
saving the latest model (epoch 42, total_steps 120000)
End of epoch 42 / 200 	 Time Taken: 188 sec
(epoch: 43, iters: 1440, time: 0.046) G_GAN: 0.705 G_GAN_Feat: 0.984 G_VGG: 2.153 D_real: 0.574 D_fake: 0.457 
End of epoch 43 / 200 	 Time Taken: 185 sec
(epoch: 44, iters: 960, time: 0.041) G_GAN: 0.570 G_GAN_Feat: 0.972 G_VGG: 2.277 D_real: 0.517 D_fake: 0.547 
End of epoch 44 / 200 	 Time Taken: 186 sec
(epoch: 45, iters: 480, time: 0.038) G_GAN: 0.635 G_GAN_Feat: 0.854 G_VGG: 2.119 D_real: 0.570 D_fake: 0.447 
(epoch: 45, iters: 2880, time: 0.043) G_GAN: 0.639 G_GAN_Feat: 1.578 G_VGG: 2.244 D_real: 0.319 D_fake: 0.492 
End of epoch 45 / 200 	 Time Taken: 185 sec
(epoch: 46, iters: 2400, time: 0.036) G_GAN: 0.469 G_GAN_Feat: 1.162 G_VGG: 2.058 D_real: 0.350 D_fake: 0.619 
saving the latest model (epoch 46, total_steps 132000)
End of epoch 46 / 200 	 Time Taken: 187 sec
(epoch: 47, iters: 1920, time: 0.042) G_GAN: 0.271 G_GAN_Feat: 1.177 G_VGG: 2.150 D_real: 0.170 D_fake: 0.912 
End of epoch 47 / 200 	 Time Taken: 185 sec
(epoch: 48, iters: 1440, time: 0.039) G_GAN: 0.597 G_GAN_Feat: 1.286 G_VGG: 2.125 D_real: 0.410 D_fake: 0.490 
End of epoch 48 / 200 	 Time Taken: 186 sec
(epoch: 49, iters: 960, time: 0.041) G_GAN: 0.523 G_GAN_Feat: 0.910 G_VGG: 2.058 D_real: 0.426 D_fake: 0.532 
End of epoch 49 / 200 	 Time Taken: 185 sec
(epoch: 50, iters: 480, time: 0.036) G_GAN: 0.734 G_GAN_Feat: 0.978 G_VGG: 1.950 D_real: 0.769 D_fake: 0.365 
(epoch: 50, iters: 2880, time: 0.043) G_GAN: 0.577 G_GAN_Feat: 0.686 G_VGG: 1.991 D_real: 0.532 D_fake: 0.452 
saving the latest model (epoch 50, total_steps 144000)
End of epoch 50 / 200 	 Time Taken: 190 sec
saving the model at the end of epoch 50, iters 144000
(epoch: 51, iters: 2400, time: 0.041) G_GAN: 0.470 G_GAN_Feat: 1.551 G_VGG: 2.048 D_real: 0.306 D_fake: 0.701 
End of epoch 51 / 200 	 Time Taken: 187 sec
(epoch: 52, iters: 1920, time: 0.043) G_GAN: 0.665 G_GAN_Feat: 0.782 G_VGG: 1.923 D_real: 0.604 D_fake: 0.425 
End of epoch 52 / 200 	 Time Taken: 185 sec
(epoch: 53, iters: 1440, time: 0.042) G_GAN: 0.702 G_GAN_Feat: 1.143 G_VGG: 1.936 D_real: 0.558 D_fake: 0.510 
End of epoch 53 / 200 	 Time Taken: 187 sec
(epoch: 54, iters: 960, time: 0.041) G_GAN: 0.473 G_GAN_Feat: 0.702 G_VGG: 1.926 D_real: 0.429 D_fake: 0.565 
End of epoch 54 / 200 	 Time Taken: 185 sec
(epoch: 55, iters: 480, time: 0.044) G_GAN: 0.527 G_GAN_Feat: 0.947 G_VGG: 1.829 D_real: 0.403 D_fake: 0.499 
saving the latest model (epoch 55, total_steps 156000)
(epoch: 55, iters: 2880, time: 0.034) G_GAN: 0.457 G_GAN_Feat: 0.702 G_VGG: 1.802 D_real: 0.432 D_fake: 0.728 
End of epoch 55 / 200 	 Time Taken: 185 sec
(epoch: 56, iters: 2400, time: 0.040) G_GAN: 0.365 G_GAN_Feat: 0.847 G_VGG: 1.858 D_real: 0.305 D_fake: 0.742 
End of epoch 56 / 200 	 Time Taken: 186 sec
(epoch: 57, iters: 1920, time: 0.038) G_GAN: 0.769 G_GAN_Feat: 1.016 G_VGG: 1.860 D_real: 0.594 D_fake: 0.346 
End of epoch 57 / 200 	 Time Taken: 186 sec
(epoch: 58, iters: 1440, time: 0.040) G_GAN: 0.975 G_GAN_Feat: 1.372 G_VGG: 1.904 D_real: 0.573 D_fake: 0.280 
End of epoch 58 / 200 	 Time Taken: 185 sec
(epoch: 59, iters: 960, time: 0.039) G_GAN: 0.743 G_GAN_Feat: 1.325 G_VGG: 1.823 D_real: 0.480 D_fake: 0.341 
saving the latest model (epoch 59, total_steps 168000)
End of epoch 59 / 200 	 Time Taken: 184 sec
(epoch: 60, iters: 480, time: 0.038) G_GAN: 0.693 G_GAN_Feat: 1.038 G_VGG: 1.821 D_real: 0.457 D_fake: 0.367 
(epoch: 60, iters: 2880, time: 0.038) G_GAN: 0.587 G_GAN_Feat: 0.766 G_VGG: 1.809 D_real: 0.519 D_fake: 0.440 
End of epoch 60 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 60, iters 172800
(epoch: 61, iters: 2400, time: 0.035) G_GAN: 0.808 G_GAN_Feat: 0.715 G_VGG: 1.744 D_real: 0.737 D_fake: 0.379 
End of epoch 61 / 200 	 Time Taken: 184 sec
(epoch: 62, iters: 1920, time: 0.038) G_GAN: 0.592 G_GAN_Feat: 0.911 G_VGG: 1.798 D_real: 0.485 D_fake: 0.586 
End of epoch 62 / 200 	 Time Taken: 185 sec
(epoch: 63, iters: 1440, time: 0.039) G_GAN: 0.572 G_GAN_Feat: 0.777 G_VGG: 1.661 D_real: 0.478 D_fake: 0.456 
saving the latest model (epoch 63, total_steps 180000)
End of epoch 63 / 200 	 Time Taken: 187 sec
(epoch: 64, iters: 960, time: 0.040) G_GAN: 0.623 G_GAN_Feat: 1.579 G_VGG: 1.842 D_real: 0.277 D_fake: 0.524 
End of epoch 64 / 200 	 Time Taken: 183 sec
(epoch: 65, iters: 480, time: 0.041) G_GAN: 1.226 G_GAN_Feat: 2.379 G_VGG: 2.846 D_real: 0.661 D_fake: 0.215 
(epoch: 65, iters: 2880, time: 0.047) G_GAN: 0.373 G_GAN_Feat: 1.174 G_VGG: 2.139 D_real: 0.285 D_fake: 0.859 
End of epoch 65 / 200 	 Time Taken: 188 sec
(epoch: 66, iters: 2400, time: 0.038) G_GAN: 0.629 G_GAN_Feat: 1.053 G_VGG: 1.856 D_real: 0.454 D_fake: 0.477 
End of epoch 66 / 200 	 Time Taken: 187 sec
(epoch: 67, iters: 1920, time: 0.037) G_GAN: 0.602 G_GAN_Feat: 1.658 G_VGG: 1.965 D_real: 0.290 D_fake: 0.489 
saving the latest model (epoch 67, total_steps 192000)
End of epoch 67 / 200 	 Time Taken: 184 sec
(epoch: 68, iters: 1440, time: 0.041) G_GAN: 0.625 G_GAN_Feat: 0.983 G_VGG: 1.800 D_real: 0.462 D_fake: 0.432 
End of epoch 68 / 200 	 Time Taken: 186 sec
(epoch: 69, iters: 960, time: 0.039) G_GAN: 0.931 G_GAN_Feat: 1.471 G_VGG: 1.893 D_real: 0.625 D_fake: 0.318 
End of epoch 69 / 200 	 Time Taken: 185 sec
(epoch: 70, iters: 480, time: 0.040) G_GAN: 0.657 G_GAN_Feat: 1.287 G_VGG: 1.762 D_real: 0.384 D_fake: 0.430 
(epoch: 70, iters: 2880, time: 0.034) G_GAN: 1.386 G_GAN_Feat: 2.813 G_VGG: 2.143 D_real: 0.172 D_fake: 0.146 
End of epoch 70 / 200 	 Time Taken: 185 sec
saving the model at the end of epoch 70, iters 201600
(epoch: 71, iters: 2400, time: 0.039) G_GAN: 0.851 G_GAN_Feat: 1.955 G_VGG: 1.921 D_real: 0.199 D_fake: 0.334 
saving the latest model (epoch 71, total_steps 204000)
End of epoch 71 / 200 	 Time Taken: 185 sec
(epoch: 72, iters: 1920, time: 0.038) G_GAN: 0.580 G_GAN_Feat: 1.410 G_VGG: 1.825 D_real: 0.346 D_fake: 0.683 
End of epoch 72 / 200 	 Time Taken: 185 sec
(epoch: 73, iters: 1440, time: 0.042) G_GAN: 1.843 G_GAN_Feat: 2.900 G_VGG: 1.873 D_real: 0.316 D_fake: 0.043 
End of epoch 73 / 200 	 Time Taken: 185 sec
(epoch: 74, iters: 960, time: 0.039) G_GAN: 1.768 G_GAN_Feat: 3.334 G_VGG: 2.580 D_real: 0.143 D_fake: 0.060 
End of epoch 74 / 200 	 Time Taken: 184 sec
(epoch: 75, iters: 480, time: 0.036) G_GAN: 1.451 G_GAN_Feat: 2.458 G_VGG: 2.109 D_real: 0.391 D_fake: 0.124 
(epoch: 75, iters: 2880, time: 0.037) G_GAN: 1.013 G_GAN_Feat: 2.253 G_VGG: 1.942 D_real: 0.122 D_fake: 0.318 
saving the latest model (epoch 75, total_steps 216000)
End of epoch 75 / 200 	 Time Taken: 189 sec
(epoch: 76, iters: 2400, time: 0.041) G_GAN: 0.759 G_GAN_Feat: 1.782 G_VGG: 1.815 D_real: 0.094 D_fake: 0.430 
End of epoch 76 / 200 	 Time Taken: 185 sec
(epoch: 77, iters: 1920, time: 0.042) G_GAN: 1.385 G_GAN_Feat: 2.951 G_VGG: 1.828 D_real: 0.059 D_fake: 0.149 
End of epoch 77 / 200 	 Time Taken: 185 sec
(epoch: 78, iters: 1440, time: 0.045) G_GAN: 1.257 G_GAN_Feat: 2.052 G_VGG: 1.749 D_real: 0.410 D_fake: 0.160 
End of epoch 78 / 200 	 Time Taken: 183 sec
(epoch: 79, iters: 960, time: 0.043) G_GAN: 1.073 G_GAN_Feat: 2.138 G_VGG: 1.910 D_real: 0.216 D_fake: 0.262 
End of epoch 79 / 200 	 Time Taken: 186 sec
(epoch: 80, iters: 480, time: 0.036) G_GAN: 1.967 G_GAN_Feat: 3.100 G_VGG: 1.906 D_real: 0.185 D_fake: 0.064 
saving the latest model (epoch 80, total_steps 228000)
(epoch: 80, iters: 2880, time: 0.039) G_GAN: 1.339 G_GAN_Feat: 2.692 G_VGG: 1.896 D_real: 0.186 D_fake: 0.169 
End of epoch 80 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 80, iters 230400
(epoch: 81, iters: 2400, time: 0.042) G_GAN: 1.698 G_GAN_Feat: 2.808 G_VGG: 1.891 D_real: 0.435 D_fake: 0.127 
End of epoch 81 / 200 	 Time Taken: 185 sec
(epoch: 82, iters: 1920, time: 0.042) G_GAN: 1.067 G_GAN_Feat: 2.343 G_VGG: 1.788 D_real: 0.207 D_fake: 0.605 
End of epoch 82 / 200 	 Time Taken: 184 sec
(epoch: 83, iters: 1440, time: 0.037) G_GAN: 1.223 G_GAN_Feat: 2.105 G_VGG: 1.753 D_real: 0.193 D_fake: 0.436 
End of epoch 83 / 200 	 Time Taken: 184 sec
(epoch: 84, iters: 960, time: 0.039) G_GAN: 0.577 G_GAN_Feat: 0.826 G_VGG: 1.717 D_real: 0.480 D_fake: 0.451 
saving the latest model (epoch 84, total_steps 240000)
End of epoch 84 / 200 	 Time Taken: 186 sec
(epoch: 85, iters: 480, time: 0.042) G_GAN: 0.639 G_GAN_Feat: 1.745 G_VGG: 1.648 D_real: 0.474 D_fake: 0.501 
(epoch: 85, iters: 2880, time: 0.042) G_GAN: 1.347 G_GAN_Feat: 1.977 G_VGG: 1.683 D_real: 0.409 D_fake: 0.373 
End of epoch 85 / 200 	 Time Taken: 185 sec
(epoch: 86, iters: 2400, time: 0.042) G_GAN: 1.732 G_GAN_Feat: 3.177 G_VGG: 1.837 D_real: 0.047 D_fake: 0.116 
End of epoch 86 / 200 	 Time Taken: 185 sec
(epoch: 87, iters: 1920, time: 0.040) G_GAN: 0.507 G_GAN_Feat: 0.751 G_VGG: 1.653 D_real: 0.423 D_fake: 0.524 
End of epoch 87 / 200 	 Time Taken: 183 sec
(epoch: 88, iters: 1440, time: 0.041) G_GAN: 1.353 G_GAN_Feat: 2.217 G_VGG: 1.689 D_real: 0.230 D_fake: 0.169 
saving the latest model (epoch 88, total_steps 252000)
End of epoch 88 / 200 	 Time Taken: 184 sec
(epoch: 89, iters: 960, time: 0.041) G_GAN: 1.714 G_GAN_Feat: 2.053 G_VGG: 1.749 D_real: 0.932 D_fake: 0.065 
End of epoch 89 / 200 	 Time Taken: 185 sec
(epoch: 90, iters: 480, time: 0.039) G_GAN: 1.095 G_GAN_Feat: 2.566 G_VGG: 1.741 D_real: 0.049 D_fake: 0.248 
(epoch: 90, iters: 2880, time: 0.038) G_GAN: 0.568 G_GAN_Feat: 1.426 G_VGG: 1.731 D_real: 0.268 D_fake: 0.606 
End of epoch 90 / 200 	 Time Taken: 185 sec
saving the model at the end of epoch 90, iters 259200
(epoch: 91, iters: 2400, time: 0.041) G_GAN: 0.791 G_GAN_Feat: 1.967 G_VGG: 1.665 D_real: 0.072 D_fake: 0.409 
End of epoch 91 / 200 	 Time Taken: 185 sec
(epoch: 92, iters: 1920, time: 0.040) G_GAN: 2.252 G_GAN_Feat: 3.130 G_VGG: 1.785 D_real: 0.244 D_fake: 0.048 
saving the latest model (epoch 92, total_steps 264000)
End of epoch 92 / 200 	 Time Taken: 186 sec
(epoch: 93, iters: 1440, time: 0.042) G_GAN: 2.094 G_GAN_Feat: 3.422 G_VGG: 1.716 D_real: 0.052 D_fake: 0.031 
End of epoch 93 / 200 	 Time Taken: 184 sec
(epoch: 94, iters: 960, time: 0.042) G_GAN: 1.270 G_GAN_Feat: 1.704 G_VGG: 1.655 D_real: 0.417 D_fake: 0.276 
End of epoch 94 / 200 	 Time Taken: 185 sec
(epoch: 95, iters: 480, time: 0.037) G_GAN: 1.591 G_GAN_Feat: 2.683 G_VGG: 1.729 D_real: 0.251 D_fake: 0.060 
(epoch: 95, iters: 2880, time: 0.040) G_GAN: 2.090 G_GAN_Feat: 2.763 G_VGG: 1.769 D_real: 0.459 D_fake: 0.079 
End of epoch 95 / 200 	 Time Taken: 183 sec
(epoch: 96, iters: 2400, time: 0.039) G_GAN: 0.514 G_GAN_Feat: 2.179 G_VGG: 1.644 D_real: 0.060 D_fake: 0.609 
saving the latest model (epoch 96, total_steps 276000)
End of epoch 96 / 200 	 Time Taken: 186 sec
(epoch: 97, iters: 1920, time: 0.043) G_GAN: 0.908 G_GAN_Feat: 2.627 G_VGG: 1.669 D_real: 0.122 D_fake: 0.521 
End of epoch 97 / 200 	 Time Taken: 184 sec
(epoch: 98, iters: 1440, time: 0.043) G_GAN: 1.086 G_GAN_Feat: 2.775 G_VGG: 1.698 D_real: 0.075 D_fake: 0.358 
End of epoch 98 / 200 	 Time Taken: 185 sec
(epoch: 99, iters: 960, time: 0.044) G_GAN: 2.238 G_GAN_Feat: 3.341 G_VGG: 1.704 D_real: 0.226 D_fake: 0.053 
End of epoch 99 / 200 	 Time Taken: 182 sec
(epoch: 100, iters: 480, time: 0.038) G_GAN: 1.387 G_GAN_Feat: 2.488 G_VGG: 1.691 D_real: 0.266 D_fake: 0.177 
(epoch: 100, iters: 2880, time: 0.039) G_GAN: 1.491 G_GAN_Feat: 3.674 G_VGG: 1.626 D_real: 0.091 D_fake: 0.098 
saving the latest model (epoch 100, total_steps 288000)
End of epoch 100 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 100, iters 288000
(epoch: 101, iters: 2400, time: 0.036) G_GAN: 1.673 G_GAN_Feat: 2.194 G_VGG: 1.634 D_real: 0.914 D_fake: 0.095 
End of epoch 101 / 200 	 Time Taken: 186 sec
(epoch: 102, iters: 1920, time: 0.045) G_GAN: 1.547 G_GAN_Feat: 2.971 G_VGG: 1.621 D_real: 0.077 D_fake: 0.097 
End of epoch 102 / 200 	 Time Taken: 180 sec
(epoch: 103, iters: 1440, time: 0.043) G_GAN: 1.138 G_GAN_Feat: 2.161 G_VGG: 1.783 D_real: 0.150 D_fake: 0.298 
End of epoch 103 / 200 	 Time Taken: 182 sec
(epoch: 104, iters: 960, time: 0.043) G_GAN: 0.528 G_GAN_Feat: 1.020 G_VGG: 1.643 D_real: 0.381 D_fake: 0.644 
End of epoch 104 / 200 	 Time Taken: 186 sec
(epoch: 105, iters: 480, time: 0.042) G_GAN: 1.777 G_GAN_Feat: 2.844 G_VGG: 1.651 D_real: 0.233 D_fake: 0.049 
saving the latest model (epoch 105, total_steps 300000)
(epoch: 105, iters: 2880, time: 0.045) G_GAN: 2.123 G_GAN_Feat: 3.845 G_VGG: 1.659 D_real: 0.068 D_fake: 0.022 
End of epoch 105 / 200 	 Time Taken: 186 sec
(epoch: 106, iters: 2400, time: 0.041) G_GAN: 1.141 G_GAN_Feat: 2.788 G_VGG: 1.628 D_real: 0.214 D_fake: 0.200 
End of epoch 106 / 200 	 Time Taken: 185 sec
(epoch: 107, iters: 1920, time: 0.042) G_GAN: 0.978 G_GAN_Feat: 2.582 G_VGG: 1.585 D_real: 0.061 D_fake: 0.309 
End of epoch 107 / 200 	 Time Taken: 183 sec
(epoch: 108, iters: 1440, time: 0.039) G_GAN: 1.384 G_GAN_Feat: 2.596 G_VGG: 1.608 D_real: 0.031 D_fake: 0.171 
End of epoch 108 / 200 	 Time Taken: 185 sec
(epoch: 109, iters: 960, time: 0.044) G_GAN: 2.015 G_GAN_Feat: 4.460 G_VGG: 1.676 D_real: 0.033 D_fake: 0.017 
saving the latest model (epoch 109, total_steps 312000)
End of epoch 109 / 200 	 Time Taken: 186 sec
(epoch: 110, iters: 480, time: 0.037) G_GAN: 1.470 G_GAN_Feat: 2.373 G_VGG: 1.678 D_real: 0.191 D_fake: 0.155 
(epoch: 110, iters: 2880, time: 0.044) G_GAN: 1.043 G_GAN_Feat: 1.590 G_VGG: 1.636 D_real: 0.339 D_fake: 0.222 
End of epoch 110 / 200 	 Time Taken: 184 sec
saving the model at the end of epoch 110, iters 316800
(epoch: 111, iters: 2400, time: 0.039) G_GAN: 1.900 G_GAN_Feat: 4.530 G_VGG: 1.680 D_real: 0.020 D_fake: 0.014 
End of epoch 111 / 200 	 Time Taken: 186 sec
(epoch: 112, iters: 1920, time: 0.039) G_GAN: 1.324 G_GAN_Feat: 2.090 G_VGG: 1.608 D_real: 0.294 D_fake: 0.142 
End of epoch 112 / 200 	 Time Taken: 183 sec
(epoch: 113, iters: 1440, time: 0.037) G_GAN: 0.399 G_GAN_Feat: 1.381 G_VGG: 1.557 D_real: 0.049 D_fake: 0.848 
saving the latest model (epoch 113, total_steps 324000)
End of epoch 113 / 200 	 Time Taken: 184 sec
(epoch: 114, iters: 960, time: 0.046) G_GAN: 0.456 G_GAN_Feat: 1.939 G_VGG: 1.600 D_real: 0.066 D_fake: 0.903 
End of epoch 114 / 200 	 Time Taken: 184 sec
(epoch: 115, iters: 480, time: 0.040) G_GAN: 2.006 G_GAN_Feat: 3.909 G_VGG: 1.660 D_real: 0.030 D_fake: 0.025 
(epoch: 115, iters: 2880, time: 0.044) G_GAN: 1.032 G_GAN_Feat: 2.631 G_VGG: 1.678 D_real: 0.118 D_fake: 0.306 
End of epoch 115 / 200 	 Time Taken: 184 sec
(epoch: 116, iters: 2400, time: 0.035) G_GAN: 1.513 G_GAN_Feat: 3.007 G_VGG: 1.569 D_real: 0.034 D_fake: 0.204 
End of epoch 116 / 200 	 Time Taken: 183 sec
(epoch: 117, iters: 1920, time: 0.043) G_GAN: 1.782 G_GAN_Feat: 3.432 G_VGG: 1.575 D_real: 0.021 D_fake: 0.061 
saving the latest model (epoch 117, total_steps 336000)
End of epoch 117 / 200 	 Time Taken: 185 sec
(epoch: 118, iters: 1440, time: 0.038) G_GAN: 1.161 G_GAN_Feat: 2.371 G_VGG: 1.584 D_real: 0.131 D_fake: 0.189 
End of epoch 118 / 200 	 Time Taken: 185 sec
(epoch: 119, iters: 960, time: 0.036) G_GAN: 0.887 G_GAN_Feat: 1.809 G_VGG: 1.586 D_real: 0.079 D_fake: 0.301 
End of epoch 119 / 200 	 Time Taken: 184 sec
(epoch: 120, iters: 480, time: 0.035) G_GAN: 1.075 G_GAN_Feat: 2.078 G_VGG: 1.524 D_real: 0.036 D_fake: 0.395 
(epoch: 120, iters: 2880, time: 0.040) G_GAN: 2.012 G_GAN_Feat: 3.176 G_VGG: 1.636 D_real: 0.037 D_fake: 0.044 
End of epoch 120 / 200 	 Time Taken: 186 sec
saving the model at the end of epoch 120, iters 345600
(epoch: 121, iters: 2400, time: 0.038) G_GAN: 1.121 G_GAN_Feat: 2.199 G_VGG: 1.573 D_real: 0.022 D_fake: 0.329 
saving the latest model (epoch 121, total_steps 348000)
End of epoch 121 / 200 	 Time Taken: 186 sec
(epoch: 122, iters: 1920, time: 0.040) G_GAN: 1.688 G_GAN_Feat: 2.708 G_VGG: 1.534 D_real: 0.050 D_fake: 0.114 
End of epoch 122 / 200 	 Time Taken: 186 sec
(epoch: 123, iters: 1440, time: 0.041) G_GAN: 0.856 G_GAN_Feat: 1.736 G_VGG: 1.532 D_real: 0.317 D_fake: 0.471 
End of epoch 123 / 200 	 Time Taken: 183 sec
(epoch: 124, iters: 960, time: 0.047) G_GAN: 0.930 G_GAN_Feat: 1.778 G_VGG: 1.496 D_real: 0.261 D_fake: 0.366 
End of epoch 124 / 200 	 Time Taken: 183 sec
(epoch: 125, iters: 480, time: 0.036) G_GAN: 0.895 G_GAN_Feat: 1.643 G_VGG: 1.572 D_real: 0.396 D_fake: 0.389 
(epoch: 125, iters: 2880, time: 0.037) G_GAN: 0.849 G_GAN_Feat: 2.041 G_VGG: 1.604 D_real: 0.057 D_fake: 0.459 
saving the latest model (epoch 125, total_steps 360000)
End of epoch 125 / 200 	 Time Taken: 188 sec
(epoch: 126, iters: 2400, time: 0.034) G_GAN: 1.501 G_GAN_Feat: 2.874 G_VGG: 1.596 D_real: 0.032 D_fake: 0.099 
End of epoch 126 / 200 	 Time Taken: 182 sec
(epoch: 127, iters: 1920, time: 0.038) G_GAN: 0.532 G_GAN_Feat: 1.696 G_VGG: 1.578 D_real: 0.053 D_fake: 0.707 
End of epoch 127 / 200 	 Time Taken: 183 sec
(epoch: 128, iters: 1440, time: 0.046) G_GAN: 2.030 G_GAN_Feat: 2.798 G_VGG: 1.547 D_real: 0.176 D_fake: 0.045 
End of epoch 128 / 200 	 Time Taken: 182 sec
(epoch: 129, iters: 960, time: 0.041) G_GAN: 1.857 G_GAN_Feat: 3.972 G_VGG: 1.500 D_real: 0.021 D_fake: 0.026 
End of epoch 129 / 200 	 Time Taken: 183 sec
(epoch: 130, iters: 480, time: 0.037) G_GAN: 1.701 G_GAN_Feat: 2.400 G_VGG: 1.601 D_real: 0.640 D_fake: 0.079 
saving the latest model (epoch 130, total_steps 372000)
(epoch: 130, iters: 2880, time: 0.040) G_GAN: 0.581 G_GAN_Feat: 1.874 G_VGG: 1.534 D_real: 0.039 D_fake: 0.709 
End of epoch 130 / 200 	 Time Taken: 187 sec
saving the model at the end of epoch 130, iters 374400
(epoch: 131, iters: 2400, time: 0.042) G_GAN: 1.300 G_GAN_Feat: 2.110 G_VGG: 1.524 D_real: 0.084 D_fake: 0.248 
End of epoch 131 / 200 	 Time Taken: 185 sec
(epoch: 132, iters: 1920, time: 0.044) G_GAN: 1.013 G_GAN_Feat: 2.236 G_VGG: 1.532 D_real: 0.030 D_fake: 0.451 
End of epoch 132 / 200 	 Time Taken: 184 sec
(epoch: 133, iters: 1440, time: 0.041) G_GAN: 1.204 G_GAN_Feat: 2.166 G_VGG: 1.584 D_real: 0.051 D_fake: 0.177 
End of epoch 133 / 200 	 Time Taken: 185 sec
(epoch: 134, iters: 960, time: 0.036) G_GAN: 1.110 G_GAN_Feat: 1.927 G_VGG: 1.580 D_real: 0.129 D_fake: 0.297 
saving the latest model (epoch 134, total_steps 384000)
End of epoch 134 / 200 	 Time Taken: 184 sec
(epoch: 135, iters: 480, time: 0.040) G_GAN: 1.080 G_GAN_Feat: 1.863 G_VGG: 1.540 D_real: 0.196 D_fake: 0.330 
(epoch: 135, iters: 2880, time: 0.043) G_GAN: 1.504 G_GAN_Feat: 2.109 G_VGG: 1.542 D_real: 0.138 D_fake: 0.118 
End of epoch 135 / 200 	 Time Taken: 186 sec
(epoch: 136, iters: 2400, time: 0.041) G_GAN: 1.465 G_GAN_Feat: 2.362 G_VGG: 1.487 D_real: 0.110 D_fake: 0.166 
End of epoch 136 / 200 	 Time Taken: 184 sec
(epoch: 137, iters: 1920, time: 0.044) G_GAN: 0.891 G_GAN_Feat: 1.819 G_VGG: 1.457 D_real: 0.054 D_fake: 0.515 
End of epoch 137 / 200 	 Time Taken: 185 sec
(epoch: 138, iters: 1440, time: 0.043) G_GAN: 2.007 G_GAN_Feat: 2.607 G_VGG: 1.526 D_real: 0.441 D_fake: 0.038 
saving the latest model (epoch 138, total_steps 396000)
End of epoch 138 / 200 	 Time Taken: 186 sec
(epoch: 139, iters: 960, time: 0.045) G_GAN: 1.323 G_GAN_Feat: 1.820 G_VGG: 1.561 D_real: 0.520 D_fake: 0.174 
End of epoch 139 / 200 	 Time Taken: 184 sec
(epoch: 140, iters: 480, time: 0.040) G_GAN: 0.669 G_GAN_Feat: 1.602 G_VGG: 1.516 D_real: 0.049 D_fake: 0.598 
(epoch: 140, iters: 2880, time: 0.039) G_GAN: 0.739 G_GAN_Feat: 1.629 G_VGG: 1.470 D_real: 0.141 D_fake: 0.515 
End of epoch 140 / 200 	 Time Taken: 185 sec
saving the model at the end of epoch 140, iters 403200
(epoch: 141, iters: 2400, time: 0.043) G_GAN: 0.726 G_GAN_Feat: 1.198 G_VGG: 1.490 D_real: 0.510 D_fake: 0.423 
End of epoch 141 / 200 	 Time Taken: 185 sec
(epoch: 142, iters: 1920, time: 0.036) G_GAN: 2.001 G_GAN_Feat: 2.202 G_VGG: 1.531 D_real: 0.514 D_fake: 0.056 
saving the latest model (epoch 142, total_steps 408000)
End of epoch 142 / 200 	 Time Taken: 184 sec
(epoch: 143, iters: 1440, time: 0.040) G_GAN: 0.772 G_GAN_Feat: 1.784 G_VGG: 1.508 D_real: 0.117 D_fake: 0.576 
End of epoch 143 / 200 	 Time Taken: 183 sec
(epoch: 144, iters: 960, time: 0.043) G_GAN: 1.091 G_GAN_Feat: 1.871 G_VGG: 1.539 D_real: 0.140 D_fake: 0.239 
End of epoch 144 / 200 	 Time Taken: 181 sec
(epoch: 145, iters: 480, time: 0.042) G_GAN: 2.194 G_GAN_Feat: 2.533 G_VGG: 1.536 D_real: 0.226 D_fake: 0.059 
(epoch: 145, iters: 2880, time: 0.042) G_GAN: 1.017 G_GAN_Feat: 1.399 G_VGG: 1.536 D_real: 0.477 D_fake: 0.243 
End of epoch 145 / 200 	 Time Taken: 184 sec
(epoch: 146, iters: 2400, time: 0.038) G_GAN: 1.534 G_GAN_Feat: 2.171 G_VGG: 1.528 D_real: 0.090 D_fake: 0.088 
saving the latest model (epoch 146, total_steps 420000)
End of epoch 146 / 200 	 Time Taken: 184 sec
(epoch: 147, iters: 1920, time: 0.043) G_GAN: 0.771 G_GAN_Feat: 1.932 G_VGG: 1.568 D_real: 0.309 D_fake: 0.430 
End of epoch 147 / 200 	 Time Taken: 184 sec
(epoch: 148, iters: 1440, time: 0.041) G_GAN: 1.698 G_GAN_Feat: 2.174 G_VGG: 1.538 D_real: 0.101 D_fake: 0.070


## log for 200 epochs with 16 GPUs
```

export HOME='/mnt/old/home-hryu' && cd /mnt/old/git/pix2pixHD && 
python train.py --name label2city_512p-16gpu--batch64 --batchSize 64 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15


export HOME='/mnt/old/home-hryu' && cd /mnt/old/git/pix2pixHD && python train.py --name label2city_512p-16gpu--batch16 --batchSize 16 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15



python train.py --name label2city_512p-16gpu--batch64 --batchSize 64 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
cd /mnt/old/git/pix2pixHD

I have no name!@b2a4cebbffdd:/mnt/old/git/pix2pixHD$ python train.py --name label2city_512p-16gpu--batch64 --batchSize 64 --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
------------ Options -------------
batchSize: 64
beta1: 0.5
checkpoints_dir: ./checkpoints
continue_train: False
data_type: 32
dataroot: ./datasets/cityscapes/
debug: False
display_freq: 100
display_winsize: 512
feat_num: 3
fineSize: 512
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
input_nc: 3
instance_feat: False
isTrain: True
label_feat: False
label_nc: 35
lambda_feat: 10.0
loadSize: 1024
load_features: False
load_pretrain: 
lr: 0.0002
max_dataset_size: inf
model: pix2pixHD
nThreads: 2
n_blocks_global: 9
n_blocks_local: 3
n_clusters: 10
n_downsample_E: 4
n_downsample_global: 4
n_layers_D: 3
n_local_enhancers: 1
name: label2city_512p-16gpu--batch64
ndf: 64
nef: 16
netG: global
ngf: 64
niter: 100
niter_decay: 100
niter_fix_global: 0
no_flip: False
no_ganFeat_loss: False
no_html: False
no_instance: False
no_lsgan: False
no_vgg_loss: False
norm: instance
num_D: 2
output_nc: 3
phase: train
pool_size: 0
print_freq: 100
resize_or_crop: scale_width
save_epoch_freq: 10
save_latest_freq: 1000
serial_batches: False
tf_log: False
use_dropout: False
verbose: False
which_epoch: latest
-------------- End ----------------
CustomDatasetDataLoader
dataset [AlignedDataset] was created
#training images = 2944
GlobalGenerator(
  (model): Sequential(
    (0): ReflectionPad2d((3, 3, 3, 3))
    (1): Conv2d(36, 64, kernel_size=(7, 7), stride=(1, 1))
    (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (3): ReLU(inplace)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (6): ReLU(inplace)
    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (9): ReLU(inplace)
    (10): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (11): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (12): ReLU(inplace)
    (13): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (14): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (15): ReLU(inplace)
    (16): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (17): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (18): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (19): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (20): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (21): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (22): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (23): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (24): ResnetBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (25): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (26): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (27): ReLU(inplace)
    (28): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (29): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (30): ReLU(inplace)
    (31): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (32): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (33): ReLU(inplace)
    (34): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (35): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (36): ReLU(inplace)
    (37): ReflectionPad2d((3, 3, 3, 3))
    (38): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
    (39): Tanh()
  )
)
MultiscaleDiscriminator(
  (scale0_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale0_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (scale1_layer0): Sequential(
    (0): Conv2d(39, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (scale1_layer4): Sequential(
    (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
  )
  (downsample): AvgPool2d(kernel_size=3, stride=2, padding=[1, 1])
)
create web directory ./checkpoints/label2city_512p-16gpu--batch64/web...
/opt/conda/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:58: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '


(epoch: 59, iters: 1920, time: 0.093) G_GAN: 0.984 G_GAN_Feat: 1.836 G_VGG: 1.243 D_real: 1.047 D_fake: 0.284 
(epoch: 59, iters: 2320, time: 0.086) G_GAN: 2.279 G_GAN_Feat: 2.848 G_VGG: 1.217 D_real: 0.225 D_fake: 0.063 
(epoch: 59, iters: 2720, time: 0.086) G_GAN: 0.570 G_GAN_Feat: 1.300 G_VGG: 1.149 D_real: 0.291 D_fake: 0.537 
(epoch: 60, iters: 160, time: 0.087) G_GAN: 1.326 G_GAN_Feat: 2.277 G_VGG: 1.139 D_real: 0.210 D_fake: 0.149 
(epoch: 60, iters: 560, time: 0.091) G_GAN: 1.398 G_GAN_Feat: 2.265 G_VGG: 1.298 D_real: 0.395 D_fake: 0.156 
(epoch: 60, iters: 960, time: 0.085) G_GAN: 1.305 G_GAN_Feat: 1.948 G_VGG: 1.121 D_real: 0.322 D_fake: 0.279 
(epoch: 60, iters: 1360, time: 0.087) G_GAN: 1.352 G_GAN_Feat: 2.384 G_VGG: 1.194 D_real: 0.162 D_fake: 0.196 
(epoch: 60, iters: 1760, time: 0.096) G_GAN: 1.211 G_GAN_Feat: 2.890 G_VGG: 1.353 D_real: 0.033 D_fake: 0.212 
(epoch: 60, iters: 2160, time: 0.095) G_GAN: 1.122 G_GAN_Feat: 2.974 G_VGG: 1.300 D_real: 0.204 D_fake: 0.202 
(epoch: 60, iters: 2560, time: 0.088) G_GAN: 0.990 G_GAN_Feat: 1.787 G_VGG: 1.171 D_real: 0.696 D_fake: 0.370 
(epoch: 60, iters: 2960, time: 0.084) G_GAN: 1.525 G_GAN_Feat: 2.726 G_VGG: 1.257 D_real: 0.045 D_fake: 0.224 
(epoch: 61, iters: 400, time: 0.087) G_GAN: 0.550 G_GAN_Feat: 1.496 G_VGG: 1.253 D_real: 0.338 D_fake: 0.503 
(epoch: 61, iters: 800, time: 0.093) G_GAN: 1.284 G_GAN_Feat: 2.760 G_VGG: 1.266 D_real: 0.047 D_fake: 0.168 
(epoch: 61, iters: 1200, time: 0.086) G_GAN: 1.193 G_GAN_Feat: 2.456 G_VGG: 1.275 D_real: 0.192 D_fake: 0.168 
(epoch: 61, iters: 1600, time: 0.094) G_GAN: 0.937 G_GAN_Feat: 1.711 G_VGG: 1.108 D_real: 0.033 D_fake: 0.413 
(epoch: 61, iters: 2000, time: 0.101) G_GAN: 2.731 G_GAN_Feat: 3.216 G_VGG: 1.334 D_real: 0.470 D_fake: 0.075 
(epoch: 61, iters: 2400, time: 0.095) G_GAN: 2.051 G_GAN_Feat: 3.244 G_VGG: 1.116 D_real: 0.054 D_fake: 0.057 
(epoch: 61, iters: 2800, time: 0.098) G_GAN: 1.788 G_GAN_Feat: 2.342 G_VGG: 1.224 D_real: 0.293 D_fake: 0.132 
(epoch: 62, iters: 240, time: 0.094) G_GAN: 0.579 G_GAN_Feat: 2.061 G_VGG: 1.247 D_real: 0.155 D_fake: 0.553 
(epoch: 62, iters: 640, time: 0.099) G_GAN: 1.270 G_GAN_Feat: 1.955 G_VGG: 1.217 D_real: 0.491 D_fake: 0.347 
(epoch: 62, iters: 1040, time: 0.090) G_GAN: 1.719 G_GAN_Feat: 3.637 G_VGG: 1.332 D_real: 0.056 D_fake: 0.174 
(epoch: 62, iters: 1440, time: 0.091) G_GAN: 0.813 G_GAN_Feat: 1.660 G_VGG: 1.051 D_real: 0.250 D_fake: 0.383 
(epoch: 62, iters: 1840, time: 0.097) G_GAN: 1.310 G_GAN_Feat: 2.501 G_VGG: 1.229 D_real: 0.168 D_fake: 0.223 
(epoch: 62, iters: 2240, time: 0.095) G_GAN: 0.714 G_GAN_Feat: 1.656 G_VGG: 1.249 D_real: 0.402 D_fake: 0.456 
(epoch: 62, iters: 2640, time: 0.086) G_GAN: 0.467 G_GAN_Feat: 1.869 G_VGG: 1.154 D_real: 0.157 D_fake: 0.775 
(epoch: 63, iters: 80, time: 0.093) G_GAN: 1.126 G_GAN_Feat: 2.142 G_VGG: 1.229 D_real: 0.273 D_fake: 0.305 
(epoch: 63, iters: 480, time: 0.087) G_GAN: 0.926 G_GAN_Feat: 1.802 G_VGG: 1.161 D_real: 0.231 D_fake: 0.290 
(epoch: 63, iters: 880, time: 0.091) G_GAN: 1.464 G_GAN_Feat: 2.016 G_VGG: 1.175 D_real: 0.615 D_fake: 0.372 
(epoch: 63, iters: 1280, time: 0.089) G_GAN: 1.750 G_GAN_Feat: 2.853 G_VGG: 1.294 D_real: 0.611 D_fake: 0.086 
(epoch: 63, iters: 1680, time: 0.097) G_GAN: 0.721 G_GAN_Feat: 2.073 G_VGG: 1.238 D_real: 0.395 D_fake: 0.433 
(epoch: 63, iters: 2080, time: 0.088) G_GAN: 1.857 G_GAN_Feat: 2.599 G_VGG: 1.261 D_real: 0.242 D_fake: 0.048 
(epoch: 63, iters: 2480, time: 0.087) G_GAN: 0.789 G_GAN_Feat: 1.791 G_VGG: 1.282 D_real: 0.186 D_fake: 0.476 
(epoch: 63, iters: 2880, time: 0.100) G_GAN: 2.409 G_GAN_Feat: 2.937 G_VGG: 1.214 D_real: 0.302 D_fake: 0.058 
(epoch: 64, iters: 320, time: 0.098) G_GAN: 1.444 G_GAN_Feat: 2.777 G_VGG: 1.116 D_real: 0.439 D_fake: 0.101 
(epoch: 64, iters: 720, time: 0.085) G_GAN: 0.974 G_GAN_Feat: 1.639 G_VGG: 1.124 D_real: 0.460 D_fake: 0.456 
(epoch: 64, iters: 1120, time: 0.091) G_GAN: 1.224 G_GAN_Feat: 2.238 G_VGG: 1.095 D_real: 0.069 D_fake: 0.153 
(epoch: 64, iters: 1520, time: 0.082) G_GAN: 1.470 G_GAN_Feat: 1.691 G_VGG: 1.170 D_real: 0.353 D_fake: 0.161 
(epoch: 64, iters: 1920, time: 0.092) G_GAN: 0.754 G_GAN_Feat: 1.171 G_VGG: 1.065 D_real: 0.369 D_fake: 0.386 
(epoch: 64, iters: 2320, time: 0.090) G_GAN: 1.127 G_GAN_Feat: 1.885 G_VGG: 1.155 D_real: 0.329 D_fake: 0.392 
(epoch: 64, iters: 2720, time: 0.093) G_GAN: 1.665 G_GAN_Feat: 2.734 G_VGG: 1.220 D_real: 0.082 D_fake: 0.072 
(epoch: 65, iters: 160, time: 0.093) G_GAN: 0.803 G_GAN_Feat: 2.078 G_VGG: 1.313 D_real: 0.174 D_fake: 0.456 
(epoch: 65, iters: 560, time: 0.092) G_GAN: 1.131 G_GAN_Feat: 2.203 G_VGG: 1.264 D_real: 0.108 D_fake: 0.219 
(epoch: 65, iters: 960, time: 0.087) G_GAN: 1.550 G_GAN_Feat: 2.122 G_VGG: 1.060 D_real: 0.310 D_fake: 0.285 
(epoch: 65, iters: 1360, time: 0.093) G_GAN: 1.008 G_GAN_Feat: 2.303 G_VGG: 1.068 D_real: 0.143 D_fake: 0.232 
(epoch: 65, iters: 1760, time: 0.094) G_GAN: 1.445 G_GAN_Feat: 1.868 G_VGG: 1.106 D_real: 0.365 D_fake: 0.169 
(epoch: 65, iters: 2160, time: 0.093) G_GAN: 1.416 G_GAN_Feat: 2.065 G_VGG: 1.067 D_real: 0.068 D_fake: 0.302 
(epoch: 65, iters: 2560, time: 0.088) G_GAN: 0.654 G_GAN_Feat: 2.297 G_VGG: 1.147 D_real: 0.110 D_fake: 0.504 
(epoch: 65, iters: 2960, time: 0.087) G_GAN: 1.424 G_GAN_Feat: 1.389 G_VGG: 1.146 D_real: 0.968 D_fake: 0.190 
(epoch: 66, iters: 400, time: 0.087) G_GAN: 1.464 G_GAN_Feat: 2.717 G_VGG: 1.021 D_real: 0.024 D_fake: 0.101 
(epoch: 66, iters: 800, time: 0.087) G_GAN: 1.245 G_GAN_Feat: 2.524 G_VGG: 1.117 D_real: 0.067 D_fake: 0.158 
(epoch: 66, iters: 1200, time: 0.089) G_GAN: 0.723 G_GAN_Feat: 1.549 G_VGG: 1.158 D_real: 0.244 D_fake: 0.426 
(epoch: 66, iters: 1600, time: 0.091) G_GAN: 0.972 G_GAN_Feat: 1.759 G_VGG: 1.235 D_real: 0.368 D_fake: 0.276 
(epoch: 66, iters: 2000, time: 0.094) G_GAN: 1.910 G_GAN_Feat: 2.947 G_VGG: 1.191 D_real: 0.105 D_fake: 0.033 
(epoch: 66, iters: 2400, time: 0.098) G_GAN: 1.641 G_GAN_Feat: 2.300 G_VGG: 1.131 D_real: 0.087 D_fake: 0.193 
(epoch: 66, iters: 2800, time: 0.086) G_GAN: 1.282 G_GAN_Feat: 1.772 G_VGG: 1.153 D_real: 0.370 D_fake: 0.383 
(epoch: 67, iters: 240, time: 0.091) G_GAN: 1.649 G_GAN_Feat: 1.985 G_VGG: 1.101 D_real: 0.265 D_fake: 0.080 
(epoch: 67, iters: 640, time: 0.096) G_GAN: 1.326 G_GAN_Feat: 2.597 G_VGG: 1.182 D_real: 0.416 D_fake: 0.283 
(epoch: 67, iters: 1040, time: 0.094) G_GAN: 0.613 G_GAN_Feat: 1.805 G_VGG: 1.122 D_real: 0.051 D_fake: 0.658 
(epoch: 67, iters: 1440, time: 0.085) G_GAN: 1.022 G_GAN_Feat: 2.205 G_VGG: 1.217 D_real: 0.125 D_fake: 0.348 
(epoch: 67, iters: 1840, time: 0.095) G_GAN: 1.623 G_GAN_Feat: 2.088 G_VGG: 1.136 D_real: 0.293 D_fake: 0.152 
(epoch: 67, iters: 2240, time: 0.083) G_GAN: 1.220 G_GAN_Feat: 1.836 G_VGG: 1.146 D_real: 0.453 D_fake: 0.228 
(epoch: 67, iters: 2640, time: 0.088) G_GAN: 1.693 G_GAN_Feat: 2.311 G_VGG: 1.128 D_real: 0.654 D_fake: 0.055 
(epoch: 68, iters: 80, time: 0.104) G_GAN: 1.101 G_GAN_Feat: 2.294 G_VGG: 1.176 D_real: 0.119 D_fake: 0.309 
(epoch: 68, iters: 480, time: 0.101) G_GAN: 0.938 G_GAN_Feat: 2.145 G_VGG: 1.233 D_real: 0.075 D_fake: 0.400 
(epoch: 68, iters: 880, time: 0.095) G_GAN: 0.990 G_GAN_Feat: 3.171 G_VGG: 1.159 D_real: 0.542 D_fake: 0.257 
(epoch: 68, iters: 1280, time: 0.085) G_GAN: 0.687 G_GAN_Feat: 1.334 G_VGG: 1.124 D_real: 0.443 D_fake: 0.700 
(epoch: 68, iters: 1680, time: 0.093) G_GAN: 1.906 G_GAN_Feat: 2.936 G_VGG: 1.406 D_real: 0.237 D_fake: 0.074 
(epoch: 68, iters: 2080, time: 0.096) G_GAN: 1.096 G_GAN_Feat: 2.122 G_VGG: 1.210 D_real: 0.370 D_fake: 0.267 
(epoch: 68, iters: 2480, time: 0.091) G_GAN: 1.199 G_GAN_Feat: 1.354 G_VGG: 0.993 D_real: 0.698 D_fake: 0.329 
(epoch: 68, iters: 2880, time: 0.092) G_GAN: 1.023 G_GAN_Feat: 1.829 G_VGG: 1.103 D_real: 0.287 D_fake: 0.348 
(epoch: 69, iters: 320, time: 0.091) G_GAN: 2.204 G_GAN_Feat: 2.688 G_VGG: 1.166 D_real: 0.315 D_fake: 0.037 
(epoch: 69, iters: 720, time: 0.100) G_GAN: 0.882 G_GAN_Feat: 1.887 G_VGG: 0.977 D_real: 0.301 D_fake: 0.342 
(epoch: 69, iters: 1120, time: 0.092) G_GAN: 1.249 G_GAN_Feat: 2.605 G_VGG: 1.067 D_real: 0.075 D_fake: 0.217 
(epoch: 69, iters: 1520, time: 0.094) G_GAN: 0.914 G_GAN_Feat: 2.181 G_VGG: 1.077 D_real: 0.058 D_fake: 0.428 
(epoch: 69, iters: 1920, time: 0.095) G_GAN: 0.773 G_GAN_Feat: 1.735 G_VGG: 1.148 D_real: 0.100 D_fake: 0.405 
(epoch: 69, iters: 2320, time: 0.093) G_GAN: 0.927 G_GAN_Feat: 1.947 G_VGG: 1.097 D_real: 0.222 D_fake: 0.347 
(epoch: 69, iters: 2720, time: 0.093) G_GAN: 1.613 G_GAN_Feat: 2.476 G_VGG: 1.171 D_real: 0.751 D_fake: 0.198 
(epoch: 70, iters: 160, time: 0.095) G_GAN: 1.173 G_GAN_Feat: 2.450 G_VGG: 1.032 D_real: 0.069 D_fake: 0.431 
(epoch: 70, iters: 560, time: 0.092) G_GAN: 0.448 G_GAN_Feat: 1.791 G_VGG: 1.137 D_real: 0.063 D_fake: 0.782 
(epoch: 70, iters: 960, time: 0.085) G_GAN: 1.069 G_GAN_Feat: 2.103 G_VGG: 1.098 D_real: 0.089 D_fake: 0.257 
(epoch: 70, iters: 1360, time: 0.085) G_GAN: 1.522 G_GAN_Feat: 3.174 G_VGG: 1.153 D_real: 0.521 D_fake: 0.082 
(epoch: 70, iters: 1760, time: 0.094) G_GAN: 1.271 G_GAN_Feat: 2.571 G_VGG: 1.173 D_real: 0.120 D_fake: 0.161 
(epoch: 70, iters: 2160, time: 0.096) G_GAN: 1.298 G_GAN_Feat: 2.209 G_VGG: 1.131 D_real: 0.065 D_fake: 0.293 
(epoch: 70, iters: 2560, time: 0.096) G_GAN: 0.939 G_GAN_Feat: 1.796 G_VGG: 1.157 D_real: 0.205 D_fake: 0.283 
(epoch: 70, iters: 2960, time: 0.086) G_GAN: 1.538 G_GAN_Feat: 2.119 G_VGG: 1.086 D_real: 0.309 D_fake: 0.150 
(epoch: 71, iters: 400, time: 0.086) G_GAN: 0.509 G_GAN_Feat: 1.058 G_VGG: 1.095 D_real: 0.375 D_fake: 0.542 
(epoch: 71, iters: 800, time: 0.087) G_GAN: 0.776 G_GAN_Feat: 1.149 G_VGG: 1.001 D_real: 0.525 D_fake: 0.368 
(epoch: 71, iters: 1200, time: 0.092) G_GAN: 2.064 G_GAN_Feat: 3.199 G_VGG: 1.174 D_real: 0.027 D_fake: 0.047 
(epoch: 71, iters: 1600, time: 0.090) G_GAN: 1.471 G_GAN_Feat: 2.207 G_VGG: 1.016 D_real: 0.113 D_fake: 0.092 
(epoch: 71, iters: 2000, time: 0.089) G_GAN: 0.616 G_GAN_Feat: 1.802 G_VGG: 1.035 D_real: 0.563 D_fake: 0.492 
(epoch: 71, iters: 2400, time: 0.094) G_GAN: 1.019 G_GAN_Feat: 2.081 G_VGG: 1.041 D_real: 0.034 D_fake: 0.613 
(epoch: 71, iters: 2800, time: 0.088) G_GAN: 1.060 G_GAN_Feat: 2.088 G_VGG: 1.009 D_real: 0.178 D_fake: 0.266 
(epoch: 72, iters: 240, time: 0.082) G_GAN: 0.801 G_GAN_Feat: 2.065 G_VGG: 1.082 D_real: 0.046 D_fake: 0.443 
(epoch: 72, iters: 640, time: 0.093) G_GAN: 1.173 G_GAN_Feat: 1.609 G_VGG: 1.047 D_real: 0.502 D_fake: 0.176 
(epoch: 72, iters: 1040, time: 0.084) G_GAN: 1.548 G_GAN_Feat: 2.907 G_VGG: 1.214 D_real: 0.345 D_fake: 0.088 
(epoch: 72, iters: 1440, time: 0.084) G_GAN: 0.537 G_GAN_Feat: 1.741 G_VGG: 1.075 D_real: 0.387 D_fake: 0.567 
(epoch: 72, iters: 1840, time: 0.087) G_GAN: 0.905 G_GAN_Feat: 1.111 G_VGG: 1.006 D_real: 0.356 D_fake: 0.305 
(epoch: 72, iters: 2240, time: 0.086) G_GAN: 1.224 G_GAN_Feat: 1.357 G_VGG: 0.996 D_real: 0.430 D_fake: 0.214 
(epoch: 72, iters: 2640, time: 0.086) G_GAN: 1.984 G_GAN_Feat: 3.198 G_VGG: 1.100 D_real: 0.202 D_fake: 0.043 
(epoch: 73, iters: 80, time: 0.095) G_GAN: 0.557 G_GAN_Feat: 1.148 G_VGG: 0.998 D_real: 0.420 D_fake: 0.537 
(epoch: 73, iters: 480, time: 0.099) G_GAN: 0.615 G_GAN_Feat: 1.311 G_VGG: 1.159 D_real: 0.460 D_fake: 0.490 
(epoch: 73, iters: 880, time: 0.096) G_GAN: 1.505 G_GAN_Feat: 2.306 G_VGG: 1.059 D_real: 0.171 D_fake: 0.095 
(epoch: 73, iters: 1280, time: 0.097) G_GAN: 0.623 G_GAN_Feat: 1.599 G_VGG: 1.003 D_real: 0.270 D_fake: 0.495 
(epoch: 73, iters: 1680, time: 0.092) G_GAN: 1.361 G_GAN_Feat: 1.863 G_VGG: 1.102 D_real: 0.282 D_fake: 0.220 
(epoch: 73, iters: 2080, time: 0.087) G_GAN: 1.071 G_GAN_Feat: 1.951 G_VGG: 1.184 D_real: 0.276 D_fake: 0.317 
(epoch: 73, iters: 2480, time: 0.092) G_GAN: 1.052 G_GAN_Feat: 2.031 G_VGG: 1.018 D_real: 0.161 D_fake: 0.417 
(epoch: 73, iters: 2880, time: 0.087) G_GAN: 0.871 G_GAN_Feat: 1.509 G_VGG: 1.027 D_real: 0.366 D_fake: 0.366 
(epoch: 74, iters: 320, time: 0.093) G_GAN: 1.048 G_GAN_Feat: 1.717 G_VGG: 1.089 D_real: 0.226 D_fake: 0.274 
(epoch: 74, iters: 720, time: 0.085) G_GAN: 1.255 G_GAN_Feat: 2.186 G_VGG: 1.104 D_real: 1.051 D_fake: 0.160 
(epoch: 74, iters: 1120, time: 0.096) G_GAN: 1.085 G_GAN_Feat: 1.825 G_VGG: 0.986 D_real: 0.149 D_fake: 0.246 
(epoch: 74, iters: 1520, time: 0.094) G_GAN: 1.373 G_GAN_Feat: 2.546 G_VGG: 1.105 D_real: 0.085 D_fake: 0.212 
(epoch: 74, iters: 1920, time: 0.096) G_GAN: 0.647 G_GAN_Feat: 2.277 G_VGG: 1.093 D_real: 0.233 D_fake: 0.845 
(epoch: 74, iters: 2320, time: 0.096) G_GAN: 1.301 G_GAN_Feat: 2.635 G_VGG: 1.052 D_real: 0.029 D_fake: 0.288 
(epoch: 74, iters: 2720, time: 0.101) G_GAN: 0.660 G_GAN_Feat: 1.708 G_VGG: 1.175 D_real: 0.427 D_fake: 0.410 
(epoch: 75, iters: 160, time: 0.093) G_GAN: 0.513 G_GAN_Feat: 1.409 G_VGG: 0.984 D_real: 0.498 D_fake: 0.626 
(epoch: 75, iters: 560, time: 0.093) G_GAN: 0.825 G_GAN_Feat: 2.175 G_VGG: 1.073 D_real: 0.208 D_fake: 0.360 
(epoch: 75, iters: 960, time: 0.087) G_GAN: 1.638 G_GAN_Feat: 2.828 G_VGG: 1.124 D_real: 0.445 D_fake: 0.063 
(epoch: 75, iters: 1360, time: 0.086) G_GAN: 1.347 G_GAN_Feat: 1.863 G_VGG: 0.990 D_real: 0.501 D_fake: 0.148 
(epoch: 75, iters: 1760, time: 0.095) G_GAN: 0.949 G_GAN_Feat: 1.229 G_VGG: 1.007 D_real: 0.288 D_fake: 0.357 
(epoch: 75, iters: 2160, time: 0.097) G_GAN: 0.657 G_GAN_Feat: 1.114 G_VGG: 0.996 D_real: 0.333 D_fake: 0.444 
(epoch: 75, iters: 2560, time: 0.093) G_GAN: 0.921 G_GAN_Feat: 1.594 G_VGG: 1.084 D_real: 0.212 D_fake: 0.285 
(epoch: 75, iters: 2960, time: 0.087) G_GAN: 1.197 G_GAN_Feat: 2.373 G_VGG: 1.082 D_real: 0.070 D_fake: 0.304 
(epoch: 76, iters: 400, time: 0.103) G_GAN: 1.476 G_GAN_Feat: 2.371 G_VGG: 1.101 D_real: 0.083 D_fake: 0.103 
(epoch: 76, iters: 800, time: 0.091) G_GAN: 1.401 G_GAN_Feat: 2.602 G_VGG: 0.931 D_real: 0.129 D_fake: 0.116 
(epoch: 76, iters: 1200, time: 0.094) G_GAN: 1.192 G_GAN_Feat: 2.285 G_VGG: 1.053 D_real: 0.054 D_fake: 0.222 
(epoch: 76, iters: 1600, time: 0.092) G_GAN: 1.301 G_GAN_Feat: 2.422 G_VGG: 1.026 D_real: 0.489 D_fake: 0.151 
(epoch: 76, iters: 2000, time: 0.088) G_GAN: 0.781 G_GAN_Feat: 1.517 G_VGG: 1.090 D_real: 0.277 D_fake: 0.400 
(epoch: 76, iters: 2400, time: 0.093) G_GAN: 0.453 G_GAN_Feat: 1.704 G_VGG: 1.089 D_real: 0.051 D_fake: 1.139 
(epoch: 76, iters: 2800, time: 0.097) G_GAN: 0.926 G_GAN_Feat: 2.011 G_VGG: 1.050 D_real: 0.242 D_fake: 0.389 
(epoch: 77, iters: 240, time: 0.092) G_GAN: 1.173 G_GAN_Feat: 3.020 G_VGG: 1.226 D_real: 0.168 D_fake: 0.189 
(epoch: 77, iters: 640, time: 0.090) G_GAN: 0.901 G_GAN_Feat: 1.328 G_VGG: 1.024 D_real: 0.448 D_fake: 0.297 
(epoch: 77, iters: 1040, time: 0.098) G_GAN: 1.693 G_GAN_Feat: 2.252 G_VGG: 1.089 D_real: 0.174 D_fake: 0.058 
(epoch: 77, iters: 1440, time: 0.092) G_GAN: 1.031 G_GAN_Feat: 1.347 G_VGG: 1.031 D_real: 0.589 D_fake: 0.284 
(epoch: 77, iters: 1840, time: 0.092) G_GAN: 1.886 G_GAN_Feat: 2.748 G_VGG: 1.122 D_real: 0.179 D_fake: 0.053 
(epoch: 77, iters: 2240, time: 0.088) G_GAN: 0.572 G_GAN_Feat: 1.345 G_VGG: 1.192 D_real: 0.299 D_fake: 0.530 
(epoch: 77, iters: 2640, time: 0.093) G_GAN: 1.085 G_GAN_Feat: 0.798 G_VGG: 0.967 D_real: 1.078 D_fake: 1.033 
(epoch: 78, iters: 80, time: 0.092) G_GAN: 0.777 G_GAN_Feat: 1.924 G_VGG: 1.126 D_real: 0.267 D_fake: 0.536 
(epoch: 78, iters: 480, time: 0.091) G_GAN: 0.762 G_GAN_Feat: 2.217 G_VGG: 0.984 D_real: 0.072 D_fake: 0.423 
(epoch: 78, iters: 880, time: 0.095) G_GAN: 1.363 G_GAN_Feat: 2.385 G_VGG: 1.076 D_real: 0.500 D_fake: 0.207 
(epoch: 78, iters: 1280, time: 0.091) G_GAN: 0.796 G_GAN_Feat: 1.810 G_VGG: 1.067 D_real: 0.304 D_fake: 0.387 
(epoch: 78, iters: 1680, time: 0.089) G_GAN: 1.792 G_GAN_Feat: 2.852 G_VGG: 1.052 D_real: 0.576 D_fake: 0.091 
(epoch: 78, iters: 2080, time: 0.084) G_GAN: 1.919 G_GAN_Feat: 3.442 G_VGG: 1.116 D_real: 0.147 D_fake: 0.043 
(epoch: 78, iters: 2480, time: 0.095) G_GAN: 1.009 G_GAN_Feat: 2.533 G_VGG: 1.106 D_real: 0.050 D_fake: 0.558 
(epoch: 78, iters: 2880, time: 0.098) G_GAN: 0.505 G_GAN_Feat: 0.929 G_VGG: 1.011 D_real: 0.466 D_fake: 0.526 
(epoch: 79, iters: 320, time: 0.094) G_GAN: 1.246 G_GAN_Feat: 2.118 G_VGG: 1.010 D_real: 0.105 D_fake: 0.218 
(epoch: 79, iters: 720, time: 0.099) G_GAN: 0.883 G_GAN_Feat: 2.057 G_VGG: 1.049 D_real: 0.126 D_fake: 0.359 
(epoch: 79, iters: 1120, time: 0.091) G_GAN: 1.035 G_GAN_Feat: 2.834 G_VGG: 0.915 D_real: 0.072 D_fake: 0.294 
(epoch: 79, iters: 1520, time: 0.091) G_GAN: 0.822 G_GAN_Feat: 1.470 G_VGG: 1.089 D_real: 0.250 D_fake: 0.397 
(epoch: 79, iters: 1920, time: 0.091) G_GAN: 1.166 G_GAN_Feat: 1.919 G_VGG: 1.032 D_real: 0.339 D_fake: 0.241 
(epoch: 79, iters: 2320, time: 0.093) G_GAN: 1.050 G_GAN_Feat: 1.881 G_VGG: 0.965 D_real: 0.407 D_fake: 0.273 
(epoch: 79, iters: 2720, time: 0.084) G_GAN: 1.093 G_GAN_Feat: 1.643 G_VGG: 1.038 D_real: 0.154 D_fake: 0.361 
(epoch: 80, iters: 160, time: 0.093) G_GAN: 1.001 G_GAN_Feat: 1.975 G_VGG: 1.003 D_real: 1.076 D_fake: 0.376 
(epoch: 80, iters: 560, time: 0.086) G_GAN: 1.035 G_GAN_Feat: 2.440 G_VGG: 0.972 D_real: 0.222 D_fake: 0.208 
(epoch: 80, iters: 960, time: 0.090) G_GAN: 1.249 G_GAN_Feat: 2.905 G_VGG: 1.092 D_real: 0.218 D_fake: 0.256 
(epoch: 80, iters: 1360, time: 0.095) G_GAN: 0.951 G_GAN_Feat: 1.323 G_VGG: 0.999 D_real: 0.458 D_fake: 0.324 
(epoch: 80, iters: 1760, time: 0.088) G_GAN: 1.274 G_GAN_Feat: 2.389 G_VGG: 1.160 D_real: 0.146 D_fake: 0.177 
(epoch: 80, iters: 2160, time: 0.094) G_GAN: 0.731 G_GAN_Feat: 2.232 G_VGG: 0.963 D_real: 0.251 D_fake: 0.447 
(epoch: 80, iters: 2560, time: 0.094) G_GAN: 1.102 G_GAN_Feat: 1.945 G_VGG: 1.114 D_real: 0.283 D_fake: 0.252 
(epoch: 80, iters: 2960, time: 0.086) G_GAN: 0.570 G_GAN_Feat: 2.143 G_VGG: 0.986 D_real: 0.398 D_fake: 0.564 
(epoch: 81, iters: 400, time: 0.086) G_GAN: 0.871 G_GAN_Feat: 1.443 G_VGG: 0.996 D_real: 0.244 D_fake: 0.401 
(epoch: 81, iters: 800, time: 0.085) G_GAN: 1.273 G_GAN_Feat: 1.974 G_VGG: 1.199 D_real: 0.493 D_fake: 0.189 
(epoch: 81, iters: 1200, time: 0.095) G_GAN: 1.999 G_GAN_Feat: 2.545 G_VGG: 1.135 D_real: 0.579 D_fake: 0.062 
(epoch: 81, iters: 1600, time: 0.098) G_GAN: 0.751 G_GAN_Feat: 1.728 G_VGG: 1.001 D_real: 0.263 D_fake: 0.366 
(epoch: 81, iters: 2000, time: 0.102) G_GAN: 0.924 G_GAN_Feat: 1.655 G_VGG: 1.077 D_real: 0.271 D_fake: 0.350 
(epoch: 81, iters: 2400, time: 0.087) G_GAN: 0.688 G_GAN_Feat: 1.589 G_VGG: 1.054 D_real: 0.226 D_fake: 0.405 
(epoch: 81, iters: 2800, time: 0.092) G_GAN: 1.490 G_GAN_Feat: 1.788 G_VGG: 1.108 D_real: 0.596 D_fake: 0.122 
(epoch: 82, iters: 240, time: 0.087) G_GAN: 1.136 G_GAN_Feat: 1.974 G_VGG: 0.945 D_real: 0.386 D_fake: 0.169 
(epoch: 82, iters: 640, time: 0.094) G_GAN: 0.993 G_GAN_Feat: 2.122 G_VGG: 1.127 D_real: 0.502 D_fake: 0.317 
(epoch: 82, iters: 1040, time: 0.097) G_GAN: 1.660 G_GAN_Feat: 2.670 G_VGG: 1.136 D_real: 0.093 D_fake: 0.077 
(epoch: 82, iters: 1440, time: 0.085) G_GAN: 1.861 G_GAN_Feat: 3.028 G_VGG: 1.166 D_real: 0.875 D_fake: 0.032 
(epoch: 82, iters: 1840, time: 0.093) G_GAN: 1.163 G_GAN_Feat: 2.099 G_VGG: 1.076 D_real: 0.182 D_fake: 0.285 
(epoch: 82, iters: 2240, time: 0.099) G_GAN: 1.018 G_GAN_Feat: 1.866 G_VGG: 1.006 D_real: 0.300 D_fake: 0.289 
(epoch: 82, iters: 2640, time: 0.091) G_GAN: 1.312 G_GAN_Feat: 2.297 G_VGG: 0.995 D_real: 0.426 D_fake: 0.177 
(epoch: 83, iters: 80, time: 0.084) G_GAN: 1.076 G_GAN_Feat: 2.410 G_VGG: 1.104 D_real: 0.081 D_fake: 0.312 
(epoch: 83, iters: 480, time: 0.101) G_GAN: 0.822 G_GAN_Feat: 2.301 G_VGG: 1.006 D_real: 0.025 D_fake: 0.432 
(epoch: 83, iters: 880, time: 0.093) G_GAN: 1.112 G_GAN_Feat: 2.300 G_VGG: 0.988 D_real: 0.888 D_fake: 0.173 
(epoch: 83, iters: 1280, time: 0.093) G_GAN: 1.087 G_GAN_Feat: 2.551 G_VGG: 1.129 D_real: 0.207 D_fake: 0.251 
(epoch: 83, iters: 1680, time: 0.092) G_GAN: 1.334 G_GAN_Feat: 2.644 G_VGG: 1.015 D_real: 0.102 D_fake: 0.132 
(epoch: 83, iters: 2080, time: 0.085) G_GAN: 1.106 G_GAN_Feat: 2.727 G_VGG: 1.065 D_real: 0.021 D_fake: 0.327 
(epoch: 83, iters: 2480, time: 0.088) G_GAN: 0.693 G_GAN_Feat: 1.802 G_VGG: 1.069 D_real: 0.027 D_fake: 0.598 
(epoch: 83, iters: 2880, time: 0.090) G_GAN: 1.295 G_GAN_Feat: 2.060 G_VGG: 1.053 D_real: 0.351 D_fake: 0.164 
(epoch: 84, iters: 320, time: 0.088) G_GAN: 1.108 G_GAN_Feat: 1.881 G_VGG: 1.012 D_real: 0.059 D_fake: 0.238 
(epoch: 84, iters: 720, time: 0.092) G_GAN: 0.977 G_GAN_Feat: 1.795 G_VGG: 0.930 D_real: 0.412 D_fake: 0.290 
(epoch: 84, iters: 1120, time: 0.094) G_GAN: 1.337 G_GAN_Feat: 2.203 G_VGG: 1.006 D_real: 0.169 D_fake: 0.133 
(epoch: 84, iters: 1520, time: 0.091) G_GAN: 1.316 G_GAN_Feat: 1.800 G_VGG: 1.025 D_real: 0.582 D_fake: 0.144 
(epoch: 84, iters: 1920, time: 0.093) G_GAN: 1.412 G_GAN_Feat: 2.658 G_VGG: 1.112 D_real: 0.256 D_fake: 0.278 
(epoch: 84, iters: 2320, time: 0.095) G_GAN: 0.917 G_GAN_Feat: 2.190 G_VGG: 0.974 D_real: 0.152 D_fake: 0.489 
(epoch: 84, iters: 2720, time: 0.088) G_GAN: 0.687 G_GAN_Feat: 1.747 G_VGG: 1.016 D_real: 0.066 D_fake: 0.692 
(epoch: 85, iters: 160, time: 0.086) G_GAN: 1.881 G_GAN_Feat: 4.153 G_VGG: 1.242 D_real: 0.161 D_fake: 0.062 
(epoch: 85, iters: 560, time: 0.092) G_GAN: 1.495 G_GAN_Feat: 2.967 G_VGG: 1.124 D_real: 0.183 D_fake: 0.139 
(epoch: 85, iters: 960, time: 0.093) G_GAN: 1.168 G_GAN_Feat: 1.763 G_VGG: 1.119 D_real: 0.534 D_fake: 0.324 
(epoch: 85, iters: 1360, time: 0.100) G_GAN: 1.174 G_GAN_Feat: 2.510 G_VGG: 0.999 D_real: 0.052 D_fake: 0.188 
(epoch: 85, iters: 1760, time: 0.084) G_GAN: 1.410 G_GAN_Feat: 1.833 G_VGG: 1.009 D_real: 0.596 D_fake: 0.188 
(epoch: 85, iters: 2160, time: 0.093) G_GAN: 0.777 G_GAN_Feat: 1.484 G_VGG: 1.032 D_real: 0.182 D_fake: 0.450 
(epoch: 85, iters: 2560, time: 0.095) G_GAN: 1.619 G_GAN_Feat: 2.426 G_VGG: 1.148 D_real: 0.185 D_fake: 0.071 
(epoch: 85, iters: 2960, time: 0.084) G_GAN: 1.196 G_GAN_Feat: 2.333 G_VGG: 1.018 D_real: 0.148 D_fake: 0.304 
(epoch: 86, iters: 400, time: 0.089) G_GAN: 1.010 G_GAN_Feat: 2.231 G_VGG: 1.083 D_real: 0.767 D_fake: 0.333 
(epoch: 86, iters: 800, time: 0.091) G_GAN: 1.295 G_GAN_Feat: 2.265 G_VGG: 1.056 D_real: 0.270 D_fake: 0.163 
(epoch: 86, iters: 1200, time: 0.098) G_GAN: 1.203 G_GAN_Feat: 1.627 G_VGG: 1.014 D_real: 0.411 D_fake: 0.197 
(epoch: 86, iters: 1600, time: 0.091) G_GAN: 1.492 G_GAN_Feat: 2.400 G_VGG: 1.079 D_real: 0.081 D_fake: 0.095 
(epoch: 86, iters: 2000, time: 0.099) G_GAN: 0.851 G_GAN_Feat: 1.914 G_VGG: 1.001 D_real: 0.204 D_fake: 0.409 
(epoch: 86, iters: 2400, time: 0.095) G_GAN: 1.486 G_GAN_Feat: 2.086 G_VGG: 1.012 D_real: 0.068 D_fake: 0.381 
(epoch: 86, iters: 2800, time: 0.091) G_GAN: 0.952 G_GAN_Feat: 2.042 G_VGG: 0.978 D_real: 0.534 D_fake: 0.306 
(epoch: 87, iters: 240, time: 0.086) G_GAN: 0.769 G_GAN_Feat: 1.679 G_VGG: 0.992 D_real: 0.364 D_fake: 0.415 
(epoch: 87, iters: 640, time: 0.097) G_GAN: 1.153 G_GAN_Feat: 2.146 G_VGG: 1.094 D_real: 0.046 D_fake: 0.549 
(epoch: 87, iters: 1040, time: 0.090) G_GAN: 1.489 G_GAN_Feat: 2.186 G_VGG: 1.020 D_real: 0.139 D_fake: 0.159 
(epoch: 87, iters: 1440, time: 0.088) G_GAN: 1.453 G_GAN_Feat: 2.232 G_VGG: 0.990 D_real: 0.285 D_fake: 0.087 
(epoch: 87, iters: 1840, time: 0.097) G_GAN: 2.238 G_GAN_Feat: 3.473 G_VGG: 1.041 D_real: 0.233 D_fake: 0.038 
(epoch: 87, iters: 2240, time: 0.092) G_GAN: 1.736 G_GAN_Feat: 2.639 G_VGG: 1.016 D_real: 0.459 D_fake: 0.084 
(epoch: 87, iters: 2640, time: 0.091) G_GAN: 1.741 G_GAN_Feat: 2.565 G_VGG: 1.067 D_real: 0.067 D_fake: 0.074 
(epoch: 88, iters: 80, time: 0.100) G_GAN: 0.841 G_GAN_Feat: 2.228 G_VGG: 0.946 D_real: 0.062 D_fake: 0.578 
(epoch: 88, iters: 480, time: 0.084) G_GAN: 0.856 G_GAN_Feat: 1.592 G_VGG: 0.968 D_real: 0.206 D_fake: 0.334 
(epoch: 88, iters: 880, time: 0.090) G_GAN: 0.363 G_GAN_Feat: 1.951 G_VGG: 0.988 D_real: 0.047 D_fake: 0.859 
(epoch: 88, iters: 1280, time: 0.091) G_GAN: 0.872 G_GAN_Feat: 1.467 G_VGG: 1.005 D_real: 0.274 D_fake: 0.360 
(epoch: 88, iters: 1680, time: 0.092) G_GAN: 1.201 G_GAN_Feat: 3.216 G_VGG: 1.189 D_real: 0.060 D_fake: 0.189 
(epoch: 88, iters: 2080, time: 0.091) G_GAN: 1.872 G_GAN_Feat: 2.074 G_VGG: 1.047 D_real: 0.783 D_fake: 0.152 
(epoch: 88, iters: 2480, time: 0.088) G_GAN: 1.608 G_GAN_Feat: 2.904 G_VGG: 1.123 D_real: 0.501 D_fake: 0.110 
(epoch: 88, iters: 2880, time: 0.089) G_GAN: 1.756 G_GAN_Feat: 2.318 G_VGG: 1.025 D_real: 0.467 D_fake: 0.060 
(epoch: 89, iters: 320, time: 0.084) G_GAN: 1.189 G_GAN_Feat: 2.016 G_VGG: 0.993 D_real: 0.254 D_fake: 0.198 
(epoch: 89, iters: 720, time: 0.084) G_GAN: 1.204 G_GAN_Feat: 2.284 G_VGG: 0.956 D_real: 0.026 D_fake: 0.321 
(epoch: 89, iters: 1120, time: 0.091) G_GAN: 1.256 G_GAN_Feat: 2.133 G_VGG: 1.039 D_real: 0.232 D_fake: 0.206 
(epoch: 89, iters: 1520, time: 0.102) G_GAN: 1.125 G_GAN_Feat: 2.740 G_VGG: 0.980 D_real: 0.087 D_fake: 0.268 
(epoch: 89, iters: 1920, time: 0.086) G_GAN: 1.115 G_GAN_Feat: 2.307 G_VGG: 1.081 D_real: 0.365 D_fake: 0.276 
(epoch: 89, iters: 2320, time: 0.088) G_GAN: 1.594 G_GAN_Feat: 2.161 G_VGG: 0.938 D_real: 0.343 D_fake: 0.105 
(epoch: 89, iters: 2720, time: 0.092) G_GAN: 1.519 G_GAN_Feat: 2.930 G_VGG: 1.085 D_real: 0.027 D_fake: 0.138 
(epoch: 90, iters: 160, time: 0.089) G_GAN: 1.177 G_GAN_Feat: 2.243 G_VGG: 1.035 D_real: 0.208 D_fake: 0.213 
(epoch: 90, iters: 560, time: 0.090) G_GAN: 1.273 G_GAN_Feat: 2.064 G_VGG: 0.929 D_real: 0.493 D_fake: 0.286 
(epoch: 90, iters: 960, time: 0.088) G_GAN: 0.991 G_GAN_Feat: 1.856 G_VGG: 1.019 D_real: 0.308 D_fake: 0.315 
(epoch: 90, iters: 1360, time: 0.090) G_GAN: 2.143 G_GAN_Feat: 2.605 G_VGG: 1.012 D_real: 0.164 D_fake: 0.023 
(epoch: 90, iters: 1760, time: 0.085) G_GAN: 0.742 G_GAN_Feat: 1.097 G_VGG: 1.023 D_real: 0.290 D_fake: 0.393 
(epoch: 90, iters: 2160, time: 0.089) G_GAN: 0.813 G_GAN_Feat: 1.335 G_VGG: 0.974 D_real: 0.254 D_fake: 0.339 
(epoch: 90, iters: 2560, time: 0.092) G_GAN: 0.845 G_GAN_Feat: 2.187 G_VGG: 1.086 D_real: 0.324 D_fake: 0.442 
(epoch: 90, iters: 2960, time: 0.088) G_GAN: 0.672 G_GAN_Feat: 1.782 G_VGG: 1.073 D_real: 0.118 D_fake: 0.553 
(epoch: 91, iters: 400, time: 0.097) G_GAN: 1.604 G_GAN_Feat: 2.139 G_VGG: 0.998 D_real: 0.306 D_fake: 0.331 
(epoch: 91, iters: 800, time: 0.088) G_GAN: 0.618 G_GAN_Feat: 1.267 G_VGG: 0.912 D_real: 0.245 D_fake: 0.466 
(epoch: 91, iters: 1200, time: 0.097) G_GAN: 1.099 G_GAN_Feat: 2.416 G_VGG: 0.961 D_real: 0.160 D_fake: 0.424 
(epoch: 91, iters: 1600, time: 0.084) G_GAN: 0.838 G_GAN_Feat: 1.307 G_VGG: 0.981 D_real: 0.655 D_fake: 0.327 
(epoch: 91, iters: 2000, time: 0.093) G_GAN: 0.642 G_GAN_Feat: 1.342 G_VGG: 1.049 D_real: 0.439 D_fake: 0.413 
(epoch: 91, iters: 2400, time: 0.092) G_GAN: 0.903 G_GAN_Feat: 1.534 G_VGG: 0.983 D_real: 0.642 D_fake: 0.443 
(epoch: 91, iters: 2800, time: 0.092) G_GAN: 0.846 G_GAN_Feat: 1.047 G_VGG: 0.964 D_real: 0.439 D_fake: 0.344 
(epoch: 92, iters: 240, time: 0.099) G_GAN: 0.458 G_GAN_Feat: 1.289 G_VGG: 1.017 D_real: 0.262 D_fake: 0.610 
(epoch: 92, iters: 640, time: 0.088) G_GAN: 1.002 G_GAN_Feat: 1.122 G_VGG: 0.914 D_real: 0.516 D_fake: 0.321 
(epoch: 92, iters: 1040, time: 0.096) G_GAN: 0.754 G_GAN_Feat: 1.527 G_VGG: 0.980 D_real: 0.181 D_fake: 0.373 
(epoch: 92, iters: 1440, time: 0.098) G_GAN: 0.992 G_GAN_Feat: 1.790 G_VGG: 1.058 D_real: 0.129 D_fake: 0.308 
(epoch: 92, iters: 1840, time: 0.091) G_GAN: 1.121 G_GAN_Feat: 1.637 G_VGG: 1.012 D_real: 0.547 D_fake: 0.259 
(epoch: 92, iters: 2240, time: 0.094) G_GAN: 1.280 G_GAN_Feat: 1.658 G_VGG: 0.873 D_real: 0.533 D_fake: 0.281 
(epoch: 92, iters: 2640, time: 0.093) G_GAN: 1.625 G_GAN_Feat: 2.011 G_VGG: 1.048 D_real: 0.476 D_fake: 0.122 
(epoch: 93, iters: 80, time: 0.096) G_GAN: 1.206 G_GAN_Feat: 1.550 G_VGG: 0.933 D_real: 0.355 D_fake: 0.168 
(epoch: 93, iters: 480, time: 0.086) G_GAN: 1.263 G_GAN_Feat: 2.614 G_VGG: 1.024 D_real: 0.147 D_fake: 0.236 
(epoch: 93, iters: 880, time: 0.087) G_GAN: 2.012 G_GAN_Feat: 2.319 G_VGG: 1.031 D_real: 0.649 D_fake: 0.063 
(epoch: 93, iters: 1280, time: 0.091) G_GAN: 1.496 G_GAN_Feat: 2.219 G_VGG: 0.969 D_real: 0.582 D_fake: 0.117 
(epoch: 93, iters: 1680, time: 0.099) G_GAN: 1.273 G_GAN_Feat: 1.994 G_VGG: 0.957 D_real: 0.118 D_fake: 0.385 
(epoch: 93, iters: 2080, time: 0.088) G_GAN: 1.002 G_GAN_Feat: 2.395 G_VGG: 0.908 D_real: 0.111 D_fake: 0.286 
(epoch: 93, iters: 2480, time: 0.088) G_GAN: 0.544 G_GAN_Feat: 1.863 G_VGG: 1.026 D_real: 0.207 D_fake: 0.590 
(epoch: 93, iters: 2880, time: 0.098) G_GAN: 1.134 G_GAN_Feat: 2.265 G_VGG: 0.915 D_real: 0.240 D_fake: 0.482 
(epoch: 94, iters: 320, time: 0.095) G_GAN: 0.379 G_GAN_Feat: 1.852 G_VGG: 1.001 D_real: 0.363 D_fake: 0.700 
(epoch: 94, iters: 720, time: 0.089) G_GAN: 1.159 G_GAN_Feat: 2.613 G_VGG: 1.038 D_real: 0.829 D_fake: 0.207 
(epoch: 94, iters: 1120, time: 0.088) G_GAN: 0.746 G_GAN_Feat: 1.027 G_VGG: 0.957 D_real: 0.426 D_fake: 0.373 
(epoch: 94, iters: 1520, time: 0.100) G_GAN: 0.723 G_GAN_Feat: 1.198 G_VGG: 0.882 D_real: 0.501 D_fake: 0.493 
(epoch: 94, iters: 1920, time: 0.095) G_GAN: 1.499 G_GAN_Feat: 2.328 G_VGG: 1.059 D_real: 0.326 D_fake: 0.259 
(epoch: 94, iters: 2320, time: 0.090) G_GAN: 1.157 G_GAN_Feat: 1.905 G_VGG: 1.019 D_real: 0.200 D_fake: 0.196 
(epoch: 94, iters: 2720, time: 0.092) G_GAN: 0.763 G_GAN_Feat: 2.213 G_VGG: 0.965 D_real: 0.076 D_fake: 0.472 
(epoch: 95, iters: 160, time: 0.092) G_GAN: 1.393 G_GAN_Feat: 1.931 G_VGG: 0.990 D_real: 0.678 D_fake: 0.257 
(epoch: 95, iters: 560, time: 0.092) G_GAN: 1.881 G_GAN_Feat: 3.044 G_VGG: 1.017 D_real: 0.196 D_fake: 0.050 
(epoch: 95, iters: 960, time: 0.097) G_GAN: 1.341 G_GAN_Feat: 1.992 G_VGG: 1.033 D_real: 1.092 D_fake: 0.153 
(epoch: 95, iters: 1360, time: 0.085) G_GAN: 1.328 G_GAN_Feat: 2.096 G_VGG: 1.063 D_real: 0.300 D_fake: 0.156 
(epoch: 95, iters: 1760, time: 0.093) G_GAN: 1.594 G_GAN_Feat: 2.548 G_VGG: 0.908 D_real: 0.453 D_fake: 0.081 
(epoch: 95, iters: 2160, time: 0.097) G_GAN: 1.113 G_GAN_Feat: 1.887 G_VGG: 0.932 D_real: 0.344 D_fake: 0.215 
(epoch: 95, iters: 2560, time: 0.092) G_GAN: 0.884 G_GAN_Feat: 1.936 G_VGG: 0.892 D_real: 0.235 D_fake: 0.356 
(epoch: 95, iters: 2960, time: 0.096) G_GAN: 1.127 G_GAN_Feat: 1.793 G_VGG: 0.926 D_real: 0.274 D_fake: 0.215 
(epoch: 96, iters: 400, time: 0.096) G_GAN: 1.538 G_GAN_Feat: 2.287 G_VGG: 1.006 D_real: 0.669 D_fake: 0.136 
(epoch: 96, iters: 800, time: 0.093) G_GAN: 0.839 G_GAN_Feat: 2.098 G_VGG: 1.131 D_real: 0.406 D_fake: 0.446 
(epoch: 96, iters: 1200, time: 0.095) G_GAN: 1.266 G_GAN_Feat: 1.745 G_VGG: 0.992 D_real: 0.146 D_fake: 0.163 
(epoch: 96, iters: 1600, time: 0.084) G_GAN: 1.227 G_GAN_Feat: 2.553 G_VGG: 1.049 D_real: 0.319 D_fake: 0.155 
(epoch: 96, iters: 2000, time: 0.086) G_GAN: 1.064 G_GAN_Feat: 2.461 G_VGG: 1.012 D_real: 0.165 D_fake: 0.236 
(epoch: 96, iters: 2400, time: 0.093) G_GAN: 1.775 G_GAN_Feat: 2.613 G_VGG: 0.941 D_real: 0.346 D_fake: 0.049 
(epoch: 96, iters: 2800, time: 0.098) G_GAN: 0.780 G_GAN_Feat: 2.067 G_VGG: 0.977 D_real: 0.378 D_fake: 0.414 
(epoch: 97, iters: 240, time: 0.094) G_GAN: 1.223 G_GAN_Feat: 2.092 G_VGG: 0.949 D_real: 0.296 D_fake: 0.417 
(epoch: 97, iters: 640, time: 0.098) G_GAN: 1.411 G_GAN_Feat: 1.986 G_VGG: 0.873 D_real: 0.171 D_fake: 0.130 
(epoch: 97, iters: 1040, time: 0.100) G_GAN: 0.787 G_GAN_Feat: 1.404 G_VGG: 0.882 D_real: 0.460 D_fake: 0.338 
(epoch: 97, iters: 1440, time: 0.101) G_GAN: 1.240 G_GAN_Feat: 2.541 G_VGG: 0.931 D_real: 0.274 D_fake: 0.139 
(epoch: 97, iters: 1840, time: 0.090) G_GAN: 0.963 G_GAN_Feat: 1.858 G_VGG: 1.071 D_real: 0.201 D_fake: 0.288 
(epoch: 97, iters: 2240, time: 0.085) G_GAN: 1.090 G_GAN_Feat: 2.100 G_VGG: 0.951 D_real: 0.191 D_fake: 0.233 
(epoch: 97, iters: 2640, time: 0.096) G_GAN: 1.253 G_GAN_Feat: 1.908 G_VGG: 0.976 D_real: 0.264 D_fake: 0.211 
(epoch: 98, iters: 80, time: 0.093) G_GAN: 1.995 G_GAN_Feat: 2.353 G_VGG: 1.037 D_real: 0.521 D_fake: 0.069 
(epoch: 98, iters: 480, time: 0.093) G_GAN: 0.944 G_GAN_Feat: 2.211 G_VGG: 1.042 D_real: 0.225 D_fake: 0.334 
(epoch: 98, iters: 880, time: 0.091) G_GAN: 0.705 G_GAN_Feat: 1.520 G_VGG: 1.003 D_real: 0.378 D_fake: 0.407 
(epoch: 98, iters: 1280, time: 0.083) G_GAN: 1.171 G_GAN_Feat: 1.870 G_VGG: 0.865 D_real: 0.316 D_fake: 0.272 
(epoch: 98, iters: 1680, time: 0.086) G_GAN: 1.169 G_GAN_Feat: 2.017 G_VGG: 0.949 D_real: 0.278 D_fake: 0.237 
(epoch: 98, iters: 2080, time: 0.087) G_GAN: 0.921 G_GAN_Feat: 1.546 G_VGG: 1.108 D_real: 0.341 D_fake: 0.327 
(epoch: 98, iters: 2480, time: 0.091) G_GAN: 0.851 G_GAN_Feat: 1.241 G_VGG: 1.002 D_real: 0.375 D_fake: 0.358 
(epoch: 98, iters: 2880, time: 0.087) G_GAN: 0.488 G_GAN_Feat: 1.398 G_VGG: 0.990 D_real: 0.313 D_fake: 0.567 
(epoch: 99, iters: 320, time: 0.085) G_GAN: 1.046 G_GAN_Feat: 2.359 G_VGG: 1.070 D_real: 0.068 D_fake: 0.465 
(epoch: 99, iters: 720, time: 0.092) G_GAN: 0.886 G_GAN_Feat: 1.608 G_VGG: 0.942 D_real: 0.410 D_fake: 0.308 
(epoch: 99, iters: 1120, time: 0.092) G_GAN: 2.029 G_GAN_Feat: 2.642 G_VGG: 1.018 D_real: 0.263 D_fake: 0.053 
(epoch: 99, iters: 1520, time: 0.100) G_GAN: 1.546 G_GAN_Feat: 1.985 G_VGG: 0.974 D_real: 0.805 D_fake: 0.334 
(epoch: 99, iters: 1920, time: 0.090) G_GAN: 1.271 G_GAN_Feat: 2.097 G_VGG: 0.986 D_real: 0.100 D_fake: 0.223 
(epoch: 99, iters: 2320, time: 0.096) G_GAN: 0.463 G_GAN_Feat: 2.232 G_VGG: 0.944 D_real: 0.282 D_fake: 0.718 
(epoch: 99, iters: 2720, time: 0.087) G_GAN: 2.145 G_GAN_Feat: 1.927 G_VGG: 1.003 D_real: 0.854 D_fake: 0.107 
(epoch: 100, iters: 160, time: 0.085) G_GAN: 1.189 G_GAN_Feat: 1.774 G_VGG: 0.891 D_real: 0.647 D_fake: 0.280 
(epoch: 100, iters: 560, time: 0.091) G_GAN: 0.905 G_GAN_Feat: 1.636 G_VGG: 0.929 D_real: 0.527 D_fake: 0.282 
(epoch: 100, iters: 960, time: 0.096) G_GAN: 1.258 G_GAN_Feat: 2.116 G_VGG: 0.982 D_real: 0.115 D_fake: 0.226 
(epoch: 100, iters: 1360, time: 0.094) G_GAN: 1.674 G_GAN_Feat: 2.368 G_VGG: 0.907 D_real: 0.280 D_fake: 0.094 
(epoch: 100, iters: 1760, time: 0.092) G_GAN: 1.201 G_GAN_Feat: 2.054 G_VGG: 0.925 D_real: 0.503 D_fake: 0.346 
(epoch: 100, iters: 2160, time: 0.088) G_GAN: 1.403 G_GAN_Feat: 1.981 G_VGG: 1.052 D_real: 0.567 D_fake: 0.130 
(epoch: 100, iters: 2560, time: 0.093) G_GAN: 1.490 G_GAN_Feat: 1.857 G_VGG: 0.958 D_real: 0.285 D_fake: 0.168 
(epoch: 100, iters: 2960, time: 0.092) G_GAN: 1.174 G_GAN_Feat: 2.156 G_VGG: 0.886 D_real: 0.236 D_fake: 0.200 
(epoch: 101, iters: 400, time: 0.087) G_GAN: 0.646 G_GAN_Feat: 2.192 G_VGG: 1.011 D_real: 0.133 D_fake: 0.685 
(epoch: 101, iters: 800, time: 0.094) G_GAN: 0.924 G_GAN_Feat: 1.443 G_VGG: 0.973 D_real: 0.259 D_fake: 0.305 
(epoch: 101, iters: 1200, time: 0.089) G_GAN: 0.645 G_GAN_Feat: 1.831 G_VGG: 1.038 D_real: 0.254 D_fake: 0.429 
(epoch: 101, iters: 1600, time: 0.090) G_GAN: 0.487 G_GAN_Feat: 1.272 G_VGG: 0.867 D_real: 0.277 D_fake: 0.578 
(epoch: 101, iters: 2000, time: 0.093) G_GAN: 1.191 G_GAN_Feat: 1.265 G_VGG: 0.909 D_real: 0.331 D_fake: 0.298 
(epoch: 101, iters: 2400, time: 0.101) G_GAN: 0.663 G_GAN_Feat: 1.543 G_VGG: 0.997 D_real: 0.363 D_fake: 0.467 
(epoch: 101, iters: 2800, time: 0.098) G_GAN: 1.851 G_GAN_Feat: 2.125 G_VGG: 1.025 D_real: 0.328 D_fake: 0.182 
(epoch: 102, iters: 240, time: 0.091) G_GAN: 1.042 G_GAN_Feat: 2.178 G_VGG: 0.951 D_real: 0.098 D_fake: 0.394 
(epoch: 102, iters: 640, time: 0.090) G_GAN: 0.759 G_GAN_Feat: 2.256 G_VGG: 0.972 D_real: 0.100 D_fake: 0.430 
(epoch: 102, iters: 1040, time: 0.088) G_GAN: 1.040 G_GAN_Feat: 2.040 G_VGG: 0.995 D_real: 0.222 D_fake: 0.239 
(epoch: 102, iters: 1440, time: 0.092) G_GAN: 0.762 G_GAN_Feat: 2.334 G_VGG: 0.946 D_real: 0.037 D_fake: 0.449 
(epoch: 102, iters: 1840, time: 0.097) G_GAN: 0.814 G_GAN_Feat: 2.024 G_VGG: 0.931 D_real: 0.224 D_fake: 0.413 
(epoch: 102, iters: 2240, time: 0.096) G_GAN: 1.313 G_GAN_Feat: 2.007 G_VGG: 0.941 D_real: 0.262 D_fake: 0.194 
(epoch: 102, iters: 2640, time: 0.091) G_GAN: 1.025 G_GAN_Feat: 2.285 G_VGG: 1.002 D_real: 0.157 D_fake: 0.281 
(epoch: 103, iters: 80, time: 0.096) G_GAN: 0.837 G_GAN_Feat: 1.544 G_VGG: 0.969 D_real: 0.372 D_fake: 0.394 
(epoch: 103, iters: 480, time: 0.091) G_GAN: 1.291 G_GAN_Feat: 1.700 G_VGG: 1.005 D_real: 0.266 D_fake: 0.164 
(epoch: 103, iters: 880, time: 0.099) G_GAN: 0.666 G_GAN_Feat: 1.562 G_VGG: 0.991 D_real: 0.357 D_fake: 0.403 
(epoch: 103, iters: 1280, time: 0.084) G_GAN: 0.901 G_GAN_Feat: 2.002 G_VGG: 0.985 D_real: 0.360 D_fake: 0.320 
(epoch: 103, iters: 1680, time: 0.096) G_GAN: 0.962 G_GAN_Feat: 1.722 G_VGG: 1.012 D_real: 0.278 D_fake: 0.350 
(epoch: 103, iters: 2080, time: 0.087) G_GAN: 1.314 G_GAN_Feat: 2.050 G_VGG: 0.936 D_real: 0.431 D_fake: 0.122 
(epoch: 103, iters: 2480, time: 0.084) G_GAN: 1.639 G_GAN_Feat: 2.403 G_VGG: 1.013 D_real: 0.425 D_fake: 0.153 
(epoch: 103, iters: 2880, time: 0.094) G_GAN: 1.007 G_GAN_Feat: 2.012 G_VGG: 1.042 D_real: 0.651 D_fake: 0.269 
(epoch: 104, iters: 320, time: 0.093) G_GAN: 1.906 G_GAN_Feat: 2.679 G_VGG: 0.978 D_real: 0.452 D_fake: 0.042 
(epoch: 104, iters: 720, time: 0.084) G_GAN: 1.505 G_GAN_Feat: 2.409 G_VGG: 0.916 D_real: 0.245 D_fake: 0.141 
(epoch: 104, iters: 1120, time: 0.084) G_GAN: 0.891 G_GAN_Feat: 1.786 G_VGG: 0.970 D_real: 0.075 D_fake: 0.375 
(epoch: 104, iters: 1520, time: 0.084) G_GAN: 1.102 G_GAN_Feat: 2.655 G_VGG: 0.980 D_real: 0.096 D_fake: 0.180 
(epoch: 104, iters: 1920, time: 0.092) G_GAN: 0.782 G_GAN_Feat: 1.581 G_VGG: 0.946 D_real: 0.078 D_fake: 0.507 
(epoch: 104, iters: 2320, time: 0.091) G_GAN: 0.858 G_GAN_Feat: 2.382 G_VGG: 0.947 D_real: 0.160 D_fake: 0.490 
(epoch: 104, iters: 2720, time: 0.092) G_GAN: 0.376 G_GAN_Feat: 1.252 G_VGG: 0.926 D_real: 0.300 D_fake: 0.906 
(epoch: 105, iters: 160, time: 0.096) G_GAN: 0.811 G_GAN_Feat: 1.837 G_VGG: 0.930 D_real: 0.501 D_fake: 0.337 
(epoch: 105, iters: 560, time: 0.092) G_GAN: 1.100 G_GAN_Feat: 1.978 G_VGG: 0.993 D_real: 0.464 D_fake: 0.271 
(epoch: 105, iters: 960, time: 0.088) G_GAN: 0.467 G_GAN_Feat: 1.377 G_VGG: 0.893 D_real: 0.148 D_fake: 0.669 
(epoch: 105, iters: 1360, time: 0.090) G_GAN: 1.064 G_GAN_Feat: 1.725 G_VGG: 0.992 D_real: 0.299 D_fake: 0.223 
(epoch: 105, iters: 1760, time: 0.085) G_GAN: 1.266 G_GAN_Feat: 2.117 G_VGG: 0.987 D_real: 0.148 D_fake: 0.258 
(epoch: 105, iters: 2160, time: 0.087) G_GAN: 0.641 G_GAN_Feat: 1.860 G_VGG: 1.069 D_real: 0.141 D_fake: 0.479 
(epoch: 105, iters: 2560, time: 0.088) G_GAN: 0.609 G_GAN_Feat: 1.173 G_VGG: 0.941 D_real: 0.368 D_fake: 0.450 
(epoch: 105, iters: 2960, time: 0.091) G_GAN: 1.337 G_GAN_Feat: 1.816 G_VGG: 0.991 D_real: 0.387 D_fake: 0.240 
(epoch: 106, iters: 400, time: 0.090) G_GAN: 0.657 G_GAN_Feat: 1.982 G_VGG: 1.022 D_real: 0.194 D_fake: 0.537 
(epoch: 106, iters: 800, time: 0.091) G_GAN: 1.252 G_GAN_Feat: 2.335 G_VGG: 0.980 D_real: 0.258 D_fake: 0.174 
(epoch: 106, iters: 1200, time: 0.097) G_GAN: 1.372 G_GAN_Feat: 1.940 G_VGG: 0.956 D_real: 0.764 D_fake: 0.133 
(epoch: 106, iters: 1600, time: 0.088) G_GAN: 1.062 G_GAN_Feat: 1.690 G_VGG: 0.879 D_real: 0.436 D_fake: 0.252 
(epoch: 106, iters: 2000, time: 0.077) G_GAN: 1.322 G_GAN_Feat: 2.619 G_VGG: 1.064 D_real: 0.522 D_fake: 0.150 
(epoch: 106, iters: 2400, time: 0.093) G_GAN: 1.053 G_GAN_Feat: 1.762 G_VGG: 0.911 D_real: 0.153 D_fake: 0.316 
(epoch: 106, iters: 2800, time: 0.086) G_GAN: 1.197 G_GAN_Feat: 2.266 G_VGG: 0.978 D_real: 0.053 D_fake: 0.276 
(epoch: 107, iters: 240, time: 0.089) G_GAN: 0.862 G_GAN_Feat: 1.834 G_VGG: 0.894 D_real: 0.030 D_fake: 0.445 
(epoch: 107, iters: 640, time: 0.088) G_GAN: 0.895 G_GAN_Feat: 2.694 G_VGG: 0.915 D_real: 0.508 D_fake: 0.303 
(epoch: 107, iters: 1040, time: 0.086) G_GAN: 1.241 G_GAN_Feat: 1.952 G_VGG: 0.953 D_real: 0.079 D_fake: 0.220 
(epoch: 107, iters: 1440, time: 0.092) G_GAN: 1.810 G_GAN_Feat: 2.514 G_VGG: 0.846 D_real: 0.084 D_fake: 0.045 
(epoch: 107, iters: 1840, time: 0.091) G_GAN: 1.291 G_GAN_Feat: 2.318 G_VGG: 0.987 D_real: 0.116 D_fake: 0.406 
(epoch: 107, iters: 2240, time: 0.092) G_GAN: 0.502 G_GAN_Feat: 1.842 G_VGG: 0.989 D_real: 0.065 D_fake: 0.667 
(epoch: 107, iters: 2640, time: 0.100) G_GAN: 1.468 G_GAN_Feat: 1.743 G_VGG: 0.939 D_real: 0.499 D_fake: 0.114 
(epoch: 108, iters: 80, time: 0.096) G_GAN: 1.517 G_GAN_Feat: 2.661 G_VGG: 0.979 D_real: 0.165 D_fake: 0.068 
(epoch: 108, iters: 480, time: 0.099) G_GAN: 1.628 G_GAN_Feat: 2.165 G_VGG: 0.962 D_real: 0.494 D_fake: 0.217 
(epoch: 108, iters: 880, time: 0.093) G_GAN: 1.992 G_GAN_Feat: 2.634 G_VGG: 0.946 D_real: 0.230 D_fake: 0.034 
(epoch: 108, iters: 1280, time: 0.090) G_GAN: 1.723 G_GAN_Feat: 2.061 G_VGG: 0.915 D_real: 0.599 D_fake: 0.043 
(epoch: 108, iters: 1680, time: 0.100) G_GAN: 1.049 G_GAN_Feat: 1.961 G_VGG: 1.023 D_real: 0.328 D_fake: 0.250 
(epoch: 108, iters: 2080, time: 0.091) G_GAN: 0.990 G_GAN_Feat: 1.922 G_VGG: 1.014 D_real: 0.125 D_fake: 0.437 
(epoch: 108, iters: 2480, time: 0.095) G_GAN: 1.113 G_GAN_Feat: 1.874 G_VGG: 0.915 D_real: 0.519 D_fake: 0.303 
(epoch: 108, iters: 2880, time: 0.092) G_GAN: 2.437 G_GAN_Feat: 2.962 G_VGG: 0.846 D_real: 0.424 D_fake: 0.049 
(epoch: 109, iters: 320, time: 0.096) G_GAN: 1.684 G_GAN_Feat: 2.235 G_VGG: 0.914 D_real: 0.133 D_fake: 0.067 
(epoch: 109, iters: 720, time: 0.090) G_GAN: 1.287 G_GAN_Feat: 1.925 G_VGG: 0.871 D_real: 0.198 D_fake: 0.238 
(epoch: 109, iters: 1120, time: 0.090) G_GAN: 1.302 G_GAN_Feat: 1.928 G_VGG: 0.903 D_real: 0.079 D_fake: 0.183 
(epoch: 109, iters: 1520, time: 0.084) G_GAN: 0.733 G_GAN_Feat: 1.069 G_VGG: 0.940 D_real: 0.335 D_fake: 0.394 
(epoch: 109, iters: 1920, time: 0.097) G_GAN: 1.043 G_GAN_Feat: 1.064 G_VGG: 0.890 D_real: 0.437 D_fake: 0.286 
(epoch: 109, iters: 2320, time: 0.091) G_GAN: 1.042 G_GAN_Feat: 1.172 G_VGG: 0.932 D_real: 0.753 D_fake: 0.291 
(epoch: 109, iters: 2720, time: 0.084) G_GAN: 0.878 G_GAN_Feat: 1.534 G_VGG: 0.962 D_real: 0.209 D_fake: 0.336 
(epoch: 110, iters: 160, time: 0.089) G_GAN: 0.826 G_GAN_Feat: 1.614 G_VGG: 0.898 D_real: 0.253 D_fake: 0.348 
(epoch: 110, iters: 560, time: 0.090) G_GAN: 1.202 G_GAN_Feat: 2.330 G_VGG: 0.880 D_real: 0.191 D_fake: 0.424 
(epoch: 110, iters: 960, time: 0.089) G_GAN: 0.601 G_GAN_Feat: 2.155 G_VGG: 0.996 D_real: 0.417 D_fake: 0.495 
(epoch: 110, iters: 1360, time: 0.088) G_GAN: 1.055 G_GAN_Feat: 2.220 G_VGG: 0.968 D_real: 0.033 D_fake: 0.225 
(epoch: 110, iters: 1760, time: 0.095) G_GAN: 1.543 G_GAN_Feat: 2.182 G_VGG: 1.015 D_real: 0.135 D_fake: 0.095 
(epoch: 110, iters: 2160, time: 0.092) G_GAN: 1.275 G_GAN_Feat: 2.349 G_VGG: 0.935 D_real: 0.186 D_fake: 0.153 
(epoch: 110, iters: 2560, time: 0.092) G_GAN: 1.066 G_GAN_Feat: 1.868 G_VGG: 0.992 D_real: 0.261 D_fake: 0.316 
(epoch: 110, iters: 2960, time: 0.092) G_GAN: 1.227 G_GAN_Feat: 1.566 G_VGG: 0.947 D_real: 0.427 D_fake: 0.143 
(epoch: 111, iters: 400, time: 0.084) G_GAN: 1.025 G_GAN_Feat: 1.879 G_VGG: 0.837 D_real: 0.135 D_fake: 0.266 
(epoch: 111, iters: 800, time: 0.092) G_GAN: 0.905 G_GAN_Feat: 1.745 G_VGG: 0.943 D_real: 0.109 D_fake: 0.510 
(epoch: 111, iters: 1200, time: 0.095) G_GAN: 1.019 G_GAN_Feat: 2.086 G_VGG: 0.959 D_real: 0.834 D_fake: 0.290 
(epoch: 111, iters: 1600, time: 0.091) G_GAN: 1.042 G_GAN_Feat: 1.746 G_VGG: 0.882 D_real: 0.069 D_fake: 0.494 
(epoch: 111, iters: 2000, time: 0.091) G_GAN: 2.554 G_GAN_Feat: 2.316 G_VGG: 1.025 D_real: 0.582 D_fake: 0.062 
(epoch: 111, iters: 2400, time: 0.098) G_GAN: 1.381 G_GAN_Feat: 1.678 G_VGG: 1.062 D_real: 0.531 D_fake: 0.184 
(epoch: 111, iters: 2800, time: 0.096) G_GAN: 1.991 G_GAN_Feat: 2.251 G_VGG: 0.967 D_real: 0.323 D_fake: 0.070 
(epoch: 112, iters: 240, time: 0.093) G_GAN: 0.571 G_GAN_Feat: 1.382 G_VGG: 0.896 D_real: 0.251 D_fake: 0.511 
(epoch: 112, iters: 640, time: 0.089) G_GAN: 0.732 G_GAN_Feat: 1.825 G_VGG: 0.953 D_real: 0.237 D_fake: 0.444 
(epoch: 112, iters: 1040, time: 0.094) G_GAN: 1.004 G_GAN_Feat: 2.196 G_VGG: 0.934 D_real: 0.077 D_fake: 0.270 
(epoch: 112, iters: 1440, time: 0.093) G_GAN: 1.040 G_GAN_Feat: 1.833 G_VGG: 0.956 D_real: 0.034 D_fake: 0.364 
(epoch: 112, iters: 1840, time: 0.095) G_GAN: 1.634 G_GAN_Feat: 2.587 G_VGG: 0.969 D_real: 0.097 D_fake: 0.073 
(epoch: 112, iters: 2240, time: 0.085) G_GAN: 0.950 G_GAN_Feat: 2.197 G_VGG: 0.947 D_real: 0.527 D_fake: 0.327 
(epoch: 112, iters: 2640, time: 0.089) G_GAN: 0.810 G_GAN_Feat: 1.811 G_VGG: 0.913 D_real: 0.308 D_fake: 0.456 
(epoch: 113, iters: 80, time: 0.090) G_GAN: 0.818 G_GAN_Feat: 1.546 G_VGG: 0.818 D_real: 0.597 D_fake: 0.324 
(epoch: 113, iters: 480, time: 0.084) G_GAN: 0.577 G_GAN_Feat: 1.033 G_VGG: 0.829 D_real: 0.465 D_fake: 0.460 
(epoch: 113, iters: 880, time: 0.100) G_GAN: 0.566 G_GAN_Feat: 1.845 G_VGG: 0.825 D_real: 0.443 D_fake: 0.540 
(epoch: 113, iters: 1280, time: 0.092) G_GAN: 0.667 G_GAN_Feat: 2.071 G_VGG: 0.992 D_real: 0.049 D_fake: 0.488 
(epoch: 113, iters: 1680, time: 0.087) G_GAN: 1.289 G_GAN_Feat: 1.427 G_VGG: 0.911 D_real: 0.790 D_fake: 0.336 
(epoch: 113, iters: 2080, time: 0.084) G_GAN: 1.012 G_GAN_Feat: 1.609 G_VGG: 0.907 D_real: 0.517 D_fake: 0.369 
(epoch: 113, iters: 2480, time: 0.086) G_GAN: 1.482 G_GAN_Feat: 1.989 G_VGG: 0.906 D_real: 0.352 D_fake: 0.382 
(epoch: 113, iters: 2880, time: 0.085) G_GAN: 0.746 G_GAN_Feat: 1.893 G_VGG: 0.983 D_real: 0.074 D_fake: 0.677 
(epoch: 114, iters: 320, time: 0.095) G_GAN: 0.952 G_GAN_Feat: 2.318 G_VGG: 0.991 D_real: 0.075 D_fake: 0.308 
(epoch: 114, iters: 720, time: 0.088) G_GAN: 0.581 G_GAN_Feat: 1.880 G_VGG: 0.860 D_real: 0.117 D_fake: 0.600 
(epoch: 114, iters: 1120, time: 0.083) G_GAN: 1.627 G_GAN_Feat: 2.856 G_VGG: 0.951 D_real: 0.042 D_fake: 0.074 
(epoch: 114, iters: 1520, time: 0.093) G_GAN: 1.581 G_GAN_Feat: 2.331 G_VGG: 0.981 D_real: 0.415 D_fake: 0.173 
(epoch: 114, iters: 1920, time: 0.090) G_GAN: 1.081 G_GAN_Feat: 2.165 G_VGG: 0.987 D_real: 0.131 D_fake: 0.458 
(epoch: 114, iters: 2320, time: 0.084) G_GAN: 1.021 G_GAN_Feat: 2.093 G_VGG: 0.918 D_real: 0.361 D_fake: 0.302 
(epoch: 114, iters: 2720, time: 0.088) G_GAN: 0.620 G_GAN_Feat: 1.470 G_VGG: 0.881 D_real: 0.384 D_fake: 0.506 
(epoch: 115, iters: 160, time: 0.090) G_GAN: 1.207 G_GAN_Feat: 2.221 G_VGG: 0.907 D_real: 0.064 D_fake: 0.187 
(epoch: 115, iters: 560, time: 0.096) G_GAN: 1.300 G_GAN_Feat: 2.141 G_VGG: 0.877 D_real: 0.071 D_fake: 0.263 
(epoch: 115, iters: 960, time: 0.094) G_GAN: 1.543 G_GAN_Feat: 2.068 G_VGG: 0.960 D_real: 0.460 D_fake: 0.095 
(epoch: 115, iters: 1360, time: 0.091) G_GAN: 1.685 G_GAN_Feat: 2.121 G_VGG: 0.888 D_real: 0.777 D_fake: 0.082 
(epoch: 115, iters: 1760, time: 0.085) G_GAN: 1.999 G_GAN_Feat: 2.082 G_VGG: 0.896 D_real: 1.515 D_fake: 0.109 
(epoch: 115, iters: 2160, time: 0.077) G_GAN: 1.551 G_GAN_Feat: 1.879 G_VGG: 0.991 D_real: 0.626 D_fake: 0.216 
(epoch: 115, iters: 2560, time: 0.099) G_GAN: 1.164 G_GAN_Feat: 1.932 G_VGG: 0.888 D_real: 0.167 D_fake: 0.283 
(epoch: 115, iters: 2960, time: 0.089) G_GAN: 0.923 G_GAN_Feat: 1.947 G_VGG: 0.860 D_real: 0.055 D_fake: 0.321 
(epoch: 116, iters: 400, time: 0.097) G_GAN: 1.191 G_GAN_Feat: 1.987 G_VGG: 0.912 D_real: 0.364 D_fake: 0.291 
(epoch: 116, iters: 800, time: 0.084) G_GAN: 1.060 G_GAN_Feat: 2.349 G_VGG: 0.885 D_real: 0.121 D_fake: 0.278 
(epoch: 116, iters: 1200, time: 0.095) G_GAN: 0.962 G_GAN_Feat: 2.122 G_VGG: 0.963 D_real: 0.159 D_fake: 0.586 
(epoch: 116, iters: 1600, time: 0.083) G_GAN: 1.083 G_GAN_Feat: 1.664 G_VGG: 0.951 D_real: 0.147 D_fake: 0.351 
(epoch: 116, iters: 2000, time: 0.087) G_GAN: 1.286 G_GAN_Feat: 2.337 G_VGG: 1.108 D_real: 0.308 D_fake: 0.239 
(epoch: 116, iters: 2400, time: 0.085) G_GAN: 1.124 G_GAN_Feat: 2.153 G_VGG: 0.963 D_real: 0.319 D_fake: 0.281 
(epoch: 116, iters: 2800, time: 0.090) G_GAN: 1.384 G_GAN_Feat: 2.265 G_VGG: 0.885 D_real: 0.143 D_fake: 0.117 
(epoch: 117, iters: 240, time: 0.091) G_GAN: 1.039 G_GAN_Feat: 2.175 G_VGG: 0.941 D_real: 0.204 D_fake: 0.195 
(epoch: 117, iters: 640, time: 0.083) G_GAN: 0.806 G_GAN_Feat: 1.766 G_VGG: 0.928 D_real: 0.024 D_fake: 0.557 
(epoch: 117, iters: 1040, time: 0.084) G_GAN: 1.596 G_GAN_Feat: 2.263 G_VGG: 0.832 D_real: 0.763 D_fake: 0.110 
(epoch: 117, iters: 1440, time: 0.096) G_GAN: 0.961 G_GAN_Feat: 2.028 G_VGG: 0.789 D_real: 0.054 D_fake: 0.260 
(epoch: 117, iters: 1840, time: 0.094) G_GAN: 1.575 G_GAN_Feat: 1.955 G_VGG: 0.900 D_real: 0.225 D_fake: 0.217 
(epoch: 117, iters: 2240, time: 0.099) G_GAN: 1.302 G_GAN_Feat: 2.025 G_VGG: 0.886 D_real: 0.580 D_fake: 0.197 
(epoch: 117, iters: 2640, time: 0.085) G_GAN: 0.916 G_GAN_Feat: 2.100 G_VGG: 0.962 D_real: 0.270 D_fake: 0.279 
(epoch: 118, iters: 80, time: 0.093) G_GAN: 1.123 G_GAN_Feat: 1.637 G_VGG: 0.967 D_real: 0.236 D_fake: 0.242 
(epoch: 118, iters: 480, time: 0.095) G_GAN: 0.850 G_GAN_Feat: 1.876 G_VGG: 0.961 D_real: 0.066 D_fake: 0.534 
(epoch: 118, iters: 880, time: 0.093) G_GAN: 1.588 G_GAN_Feat: 2.251 G_VGG: 0.940 D_real: 0.362 D_fake: 0.084 
(epoch: 118, iters: 1280, time: 0.078) G_GAN: 1.219 G_GAN_Feat: 1.875 G_VGG: 0.785 D_real: 0.439 D_fake: 0.229 
(epoch: 118, iters: 1680, time: 0.085) G_GAN: 1.124 G_GAN_Feat: 2.027 G_VGG: 0.873 D_real: 0.093 D_fake: 0.344 
(epoch: 118, iters: 2080, time: 0.100) G_GAN: 1.054 G_GAN_Feat: 1.949 G_VGG: 0.927 D_real: 0.174 D_fake: 0.371 
(epoch: 118, iters: 2480, time: 0.093) G_GAN: 0.638 G_GAN_Feat: 1.818 G_VGG: 0.835 D_real: 0.043 D_fake: 0.526 
(epoch: 118, iters: 2880, time: 0.089) G_GAN: 1.810 G_GAN_Feat: 1.739 G_VGG: 0.937 D_real: 0.482 D_fake: 0.090 
(epoch: 119, iters: 320, time: 0.097) G_GAN: 0.861 G_GAN_Feat: 2.067 G_VGG: 0.949 D_real: 0.473 D_fake: 0.308 
(epoch: 119, iters: 720, time: 0.091) G_GAN: 1.238 G_GAN_Feat: 2.155 G_VGG: 0.852 D_real: 0.107 D_fake: 0.189 
(epoch: 119, iters: 1120, time: 0.096) G_GAN: 0.740 G_GAN_Feat: 1.605 G_VGG: 0.899 D_real: 0.234 D_fake: 0.410 
(epoch: 119, iters: 1520, time: 0.094) G_GAN: 0.439 G_GAN_Feat: 1.526 G_VGG: 0.817 D_real: 0.063 D_fake: 0.775 
(epoch: 119, iters: 1920, time: 0.099) G_GAN: 2.060 G_GAN_Feat: 2.334 G_VGG: 0.872 D_real: 0.804 D_fake: 0.045 
(epoch: 119, iters: 2320, time: 0.086) G_GAN: 1.052 G_GAN_Feat: 2.206 G_VGG: 1.005 D_real: 0.044 D_fake: 0.437 
(epoch: 119, iters: 2720, time: 0.093) G_GAN: 1.544 G_GAN_Feat: 2.195 G_VGG: 0.894 D_real: 0.178 D_fake: 0.152 
(epoch: 120, iters: 160, time: 0.095) G_GAN: 1.339 G_GAN_Feat: 2.150 G_VGG: 0.867 D_real: 0.241 D_fake: 0.157 
(epoch: 120, iters: 560, time: 0.094) G_GAN: 1.002 G_GAN_Feat: 1.995 G_VGG: 0.898 D_real: 0.050 D_fake: 0.428 
(epoch: 120, iters: 960, time: 0.085) G_GAN: 1.577 G_GAN_Feat: 2.286 G_VGG: 0.855 D_real: 0.239 D_fake: 0.144 
(epoch: 120, iters: 1360, time: 0.096) G_GAN: 1.137 G_GAN_Feat: 2.365 G_VGG: 0.927 D_real: 0.437 D_fake: 0.218 
(epoch: 120, iters: 1760, time: 0.097) G_GAN: 0.743 G_GAN_Feat: 2.410 G_VGG: 0.919 D_real: 0.037 D_fake: 0.510 
(epoch: 120, iters: 2160, time: 0.097) G_GAN: 0.940 G_GAN_Feat: 2.454 G_VGG: 0.924 D_real: 0.184 D_fake: 0.301 
(epoch: 120, iters: 2560, time: 0.091) G_GAN: 0.794 G_GAN_Feat: 1.338 G_VGG: 0.890 D_real: 0.299 D_fake: 0.347 
(epoch: 120, iters: 2960, time: 0.092) G_GAN: 1.299 G_GAN_Feat: 1.942 G_VGG: 0.971 D_real: 0.538 D_fake: 0.206 
(epoch: 121, iters: 400, time: 0.097) G_GAN: 1.325 G_GAN_Feat: 2.373 G_VGG: 1.030 D_real: 0.467 D_fake: 0.430 
(epoch: 121, iters: 800, time: 0.099) G_GAN: 1.272 G_GAN_Feat: 1.901 G_VGG: 0.975 D_real: 0.320 D_fake: 0.154 
(epoch: 121, iters: 1200, time: 0.094) G_GAN: 1.015 G_GAN_Feat: 2.006 G_VGG: 0.893 D_real: 0.163 D_fake: 0.366 
(epoch: 121, iters: 1600, time: 0.084) G_GAN: 1.666 G_GAN_Feat: 1.958 G_VGG: 0.981 D_real: 0.418 D_fake: 0.179 
(epoch: 121, iters: 2000, time: 0.086) G_GAN: 1.207 G_GAN_Feat: 2.219 G_VGG: 0.929 D_real: 0.500 D_fake: 0.158 
(epoch: 121, iters: 2400, time: 0.101) G_GAN: 0.908 G_GAN_Feat: 2.273 G_VGG: 0.852 D_real: 0.253 D_fake: 0.312 
(epoch: 121, iters: 2800, time: 0.089) G_GAN: 1.103 G_GAN_Feat: 2.052 G_VGG: 1.024 D_real: 0.083 D_fake: 0.427 
(epoch: 122, iters: 240, time: 0.091) G_GAN: 0.800 G_GAN_Feat: 1.913 G_VGG: 0.838 D_real: 0.026 D_fake: 0.414 
(epoch: 122, iters: 640, time: 0.097) G_GAN: 1.151 G_GAN_Feat: 1.897 G_VGG: 0.876 D_real: 0.352 D_fake: 0.244 
(epoch: 122, iters: 1040, time: 0.097) G_GAN: 1.066 G_GAN_Feat: 2.115 G_VGG: 0.964 D_real: 0.394 D_fake: 0.222 
(epoch: 122, iters: 1440, time: 0.096) G_GAN: 0.935 G_GAN_Feat: 2.159 G_VGG: 0.880 D_real: 0.818 D_fake: 0.266 
(epoch: 122, iters: 1840, time: 0.096) G_GAN: 1.217 G_GAN_Feat: 2.566 G_VGG: 0.937 D_real: 0.206 D_fake: 0.144 
(epoch: 122, iters: 2240, time: 0.086) G_GAN: 0.754 G_GAN_Feat: 2.195 G_VGG: 0.926 D_real: 0.293 D_fake: 0.452 
(epoch: 122, iters: 2640, time: 0.091) G_GAN: 1.357 G_GAN_Feat: 2.168 G_VGG: 0.922 D_real: 0.148 D_fake: 0.131 
(epoch: 123, iters: 80, time: 0.089) G_GAN: 0.807 G_GAN_Feat: 1.771 G_VGG: 0.869 D_real: 0.329 D_fake: 0.388 
(epoch: 123, iters: 480, time: 0.091) G_GAN: 1.634 G_GAN_Feat: 2.481 G_VGG: 0.912 D_real: 0.373 D_fake: 0.091 
(epoch: 123, iters: 880, time: 0.096) G_GAN: 0.724 G_GAN_Feat: 1.940 G_VGG: 1.034 D_real: 0.143 D_fake: 0.445 
(epoch: 123, iters: 1280, time: 0.093) G_GAN: 1.411 G_GAN_Feat: 2.749 G_VGG: 0.942 D_real: 0.090 D_fake: 0.221 
(epoch: 123, iters: 1680, time: 0.087) G_GAN: 1.568 G_GAN_Feat: 2.860 G_VGG: 0.916 D_real: 0.037 D_fake: 0.223 
(epoch: 123, iters: 2080, time: 0.096) G_GAN: 1.006 G_GAN_Feat: 2.009 G_VGG: 0.940 D_real: 0.082 D_fake: 0.286 
(epoch: 123, iters: 2480, time: 0.100) G_GAN: 1.132 G_GAN_Feat: 1.959 G_VGG: 0.992 D_real: 0.218 D_fake: 0.206 
(epoch: 123, iters: 2880, time: 0.097) G_GAN: 1.002 G_GAN_Feat: 2.163 G_VGG: 0.942 D_real: 0.441 D_fake: 0.245 
(epoch: 124, iters: 320, time: 0.098) G_GAN: 0.727 G_GAN_Feat: 2.077 G_VGG: 0.970 D_real: 0.213 D_fake: 0.430 
(epoch: 124, iters: 720, time: 0.092) G_GAN: 1.113 G_GAN_Feat: 1.729 G_VGG: 0.874 D_real: 0.134 D_fake: 0.396 
(epoch: 124, iters: 1120, time: 0.088) G_GAN: 1.163 G_GAN_Feat: 1.983 G_VGG: 0.958 D_real: 0.139 D_fake: 0.336 
(epoch: 124, iters: 1520, time: 0.088) G_GAN: 0.963 G_GAN_Feat: 2.234 G_VGG: 0.819 D_real: 0.052 D_fake: 0.356 
(epoch: 124, iters: 1920, time: 0.090) G_GAN: 0.921 G_GAN_Feat: 2.012 G_VGG: 0.915 D_real: 0.221 D_fake: 0.312 
(epoch: 124, iters: 2320, time: 0.091) G_GAN: 1.301 G_GAN_Feat: 2.110 G_VGG: 0.871 D_real: 0.048 D_fake: 0.221 
(epoch: 124, iters: 2720, time: 0.091) G_GAN: 1.494 G_GAN_Feat: 1.990 G_VGG: 0.818 D_real: 0.279 D_fake: 0.278 
(epoch: 125, iters: 160, time: 0.096) G_GAN: 1.326 G_GAN_Feat: 2.723 G_VGG: 1.036 D_real: 0.057 D_fake: 0.142 
(epoch: 125, iters: 560, time: 0.087) G_GAN: 0.795 G_GAN_Feat: 1.591 G_VGG: 0.862 D_real: 0.280 D_fake: 0.352 
(epoch: 125, iters: 960, time: 0.093) G_GAN: 1.132 G_GAN_Feat: 2.215 G_VGG: 0.929 D_real: 0.275 D_fake: 0.326 
(epoch: 125, iters: 1360, time: 0.092) G_GAN: 1.597 G_GAN_Feat: 2.377 G_VGG: 0.971 D_real: 0.171 D_fake: 0.144 
(epoch: 125, iters: 1760, time: 0.093) G_GAN: 0.983 G_GAN_Feat: 2.777 G_VGG: 0.965 D_real: 0.225 D_fake: 0.256 
(epoch: 125, iters: 2160, time: 0.094) G_GAN: 0.871 G_GAN_Feat: 2.048 G_VGG: 0.861 D_real: 0.030 D_fake: 0.856 
(epoch: 125, iters: 2560, time: 0.094) G_GAN: 1.371 G_GAN_Feat: 2.157 G_VGG: 0.897 D_real: 0.724 D_fake: 0.144 
(epoch: 125, iters: 2960, time: 0.091) G_GAN: 1.049 G_GAN_Feat: 1.868 G_VGG: 0.912 D_real: 0.327 D_fake: 0.255 
(epoch: 126, iters: 400, time: 0.094) G_GAN: 1.064 G_GAN_Feat: 2.158 G_VGG: 0.850 D_real: 0.083 D_fake: 0.370 
(epoch: 126, iters: 800, time: 0.093) G_GAN: 1.560 G_GAN_Feat: 2.449 G_VGG: 0.959 D_real: 0.206 D_fake: 0.139 
(epoch: 126, iters: 1200, time: 0.101) G_GAN: 0.973 G_GAN_Feat: 2.281 G_VGG: 0.914 D_real: 0.176 D_fake: 0.286 
(epoch: 126, iters: 1600, time: 0.085) G_GAN: 1.296 G_GAN_Feat: 2.721 G_VGG: 0.874 D_real: 0.031 D_fake: 0.201 
(epoch: 126, iters: 2000, time: 0.086) G_GAN: 1.205 G_GAN_Feat: 1.828 G_VGG: 0.940 D_real: 0.247 D_fake: 0.333 
(epoch: 126, iters: 2400, time: 0.094) G_GAN: 1.434 G_GAN_Feat: 2.198 G_VGG: 0.890 D_real: 0.144 D_fake: 0.333 
(epoch: 126, iters: 2800, time: 0.091) G_GAN: 0.874 G_GAN_Feat: 1.805 G_VGG: 0.828 D_real: 0.077 D_fake: 0.484 
(epoch: 127, iters: 240, time: 0.086) G_GAN: 0.944 G_GAN_Feat: 2.283 G_VGG: 0.827 D_real: 0.050 D_fake: 0.395 
(epoch: 127, iters: 640, time: 0.090) G_GAN: 1.143 G_GAN_Feat: 2.244 G_VGG: 0.980 D_real: 0.048 D_fake: 0.363 
(epoch: 127, iters: 1040, time: 0.086) G_GAN: 0.739 G_GAN_Feat: 2.524 G_VGG: 0.825 D_real: 0.056 D_fake: 0.421 
(epoch: 127, iters: 1440, time: 0.090) G_GAN: 1.486 G_GAN_Feat: 2.548 G_VGG: 0.920 D_real: 0.157 D_fake: 0.203 
(epoch: 127, iters: 1840, time: 0.087) G_GAN: 1.585 G_GAN_Feat: 2.294 G_VGG: 0.906 D_real: 0.188 D_fake: 0.324 
(epoch: 127, iters: 2240, time: 0.091) G_GAN: 1.489 G_GAN_Feat: 2.599 G_VGG: 0.901 D_real: 0.082 D_fake: 0.210 
(epoch: 127, iters: 2640, time: 0.093) G_GAN: 1.526 G_GAN_Feat: 2.220 G_VGG: 0.893 D_real: 0.206 D_fake: 0.287 
(epoch: 128, iters: 80, time: 0.090) G_GAN: 1.924 G_GAN_Feat: 2.360 G_VGG: 0.789 D_real: 0.452 D_fake: 0.071 
(epoch: 128, iters: 480, time: 0.094) G_GAN: 1.232 G_GAN_Feat: 2.147 G_VGG: 0.870 D_real: 0.083 D_fake: 0.245 
(epoch: 128, iters: 880, time: 0.094) G_GAN: 1.447 G_GAN_Feat: 2.374 G_VGG: 0.814 D_real: 0.316 D_fake: 0.148 
(epoch: 128, iters: 1280, time: 0.088) G_GAN: 1.375 G_GAN_Feat: 2.390 G_VGG: 0.915 D_real: 0.385 D_fake: 0.154 
(epoch: 128, iters: 1680, time: 0.094) G_GAN: 1.258 G_GAN_Feat: 2.707 G_VGG: 0.949 D_real: 0.089 D_fake: 0.194 
(epoch: 128, iters: 2080, time: 0.093) G_GAN: 1.397 G_GAN_Feat: 2.466 G_VGG: 0.895 D_real: 0.677 D_fake: 0.154 
(epoch: 128, iters: 2480, time: 0.088) G_GAN: 1.254 G_GAN_Feat: 2.487 G_VGG: 0.892 D_real: 0.050 D_fake: 0.189 
(epoch: 128, iters: 2880, time: 0.092) G_GAN: 1.040 G_GAN_Feat: 1.526 G_VGG: 0.756 D_real: 0.224 D_fake: 0.287 
(epoch: 129, iters: 320, time: 0.092) G_GAN: 0.995 G_GAN_Feat: 2.198 G_VGG: 0.947 D_real: 0.138 D_fake: 0.368 
(epoch: 129, iters: 720, time: 0.101) G_GAN: 1.246 G_GAN_Feat: 2.135 G_VGG: 0.879 D_real: 0.160 D_fake: 0.235 
(epoch: 129, iters: 1120, time: 0.085) G_GAN: 1.843 G_GAN_Feat: 2.282 G_VGG: 0.800 D_real: 0.282 D_fake: 0.248 
(epoch: 129, iters: 1520, time: 0.095) G_GAN: 0.712 G_GAN_Feat: 1.847 G_VGG: 0.870 D_real: 0.246 D_fake: 0.439 
(epoch: 129, iters: 1920, time: 0.085) G_GAN: 0.915 G_GAN_Feat: 2.097 G_VGG: 1.021 D_real: 0.105 D_fake: 0.448 
(epoch: 129, iters: 2320, time: 0.099) G_GAN: 1.137 G_GAN_Feat: 2.721 G_VGG: 0.991 D_real: 0.094 D_fake: 0.214 
(epoch: 129, iters: 2720, time: 0.090) G_GAN: 1.783 G_GAN_Feat: 2.559 G_VGG: 0.853 D_real: 0.298 D_fake: 0.044 
(epoch: 130, iters: 160, time: 0.096) G_GAN: 2.020 G_GAN_Feat: 3.279 G_VGG: 0.889 D_real: 0.152 D_fake: 0.090 
(epoch: 130, iters: 560, time: 0.099) G_GAN: 1.948 G_GAN_Feat: 3.309 G_VGG: 0.894 D_real: 0.064 D_fake: 0.017 
(epoch: 130, iters: 960, time: 0.092) G_GAN: 1.607 G_GAN_Feat: 2.737 G_VGG: 0.889 D_real: 0.192 D_fake: 0.170 
(epoch: 130, iters: 1360, time: 0.093) G_GAN: 1.686 G_GAN_Feat: 2.404 G_VGG: 0.886 D_real: 0.165 D_fake: 0.249 
(epoch: 130, iters: 1760, time: 0.093) G_GAN: 1.481 G_GAN_Feat: 2.203 G_VGG: 0.823 D_real: 0.275 D_fake: 0.395 
(epoch: 130, iters: 2160, time: 0.095) G_GAN: 2.044 G_GAN_Feat: 2.922 G_VGG: 0.959 D_real: 0.239 D_fake: 0.090 
(epoch: 130, iters: 2560, time: 0.097) G_GAN: 1.695 G_GAN_Feat: 2.897 G_VGG: 0.886 D_real: 0.027 D_fake: 0.102 
(epoch: 130, iters: 2960, time: 0.096) G_GAN: 1.518 G_GAN_Feat: 2.432 G_VGG: 0.825 D_real: 0.597 D_fake: 0.087 
(epoch: 131, iters: 400, time: 0.086) G_GAN: 1.067 G_GAN_Feat: 2.681 G_VGG: 0.860 D_real: 0.571 D_fake: 0.278 
(epoch: 131, iters: 800, time: 0.086) G_GAN: 0.886 G_GAN_Feat: 1.873 G_VGG: 0.761 D_real: 0.156 D_fake: 0.474 
(epoch: 131, iters: 1200, time: 0.085) G_GAN: 2.059 G_GAN_Feat: 2.925 G_VGG: 1.032 D_real: 0.152 D_fake: 0.045 
(epoch: 131, iters: 1600, time: 0.089) G_GAN: 1.032 G_GAN_Feat: 2.098 G_VGG: 0.916 D_real: 0.192 D_fake: 0.557 
(epoch: 131, iters: 2000, time: 0.099) G_GAN: 1.221 G_GAN_Feat: 2.832 G_VGG: 0.851 D_real: 0.165 D_fake: 0.203 
(epoch: 131, iters: 2400, time: 0.093) G_GAN: 1.069 G_GAN_Feat: 2.744 G_VGG: 0.921 D_real: 0.074 D_fake: 0.839 
(epoch: 131, iters: 2800, time: 0.095) G_GAN: 1.071 G_GAN_Feat: 2.150 G_VGG: 0.913 D_real: 0.045 D_fake: 0.309 
(epoch: 132, iters: 240, time: 0.090) G_GAN: 0.806 G_GAN_Feat: 2.127 G_VGG: 0.885 D_real: 0.193 D_fake: 0.465 
(epoch: 132, iters: 640, time: 0.092) G_GAN: 1.538 G_GAN_Feat: 2.577 G_VGG: 0.890 D_real: 0.100 D_fake: 0.321 
(epoch: 132, iters: 1040, time: 0.084) G_GAN: 1.022 G_GAN_Feat: 1.866 G_VGG: 0.847 D_real: 0.186 D_fake: 0.299 
(epoch: 132, iters: 1440, time: 0.091) G_GAN: 1.063 G_GAN_Feat: 2.335 G_VGG: 0.936 D_real: 0.053 D_fake: 0.475 
(epoch: 132, iters: 1840, time: 0.093) G_GAN: 1.376 G_GAN_Feat: 1.960 G_VGG: 0.895 D_real: 0.203 D_fake: 0.360 
(epoch: 132, iters: 2240, time: 0.099) G_GAN: 1.343 G_GAN_Feat: 2.645 G_VGG: 0.971 D_real: 0.207 D_fake: 0.157 
(epoch: 132, iters: 2640, time: 0.088) G_GAN: 1.315 G_GAN_Feat: 2.411 G_VGG: 0.848 D_real: 0.140 D_fake: 0.218 
(epoch: 133, iters: 80, time: 0.101) G_GAN: 1.466 G_GAN_Feat: 2.478 G_VGG: 0.874 D_real: 0.524 D_fake: 0.294 
(epoch: 133, iters: 480, time: 0.085) G_GAN: 1.401 G_GAN_Feat: 2.861 G_VGG: 0.865 D_real: 0.287 D_fake: 0.236 
(epoch: 133, iters: 880, time: 0.090) G_GAN: 2.081 G_GAN_Feat: 2.980 G_VGG: 0.908 D_real: 0.393 D_fake: 0.036 
(epoch: 133, iters: 1280, time: 0.090) G_GAN: 1.517 G_GAN_Feat: 2.462 G_VGG: 0.881 D_real: 0.103 D_fake: 0.097 
(epoch: 133, iters: 1680, time: 0.085) G_GAN: 1.922 G_GAN_Feat: 2.683 G_VGG: 0.857 D_real: 0.320 D_fake: 0.103 
(epoch: 133, iters: 2080, time: 0.092) G_GAN: 1.169 G_GAN_Feat: 2.166 G_VGG: 0.877 D_real: 0.241 D_fake: 0.375 
(epoch: 133, iters: 2480, time: 0.086) G_GAN: 1.504 G_GAN_Feat: 2.233 G_VGG: 0.809 D_real: 0.597 D_fake: 0.110 
(epoch: 133, iters: 2880, time: 0.087) G_GAN: 1.047 G_GAN_Feat: 2.091 G_VGG: 0.881 D_real: 0.326 D_fake: 0.270 
(epoch: 134, iters: 320, time: 0.097) G_GAN: 1.204 G_GAN_Feat: 2.565 G_VGG: 0.927 D_real: 0.158 D_fake: 0.455 
(epoch: 134, iters: 720, time: 0.086) G_GAN: 1.012 G_GAN_Feat: 2.581 G_VGG: 0.781 D_real: 0.342 D_fake: 0.331 
(epoch: 134, iters: 1120, time: 0.081) G_GAN: 1.066 G_GAN_Feat: 2.079 G_VGG: 0.780 D_real: 0.179 D_fake: 0.330 
(epoch: 134, iters: 1520, time: 0.095) G_GAN: 1.718 G_GAN_Feat: 2.662 G_VGG: 0.861 D_real: 0.248 D_fake: 0.049 
(epoch: 134, iters: 1920, time: 0.091) G_GAN: 1.314 G_GAN_Feat: 2.259 G_VGG: 0.926 D_real: 0.260 D_fake: 0.237 
(epoch: 134, iters: 2320, time: 0.093) G_GAN: 1.522 G_GAN_Feat: 3.116 G_VGG: 0.831 D_real: 0.035 D_fake: 0.084 
(epoch: 134, iters: 2720, time: 0.104) G_GAN: 1.683 G_GAN_Feat: 2.857 G_VGG: 0.885 D_real: 0.227 D_fake: 0.215 
(epoch: 135, iters: 160, time: 0.098) G_GAN: 0.650 G_GAN_Feat: 2.007 G_VGG: 0.869 D_real: 0.128 D_fake: 0.488 
(epoch: 135, iters: 560, time: 0.086) G_GAN: 1.510 G_GAN_Feat: 2.341 G_VGG: 0.895 D_real: 0.301 D_fake: 0.113 
(epoch: 135, iters: 960, time: 0.089) G_GAN: 1.378 G_GAN_Feat: 2.090 G_VGG: 0.802 D_real: 0.131 D_fake: 0.186 
(epoch: 135, iters: 1360, time: 0.092) G_GAN: 1.690 G_GAN_Feat: 2.974 G_VGG: 1.032 D_real: 0.774 D_fake: 0.061 
(epoch: 135, iters: 1760, time: 0.093) G_GAN: 1.189 G_GAN_Feat: 2.283 G_VGG: 0.848 D_real: 0.052 D_fake: 0.569 
(epoch: 135, iters: 2160, time: 0.086) G_GAN: 1.268 G_GAN_Feat: 2.308 G_VGG: 0.867 D_real: 0.620 D_fake: 0.210 
(epoch: 135, iters: 2560, time: 0.090) G_GAN: 0.950 G_GAN_Feat: 1.972 G_VGG: 0.886 D_real: 0.252 D_fake: 0.324 
(epoch: 135, iters: 2960, time: 0.089) G_GAN: 1.517 G_GAN_Feat: 2.429 G_VGG: 0.859 D_real: 0.314 D_fake: 0.220 
(epoch: 136, iters: 400, time: 0.088) G_GAN: 0.938 G_GAN_Feat: 1.865 G_VGG: 0.829 D_real: 0.069 D_fake: 0.454 
(epoch: 136, iters: 800, time: 0.092) G_GAN: 1.400 G_GAN_Feat: 2.198 G_VGG: 0.887 D_real: 0.326 D_fake: 0.181 
(epoch: 136, iters: 1200, time: 0.089) G_GAN: 1.478 G_GAN_Feat: 2.539 G_VGG: 0.947 D_real: 0.172 D_fake: 0.169 
(epoch: 136, iters: 1600, time: 0.085) G_GAN: 1.489 G_GAN_Feat: 2.640 G_VGG: 0.966 D_real: 0.212 D_fake: 0.110 
(epoch: 136, iters: 2000, time: 0.092) G_GAN: 1.206 G_GAN_Feat: 2.492 G_VGG: 0.856 D_real: 0.035 D_fake: 0.291 
(epoch: 136, iters: 2400, time: 0.096) G_GAN: 1.406 G_GAN_Feat: 2.589 G_VGG: 0.854 D_real: 0.201 D_fake: 0.174 
(epoch: 136, iters: 2800, time: 0.089) G_GAN: 2.019 G_GAN_Feat: 2.725 G_VGG: 0.795 D_real: 0.319 D_fake: 0.027 
(epoch: 137, iters: 240, time: 0.088) G_GAN: 1.042 G_GAN_Feat: 2.043 G_VGG: 0.837 D_real: 0.101 D_fake: 0.435 
(epoch: 137, iters: 640, time: 0.096) G_GAN: 1.146 G_GAN_Feat: 2.447 G_VGG: 0.822 D_real: 0.376 D_fake: 0.411 
(epoch: 137, iters: 1040, time: 0.093) G_GAN: 1.114 G_GAN_Feat: 1.741 G_VGG: 0.753 D_real: 0.778 D_fake: 0.509 
(epoch: 137, iters: 1440, time: 0.084) G_GAN: 0.998 G_GAN_Feat: 1.961 G_VGG: 0.824 D_real: 0.146 D_fake: 0.309 
(epoch: 137, iters: 1840, time: 0.096) G_GAN: 1.356 G_GAN_Feat: 2.354 G_VGG: 0.892 D_real: 0.373 D_fake: 0.121 
(epoch: 137, iters: 2240, time: 0.091) G_GAN: 1.752 G_GAN_Feat: 2.925 G_VGG: 0.873 D_real: 0.463 D_fake: 0.180 
(epoch: 137, iters: 2640, time: 0.096) G_GAN: 1.715 G_GAN_Feat: 2.479 G_VGG: 0.843 D_real: 0.140 D_fake: 0.078 
(epoch: 138, iters: 80, time: 0.087) G_GAN: 1.249 G_GAN_Feat: 2.243 G_VGG: 0.826 D_real: 0.273 D_fake: 0.227 
(epoch: 138, iters: 480, time: 0.088) G_GAN: 1.035 G_GAN_Feat: 1.556 G_VGG: 0.769 D_real: 0.314 D_fake: 0.256 
(epoch: 138, iters: 880, time: 0.090) G_GAN: 1.227 G_GAN_Feat: 2.240 G_VGG: 0.905 D_real: 0.187 D_fake: 0.167 
(epoch: 138, iters: 1280, time: 0.089) G_GAN: 1.050 G_GAN_Feat: 2.216 G_VGG: 0.899 D_real: 0.113 D_fake: 0.354 
(epoch: 138, iters: 1680, time: 0.086) G_GAN: 1.203 G_GAN_Feat: 2.221 G_VGG: 0.878 D_real: 0.107 D_fake: 0.251 
(epoch: 138, iters: 2080, time: 0.084) G_GAN: 1.431 G_GAN_Feat: 2.501 G_VGG: 0.837 D_real: 0.266 D_fake: 0.184 
(epoch: 138, iters: 2480, time: 0.083) G_GAN: 1.340 G_GAN_Feat: 2.454 G_VGG: 0.870 D_real: 0.328 D_fake: 0.309 
(epoch: 138, iters: 2880, time: 0.087) G_GAN: 1.611 G_GAN_Feat: 2.614 G_VGG: 0.843 D_real: 0.083 D_fake: 0.066 
(epoch: 139, iters: 320, time: 0.096) G_GAN: 2.001 G_GAN_Feat: 3.170 G_VGG: 0.854 D_real: 0.018 D_fake: 0.031 
(epoch: 139, iters: 720, time: 0.093) G_GAN: 1.154 G_GAN_Feat: 2.221 G_VGG: 0.914 D_real: 0.173 D_fake: 0.310 
(epoch: 139, iters: 1120, time: 0.087) G_GAN: 1.576 G_GAN_Feat: 2.263 G_VGG: 0.752 D_real: 0.503 D_fake: 0.330 
(epoch: 139, iters: 1520, time: 0.096) G_GAN: 1.240 G_GAN_Feat: 2.225 G_VGG: 0.865 D_real: 0.174 D_fake: 0.249 
(epoch: 139, iters: 1920, time: 0.095) G_GAN: 1.821 G_GAN_Feat: 2.321 G_VGG: 0.811 D_real: 0.246 D_fake: 0.089 
(epoch: 139, iters: 2320, time: 0.094) G_GAN: 1.919 G_GAN_Feat: 3.238 G_VGG: 0.865 D_real: 0.369 D_fake: 0.082 
(epoch: 139, iters: 2720, time: 0.095) G_GAN: 2.112 G_GAN_Feat: 2.632 G_VGG: 0.847 D_real: 0.401 D_fake: 0.080 
(epoch: 140, iters: 160, time: 0.101) G_GAN: 1.796 G_GAN_Feat: 2.923 G_VGG: 0.825 D_real: 0.055 D_fake: 0.058 
(epoch: 140, iters: 560, time: 0.088) G_GAN: 1.253 G_GAN_Feat: 2.185 G_VGG: 0.866 D_real: 0.109 D_fake: 0.368 
(epoch: 140, iters: 960, time: 0.086) G_GAN: 1.593 G_GAN_Feat: 2.680 G_VGG: 0.815 D_real: 0.040 D_fake: 0.108 
(epoch: 140, iters: 1360, time: 0.088) G_GAN: 1.525 G_GAN_Feat: 2.011 G_VGG: 0.827 D_real: 0.327 D_fake: 0.150 
(epoch: 140, iters: 1760, time: 0.089) G_GAN: 1.929 G_GAN_Feat: 3.112 G_VGG: 0.814 D_real: 0.027 D_fake: 0.026 
(epoch: 140, iters: 2160, time: 0.091) G_GAN: 1.233 G_GAN_Feat: 2.452 G_VGG: 0.894 D_real: 0.278 D_fake: 0.377 
(epoch: 140, iters: 2560, time: 0.092) G_GAN: 1.320 G_GAN_Feat: 2.411 G_VGG: 0.857 D_real: 0.088 D_fake: 0.462 
(epoch: 140, iters: 2960, time: 0.092) G_GAN: 1.497 G_GAN_Feat: 2.451 G_VGG: 0.855 D_real: 0.413 D_fake: 0.224 
(epoch: 141, iters: 400, time: 0.083) G_GAN: 1.274 G_GAN_Feat: 2.547 G_VGG: 0.880 D_real: 0.076 D_fake: 0.160 
(epoch: 141, iters: 800, time: 0.087) G_GAN: 2.608 G_GAN_Feat: 2.827 G_VGG: 0.811 D_real: 0.470 D_fake: 0.071 
(epoch: 141, iters: 1200, time: 0.087) G_GAN: 0.712 G_GAN_Feat: 1.866 G_VGG: 0.814 D_real: 0.124 D_fake: 0.579 
(epoch: 141, iters: 1600, time: 0.091) G_GAN: 1.212 G_GAN_Feat: 2.496 G_VGG: 0.872 D_real: 0.300 D_fake: 0.251 
(epoch: 141, iters: 2000, time: 0.092) G_GAN: 1.362 G_GAN_Feat: 2.371 G_VGG: 0.807 D_real: 0.188 D_fake: 0.406 
(epoch: 141, iters: 2400, time: 0.082) G_GAN: 1.196 G_GAN_Feat: 2.121 G_VGG: 0.900 D_real: 0.136 D_fake: 0.321 
(epoch: 141, iters: 2800, time: 0.098) G_GAN: 1.425 G_GAN_Feat: 2.285 G_VGG: 0.886 D_real: 0.403 D_fake: 0.229 
(epoch: 142, iters: 240, time: 0.085) G_GAN: 1.541 G_GAN_Feat: 2.139 G_VGG: 0.907 D_real: 0.293 D_fake: 0.201 
(epoch: 142, iters: 640, time: 0.086) G_GAN: 1.897 G_GAN_Feat: 2.943 G_VGG: 0.871 D_real: 0.150 D_fake: 0.094 
(epoch: 142, iters: 1040, time: 0.101) G_GAN: 1.080 G_GAN_Feat: 2.879 G_VGG: 0.918 D_real: 0.058 D_fake: 0.214 
(epoch: 142, iters: 1440, time: 0.096) G_GAN: 1.316 G_GAN_Feat: 2.153 G_VGG: 0.889 D_real: 0.099 D_fake: 0.334 
(epoch: 142, iters: 1840, time: 0.095) G_GAN: 1.691 G_GAN_Feat: 2.743 G_VGG: 0.841 D_real: 0.395 D_fake: 0.056 
(epoch: 142, iters: 2240, time: 0.096) G_GAN: 2.031 G_GAN_Feat: 3.010 G_VGG: 0.889 D_real: 0.161 D_fake: 0.020 
(epoch: 142, iters: 2640, time: 0.085) G_GAN: 1.389 G_GAN_Feat: 1.903 G_VGG: 0.816 D_real: 0.461 D_fake: 0.153 
(epoch: 143, iters: 80, time: 0.091) G_GAN: 1.960 G_GAN_Feat: 3.220 G_VGG: 0.857 D_real: 0.036 D_fake: 0.021 
(epoch: 143, iters: 480, time: 0.086) G_GAN: 0.943 G_GAN_Feat: 2.020 G_VGG: 0.786 D_real: 0.083 D_fake: 0.383 
(epoch: 143, iters: 880, time: 0.095) G_GAN: 1.503 G_GAN_Feat: 2.255 G_VGG: 0.866 D_real: 0.313 D_fake: 0.173 
(epoch: 143, iters: 1280, time: 0.094) G_GAN: 1.891 G_GAN_Feat: 2.811 G_VGG: 0.794 D_real: 0.230 D_fake: 0.085 
(epoch: 143, iters: 1680, time: 0.096) G_GAN: 1.458 G_GAN_Feat: 2.520 G_VGG: 0.798 D_real: 0.092 D_fake: 0.106 
(epoch: 143, iters: 2080, time: 0.089) G_GAN: 0.787 G_GAN_Feat: 1.925 G_VGG: 0.845 D_real: 0.143 D_fake: 0.398 
(epoch: 143, iters: 2480, time: 0.098) G_GAN: 2.086 G_GAN_Feat: 2.678 G_VGG: 0.869 D_real: 0.157 D_fake: 0.048 
(epoch: 143, iters: 2880, time: 0.091) G_GAN: 1.243 G_GAN_Feat: 3.141 G_VGG: 0.773 D_real: 0.326 D_fake: 0.215 
(epoch: 144, iters: 320, time: 0.097) G_GAN: 1.456 G_GAN_Feat: 2.227 G_VGG: 0.747 D_real: 0.150 D_fake: 0.116 
(epoch: 144, iters: 720, time: 0.087) G_GAN: 1.740 G_GAN_Feat: 2.253 G_VGG: 0.880 D_real: 0.059 D_fake: 0.108 
(epoch: 144, iters: 1120, time: 0.087) G_GAN: 1.022 G_GAN_Feat: 1.871 G_VGG: 0.947 D_real: 0.689 D_fake: 0.300 
(epoch: 144, iters: 1520, time: 0.090) G_GAN: 1.507 G_GAN_Feat: 2.094 G_VGG: 0.780 D_real: 0.249 D_fake: 0.172 
(epoch: 144, iters: 1920, time: 0.095) G_GAN: 1.035 G_GAN_Feat: 2.059 G_VGG: 0.857 D_real: 0.132 D_fake: 0.545 
(epoch: 144, iters: 2320, time: 0.094) G_GAN: 1.035 G_GAN_Feat: 2.046 G_VGG: 0.848 D_real: 0.043 D_fake: 0.469 
(epoch: 144, iters: 2720, time: 0.090) G_GAN: 1.892 G_GAN_Feat: 2.943 G_VGG: 0.848 D_real: 0.144 D_fake: 0.023 
(epoch: 145, iters: 160, time: 0.091) G_GAN: 1.201 G_GAN_Feat: 2.465 G_VGG: 0.827 D_real: 0.110 D_fake: 0.167 
(epoch: 145, iters: 560, time: 0.099) G_GAN: 0.923 G_GAN_Feat: 1.766 G_VGG: 0.770 D_real: 0.353 D_fake: 0.302 
(epoch: 145, iters: 960, time: 0.096) G_GAN: 1.139 G_GAN_Feat: 2.244 G_VGG: 0.774 D_real: 0.082 D_fake: 0.234 
(epoch: 145, iters: 1360, time: 0.088) G_GAN: 1.719 G_GAN_Feat: 2.375 G_VGG: 0.815 D_real: 0.087 D_fake: 0.120 
(epoch: 145, iters: 1760, time: 0.096) G_GAN: 1.329 G_GAN_Feat: 2.554 G_VGG: 0.869 D_real: 0.057 D_fake: 0.108 
(epoch: 145, iters: 2160, time: 0.092) G_GAN: 1.587 G_GAN_Feat: 2.330 G_VGG: 0.809 D_real: 0.106 D_fake: 0.101 
(epoch: 145, iters: 2560, time: 0.093) G_GAN: 1.699 G_GAN_Feat: 2.417 G_VGG: 0.852 D_real: 0.268 D_fake: 0.123 
(epoch: 145, iters: 2960, time: 0.094) G_GAN: 1.776 G_GAN_Feat: 2.634 G_VGG: 0.790 D_real: 0.365 D_fake: 0.040 
(epoch: 146, iters: 400, time: 0.094) G_GAN: 1.200 G_GAN_Feat: 1.861 G_VGG: 0.808 D_real: 0.251 D_fake: 0.277 
(epoch: 146, iters: 800, time: 0.095) G_GAN: 0.982 G_GAN_Feat: 1.774 G_VGG: 0.794 D_real: 0.252 D_fake: 0.280 
(epoch: 146, iters: 1200, time: 0.080) G_GAN: 1.384 G_GAN_Feat: 2.163 G_VGG: 0.847 D_real: 0.211 D_fake: 0.226 
(epoch: 146, iters: 1600, time: 0.090) G_GAN: 1.090 G_GAN_Feat: 1.854 G_VGG: 0.825 D_real: 0.212 D_fake: 0.327 
(epoch: 146, iters: 2000, time: 0.092) G_GAN: 1.214 G_GAN_Feat: 2.088 G_VGG: 0.835 D_real: 0.114 D_fake: 0.382 
(epoch: 146, iters: 2400, time: 0.096) G_GAN: 1.043 G_GAN_Feat: 2.061 G_VGG: 0.853 D_real: 0.080 D_fake: 0.374 
(epoch: 146, iters: 2800, time: 0.092) G_GAN: 1.293 G_GAN_Feat: 2.454 G_VGG: 0.830 D_real: 0.041 D_fake: 0.351 
(epoch: 147, iters: 240, time: 0.096) G_GAN: 1.529 G_GAN_Feat: 2.618 G_VGG: 0.808 D_real: 0.117 D_fake: 0.099 
(epoch: 147, iters: 640, time: 0.094) G_GAN: 1.780 G_GAN_Feat: 2.378 G_VGG: 0.873 D_real: 0.245 D_fake: 0.093 
(epoch: 147, iters: 1040, time: 0.091) G_GAN: 1.710 G_GAN_Feat: 2.350 G_VGG: 0.850 D_real: 0.154 D_fake: 0.067 
(epoch: 147, iters: 1440, time: 0.098) G_GAN: 1.176 G_GAN_Feat: 1.798 G_VGG: 0.813 D_real: 0.290 D_fake: 0.261 
(epoch: 147, iters: 1840, time: 0.085) G_GAN: 1.312 G_GAN_Feat: 2.027 G_VGG: 0.774 D_real: 0.159 D_fake: 0.194 
(epoch: 147, iters: 2240, time: 0.088) G_GAN: 1.382 G_GAN_Feat: 2.037 G_VGG: 0.774 D_real: 0.284 D_fake: 0.237 
(epoch: 147, iters: 2640, time: 0.092) G_GAN: 1.759 G_GAN_Feat: 2.723 G_VGG: 0.862 D_real: 0.070 D_fake: 0.065 
(epoch: 148, iters: 80, time: 0.095) G_GAN: 1.213 G_GAN_Feat: 2.444 G_VGG: 0.807 D_real: 0.176 D_fake: 0.394 
(epoch: 148, iters: 480, time: 0.086) G_GAN: 1.016 G_GAN_Feat: 1.996 G_VGG: 0.811 D_real: 0.250 D_fake: 0.557 
(epoch: 148, iters: 880, time: 0.084) G_GAN: 1.665 G_GAN_Feat: 2.567 G_VGG: 0.837 D_real: 0.254 D_fake: 0.052 
(epoch: 148, iters: 1280, time: 0.089) G_GAN: 1.217 G_GAN_Feat: 2.201 G_VGG: 0.955 D_real: 0.241 D_fake: 0.226 
(epoch: 148, iters: 1680, time: 0.085) G_GAN: 1.618 G_GAN_Feat: 2.404 G_VGG: 0.755 D_real: 0.158 D_fake: 0.059 
(epoch: 148, iters: 2080, time: 0.100) G_GAN: 1.187 G_GAN_Feat: 2.434 G_VGG: 0.819 D_real: 0.105 D_fake: 0.367 
(epoch: 148, iters: 2480, time: 0.088) G_GAN: 1.696 G_GAN_Feat: 2.201 G_VGG: 0.893 D_real: 0.271 D_fake: 0.078 
(epoch: 148, iters: 2880, time: 0.088) G_GAN: 1.080 G_GAN_Feat: 2.451 G_VGG: 0.832 D_real: 0.197 D_fake: 0.355 
(epoch: 149, iters: 320, time: 0.091) G_GAN: 1.215 G_GAN_Feat: 2.129 G_VGG: 0.759 D_real: 0.218 D_fake: 0.163 
(epoch: 149, iters: 720, time: 0.090) G_GAN: 1.214 G_GAN_Feat: 2.676 G_VGG: 0.831 D_real: 0.021 D_fake: 0.497 
(epoch: 149, iters: 1120, time: 0.095) G_GAN: 1.746 G_GAN_Feat: 2.562 G_VGG: 0.752 D_real: 0.071 D_fake: 0.089 
(epoch: 149, iters: 1520, time: 0.093) G_GAN: 1.352 G_GAN_Feat: 2.303 G_VGG: 0.863 D_real: 0.266 D_fake: 0.378 
(epoch: 149, iters: 1920, time: 0.091) G_GAN: 2.209 G_GAN_Feat: 2.618 G_VGG: 0.854 D_real: 0.150 D_fake: 0.032 
(epoch: 149, iters: 2320, time: 0.093) G_GAN: 1.734 G_GAN_Feat: 2.559 G_VGG: 0.852 D_real: 0.465 D_fake: 0.073 
(epoch: 149, iters: 2720, time: 0.090) G_GAN: 1.558 G_GAN_Feat: 2.890 G_VGG: 0.800 D_real: 0.029 D_fake: 0.069 
(epoch: 150, iters: 160, time: 0.092) G_GAN: 1.625 G_GAN_Feat: 2.503 G_VGG: 0.900 D_real: 0.201 D_fake: 0.122 
(epoch: 150, iters: 560, time: 0.088) G_GAN: 1.436 G_GAN_Feat: 2.137 G_VGG: 0.809 D_real: 0.327 D_fake: 0.204 
(epoch: 150, iters: 960, time: 0.093) G_GAN: 1.957 G_GAN_Feat: 2.497 G_VGG: 0.842 D_real: 0.416 D_fake: 0.052 
(epoch: 150, iters: 1360, time: 0.096) G_GAN: 1.411 G_GAN_Feat: 1.983 G_VGG: 0.891 D_real: 0.463 D_fake: 0.139 
(epoch: 150, iters: 1760, time: 0.095) G_GAN: 1.387 G_GAN_Feat: 2.156 G_VGG: 0.776 D_real: 0.204 D_fake: 0.130 
(epoch: 150, iters: 2160, time: 0.094) G_GAN: 1.241 G_GAN_Feat: 1.897 G_VGG: 0.775 D_real: 0.392 D_fake: 0.190 
(epoch: 150, iters: 2560, time: 0.100) G_GAN: 1.447 G_GAN_Feat: 2.254 G_VGG: 0.857 D_real: 0.118 D_fake: 0.451 
(epoch: 150, iters: 2960, time: 0.097) G_GAN: 1.967 G_GAN_Feat: 2.269 G_VGG: 0.873 D_real: 0.265 D_fake: 0.036 
(epoch: 151, iters: 400, time: 0.094) G_GAN: 1.259 G_GAN_Feat: 2.030 G_VGG: 0.790 D_real: 0.028 D_fake: 0.270 
(epoch: 151, iters: 800, time: 0.093) G_GAN: 1.173 G_GAN_Feat: 2.060 G_VGG: 0.823 D_real: 0.123 D_fake: 0.247 
(epoch: 151, iters: 1200, time: 0.086) G_GAN: 1.096 G_GAN_Feat: 2.412 G_VGG: 0.795 D_real: 0.169 D_fake: 0.351 
(epoch: 151, iters: 1600, time: 0.101) G_GAN: 0.922 G_GAN_Feat: 1.889 G_VGG: 0.771 D_real: 0.094 D_fake: 0.326 
(epoch: 151, iters: 2000, time: 0.086) G_GAN: 1.476 G_GAN_Feat: 2.269 G_VGG: 0.762 D_real: 0.050 D_fake: 0.108 
(epoch: 151, iters: 2400, time: 0.086) G_GAN: 0.924 G_GAN_Feat: 1.898 G_VGG: 0.835 D_real: 0.120 D_fake: 0.353 
(epoch: 151, iters: 2800, time: 0.098) G_GAN: 1.352 G_GAN_Feat: 2.016 G_VGG: 0.776 D_real: 0.226 D_fake: 0.232 
(epoch: 152, iters: 240, time: 0.098) G_GAN: 1.265 G_GAN_Feat: 2.248 G_VGG: 0.798 D_real: 0.047 D_fake: 0.283 
(epoch: 152, iters: 640, time: 0.095) G_GAN: 1.062 G_GAN_Feat: 2.057 G_VGG: 0.821 D_real: 0.081 D_fake: 0.468 
(epoch: 152, iters: 1040, time: 0.094) G_GAN: 1.134 G_GAN_Feat: 1.826 G_VGG: 0.777 D_real: 0.159 D_fake: 0.394 
(epoch: 152, iters: 1440, time: 0.087) G_GAN: 1.795 G_GAN_Feat: 2.391 G_VGG: 0.829 D_real: 0.124 D_fake: 0.038 
(epoch: 152, iters: 1840, time: 0.091) G_GAN: 1.257 G_GAN_Feat: 2.061 G_VGG: 0.758 D_real: 0.208 D_fake: 0.244 
(epoch: 152, iters: 2240, time: 0.091) G_GAN: 1.298 G_GAN_Feat: 1.928 G_VGG: 0.805 D_real: 0.131 D_fake: 0.250 
(epoch: 152, iters: 2640, time: 0.094) G_GAN: 1.435 G_GAN_Feat: 1.937 G_VGG: 0.811 D_real: 0.284 D_fake: 0.188 
(epoch: 153, iters: 80, time: 0.092) G_GAN: 1.633 G_GAN_Feat: 2.642 G_VGG: 0.870 D_real: 0.133 D_fake: 0.109 
(epoch: 153, iters: 480, time: 0.091) G_GAN: 1.051 G_GAN_Feat: 1.938 G_VGG: 0.806 D_real: 0.223 D_fake: 0.418 
(epoch: 153, iters: 880, time: 0.084) G_GAN: 2.030 G_GAN_Feat: 3.036 G_VGG: 0.853 D_real: 0.055 D_fake: 0.030 
(epoch: 153, iters: 1280, time: 0.092) G_GAN: 1.403 G_GAN_Feat: 2.330 G_VGG: 0.848 D_real: 0.201 D_fake: 0.192 
(epoch: 153, iters: 1680, time: 0.087) G_GAN: 1.127 G_GAN_Feat: 1.887 G_VGG: 0.817 D_real: 0.078 D_fake: 0.517 
(epoch: 153, iters: 2080, time: 0.086) G_GAN: 1.764 G_GAN_Feat: 2.733 G_VGG: 0.781 D_real: 0.103 D_fake: 0.042 
(epoch: 153, iters: 2480, time: 0.094) G_GAN: 1.069 G_GAN_Feat: 1.957 G_VGG: 0.809 D_real: 0.040 D_fake: 0.951 
(epoch: 153, iters: 2880, time: 0.087) G_GAN: 1.960 G_GAN_Feat: 2.727 G_VGG: 0.905 D_real: 0.384 D_fake: 0.026 
(epoch: 154, iters: 320, time: 0.094) G_GAN: 1.285 G_GAN_Feat: 2.172 G_VGG: 0.816 D_real: 0.114 D_fake: 0.448 
(epoch: 154, iters: 720, time: 0.097) G_GAN: 1.416 G_GAN_Feat: 2.101 G_VGG: 0.821 D_real: 0.294 D_fake: 0.181 
(epoch: 154, iters: 1120, time: 0.083) G_GAN: 1.906 G_GAN_Feat: 2.674 G_VGG: 0.799 D_real: 0.320 D_fake: 0.023 
(epoch: 154, iters: 1520, time: 0.095) G_GAN: 2.261 G_GAN_Feat: 2.778 G_VGG: 0.730 D_real: 0.141 D_fake: 0.027 
(epoch: 154, iters: 1920, time: 0.098) G_GAN: 1.736 G_GAN_Feat: 2.452 G_VGG: 0.776 D_real: 0.082 D_fake: 0.063 
(epoch: 154, iters: 2320, time: 0.092) G_GAN: 1.455 G_GAN_Feat: 2.227 G_VGG: 0.845 D_real: 0.163 D_fake: 0.188 
(epoch: 154, iters: 2720, time: 0.088) G_GAN: 1.563 G_GAN_Feat: 2.485 G_VGG: 0.951 D_real: 0.223 D_fake: 0.125 
(epoch: 155, iters: 160, time: 0.077) G_GAN: 2.161 G_GAN_Feat: 2.431 G_VGG: 0.807 D_real: 0.455 D_fake: 0.022 
(epoch: 155, iters: 560, time: 0.085) G_GAN: 1.363 G_GAN_Feat: 2.291 G_VGG: 0.760 D_real: 0.225 D_fake: 0.181 
(epoch: 155, iters: 960, time: 0.091) G_GAN: 0.989 G_GAN_Feat: 2.262 G_VGG: 0.773 D_real: 0.210 D_fake: 0.284 
(epoch: 155, iters: 1360, time: 0.099) G_GAN: 1.385 G_GAN_Feat: 2.114 G_VGG: 0.844 D_real: 0.135 D_fake: 0.349 
(epoch: 155, iters: 1760, time: 0.086) G_GAN: 1.210 G_GAN_Feat: 1.877 G_VGG: 0.884 D_real: 0.146 D_fake: 0.275 
(epoch: 155, iters: 2160, time: 0.096) G_GAN: 1.657 G_GAN_Feat: 2.400 G_VGG: 0.808 D_real: 0.060 D_fake: 0.101 
(epoch: 155, iters: 2560, time: 0.085) G_GAN: 1.376 G_GAN_Feat: 2.399 G_VGG: 0.854 D_real: 0.029 D_fake: 0.237 
(epoch: 155, iters: 2960, time: 0.094) G_GAN: 1.208 G_GAN_Feat: 2.131 G_VGG: 0.909 D_real: 0.072 D_fake: 0.286 
(epoch: 156, iters: 400, time: 0.090) G_GAN: 1.530 G_GAN_Feat: 2.199 G_VGG: 0.789 D_real: 0.251 D_fake: 0.218 
(epoch: 156, iters: 800, time: 0.090) G_GAN: 1.275 G_GAN_Feat: 2.065 G_VGG: 0.852 D_real: 0.082 D_fake: 0.228 
(epoch: 156, iters: 1200, time: 0.098) G_GAN: 1.939 G_GAN_Feat: 2.510 G_VGG: 0.777 D_real: 0.198 D_fake: 0.054 
(epoch: 156, iters: 1600, time: 0.095) G_GAN: 1.318 G_GAN_Feat: 2.332 G_VGG: 0.769 D_real: 0.073 D_fake: 0.194 
(epoch: 156, iters: 2000, time: 0.089) G_GAN: 1.420 G_GAN_Feat: 2.100 G_VGG: 0.813 D_real: 0.199 D_fake: 0.147 
(epoch: 156, iters: 2400, time: 0.086) G_GAN: 1.799 G_GAN_Feat: 2.456 G_VGG: 0.764 D_real: 0.045 D_fake: 0.059 
(epoch: 156, iters: 2800, time: 0.092) G_GAN: 1.575 G_GAN_Feat: 2.248 G_VGG: 0.832 D_real: 0.071 D_fake: 0.175 
(epoch: 157, iters: 240, time: 0.092) G_GAN: 1.775 G_GAN_Feat: 2.581 G_VGG: 0.761 D_real: 0.134 D_fake: 0.108 
(epoch: 157, iters: 640, time: 0.098) G_GAN: 1.192 G_GAN_Feat: 1.985 G_VGG: 0.779 D_real: 0.381 D_fake: 0.305 
(epoch: 157, iters: 1040, time: 0.092) G_GAN: 1.292 G_GAN_Feat: 2.216 G_VGG: 0.841 D_real: 0.124 D_fake: 0.208 
(epoch: 157, iters: 1440, time: 0.086) G_GAN: 1.433 G_GAN_Feat: 2.172 G_VGG: 0.884 D_real: 0.300 D_fake: 0.147 
(epoch: 157, iters: 1840, time: 0.094) G_GAN: 1.293 G_GAN_Feat: 1.914 G_VGG: 0.852 D_real: 0.207 D_fake: 0.201 
(epoch: 157, iters: 2240, time: 0.091) G_GAN: 1.460 G_GAN_Feat: 2.116 G_VGG: 0.851 D_real: 0.052 D_fake: 0.214 
(epoch: 157, iters: 2640, time: 0.094) G_GAN: 1.027 G_GAN_Feat: 2.117 G_VGG: 0.865 D_real: 0.080 D_fake: 0.405 
(epoch: 158, iters: 80, time: 0.088) G_GAN: 2.375 G_GAN_Feat: 2.500 G_VGG: 0.763 D_real: 0.316 D_fake: 0.041 
(epoch: 158, iters: 480, time: 0.087) G_GAN: 1.498 G_GAN_Feat: 2.069 G_VGG: 0.738 D_real: 0.063 D_fake: 0.140 
(epoch: 158, iters: 880, time: 0.099) G_GAN: 1.080 G_GAN_Feat: 2.006 G_VGG: 0.863 D_real: 0.119 D_fake: 0.345 
(epoch: 158, iters: 1280, time: 0.088) G_GAN: 1.709 G_GAN_Feat: 2.387 G_VGG: 0.852 D_real: 0.039 D_fake: 0.077 
(epoch: 158, iters: 1680, time: 0.093) G_GAN: 1.147 G_GAN_Feat: 1.727 G_VGG: 0.713 D_real: 0.399 D_fake: 0.283 
(epoch: 158, iters: 2080, time: 0.092) G_GAN: 1.656 G_GAN_Feat: 2.562 G_VGG: 0.777 D_real: 0.367 D_fake: 0.083 
(epoch: 158, iters: 2480, time: 0.093) G_GAN: 1.306 G_GAN_Feat: 1.986 G_VGG: 0.903 D_real: 0.223 D_fake: 0.199 
(epoch: 158, iters: 2880, time: 0.087) G_GAN: 1.576 G_GAN_Feat: 2.356 G_VGG: 0.815 D_real: 0.203 D_fake: 0.068 
(epoch: 159, iters: 320, time: 0.091) G_GAN: 1.474 G_GAN_Feat: 2.196 G_VGG: 0.761 D_real: 0.339 D_fake: 0.104 
(epoch: 159, iters: 720, time: 0.096) G_GAN: 1.060 G_GAN_Feat: 1.958 G_VGG: 0.837 D_real: 0.238 D_fake: 0.435 
(epoch: 159, iters: 1120, time: 0.089) G_GAN: 1.383 G_GAN_Feat: 1.966 G_VGG: 0.748 D_real: 0.413 D_fake: 0.146 
(epoch: 159, iters: 1520, time: 0.089) G_GAN: 0.821 G_GAN_Feat: 1.810 G_VGG: 0.857 D_real: 0.091 D_fake: 0.475 
(epoch: 159, iters: 1920, time: 0.092) G_GAN: 1.247 G_GAN_Feat: 2.125 G_VGG: 0.835 D_real: 0.168 D_fake: 0.321 
(epoch: 159, iters: 2320, time: 0.084) G_GAN: 1.105 G_GAN_Feat: 2.106 G_VGG: 0.804 D_real: 0.346 D_fake: 0.453 
(epoch: 159, iters: 2720, time: 0.092) G_GAN: 1.394 G_GAN_Feat: 2.145 G_VGG: 0.781 D_real: 0.116 D_fake: 0.189 
(epoch: 160, iters: 160, time: 0.093) G_GAN: 1.216 G_GAN_Feat: 2.349 G_VGG: 0.927 D_real: 0.205 D_fake: 0.291 
(epoch: 160, iters: 560, time: 0.087) G_GAN: 1.651 G_GAN_Feat: 2.781 G_VGG: 0.758 D_real: 0.024 D_fake: 0.046 
(epoch: 160, iters: 960, time: 0.097) G_GAN: 1.069 G_GAN_Feat: 1.897 G_VGG: 0.828 D_real: 0.402 D_fake: 0.277 
(epoch: 160, iters: 1360, time: 0.097) G_GAN: 1.199 G_GAN_Feat: 1.953 G_VGG: 0.891 D_real: 0.162 D_fake: 0.225 
(epoch: 160, iters: 1760, time: 0.092) G_GAN: 1.357 G_GAN_Feat: 2.163 G_VGG: 0.864 D_real: 0.096 D_fake: 0.317 
(epoch: 160, iters: 2160, time: 0.089) G_GAN: 1.073 G_GAN_Feat: 1.911 G_VGG: 0.803 D_real: 0.102 D_fake: 0.421 
(epoch: 160, iters: 2560, time: 0.093) G_GAN: 1.579 G_GAN_Feat: 2.094 G_VGG: 0.764 D_real: 0.212 D_fake: 0.130 
(epoch: 160, iters: 2960, time: 0.083) G_GAN: 1.714 G_GAN_Feat: 2.683 G_VGG: 0.813 D_real: 0.030 D_fake: 0.132 
(epoch: 161, iters: 400, time: 0.092) G_GAN: 1.135 G_GAN_Feat: 2.276 G_VGG: 0.784 D_real: 0.548 D_fake: 0.221 
(epoch: 161, iters: 800, time: 0.089) G_GAN: 1.222 G_GAN_Feat: 1.777 G_VGG: 0.761 D_real: 0.128 D_fake: 0.280 
(epoch: 161, iters: 1200, time: 0.092) G_GAN: 1.325 G_GAN_Feat: 1.919 G_VGG: 0.722 D_real: 0.167 D_fake: 0.250 
(epoch: 161, iters: 1600, time: 0.092) G_GAN: 1.348 G_GAN_Feat: 2.497 G_VGG: 0.895 D_real: 0.073 D_fake: 0.249 
(epoch: 161, iters: 2000, time: 0.091) G_GAN: 1.420 G_GAN_Feat: 2.437 G_VGG: 0.790 D_real: 0.059 D_fake: 0.107 
(epoch: 161, iters: 2400, time: 0.099) G_GAN: 1.497 G_GAN_Feat: 2.573 G_VGG: 0.706 D_real: 0.351 D_fake: 0.138 
(epoch: 161, iters: 2800, time: 0.091) G_GAN: 1.225 G_GAN_Feat: 2.127 G_VGG: 0.777 D_real: 0.164 D_fake: 0.190 
(epoch: 162, iters: 240, time: 0.091) G_GAN: 1.158 G_GAN_Feat: 2.075 G_VGG: 0.818 D_real: 0.078 D_fake: 0.345 
(epoch: 162, iters: 640, time: 0.091) G_GAN: 1.544 G_GAN_Feat: 2.421 G_VGG: 0.826 D_real: 0.024 D_fake: 0.198 
(epoch: 162, iters: 1040, time: 0.094) G_GAN: 1.051 G_GAN_Feat: 2.049 G_VGG: 0.876 D_real: 0.022 D_fake: 0.479 
(epoch: 162, iters: 1440, time: 0.090) G_GAN: 1.224 G_GAN_Feat: 1.775 G_VGG: 0.751 D_real: 0.211 D_fake: 0.292 
(epoch: 162, iters: 1840, time: 0.093) G_GAN: 1.349 G_GAN_Feat: 2.222 G_VGG: 0.868 D_real: 0.142 D_fake: 0.276 
(epoch: 162, iters: 2240, time: 0.088) G_GAN: 1.324 G_GAN_Feat: 2.043 G_VGG: 0.768 D_real: 0.153 D_fake: 0.254 
(epoch: 162, iters: 2640, time: 0.086) G_GAN: 0.975 G_GAN_Feat: 1.844 G_VGG: 0.752 D_real: 0.037 D_fake: 0.604 
(epoch: 163, iters: 80, time: 0.092) G_GAN: 1.023 G_GAN_Feat: 1.941 G_VGG: 0.830 D_real: 0.081 D_fake: 0.473 
(epoch: 163, iters: 480, time: 0.094) G_GAN: 1.493 G_GAN_Feat: 2.385 G_VGG: 0.832 D_real: 0.054 D_fake: 0.185 
(epoch: 163, iters: 880, time: 0.099) G_GAN: 1.454 G_GAN_Feat: 2.003 G_VGG: 0.827 D_real: 0.051 D_fake: 0.127 
(epoch: 163, iters: 1280, time: 0.092) G_GAN: 2.194 G_GAN_Feat: 2.647 G_VGG: 0.860 D_real: 0.287 D_fake: 0.027 
(epoch: 163, iters: 1680, time: 0.095) G_GAN: 1.105 G_GAN_Feat: 2.303 G_VGG: 0.859 D_real: 0.533 D_fake: 0.323 
(epoch: 163, iters: 2080, time: 0.090) G_GAN: 1.144 G_GAN_Feat: 2.074 G_VGG: 0.866 D_real: 0.164 D_fake: 0.233 
(epoch: 163, iters: 2480, time: 0.085) G_GAN: 1.396 G_GAN_Feat: 2.028 G_VGG: 0.766 D_real: 0.049 D_fake: 0.309 
(epoch: 163, iters: 2880, time: 0.092) G_GAN: 2.084 G_GAN_Feat: 2.842 G_VGG: 0.802 D_real: 0.148 D_fake: 0.035 
(epoch: 164, iters: 320, time: 0.094) G_GAN: 0.997 G_GAN_Feat: 2.109 G_VGG: 0.858 D_real: 0.155 D_fake: 0.396 
(epoch: 164, iters: 720, time: 0.086) G_GAN: 1.629 G_GAN_Feat: 1.947 G_VGG: 0.787 D_real: 0.212 D_fake: 0.111 
(epoch: 164, iters: 1120, time: 0.089) G_GAN: 1.248 G_GAN_Feat: 1.946 G_VGG: 0.817 D_real: 0.050 D_fake: 0.534 
(epoch: 164, iters: 1520, time: 0.103) G_GAN: 1.316 G_GAN_Feat: 2.047 G_VGG: 0.811 D_real: 0.048 D_fake: 0.311 
(epoch: 164, iters: 1920, time: 0.084) G_GAN: 1.381 G_GAN_Feat: 2.156 G_VGG: 0.892 D_real: 0.250 D_fake: 0.185 
(epoch: 164, iters: 2320, time: 0.093) G_GAN: 1.437 G_GAN_Feat: 2.056 G_VGG: 0.885 D_real: 0.105 D_fake: 0.145 
(epoch: 164, iters: 2720, time: 0.098) G_GAN: 1.469 G_GAN_Feat: 2.142 G_VGG: 0.853 D_real: 0.263 D_fake: 0.157 
(epoch: 165, iters: 160, time: 0.090) G_GAN: 1.174 G_GAN_Feat: 2.214 G_VGG: 0.889 D_real: 0.239 D_fake: 0.377 
(epoch: 165, iters: 560, time: 0.084) G_GAN: 0.559 G_GAN_Feat: 1.726 G_VGG: 0.832 D_real: 0.138 D_fake: 0.675 
(epoch: 165, iters: 960, time: 0.084) G_GAN: 1.374 G_GAN_Feat: 1.899 G_VGG: 0.797 D_real: 0.104 D_fake: 0.237 
(epoch: 165, iters: 1360, time: 0.088) G_GAN: 2.155 G_GAN_Feat: 2.353 G_VGG: 0.845 D_real: 0.056 D_fake: 0.065 
(epoch: 165, iters: 1760, time: 0.093) G_GAN: 1.370 G_GAN_Feat: 2.518 G_VGG: 0.807 D_real: 0.031 D_fake: 0.299 
(epoch: 165, iters: 2160, time: 0.086) G_GAN: 1.527 G_GAN_Feat: 2.517 G_VGG: 0.767 D_real: 0.098 D_fake: 0.085 
(epoch: 165, iters: 2560, time: 0.092) G_GAN: 1.830 G_GAN_Feat: 2.244 G_VGG: 0.759 D_real: 0.479 D_fake: 0.034 
(epoch: 165, iters: 2960, time: 0.085) G_GAN: 1.295 G_GAN_Feat: 2.175 G_VGG: 0.790 D_real: 0.056 D_fake: 0.195 
(epoch: 166, iters: 400, time: 0.085) G_GAN: 1.584 G_GAN_Feat: 2.106 G_VGG: 0.772 D_real: 0.155 D_fake: 0.081 
(epoch: 166, iters: 800, time: 0.095) G_GAN: 1.736 G_GAN_Feat: 2.443 G_VGG: 0.825 D_real: 0.385 D_fake: 0.078 
(epoch: 166, iters: 1200, time: 0.085) G_GAN: 1.376 G_GAN_Feat: 1.919 G_VGG: 0.772 D_real: 0.100 D_fake: 0.289 
(epoch: 166, iters: 1600, time: 0.094) G_GAN: 2.238 G_GAN_Feat: 2.824 G_VGG: 0.825 D_real: 0.228 D_fake: 0.035 
(epoch: 166, iters: 2000, time: 0.103) G_GAN: 1.908 G_GAN_Feat: 2.430 G_VGG: 0.764 D_real: 0.086 D_fake: 0.052 
(epoch: 166, iters: 2400, time: 0.095) G_GAN: 1.921 G_GAN_Feat: 2.381 G_VGG: 0.763 D_real: 0.159 D_fake: 0.064 
(epoch: 166, iters: 2800, time: 0.096) G_GAN: 0.696 G_GAN_Feat: 1.934 G_VGG: 0.849 D_real: 0.121 D_fake: 0.472 
(epoch: 167, iters: 240, time: 0.099) G_GAN: 2.150 G_GAN_Feat: 2.486 G_VGG: 0.779 D_real: 0.076 D_fake: 0.029 
(epoch: 167, iters: 640, time: 0.088) G_GAN: 1.288 G_GAN_Feat: 1.813 G_VGG: 0.729 D_real: 0.212 D_fake: 0.357 
(epoch: 167, iters: 1040, time: 0.092) G_GAN: 1.406 G_GAN_Feat: 2.013 G_VGG: 0.798 D_real: 0.071 D_fake: 0.189 
(epoch: 167, iters: 1440, time: 0.095) G_GAN: 1.449 G_GAN_Feat: 2.018 G_VGG: 0.827 D_real: 0.303 D_fake: 0.168 
(epoch: 167, iters: 1840, time: 0.086) G_GAN: 1.775 G_GAN_Feat: 2.658 G_VGG: 0.715 D_real: 0.079 D_fake: 0.039 
(epoch: 167, iters: 2240, time: 0.103) G_GAN: 1.391 G_GAN_Feat: 2.078 G_VGG: 0.784 D_real: 0.205 D_fake: 0.264 
(epoch: 167, iters: 2640, time: 0.098) G_GAN: 2.065 G_GAN_Feat: 2.821 G_VGG: 0.851 D_real: 0.077 D_fake: 0.018 
(epoch: 168, iters: 80, time: 0.087) G_GAN: 1.262 G_GAN_Feat: 2.117 G_VGG: 0.855 D_real: 0.028 D_fake: 0.474 
(epoch: 168, iters: 480, time: 0.091) G_GAN: 1.098 G_GAN_Feat: 1.897 G_VGG: 0.839 D_real: 0.118 D_fake: 0.346 
(epoch: 168, iters: 880, time: 0.096) G_GAN: 1.888 G_GAN_Feat: 2.598 G_VGG: 0.743 D_real: 0.104 D_fake: 0.071 
(epoch: 168, iters: 1280, time: 0.094) G_GAN: 1.755 G_GAN_Feat: 2.466 G_VGG: 0.821 D_real: 0.034 D_fake: 0.039 
(epoch: 168, iters: 1680, time: 0.095) G_GAN: 0.969 G_GAN_Feat: 1.942 G_VGG: 0.886 D_real: 0.120 D_fake: 0.346 
(epoch: 168, iters: 2080, time: 0.095) G_GAN: 1.202 G_GAN_Feat: 1.886 G_VGG: 0.814 D_real: 0.249 D_fake: 0.250 
(epoch: 168, iters: 2480, time: 0.095) G_GAN: 1.302 G_GAN_Feat: 1.774 G_VGG: 0.808 D_real: 0.107 D_fake: 0.279 
(epoch: 168, iters: 2880, time: 0.092) G_GAN: 1.820 G_GAN_Feat: 2.278 G_VGG: 0.769 D_real: 0.123 D_fake: 0.046 
(epoch: 169, iters: 320, time: 0.089) G_GAN: 1.526 G_GAN_Feat: 2.186 G_VGG: 0.743 D_real: 0.205 D_fake: 0.151 
(epoch: 169, iters: 720, time: 0.094) G_GAN: 1.079 G_GAN_Feat: 1.858 G_VGG: 0.855 D_real: 0.203 D_fake: 0.342 
(epoch: 169, iters: 1120, time: 0.086) G_GAN: 1.078 G_GAN_Feat: 2.169 G_VGG: 0.805 D_real: 0.038 D_fake: 0.378 
(epoch: 169, iters: 1520, time: 0.095) G_GAN: 1.380 G_GAN_Feat: 1.900 G_VGG: 0.798 D_real: 0.081 D_fake: 0.148 
(epoch: 169, iters: 1920, time: 0.086) G_GAN: 1.631 G_GAN_Feat: 2.352 G_VGG: 0.879 D_real: 0.112 D_fake: 0.099 
(epoch: 169, iters: 2320, time: 0.094) G_GAN: 1.306 G_GAN_Feat: 1.937 G_VGG: 0.826 D_real: 0.164 D_fake: 0.233 
(epoch: 169, iters: 2720, time: 0.087) G_GAN: 1.259 G_GAN_Feat: 1.908 G_VGG: 0.829 D_real: 0.214 D_fake: 0.284 
(epoch: 170, iters: 160, time: 0.094) G_GAN: 1.531 G_GAN_Feat: 2.131 G_VGG: 0.823 D_real: 0.191 D_fake: 0.161 
(epoch: 170, iters: 560, time: 0.092) G_GAN: 1.385 G_GAN_Feat: 2.160 G_VGG: 0.885 D_real: 0.168 D_fake: 0.255 
(epoch: 170, iters: 960, time: 0.091) G_GAN: 1.194 G_GAN_Feat: 2.236 G_VGG: 0.783 D_real: 0.035 D_fake: 0.401 
(epoch: 170, iters: 1360, time: 0.092) G_GAN: 1.464 G_GAN_Feat: 2.105 G_VGG: 0.816 D_real: 0.208 D_fake: 0.180 
(epoch: 170, iters: 1760, time: 0.079) G_GAN: 1.780 G_GAN_Feat: 2.319 G_VGG: 0.788 D_real: 0.204 D_fake: 0.056 
(epoch: 170, iters: 2160, time: 0.093) G_GAN: 1.038 G_GAN_Feat: 2.097 G_VGG: 0.817 D_real: 0.031 D_fake: 0.537 
(epoch: 170, iters: 2560, time: 0.091) G_GAN: 1.232 G_GAN_Feat: 2.106 G_VGG: 0.837 D_real: 0.178 D_fake: 0.215 
(epoch: 170, iters: 2960, time: 0.091) G_GAN: 1.469 G_GAN_Feat: 1.965 G_VGG: 0.830 D_real: 0.202 D_fake: 0.159 
(epoch: 171, iters: 400, time: 0.093) G_GAN: 1.083 G_GAN_Feat: 1.805 G_VGG: 0.804 D_real: 0.054 D_fake: 0.515 
(epoch: 171, iters: 800, time: 0.098) G_GAN: 2.183 G_GAN_Feat: 2.326 G_VGG: 0.806 D_real: 0.441 D_fake: 0.029 
(epoch: 171, iters: 1200, time: 0.089) G_GAN: 1.551 G_GAN_Feat: 2.207 G_VGG: 0.775 D_real: 0.436 D_fake: 0.115 
(epoch: 171, iters: 1600, time: 0.085) G_GAN: 1.861 G_GAN_Feat: 2.324 G_VGG: 0.860 D_real: 0.311 D_fake: 0.045 
(epoch: 171, iters: 2000, time: 0.093) G_GAN: 1.078 G_GAN_Feat: 1.906 G_VGG: 0.769 D_real: 0.053 D_fake: 0.385 
(epoch: 171, iters: 2400, time: 0.094) G_GAN: 1.568 G_GAN_Feat: 2.271 G_VGG: 0.761 D_real: 0.030 D_fake: 0.100 
(epoch: 171, iters: 2800, time: 0.093) G_GAN: 1.339 G_GAN_Feat: 1.950 G_VGG: 0.783 D_real: 0.165 D_fake: 0.317 
(epoch: 172, iters: 240, time: 0.097) G_GAN: 1.603 G_GAN_Feat: 2.207 G_VGG: 0.743 D_real: 0.071 D_fake: 0.137 
(epoch: 172, iters: 640, time: 0.094) G_GAN: 1.659 G_GAN_Feat: 2.275 G_VGG: 0.835 D_real: 0.069 D_fake: 0.180 
(epoch: 172, iters: 1040, time: 0.098) G_GAN: 1.616 G_GAN_Feat: 2.107 G_VGG: 0.783 D_real: 0.377 D_fake: 0.111 
(epoch: 172, iters: 1440, time: 0.091) G_GAN: 1.531 G_GAN_Feat: 2.145 G_VGG: 0.796 D_real: 0.039 D_fake: 0.164 
(epoch: 172, iters: 1840, time: 0.087) G_GAN: 1.326 G_GAN_Feat: 1.785 G_VGG: 0.753 D_real: 0.854 D_fake: 0.222 
(epoch: 172, iters: 2240, time: 0.087) G_GAN: 2.130 G_GAN_Feat: 2.275 G_VGG: 0.843 D_real: 0.170 D_fake: 0.030 
(epoch: 172, iters: 2640, time: 0.092) G_GAN: 1.544 G_GAN_Feat: 2.044 G_VGG: 0.834 D_real: 0.188 D_fake: 0.150 
(epoch: 173, iters: 80, time: 0.092) G_GAN: 1.347 G_GAN_Feat: 1.910 G_VGG: 0.807 D_real: 0.156 D_fake: 0.211 
(epoch: 173, iters: 480, time: 0.089) G_GAN: 1.437 G_GAN_Feat: 1.955 G_VGG: 0.791 D_real: 0.239 D_fake: 0.153 
(epoch: 173, iters: 880, time: 0.094) G_GAN: 1.527 G_GAN_Feat: 2.243 G_VGG: 0.808 D_real: 0.052 D_fake: 0.092 
(epoch: 173, iters: 1280, time: 0.090) G_GAN: 1.341 G_GAN_Feat: 1.816 G_VGG: 0.717 D_real: 0.230 D_fake: 0.213 
(epoch: 173, iters: 1680, time: 0.096) G_GAN: 1.292 G_GAN_Feat: 1.917 G_VGG: 0.749 D_real: 0.037 D_fake: 0.357 
(epoch: 173, iters: 2080, time: 0.091) G_GAN: 1.172 G_GAN_Feat: 2.018 G_VGG: 0.796 D_real: 0.026 D_fake: 0.310 
(epoch: 173, iters: 2480, time: 0.094) G_GAN: 1.309 G_GAN_Feat: 2.167 G_VGG: 0.857 D_real: 0.152 D_fake: 0.206 
(epoch: 173, iters: 2880, time: 0.090) G_GAN: 1.247 G_GAN_Feat: 2.094 G_VGG: 0.823 D_real: 0.077 D_fake: 0.321 
(epoch: 174, iters: 320, time: 0.084) G_GAN: 1.898 G_GAN_Feat: 2.311 G_VGG: 0.823 D_real: 0.166 D_fake: 0.033 
(epoch: 174, iters: 720, time: 0.094) G_GAN: 1.174 G_GAN_Feat: 1.773 G_VGG: 0.856 D_real: 0.784 D_fake: 0.305 
(epoch: 174, iters: 1120, time: 0.091) G_GAN: 0.670 G_GAN_Feat: 1.669 G_VGG: 0.736 D_real: 0.317 D_fake: 0.484 
(epoch: 174, iters: 1520, time: 0.088) G_GAN: 1.372 G_GAN_Feat: 2.228 G_VGG: 0.860 D_real: 0.076 D_fake: 0.221 
(epoch: 174, iters: 1920, time: 0.098) G_GAN: 1.350 G_GAN_Feat: 1.967 G_VGG: 0.812 D_real: 0.141 D_fake: 0.203 
(epoch: 174, iters: 2320, time: 0.088) G_GAN: 1.460 G_GAN_Feat: 2.118 G_VGG: 0.830 D_real: 0.058 D_fake: 0.215 
(epoch: 174, iters: 2720, time: 0.088) G_GAN: 0.839 G_GAN_Feat: 2.020 G_VGG: 0.784 D_real: 0.071 D_fake: 0.435 
(epoch: 175, iters: 160, time: 0.091) G_GAN: 1.605 G_GAN_Feat: 2.269 G_VGG: 0.888 D_real: 0.083 D_fake: 0.107 
(epoch: 175, iters: 560, time: 0.084) G_GAN: 1.489 G_GAN_Feat: 1.987 G_VGG: 0.753 D_real: 0.146 D_fake: 0.145 
(epoch: 175, iters: 960, time: 0.089) G_GAN: 1.583 G_GAN_Feat: 2.084 G_VGG: 0.809 D_real: 0.212 D_fake: 0.104 
(epoch: 175, iters: 1360, time: 0.085) G_GAN: 1.196 G_GAN_Feat: 1.671 G_VGG: 0.694 D_real: 0.056 D_fake: 0.312 
(epoch: 175, iters: 1760, time: 0.090) G_GAN: 1.250 G_GAN_Feat: 2.101 G_VGG: 0.887 D_real: 0.255 D_fake: 0.247 
(epoch: 175, iters: 2160, time: 0.091) G_GAN: 1.596 G_GAN_Feat: 1.922 G_VGG: 0.737 D_real: 0.333 D_fake: 0.138 
(epoch: 175, iters: 2560, time: 0.090) G_GAN: 1.356 G_GAN_Feat: 1.928 G_VGG: 0.773 D_real: 0.210 D_fake: 0.168 
(epoch: 175, iters: 2960, time: 0.092) G_GAN: 1.119 G_GAN_Feat: 1.686 G_VGG: 0.788 D_real: 0.331 D_fake: 0.271 
(epoch: 176, iters: 400, time: 0.092) G_GAN: 1.543 G_GAN_Feat: 2.264 G_VGG: 0.754 D_real: 0.055 D_fake: 0.096 
(epoch: 176, iters: 800, time: 0.096) G_GAN: 1.186 G_GAN_Feat: 1.850 G_VGG: 0.806 D_real: 0.063 D_fake: 0.424 
(epoch: 176, iters: 1200, time: 0.103) G_GAN: 1.513 G_GAN_Feat: 2.416 G_VGG: 0.815 D_real: 0.029 D_fake: 0.094 
(epoch: 176, iters: 1600, time: 0.085) G_GAN: 1.129 G_GAN_Feat: 1.939 G_VGG: 0.857 D_real: 0.095 D_fake: 0.487 
(epoch: 176, iters: 2000, time: 0.091) G_GAN: 1.417 G_GAN_Feat: 2.143 G_VGG: 0.784 D_real: 0.169 D_fake: 0.257 
(epoch: 176, iters: 2400, time: 0.094) G_GAN: 1.249 G_GAN_Feat: 1.621 G_VGG: 0.755 D_real: 0.246 D_fake: 0.231 
(epoch: 176, iters: 2800, time: 0.091) G_GAN: 1.371 G_GAN_Feat: 1.801 G_VGG: 0.804 D_real: 0.159 D_fake: 0.200 
(epoch: 177, iters: 240, time: 0.087) G_GAN: 1.240 G_GAN_Feat: 1.815 G_VGG: 0.824 D_real: 0.252 D_fake: 0.223 
(epoch: 177, iters: 640, time: 0.095) G_GAN: 1.788 G_GAN_Feat: 2.137 G_VGG: 0.775 D_real: 0.130 D_fake: 0.061 
(epoch: 177, iters: 1040, time: 0.098) G_GAN: 1.255 G_GAN_Feat: 1.997 G_VGG: 0.750 D_real: 0.124 D_fake: 0.246 
(epoch: 177, iters: 1440, time: 0.093) G_GAN: 1.827 G_GAN_Feat: 2.013 G_VGG: 0.727 D_real: 0.146 D_fake: 0.043 
(epoch: 177, iters: 1840, time: 0.090) G_GAN: 1.458 G_GAN_Feat: 2.052 G_VGG: 0.851 D_real: 0.206 D_fake: 0.155 
(epoch: 177, iters: 2240, time: 0.090) G_GAN: 1.444 G_GAN_Feat: 2.117 G_VGG: 0.767 D_real: 0.310 D_fake: 0.177 
(epoch: 177, iters: 2640, time: 0.092) G_GAN: 1.208 G_GAN_Feat: 2.076 G_VGG: 0.812 D_real: 0.165 D_fake: 0.356 
(epoch: 178, iters: 80, time: 0.094) G_GAN: 1.255 G_GAN_Feat: 2.066 G_VGG: 0.816 D_real: 0.064 D_fake: 0.329 
(epoch: 178, iters: 480, time: 0.095) G_GAN: 1.510 G_GAN_Feat: 2.323 G_VGG: 0.834 D_real: 0.192 D_fake: 0.200 
(epoch: 178, iters: 880, time: 0.096) G_GAN: 1.316 G_GAN_Feat: 1.916 G_VGG: 0.787 D_real: 0.157 D_fake: 0.296 
(epoch: 178, iters: 1280, time: 0.099) G_GAN: 1.045 G_GAN_Feat: 1.907 G_VGG: 0.832 D_real: 0.263 D_fake: 0.229 
(epoch: 178, iters: 1680, time: 0.091) G_GAN: 1.502 G_GAN_Feat: 1.915 G_VGG: 0.799 D_real: 0.211 D_fake: 0.126 
(epoch: 178, iters: 2080, time: 0.100) G_GAN: 1.754 G_GAN_Feat: 2.220 G_VGG: 0.768 D_real: 0.101 D_fake: 0.057 
(epoch: 178, iters: 2480, time: 0.095) G_GAN: 2.354 G_GAN_Feat: 2.455 G_VGG: 0.850 D_real: 0.307 D_fake: 0.040 
(epoch: 178, iters: 2880, time: 0.091) G_GAN: 2.034 G_GAN_Feat: 2.087 G_VGG: 0.771 D_real: 0.405 D_fake: 0.027 
(epoch: 179, iters: 320, time: 0.099) G_GAN: 1.676 G_GAN_Feat: 1.985 G_VGG: 0.707 D_real: 0.384 D_fake: 0.077 
(epoch: 179, iters: 720, time: 0.090) G_GAN: 1.271 G_GAN_Feat: 1.960 G_VGG: 0.806 D_real: 0.057 D_fake: 0.247 
(epoch: 179, iters: 1120, time: 0.086) G_GAN: 1.564 G_GAN_Feat: 1.943 G_VGG: 0.782 D_real: 0.175 D_fake: 0.098 
(epoch: 179, iters: 1520, time: 0.086) G_GAN: 1.701 G_GAN_Feat: 2.278 G_VGG: 0.826 D_real: 0.215 D_fake: 0.066 
(epoch: 179, iters: 1920, time: 0.095) G_GAN: 1.332 G_GAN_Feat: 1.903 G_VGG: 0.795 D_real: 0.056 D_fake: 0.237 
(epoch: 179, iters: 2320, time: 0.077) G_GAN: 1.296 G_GAN_Feat: 2.019 G_VGG: 0.868 D_real: 0.043 D_fake: 0.247 
(epoch: 179, iters: 2720, time: 0.094) G_GAN: 1.496 G_GAN_Feat: 2.174 G_VGG: 0.839 D_real: 0.130 D_fake: 0.201 
(epoch: 180, iters: 160, time: 0.094) G_GAN: 0.852 G_GAN_Feat: 1.726 G_VGG: 0.767 D_real: 0.469 D_fake: 0.362 
(epoch: 180, iters: 560, time: 0.094) G_GAN: 1.772 G_GAN_Feat: 2.227 G_VGG: 0.841 D_real: 0.486 D_fake: 0.057 
(epoch: 180, iters: 960, time: 0.089) G_GAN: 1.322 G_GAN_Feat: 1.918 G_VGG: 0.782 D_real: 0.136 D_fake: 0.200 
(epoch: 180, iters: 1360, time: 0.099) G_GAN: 1.422 G_GAN_Feat: 2.099 G_VGG: 0.858 D_real: 0.059 D_fake: 0.162 
(epoch: 180, iters: 1760, time: 0.085) G_GAN: 1.514 G_GAN_Feat: 2.057 G_VGG: 0.746 D_real: 0.204 D_fake: 0.126 
(epoch: 180, iters: 2160, time: 0.094) G_GAN: 1.317 G_GAN_Feat: 2.128 G_VGG: 0.875 D_real: 0.140 D_fake: 0.219 
(epoch: 180, iters: 2560, time: 0.089) G_GAN: 1.171 G_GAN_Feat: 2.068 G_VGG: 0.824 D_real: 0.044 D_fake: 0.494 
(epoch: 180, iters: 2960, time: 0.087) G_GAN: 1.171 G_GAN_Feat: 1.686 G_VGG: 0.755 D_real: 0.051 D_fake: 0.451 
(epoch: 181, iters: 400, time: 0.085) G_GAN: 1.586 G_GAN_Feat: 2.297 G_VGG: 0.905 D_real: 0.143 D_fake: 0.121 
(epoch: 181, iters: 800, time: 0.088) G_GAN: 1.892 G_GAN_Feat: 2.411 G_VGG: 0.859 D_real: 0.085 D_fake: 0.044 
(epoch: 181, iters: 1200, time: 0.091) G_GAN: 1.440 G_GAN_Feat: 2.077 G_VGG: 0.759 D_real: 0.093 D_fake: 0.150 
(epoch: 181, iters: 1600, time: 0.091) G_GAN: 1.495 G_GAN_Feat: 2.029 G_VGG: 0.737 D_real: 0.089 D_fake: 0.165 
(epoch: 181, iters: 2000, time: 0.092) G_GAN: 1.236 G_GAN_Feat: 1.962 G_VGG: 0.831 D_real: 0.077 D_fake: 0.352 
(epoch: 181, iters: 2400, time: 0.088) G_GAN: 1.345 G_GAN_Feat: 1.831 G_VGG: 0.810 D_real: 0.247 D_fake: 0.145 
(epoch: 181, iters: 2800, time: 0.086) G_GAN: 1.306 G_GAN_Feat: 1.903 G_VGG: 0.790 D_real: 0.169 D_fake: 0.228 
(epoch: 182, iters: 240, time: 0.089) G_GAN: 2.214 G_GAN_Feat: 2.230 G_VGG: 0.721 D_real: 0.170 D_fake: 0.054 
(epoch: 182, iters: 640, time: 0.097) G_GAN: 1.166 G_GAN_Feat: 1.809 G_VGG: 0.844 D_real: 0.131 D_fake: 0.284 
(epoch: 182, iters: 1040, time: 0.090) G_GAN: 0.844 G_GAN_Feat: 1.673 G_VGG: 0.762 D_real: 0.183 D_fake: 0.356 
(epoch: 182, iters: 1440, time: 0.091) G_GAN: 1.234 G_GAN_Feat: 1.908 G_VGG: 0.879 D_real: 0.198 D_fake: 0.301 
(epoch: 182, iters: 1840, time: 0.098) G_GAN: 1.421 G_GAN_Feat: 2.118 G_VGG: 0.773 D_real: 0.125 D_fake: 0.241 
(epoch: 182, iters: 2240, time: 0.092) G_GAN: 1.511 G_GAN_Feat: 2.189 G_VGG: 0.827 D_real: 0.075 D_fake: 0.156 
(epoch: 182, iters: 2640, time: 0.076) G_GAN: 1.302 G_GAN_Feat: 1.779 G_VGG: 0.711 D_real: 0.235 D_fake: 0.262 
(epoch: 183, iters: 80, time: 0.098) G_GAN: 1.320 G_GAN_Feat: 1.735 G_VGG: 0.790 D_real: 0.101 D_fake: 0.214 
(epoch: 183, iters: 480, time: 0.086) G_GAN: 1.683 G_GAN_Feat: 2.178 G_VGG: 0.794 D_real: 0.061 D_fake: 0.103 
(epoch: 183, iters: 880, time: 0.088) G_GAN: 1.286 G_GAN_Feat: 1.982 G_VGG: 0.862 D_real: 0.193 D_fake: 0.282 
(epoch: 183, iters: 1280, time: 0.094) G_GAN: 1.100 G_GAN_Feat: 1.899 G_VGG: 0.862 D_real: 0.104 D_fake: 0.366 
(epoch: 183, iters: 1680, time: 0.092) G_GAN: 1.303 G_GAN_Feat: 1.983 G_VGG: 0.844 D_real: 0.266 D_fake: 0.232 
(epoch: 183, iters: 2080, time: 0.097) G_GAN: 1.281 G_GAN_Feat: 1.858 G_VGG: 0.783 D_real: 0.144 D_fake: 0.253 
(epoch: 183, iters: 2480, time: 0.087) G_GAN: 1.567 G_GAN_Feat: 2.020 G_VGG: 0.841 D_real: 0.274 D_fake: 0.148 
(epoch: 183, iters: 2880, time: 0.092) G_GAN: 1.850 G_GAN_Feat: 2.314 G_VGG: 0.751 D_real: 0.066 D_fake: 0.053 
(epoch: 184, iters: 320, time: 0.084) G_GAN: 1.591 G_GAN_Feat: 2.017 G_VGG: 0.817 D_real: 0.095 D_fake: 0.189 
(epoch: 184, iters: 720, time: 0.092) G_GAN: 1.241 G_GAN_Feat: 1.887 G_VGG: 0.749 D_real: 0.109 D_fake: 0.262 
(epoch: 184, iters: 1120, time: 0.099) G_GAN: 1.724 G_GAN_Feat: 1.993 G_VGG: 0.796 D_real: 0.045 D_fake: 0.088 
(epoch: 184, iters: 1520, time: 0.088) G_GAN: 1.355 G_GAN_Feat: 1.857 G_VGG: 0.801 D_real: 0.107 D_fake: 0.185 
(epoch: 184, iters: 1920, time: 0.090) G_GAN: 1.279 G_GAN_Feat: 1.753 G_VGG: 0.717 D_real: 0.081 D_fake: 0.284 
(epoch: 184, iters: 2320, time: 0.093) G_GAN: 1.594 G_GAN_Feat: 2.085 G_VGG: 0.850 D_real: 0.187 D_fake: 0.131 
(epoch: 184, iters: 2720, time: 0.083) G_GAN: 1.297 G_GAN_Feat: 1.757 G_VGG: 0.719 D_real: 0.139 D_fake: 0.337 
(epoch: 185, iters: 160, time: 0.084) G_GAN: 1.911 G_GAN_Feat: 2.216 G_VGG: 0.726 D_real: 0.412 D_fake: 0.078 
(epoch: 185, iters: 560, time: 0.091) G_GAN: 1.307 G_GAN_Feat: 1.917 G_VGG: 0.801 D_real: 0.080 D_fake: 0.243 
(epoch: 185, iters: 960, time: 0.094) G_GAN: 1.583 G_GAN_Feat: 1.815 G_VGG: 0.770 D_real: 0.212 D_fake: 0.096 
(epoch: 185, iters: 1360, time: 0.086) G_GAN: 1.360 G_GAN_Feat: 1.837 G_VGG: 0.802 D_real: 0.133 D_fake: 0.206 
(epoch: 185, iters: 1760, time: 0.092) G_GAN: 1.480 G_GAN_Feat: 1.820 G_VGG: 0.738 D_real: 0.106 D_fake: 0.135 
(epoch: 185, iters: 2160, time: 0.092) G_GAN: 1.038 G_GAN_Feat: 1.657 G_VGG: 0.820 D_real: 0.211 D_fake: 0.329 
(epoch: 185, iters: 2560, time: 0.087) G_GAN: 1.176 G_GAN_Feat: 1.761 G_VGG: 0.793 D_real: 0.206 D_fake: 0.302 
(epoch: 185, iters: 2960, time: 0.097) G_GAN: 1.324 G_GAN_Feat: 1.830 G_VGG: 0.812 D_real: 0.092 D_fake: 0.224 
(epoch: 186, iters: 400, time: 0.085) G_GAN: 1.300 G_GAN_Feat: 1.697 G_VGG: 0.796 D_real: 0.188 D_fake: 0.220 
(epoch: 186, iters: 800, time: 0.086) G_GAN: 1.610 G_GAN_Feat: 2.153 G_VGG: 0.919 D_real: 0.211 D_fake: 0.095 
(epoch: 186, iters: 1200, time: 0.088) G_GAN: 1.139 G_GAN_Feat: 1.806 G_VGG: 0.773 D_real: 0.072 D_fake: 0.390 
(epoch: 186, iters: 1600, time: 0.091) G_GAN: 1.598 G_GAN_Feat: 1.854 G_VGG: 0.798 D_real: 0.083 D_fake: 0.119 
(epoch: 186, iters: 2000, time: 0.089) G_GAN: 1.435 G_GAN_Feat: 1.915 G_VGG: 0.768 D_real: 0.091 D_fake: 0.185 
(epoch: 186, iters: 2400, time: 0.091) G_GAN: 1.815 G_GAN_Feat: 2.250 G_VGG: 0.784 D_real: 0.042 D_fake: 0.068 
(epoch: 186, iters: 2800, time: 0.097) G_GAN: 1.411 G_GAN_Feat: 1.985 G_VGG: 0.893 D_real: 0.098 D_fake: 0.153 
(epoch: 187, iters: 240, time: 0.096) G_GAN: 1.637 G_GAN_Feat: 2.036 G_VGG: 0.758 D_real: 0.137 D_fake: 0.129 
(epoch: 187, iters: 640, time: 0.093) G_GAN: 1.267 G_GAN_Feat: 1.900 G_VGG: 0.818 D_real: 0.162 D_fake: 0.242 
(epoch: 187, iters: 1040, time: 0.098) G_GAN: 2.013 G_GAN_Feat: 2.227 G_VGG: 0.859 D_real: 0.134 D_fake: 0.036 
(epoch: 187, iters: 1440, time: 0.091) G_GAN: 0.976 G_GAN_Feat: 1.503 G_VGG: 0.702 D_real: 0.172 D_fake: 0.421 
(epoch: 187, iters: 1840, time: 0.098) G_GAN: 1.336 G_GAN_Feat: 1.730 G_VGG: 0.729 D_real: 0.113 D_fake: 0.244 
(epoch: 187, iters: 2240, time: 0.092) G_GAN: 1.555 G_GAN_Feat: 2.119 G_VGG: 0.823 D_real: 0.120 D_fake: 0.158 
(epoch: 187, iters: 2640, time: 0.092) G_GAN: 1.878 G_GAN_Feat: 2.244 G_VGG: 0.803 D_real: 0.114 D_fake: 0.061 
(epoch: 188, iters: 80, time: 0.093) G_GAN: 1.410 G_GAN_Feat: 2.138 G_VGG: 0.894 D_real: 0.171 D_fake: 0.214 
(epoch: 188, iters: 480, time: 0.096) G_GAN: 1.914 G_GAN_Feat: 1.984 G_VGG: 0.803 D_real: 0.157 D_fake: 0.046 
(epoch: 188, iters: 880, time: 0.086) G_GAN: 1.275 G_GAN_Feat: 1.811 G_VGG: 0.854 D_real: 0.148 D_fake: 0.269 
(epoch: 188, iters: 1280, time: 0.085) G_GAN: 1.345 G_GAN_Feat: 1.650 G_VGG: 0.716 D_real: 0.186 D_fake: 0.216 
(epoch: 188, iters: 1680, time: 0.099) G_GAN: 1.388 G_GAN_Feat: 2.024 G_VGG: 0.914 D_real: 0.220 D_fake: 0.198 
(epoch: 188, iters: 2080, time: 0.103) G_GAN: 1.371 G_GAN_Feat: 1.626 G_VGG: 0.730 D_real: 0.249 D_fake: 0.225 
(epoch: 188, iters: 2480, time: 0.090) G_GAN: 1.551 G_GAN_Feat: 1.981 G_VGG: 0.780 D_real: 0.128 D_fake: 0.097 
(epoch: 188, iters: 2880, time: 0.101) G_GAN: 1.512 G_GAN_Feat: 2.066 G_VGG: 0.765 D_real: 0.512 D_fake: 0.191 
(epoch: 189, iters: 320, time: 0.084) G_GAN: 1.360 G_GAN_Feat: 1.934 G_VGG: 0.866 D_real: 0.110 D_fake: 0.230 
(epoch: 189, iters: 720, time: 0.092) G_GAN: 1.454 G_GAN_Feat: 1.695 G_VGG: 0.745 D_real: 0.101 D_fake: 0.195 
(epoch: 189, iters: 1120, time: 0.094) G_GAN: 1.345 G_GAN_Feat: 1.913 G_VGG: 0.840 D_real: 0.134 D_fake: 0.199 
(epoch: 189, iters: 1520, time: 0.094) G_GAN: 1.170 G_GAN_Feat: 1.654 G_VGG: 0.780 D_real: 0.232 D_fake: 0.302 
(epoch: 189, iters: 1920, time: 0.092) G_GAN: 1.456 G_GAN_Feat: 1.892 G_VGG: 0.777 D_real: 0.092 D_fake: 0.172 
(epoch: 189, iters: 2320, time: 0.095) G_GAN: 1.305 G_GAN_Feat: 1.652 G_VGG: 0.741 D_real: 0.187 D_fake: 0.218 
(epoch: 189, iters: 2720, time: 0.086) G_GAN: 1.238 G_GAN_Feat: 1.527 G_VGG: 0.703 D_real: 0.238 D_fake: 0.241 
(epoch: 190, iters: 160, time: 0.093) G_GAN: 1.917 G_GAN_Feat: 1.910 G_VGG: 0.744 D_real: 0.114 D_fake: 0.039 
(epoch: 190, iters: 560, time: 0.095) G_GAN: 1.296 G_GAN_Feat: 1.757 G_VGG: 0.779 D_real: 0.071 D_fake: 0.291 
(epoch: 190, iters: 960, time: 0.095) G_GAN: 1.393 G_GAN_Feat: 1.785 G_VGG: 0.786 D_real: 0.254 D_fake: 0.188 
(epoch: 190, iters: 1360, time: 0.092) G_GAN: 1.721 G_GAN_Feat: 1.980 G_VGG: 0.810 D_real: 0.080 D_fake: 0.108 
(epoch: 190, iters: 1760, time: 0.089) G_GAN: 1.208 G_GAN_Feat: 1.765 G_VGG: 0.758 D_real: 0.108 D_fake: 0.293 
(epoch: 190, iters: 2160, time: 0.097) G_GAN: 1.926 G_GAN_Feat: 1.900 G_VGG: 0.760 D_real: 0.068 D_fake: 0.057 
(epoch: 190, iters: 2560, time: 0.083) G_GAN: 1.311 G_GAN_Feat: 1.774 G_VGG: 0.776 D_real: 0.047 D_fake: 0.257 
(epoch: 190, iters: 2960, time: 0.096) G_GAN: 1.351 G_GAN_Feat: 2.178 G_VGG: 0.853 D_real: 0.485 D_fake: 0.226 
(epoch: 191, iters: 400, time: 0.084) G_GAN: 1.308 G_GAN_Feat: 1.953 G_VGG: 0.874 D_real: 0.164 D_fake: 0.311 
(epoch: 191, iters: 800, time: 0.078) G_GAN: 1.417 G_GAN_Feat: 1.890 G_VGG: 0.831 D_real: 0.125 D_fake: 0.241 
(epoch: 191, iters: 1200, time: 0.102) G_GAN: 1.652 G_GAN_Feat: 1.939 G_VGG: 0.864 D_real: 0.186 D_fake: 0.111 
(epoch: 191, iters: 1600, time: 0.092) G_GAN: 1.651 G_GAN_Feat: 1.810 G_VGG: 0.819 D_real: 0.279 D_fake: 0.070 
(epoch: 191, iters: 2000, time: 0.084) G_GAN: 1.458 G_GAN_Feat: 1.911 G_VGG: 0.875 D_real: 0.134 D_fake: 0.213 
(epoch: 191, iters: 2400, time: 0.096) G_GAN: 1.195 G_GAN_Feat: 1.748 G_VGG: 0.803 D_real: 0.128 D_fake: 0.394 
(epoch: 191, iters: 2800, time: 0.098) G_GAN: 1.310 G_GAN_Feat: 1.688 G_VGG: 0.791 D_real: 0.128 D_fake: 0.289 
(epoch: 192, iters: 240, time: 0.087) G_GAN: 1.518 G_GAN_Feat: 1.843 G_VGG: 0.763 D_real: 0.164 D_fake: 0.141 
(epoch: 192, iters: 640, time: 0.091) G_GAN: 1.313 G_GAN_Feat: 1.733 G_VGG: 0.753 D_real: 0.084 D_fake: 0.207 
(epoch: 192, iters: 1040, time: 0.091) G_GAN: 1.607 G_GAN_Feat: 1.838 G_VGG: 0.825 D_real: 0.154 D_fake: 0.079 
(epoch: 192, iters: 1440, time: 0.085) G_GAN: 1.481 G_GAN_Feat: 1.728 G_VGG: 0.753 D_real: 0.125 D_fake: 0.186 
(epoch: 192, iters: 1840, time: 0.088) G_GAN: 1.503 G_GAN_Feat: 1.866 G_VGG: 0.750 D_real: 0.215 D_fake: 0.169 
(epoch: 192, iters: 2240, time: 0.092) G_GAN: 1.215 G_GAN_Feat: 1.991 G_VGG: 0.805 D_real: 0.093 D_fake: 0.272 
(epoch: 192, iters: 2640, time: 0.088) G_GAN: 1.663 G_GAN_Feat: 1.929 G_VGG: 0.790 D_real: 0.299 D_fake: 0.067 
(epoch: 193, iters: 80, time: 0.085) G_GAN: 1.296 G_GAN_Feat: 1.611 G_VGG: 0.714 D_real: 0.134 D_fake: 0.205 
(epoch: 193, iters: 480, time: 0.095) G_GAN: 2.011 G_GAN_Feat: 1.953 G_VGG: 0.786 D_real: 0.059 D_fake: 0.070 
(epoch: 193, iters: 880, time: 0.092) G_GAN: 1.365 G_GAN_Feat: 1.706 G_VGG: 0.759 D_real: 0.136 D_fake: 0.171 
(epoch: 193, iters: 1280, time: 0.087) G_GAN: 1.315 G_GAN_Feat: 1.816 G_VGG: 0.893 D_real: 0.196 D_fake: 0.233 
(epoch: 193, iters: 1680, time: 0.093) G_GAN: 1.421 G_GAN_Feat: 1.777 G_VGG: 0.746 D_real: 0.220 D_fake: 0.207 
(epoch: 193, iters: 2080, time: 0.084) G_GAN: 1.234 G_GAN_Feat: 1.662 G_VGG: 0.785 D_real: 0.098 D_fake: 0.325 
(epoch: 193, iters: 2480, time: 0.095) G_GAN: 1.418 G_GAN_Feat: 1.780 G_VGG: 0.828 D_real: 0.109 D_fake: 0.183 
(epoch: 193, iters: 2880, time: 0.098) G_GAN: 1.455 G_GAN_Feat: 2.081 G_VGG: 0.805 D_real: 0.142 D_fake: 0.180 
(epoch: 194, iters: 320, time: 0.086) G_GAN: 1.533 G_GAN_Feat: 1.848 G_VGG: 0.843 D_real: 0.145 D_fake: 0.170 
(epoch: 194, iters: 720, time: 0.101) G_GAN: 1.446 G_GAN_Feat: 1.863 G_VGG: 0.805 D_real: 0.182 D_fake: 0.150 
(epoch: 194, iters: 1120, time: 0.093) G_GAN: 1.633 G_GAN_Feat: 1.778 G_VGG: 0.787 D_real: 0.188 D_fake: 0.088 
(epoch: 194, iters: 1520, time: 0.092) G_GAN: 1.331 G_GAN_Feat: 1.922 G_VGG: 0.879 D_real: 0.172 D_fake: 0.221 
(epoch: 194, iters: 1920, time: 0.101) G_GAN: 1.197 G_GAN_Feat: 1.845 G_VGG: 0.841 D_real: 0.120 D_fake: 0.317 
(epoch: 194, iters: 2320, time: 0.092) G_GAN: 1.350 G_GAN_Feat: 1.720 G_VGG: 0.809 D_real: 0.152 D_fake: 0.223 
(epoch: 194, iters: 2720, time: 0.091) G_GAN: 1.250 G_GAN_Feat: 1.722 G_VGG: 0.764 D_real: 0.260 D_fake: 0.205 
(epoch: 195, iters: 160, time: 0.096) G_GAN: 1.448 G_GAN_Feat: 1.771 G_VGG: 0.799 D_real: 0.154 D_fake: 0.183 
(epoch: 195, iters: 560, time: 0.090) G_GAN: 1.505 G_GAN_Feat: 1.756 G_VGG: 0.766 D_real: 0.210 D_fake: 0.121 
(epoch: 195, iters: 960, time: 0.093) G_GAN: 1.340 G_GAN_Feat: 1.558 G_VGG: 0.740 D_real: 0.201 D_fake: 0.216 
(epoch: 195, iters: 1360, time: 0.089) G_GAN: 1.280 G_GAN_Feat: 1.732 G_VGG: 0.779 D_real: 0.118 D_fake: 0.250 
(epoch: 195, iters: 1760, time: 0.084) G_GAN: 1.492 G_GAN_Feat: 1.847 G_VGG: 0.778 D_real: 0.119 D_fake: 0.177 
(epoch: 195, iters: 2160, time: 0.085) G_GAN: 1.348 G_GAN_Feat: 1.769 G_VGG: 0.739 D_real: 0.222 D_fake: 0.195 
(epoch: 195, iters: 2560, time: 0.089) G_GAN: 1.421 G_GAN_Feat: 1.960 G_VGG: 0.798 D_real: 0.195 D_fake: 0.220 
(epoch: 195, iters: 2960, time: 0.084) G_GAN: 1.311 G_GAN_Feat: 1.690 G_VGG: 0.802 D_real: 0.198 D_fake: 0.249 
(epoch: 196, iters: 400, time: 0.088) G_GAN: 1.322 G_GAN_Feat: 1.690 G_VGG: 0.828 D_real: 0.192 D_fake: 0.233 
(epoch: 196, iters: 800, time: 0.085) G_GAN: 1.471 G_GAN_Feat: 1.779 G_VGG: 0.817 D_real: 0.210 D_fake: 0.188 
(epoch: 196, iters: 1200, time: 0.101) G_GAN: 1.460 G_GAN_Feat: 1.835 G_VGG: 0.871 D_real: 0.193 D_fake: 0.178 
(epoch: 196, iters: 1600, time: 0.088) G_GAN: 1.373 G_GAN_Feat: 1.698 G_VGG: 0.768 D_real: 0.130 D_fake: 0.228 
(epoch: 196, iters: 2000, time: 0.092) G_GAN: 1.333 G_GAN_Feat: 1.759 G_VGG: 0.817 D_real: 0.140 D_fake: 0.224 
(epoch: 196, iters: 2400, time: 0.101) G_GAN: 1.457 G_GAN_Feat: 1.861 G_VGG: 0.838 D_real: 0.194 D_fake: 0.187 
(epoch: 196, iters: 2800, time: 0.089) G_GAN: 1.475 G_GAN_Feat: 1.686 G_VGG: 0.788 D_real: 0.175 D_fake: 0.166 
(epoch: 197, iters: 240, time: 0.084) G_GAN: 1.393 G_GAN_Feat: 1.720 G_VGG: 0.776 D_real: 0.176 D_fake: 0.192 
(epoch: 197, iters: 640, time: 0.092) G_GAN: 1.389 G_GAN_Feat: 1.726 G_VGG: 0.807 D_real: 0.198 D_fake: 0.178 
(epoch: 197, iters: 1040, time: 0.085) G_GAN: 1.263 G_GAN_Feat: 1.758 G_VGG: 0.860 D_real: 0.163 D_fake: 0.274 
(epoch: 197, iters: 1440, time: 0.095) G_GAN: 1.320 G_GAN_Feat: 1.713 G_VGG: 0.831 D_real: 0.175 D_fake: 0.221 
(epoch: 197, iters: 1840, time: 0.096) G_GAN: 1.310 G_GAN_Feat: 1.929 G_VGG: 0.830 D_real: 0.180 D_fake: 0.207 
(epoch: 197, iters: 2240, time: 0.084) G_GAN: 1.357 G_GAN_Feat: 1.796 G_VGG: 0.800 D_real: 0.198 D_fake: 0.169 
(epoch: 197, iters: 2640, time: 0.095) G_GAN: 1.274 G_GAN_Feat: 1.638 G_VGG: 0.734 D_real: 0.220 D_fake: 0.207 
(epoch: 198, iters: 80, time: 0.086) G_GAN: 1.282 G_GAN_Feat: 1.650 G_VGG: 0.789 D_real: 0.179 D_fake: 0.257 
(epoch: 198, iters: 480, time: 0.095) G_GAN: 1.345 G_GAN_Feat: 1.734 G_VGG: 0.824 D_real: 0.195 D_fake: 0.213 
(epoch: 198, iters: 880, time: 0.097) G_GAN: 1.256 G_GAN_Feat: 1.565 G_VGG: 0.744 D_real: 0.220 D_fake: 0.253 
(epoch: 198, iters: 1280, time: 0.082) G_GAN: 1.369 G_GAN_Feat: 1.671 G_VGG: 0.795 D_real: 0.203 D_fake: 0.206 
(epoch: 198, iters: 1680, time: 0.088) G_GAN: 1.341 G_GAN_Feat: 1.582 G_VGG: 0.729 D_real: 0.190 D_fake: 0.205 
(epoch: 198, iters: 2080, time: 0.094) G_GAN: 1.370 G_GAN_Feat: 1.711 G_VGG: 0.794 D_real: 0.208 D_fake: 0.184 
(epoch: 198, iters: 2480, time: 0.093) G_GAN: 1.388 G_GAN_Feat: 1.711 G_VGG: 0.789 D_real: 0.199 D_fake: 0.178 
(epoch: 198, iters: 2880, time: 0.084) G_GAN: 1.492 G_GAN_Feat: 1.844 G_VGG: 0.812 D_real: 0.178 D_fake: 0.136 
(epoch: 199, iters: 320, time: 0.089) G_GAN: 1.315 G_GAN_Feat: 1.655 G_VGG: 0.816 D_real: 0.209 D_fake: 0.216 
(epoch: 199, iters: 720, time: 0.093) G_GAN: 1.308 G_GAN_Feat: 1.688 G_VGG: 0.845 D_real: 0.236 D_fake: 0.213 
(epoch: 199, iters: 1120, time: 0.088) G_GAN: 1.416 G_GAN_Feat: 1.781 G_VGG: 0.848 D_real: 0.171 D_fake: 0.170 
(epoch: 199, iters: 1520, time: 0.094) G_GAN: 1.335 G_GAN_Feat: 1.615 G_VGG: 0.769 D_real: 0.184 D_fake: 0.199 
(epoch: 199, iters: 1920, time: 0.092) G_GAN: 1.291 G_GAN_Feat: 1.735 G_VGG: 0.854 D_real: 0.225 D_fake: 0.253 
(epoch: 199, iters: 2320, time: 0.092) G_GAN: 1.272 G_GAN_Feat: 1.724 G_VGG: 0.808 D_real: 0.190 D_fake: 0.212 
(epoch: 199, iters: 2720, time: 0.088) G_GAN: 1.274 G_GAN_Feat: 1.613 G_VGG: 0.766 D_real: 0.206 D_fake: 0.232 
(epoch: 200, iters: 160, time: 0.087) G_GAN: 1.425 G_GAN_Feat: 1.657 G_VGG: 0.801 D_real: 0.220 D_fake: 0.179 
(epoch: 200, iters: 560, time: 0.093) G_GAN: 1.322 G_GAN_Feat: 1.610 G_VGG: 0.742 D_real: 0.193 D_fake: 0.202 
(epoch: 200, iters: 960, time: 0.094) G_GAN: 1.335 G_GAN_Feat: 1.607 G_VGG: 0.799 D_real: 0.215 D_fake: 0.211 
(epoch: 200, iters: 1360, time: 0.098) G_GAN: 1.325 G_GAN_Feat: 1.638 G_VGG: 0.801 D_real: 0.230 D_fake: 0.224 
(epoch: 200, iters: 1760, time: 0.091) G_GAN: 1.336 G_GAN_Feat: 1.669 G_VGG: 0.728 D_real: 0.175 D_fake: 0.220 
(epoch: 200, iters: 2160, time: 0.089) G_GAN: 1.371 G_GAN_Feat: 1.961 G_VGG: 0.906 D_real: 0.223 D_fake: 0.208 
(epoch: 200, iters: 2560, time: 0.084) G_GAN: 1.298 G_GAN_Feat: 1.458 G_VGG: 0.724 D_real: 0.235 D_fake: 0.211 
(epoch: 200, iters: 2960, time: 0.089) G_GAN: 1.325 G_GAN_Feat: 1.905 G_VGG: 0.867 D_real: 0.189 D_fake: 0.227 
```
