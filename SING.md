
# facebook research SING

## devops 

you could use my same docker image for WaveGlow and tacotron2. 

```
docker pull hryu/pytorch:t2-ngc-18.11
nvidia-docker run -p8888:8888 hryu/pytorch:t2-ngc-18.11
``` 

because SING use same libraries and use `torch.stft` instead of librosa library 
```
numpy
requests
scipy
torch>=0.4.1
tqdm
```

## copy model 
git clone https://github.com/facebookresearch/SING.git
```
git clone https://github.com/facebookresearch/SING.git
Cloning into 'SING'...
remote: Enumerating objects: 19, done.
remote: Counting objects: 100% (19/19), done.
remote: Compressing objects: 100% (19/19), done.
remote: Total 48 (delta 6), reused 1 (delta 0), pack-reused 29
Unpacking objects: 100% (48/48), done.
Checking connectivity... done.
```



reference [Magenta Nsynth](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth) 



## prepare dataset 

Magenta team provide [Nsynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) 

- Train [[tfrecord](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.tfrecord) | [json/wav](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz)]: A training set with 289,205 examples. Instruments do not overlap with valid or test.
- Valid [ [tfrecord](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.tfrecord) | [json/wav](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz)]: A validation set with 12,678 examples. Instruments do not overlap with train.
- Test [ [tfrecord](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.tfrecord) | [json/wav](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz)]: A test set with 4,096 examples. Instruments do not overlap with train.


### download and decompress with progressive bar 
```
echo "start to download nsynth dataset "
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz --show-progress 
echo "download finished.. "
echo "  "
echo "start to decompress "
pv nsynth-train.jsonwav.tar.gz | tar xzf - -C /mnt/old/dataset/nsynth/raw
echo "decompress finished .. "

```


log
```
/mnt/old/dataset/nsynth$ sh a.sh 
start to download nsynth dataset 
--2018-12-05 09:33:02--  http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
Resolving download.magenta.tensorflow.org (download.magenta.tensorflow.org)... 172.217.5.112, 2607:f8b0:4005:808::2010
Connecting to download.magenta.tensorflow.org (download.magenta.tensorflow.org)|172.217.5.112|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 23815298079 (22G) [application/gzip]
Saving to: ‘nsynth-train.jsonwav.tar.gz’

nsynth-train.jsonwav.tar.g 100%[========================================>]  22.18G  87.2MB/s    in 5m 14s  

2018-12-05 09:38:17 (72.3 MB/s) - ‘nsynth-train.jsonwav.tar.gz’ saved [23815298079/23815298079]

download finished.. 
  
start to decompress 
22.2GiB 0:13:11 [28.7MiB/s] [============================================================>] 100%            
decompress finished .. 
/mnt/old/dataset/nsynth$ ls raw
nsynth-train
/mnt/old/dataset/nsynth$ ls raw/nsynth-train/
audio  examples.json

```

## train

```
/mnt/old/git/facebookressearch/SING$ python -m sing.train --cuda --data /mnt/old/dataset/nsynth/raw/nsynth-train --checkpoint ch
ConvolutionalAE(encoder=ConvolutionalEncoder(Sequential(
  (0): WindowedConv1d(window=hann**2,conv=Conv1d(1, 4096, kernel_size=(1024,), stride=(256,)))
  (1): ReLU()
  (2): Conv1d(4096, 4096, kernel_size=(1,), stride=(1,))
  (3): ReLU()
  (4): Conv1d(4096, 4096, kernel_size=(1,), stride=(1,))
  (5): ReLU()
  (6): Conv1d(4096, 128, kernel_size=(1,), stride=(1,))
)),decoder=ConvolutionalDecoder(Sequential(
  (0): Conv1d(128, 4096, kernel_size=(9,), stride=(1,))
  (1): ReLU()
  (2): Conv1d(4096, 4096, kernel_size=(1,), stride=(1,))
  (3): ReLU()
  (4): Conv1d(4096, 4096, kernel_size=(1,), stride=(1,))
  (5): ReLU()
  (6): WindowedConvTranpose1d(window=hann**2,conv_tr=ConvTranspose1d(4096, 1, kernel_size=(1024,), stride=(256,), padding=(768,)))

Training autoencoder
  1%|▌                                                                | 2176/231212 [00:32<51:36, 73.96ex/s
  8%|████▊                                                           | 17344/231212 [04:01<47:19, 75.31ex/s
```

