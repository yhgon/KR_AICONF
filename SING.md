
# facebook research SING

## devops 

SING use `torch.stft` instead of librosa library. 
```
numpy
requests
scipy
torch>=0.4.1
tqdm
```

## copy model 
git clone https://github.com/facebookresearch/SING

reference [Magenta Nsynth](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth) 



## prepare dataset 

Magenta team provide [Nsynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) 

Train [tfrecord | [json/wav](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz)]: A training set with 289,205 examples. Instruments do not overlap with valid or test.
Valid [tfrecord | [json/wav](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz)]: A validation set with 12,678 examples. Instruments do not overlap with train.
Test [tfrecord | [json/wav](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz)]: A test set with 4,096 examples. Instruments do not overlap with train.


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
8.8GiB 0:05:16 [28.4MiB/s] [=======================>                                      ] 39% ETA 0:07:32

```

