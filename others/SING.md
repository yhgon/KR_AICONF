
# facebook research SING

duplicate facebook's SING with DGX-2

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
[train_ae][000] 10.0%, loss 0.369076                                                                        
[train_ae][000] 15.0%, loss 0.335791                                                                        
[train_ae][000] 20.0%, loss 0.305971                                                                        
[train_ae][000] 25.0%, loss 0.283139                                                                        
[train_ae][000] 30.0%, loss 0.266357                                                                        
[train_ae][000] 35.0%, loss 0.252597                                                                        
[train_ae][000] 40.0%, loss 0.241102                                                                        
[train_ae][000] 45.0%, loss 0.230529                                                                        
[train_ae][000] 50.0%, loss 0.222019                                                                        
[train_ae][000] 55.0%, loss 0.214202                                                                        
[train_ae][000] 60.0%, loss 0.207348                                                                        
[train_ae][000] 65.0%, loss 0.201420                                                                        
[train_ae][000] 70.0%, loss 0.196110                                                                        
[train_ae][000] 75.0%, loss 0.191142                                                                        
[train_ae][000] 80.0%, loss 0.186935                                                                        
[train_ae][000] 85.0%, loss 0.182847                                                                        
[train_ae][000] 90.0%, loss 0.179199                                                                        
[train_ae][000] 95.0%, loss 0.175985                                                                        
[train_ae][000] 100.0%, loss 0.172849                                                                       
100%|███████████████████████████████████████████████████████████████| 231212/231212 [54:20<00:00, 67.97ex/s]
100%|████████████████████████████████████████████████████████████████| 10000/10000 [00:44<00:00, 226.19ex/s]
[eval_train_ae][000] Evaluation: 
	spec_l1=0.083817
	spec_mse=0.112751
	wav_l1=0.117856
	wav_mse=0.050829

100%|████████████████████████████████████████████████████████████████| 29167/29167 [03:46<00:00, 133.51ex/s]
[valid_ae][000] Evaluation: 
	spec_l1=0.084367
	spec_mse=0.116608
	wav_l1=0.117584
	wav_mse=0.050546

100%|████████████████████████████████████████████████████████████████| 28826/28826 [03:47<00:00, 121.61ex/s]
[test_ae][000] Evaluation: 
	spec_l1=0.085511
	spec_mse=0.117768
	wav_l1=0.118981
	wav_mse=0.051662

[train_ae][001] 5.0%, loss 0.110845                                                                         
[train_ae][001] 10.0%, loss 0.108712                                                                        
[train_ae][001] 15.0%, loss 0.111350                                                                        
[train_ae][001] 20.0%, loss 0.111059                                                                        
[train_ae][001] 25.0%, loss 0.109674                                                                        
[train_ae][001] 30.0%, loss 0.107961                                                                        
[train_ae][001] 35.0%, loss 0.106814                                                                        
[train_ae][001] 40.0%, loss 0.105815                                                                        
[train_ae][001] 45.0%, loss 0.105225                                                                        
[train_ae][001] 50.0%, loss 0.104666                                                                        
[train_ae][001] 55.0%, loss 0.104414                                                                        
[train_ae][001] 60.0%, loss 0.103496                                                                        
[train_ae][001] 65.0%, loss 0.102737                                                                        
[train_ae][001] 70.0%, loss 0.101856                                                                        
[train_ae][001] 75.0%, loss 0.101098                                                                        
[train_ae][001] 80.0%, loss 0.100212                                                                        
[train_ae][001] 85.0%, loss 0.099681                                                                        
[train_ae][001] 90.0%, loss 0.099047                                                                        
[train_ae][001] 95.0%, loss 0.098430                                                                        
[train_ae][001] 100.0%, loss 0.097780                                                                       
100%|███████████████████████████████████████████████████████████████| 231212/231212 [43:35<00:00, 88.90ex/s]
100%|████████████████████████████████████████████████████████████████| 10000/10000 [00:44<00:00, 225.77ex/s]
[eval_train_ae][001] Evaluation: 
	spec_l1=0.069786
	spec_mse=0.085866
	wav_l1=0.122142
	wav_mse=0.055850

100%|████████████████████████████████████████████████████████████████| 29167/29167 [02:08<00:00, 226.21ex/s]
[valid_ae][001] Evaluation: 
	spec_l1=0.071161
	spec_mse=0.091293
	wav_l1=0.121677
	wav_mse=0.055485

100%|████████████████████████████████████████████████████████████████| 28826/28826 [02:07<00:00, 222.12ex/s]
[test_ae][001] Evaluation: 
	spec_l1=0.072031
	spec_mse=0.092283
	wav_l1=0.123048
	wav_mse=0.056602

[train_ae][002] 5.0%, loss 0.085987                                                                         
[train_ae][002] 10.0%, loss 0.086687                                                                        
[train_ae][002] 15.0%, loss 0.085140                                                                        
[train_ae][002] 20.0%, loss 0.083836                                                                        
[train_ae][002] 25.0%, loss 0.083140                                                                        
[train_ae][002] 30.0%, loss 0.083252                                                                        
[train_ae][002] 35.0%, loss 0.083223                                                                        
[train_ae][002] 40.0%, loss 0.082337                                                                        
[train_ae][002] 45.0%, loss 0.081733                                                                        
[train_ae][002] 50.0%, loss 0.081000                                                                        
[train_ae][002] 55.0%, loss 0.080607  
  
```

## multiGPU train in DGX-2 
will release soon. 

## inference with V100 
