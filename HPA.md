

[Kaggle HPA competition](https://www.kaggle.com/c/human-protein-atlas-image-classification )

kaggle-human-protein-atlas
### download Data (17 GB)
512x512 PNG files in train.zip and test.zip. 


Download Dataset 
- install kaggle api `sudo pip3 install kaggle`
- copy to your API file  `~/.kaggle/kaggle.json` from kaggle account site
- download `kaggle competitions download -c human-protein-atlas-image-classification`

```
 446K   sample_submission.csv
 1.3M   train.csv
 4.4G   test.zip
  14G   train.zip
```

```
unzip -o test.zip -d ./test | pv -l >/dev/null
unzip -o train.zip -d ./train | pv -l >/dev/null
```

## investigate file



```
head -n 20 train.csv 
Id,Target
00070df0-bbc3-11e8-b2bc-ac1f6b6435d0,16 0
000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0,7 1 2 0
000a9596-bbc4-11e8-b2bc-ac1f6b6435d0,5
000c99ba-bba4-11e8-b2b9-ac1f6b6435d0,1
001838f8-bbca-11e8-b2bc-ac1f6b6435d0,18
001bcdd2-bbb2-11e8-b2ba-ac1f6b6435d0,0
0020af02-bbba-11e8-b2ba-ac1f6b6435d0,25 2
002679c2-bbb6-11e8-b2ba-ac1f6b6435d0,0
00285ce4-bba0-11e8-b2b9-ac1f6b6435d0,2 0
002daad6-bbc9-11e8-b2bc-ac1f6b6435d0,7
002ff91e-bbb8-11e8-b2ba-ac1f6b6435d0,23
00301238-bbb2-11e8-b2ba-ac1f6b6435d0,21
0032a07e-bba9-11e8-b2ba-ac1f6b6435d0,24 0
00344514-bbc2-11e8-b2bb-ac1f6b6435d0,23
00357b1e-bba9-11e8-b2ba-ac1f6b6435d0,6 2
00383b44-bbbb-11e8-b2ba-ac1f6b6435d0,25
0038d6a6-bb9a-11e8-b2b9-ac1f6b6435d0,25 0
003957a8-bbb7-11e8-b2ba-ac1f6b6435d0,25
003feb6e-bbca-11e8-b2bc-ac1f6b6435d0,0


```
### for color filter  means
- the protein of interest (green) 
- plus three cellular landmarks: nucleus (blue),
- microtubules (red), 
- endoplasmic reticulum (yellow)

```
mkdir samples_1
cd samples_1
cp ../train/00301238-bbb2-11e8-b2ba-ac1f6b6435d0* .
ls -h 
     62K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_blue.png
     87K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_green.png
    142K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_red.png
    142K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_yellow.png


mkdir samples_2
cd samples_2
cp ../train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0* .
ls -h 
    34K   000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png
    93K   000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png
   127K   000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png
   131K   000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png

```

raw data : 170GB
[HPA](https://console.cloud.google.com/storage/browser/kaggle-human-protein-atlas?pli=1)
- test_full_size.7z	78.24GB	 	
- train_full_size.7z	189.13GB	 

Labels for Dataset 
```
0.  Nucleoplasm  
1.  Nuclear membrane   
2.  Nucleoli   
3.  Nucleoli fibrillar center   
4.  Nuclear speckles   
5.  Nuclear bodies   
6.  Endoplasmic reticulum   
7.  Golgi apparatus   
8.  Peroxisomes   
9.  Endosomes   
10.  Lysosomes   
11.  Intermediate filaments   
12.  Actin filaments   
13.  Focal adhesion sites   
14.  Microtubules   
15.  Microtubule ends   
16.  Cytokinetic bridge   
17.  Mitotic spindle   
18.  Microtubule organizing center   
19.  Centrosome   
20.  Lipid droplets   
21.  Plasma membrane   
22.  Cell junctions   
23.  Mitochondria   
24.  Aggresome   
25.  Cytosol   
26.  Cytoplasmic bodies   
27.  Rods & rings
```


## download raw dataset (250GB)
-  test_full_size.7z	  78.24GB	 
-  train_full_size.7z	189.13GB	

```
wget https://storage.googleapis.com/kaggle-human-protein-atlas/test_full_size.7z
wget https://storage.googleapis.com/kaggle-human-protein-atlas/train_full_size.7z
```

## decompress raw dataset 
#### test dataset(138GB)

```
mkdir full_size/test
cd full_size/test
p7zip -d ../test_full_size.7z | pv -l >/dev/null
du -h full_size/test
138G    full_size/test
```

#### train dataset 
```
mkdir full_size/train
cd full_size/train
p7zip -d ../train_full_size.7z | pv -l >/dev/null
du -h full_size/train
 

```


```
