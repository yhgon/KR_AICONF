

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
# of train files : 124,288 items(RGBY) --> 31,072 images
# of test  files :  46,808 items(RGBY) --> 11,702 images
```

labels 
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


```
# of train files : 124,288 items(RGBY) --> 31,072 images
# of test  files :  46,808 items(RGBY) --> 11,702 images
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

#### train dataset (320GB)
```
mkdir full_size/train
cd full_size/train
p7zip -d ../train_full_size.7z | pv -l >/dev/null
du -h full_size/train

```

7z is efficient compress file format but it's hard to handle single large file.
so I'll split the file.

#### move subfolders each 5000 images 
```
#!/bin/bash
c=1; 
d=1; 
mkdir -p SUB_${d}
for tif_filelist in *.tif
do
  if [ $c -eq 5000 ]
  then
    d=$(( d + 1 )); c=0; mkdir -p SUB_${d}
  fi
  mv "$tif_filelist" SUB_${d}/
  c=$(( c + 1 ))
done
```

result

test datasets
```
15G	test/SUB_8
15G	test/SUB_2
15G	test/SUB_5
15G	test/SUB_1
5.5G	test/SUB_10
15G	test/SUB_6
15G	test/SUB_4
15G	test/SUB_3
16G	test/SUB_9
15G	test/SUB_7
143G	test
```

train datasets
```
13G	train/SUB_23
13G	train/SUB_11
13G	train/SUB_1
14G	train/SUB_6
14G	train/SUB_16
14G	train/SUB_24
13G	train/SUB_20
14G	train/SUB_2
13G	train/SUB_8
13G	train/SUB_18
14G	train/SUB_12
13G	train/SUB_15
13G	train/SUB_5
13G	train/SUB_7
13G	train/SUB_17
12G	train/SUB_25
13G	train/SUB_22
13G	train/SUB_10
13G	train/SUB_14
13G	train/SUB_4
13G	train/SUB_21
14G	train/SUB_9
13G	train/SUB_3
13G	train/SUB_13
14G	train/SUB_19
324G	train

```

#### compress each files 
```
for dir in `find . -maxdepth 1 -type d  | grep -v "^\.$" `; do tar -cvzf ${dir}.tar.gz ${dir}; done
```

upload the files be careful train/test folder

#### decompress each files
```
time for file in *.tar.gz; do tar -zxf $file; done
```

if you have multicores, you could decompress with multicores. 
25 cores for train dataset, 10 cores for test dataset
```
time find . -type f -name "*.tar.gz" | xargs -I {} -P 25 tar -xf {}
time find . -type f -name "*.tar.gz" | xargs -I {} -P 10 tar -xf {}
```

#### monitor status every 10 sec during decompressing the files 
```
watch -n 10 du -h /mnt/hpa_full/test/test_tar/SUB_1 /mnt/hpa_full/train/train_tar/SUB_1

Every 10.0s: du -h /mnt/hpa_full/test/test_tar/SUB_1 /mnt/hpa_full/train/train_tar/...  Mon Dec 24 00:16:41 2018

2.6G    /mnt/hpa_full/test/test_tar/SUB_1
3.8G    /mnt/hpa_full/train/train_tar/SUB_1


Every 10.0s: du -h /mnt/hpa_full/test/test_tar/SUB_1 /mnt/hpa_full/train/train_tar/...  Mon Dec 24 00:18:00 2018

3.8G    /mnt/hpa_full/test/test_tar/SUB_1
4.5G    /mnt/hpa_full/train/train_tar/SUB_1


```


#### merge whole files in main folder 
```
find /target_dir -type f -exec mv -i -t /dest_dir {} +
```

```
mkdir merge_train
find subfolder_train -type f -exec mv -i -t merge_train {} +

find merge_train -type f | wc -l
124288

mkdir merge_test
find subfolder_test -type f -exec mv -i -t merge_test {} +

find merge_test -type f | wc -l
46808

```
#### convert tif(4MB) to png(317KB)

example of files with lossless compression
```
000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.tif  2048x2048    4MB
000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png  2048x2048  317KB
```

utility ImageMagic

```
mkdir train_full_png
cd merge_train
time find . -type f -name "*.tif" | parallel -j 24 convert -quiet  {}  ../train_full_png/{.}.png
cd ..

```

monitoring

```
monitor_convert_train.sh
DIR="train_full_png"
OUT="$(du -h  ${DIR} |  awk '{print $1}' )"

echo "convert result : ${OUT} from original 320G"

watch -n 20 monitor_convert.sh 
```

- PNG : 220GB
- TIF : 320GB

```
du -h merge_train train_full_png
320G	merge_train
220G	train_full_png


ls merge_train | wc -l
124288

ls train_full_png | wc -l
124288
```

eliminate TIFF file and folders
```
mkdir empty_folder
time rsync -a --delete ./empty_folder/ ./merge_train/
time rsync -a --delete ./empty_folder/ ./tar_files_train/

```

split and move to subfolders 
```
cd train_full_png
sh ../subfolder_png_train_full.sh

```
```
#!/bin/bash
c=1; 
d=1; 
mkdir -p train_png_sub_${d}
for png_filelist in *.png
do
  if [ $c -eq 5000 ]
  then
    d=$(( d + 1 )); c=0; mkdir -p train_png_sub_${d}
  fi
  mv "$png_filelist" train_png_sub_${d}/
  c=$(( c + 1 ))
done

```

```
du -h .
9.0G	./train_png_sub_24
9.0G	./train_png_sub_16
9.0G	./train_png_sub_2
8.9G	./train_png_sub_8
8.8G	./train_png_sub_5
8.8G	./train_png_sub_11
8.8G	./train_png_sub_23
8.9G	./train_png_sub_15
8.9G	./train_png_sub_1
9.0G	./train_png_sub_6
8.7G	./train_png_sub_18
9.0G	./train_png_sub_12
8.8G	./train_png_sub_20
8.7G	./train_png_sub_10
8.7G	./train_png_sub_22
8.9G	./train_png_sub_4
9.0G	./train_png_sub_9
8.8G	./train_png_sub_3
7.6G	./train_png_sub_25
8.7G	./train_png_sub_17
8.9G	./train_png_sub_13
9.0G	./train_png_sub_19
8.8G	./train_png_sub_21
8.8G	./train_png_sub_7
8.8G	./train_png_sub_14
220G	.
```



compress subfolders of png files in parallel 
```
cd train_full_png
time ls | parallel --no-notice -j$(ls |wc -l) tar -czf ../tar/{}.tar.gz {}

watch -n 5 ls -alh tar/train_png_sub_1.tar.gz


Every 5.0s: ls -alh train_png_sub_1.tar.gz                                              Mon Dec 24 15:30:50 2018

-rw-r--r--+ 1 25223 dip 1.9G Dec 24 15:30 train_png_sub_1.tar.gz


Every 5.0s: ls -alh train_png_sub_1.tar.gz                                              Mon Dec 24 15:31:11 2018

-rw-r--r--+ 1 25223 dip 2.2G Dec 24 15:31 train_png_sub_1.tar.gz


```


class
```
 0.  Nucleoplasm                    Nucleus-Nucleoplasm  (핵질) 
 1.  Nuclear membrane               Nucleus-Nuclear membrane-Nuclear membrane  
 2.  Nucleoli                       Nucleus-Nucleoli (인) 
 3.  Nucleoli fibrillar center      Nucleus-Nucleoli-Nucleoli fibrillar center
 4.  Nuclear speckles               Nucleus-Nucleoplasm-Nuclear speckles  
 5.  Nuclear bodies                 Nucleus-Nucleoplasm-Nuclear bodies 
 6.  Endoplasmic reticulum          secretory-Endoplasmic reticulum (소포체 ER)
 7.  Golgi apparatus                secretory-Golgi apparatus 
 8.  Peroxisomes                    secretory-Vesicles-Peroxisomes
 9.  Endosomes                      secretory-Vesicles-Endosomes
10.  Lysosomes                      secretory-Vesicles-Lysosomes (리소좀 : 골지 복합체에서 나옴. 세포질에 있음) 
11.  Intermediate filaments         cytoplasm-Intermediate filaments  (세포골격2. 중간섬유) 
12.  Actin filaments                cytoplasm-Actin filaments  (세포골격3. 미세섬유 : microfilaments) 
13.  Focal adhesion sites           cytoplasm-Actin filaments-Focal adhesion sites  (세포골격3.
14.  Microtubules                   cytoplasm-Microtubules  (세포골격1. 미세소관) 
15.  Microtubule ends               cytoplasm-Microtubules-Microtubule ends
16.  Cytokinetic bridge             cytoplasm-Microtubules-Cytokinetic bridge
17.  Mitotic spindle                cytoplasm-Microtubules-Mitotic spindle
18.  Microtubule organizing center  cytoplasm--Centrosome-Microtubule organizing center
19.  Centrosome                     cytoplasm-Centrosome (중심체) 
20.  Lipid droplets                 secretory-Vesicles-Lipid droplets
21.  Plasma membrane                secretory-Plasma Membrane
22.  Cell junctions                 secretory-Plasma Membrane -Cell junctions  
23.  Mitochondria                   cytoplasm-Mitochondria(미토콘드리아) 
24.  Aggresome                      cytoplasm-Cytosol-Aggresome
25.  Cytosol                        cytoplasm-Cytosol
26.  Cytoplasmic bodies             cytoplasm-Cytosol-Cytoplasmic bodies
27.  Rods & rings                   cytoplasm-Cytosol-Rods & rings
```

top3 sub category
```
Nucleus- 핵 
cytoplasm 세포질 
secretory 분비
```

count of each categories
```
Cytokinetic bridge : 530
Nucleoplasm  : 12885
Golgi apparatus  : 2822
Nuclear membrane: 1254
Nucleoli: 3621
Nuclear bodies: 2513
Microtubule organizing center: 902
Cytosol: 8228
Mitochondria: 2965
Plasma membrane: 3777
Aggresome: 322
Endoplasmic reticulum: 1008
Intermediate filaments: 1093
Nucleoli fibrillar center: 1561
Actin filaments: 688
Focal adhesion sites: 537
Microtubules: 1066
Nuclear speckles: 1858
Lipid droplets: 172
Cell junctions: 802
Mitotic spindle: 210
Centrosome: 1482
Peroxisomes: 53
Endosomes: 45
Lysosomes: 28
Cytoplasmic bodies: 328
Rods & rings: 11
Microtubule ends: 21
```
