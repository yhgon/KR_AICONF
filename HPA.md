

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

investigate file
```
mkdir samples
cd samples
cp ../train/00301238-bbb2-11e8-b2ba-ac1f6b6435d0* .
ls -h 
     62K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_blue.png
     87K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_green.png
    142K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_red.png
    142K  00301238-bbb2-11e8-b2ba-ac1f6b6435d0_yellow.png

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
