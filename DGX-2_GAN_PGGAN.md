
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

