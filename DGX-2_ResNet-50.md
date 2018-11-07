# ResNet-50 

```
Unpack DGX-2

docker pull  nvcr.io/nvidia/tensorflow:18.10-py3

nvidia-docker run -it   nvcr.io/nvidia/tensorflow:18.10-py3 bash

python /workspace/nvidia-examples/cnn/resnet.py  --layers=50  --precision=fp32   --num_iter=400  --iter_unit=batch  --batch_size 128

mpirun --allow-run-as-root  --bind-to socket   -np 16 python /workspace/nvidia-examples/cnn/resnet.py --data_dir=/mnt/hryu/dataset-tf/train-val-tfrecord-480-subset --log_dir=/mnt/hryu/result/try3   --layers=50  --precision=fp16   --num_iter=400  --iter_unit=batch  --batch_size 256
```
