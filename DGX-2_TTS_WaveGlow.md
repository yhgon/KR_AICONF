
# WaveGlow
check 
- [official site](https://nv-adlr.github.io/WaveGlow) and
- [WaveGlow Paper, **Ryan Prenger, Rafael Valle, and Bryan Catanzaro, WaveGlow: a Flow-based Generative Network for Speech Synthesis** arxiv 1811.00002](https://arxiv.org/abs/1811.00002)

for more information for glow model, check flow
 
- [OpenAI Glow Paper, Glow: Generative Flow with Invertible 1×1 Convolutions] (https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf) 
- [openAI glow] (https://github.com/openai/glow) 
- [RealNVP paper](https://arxiv.org/abs/1605.08803) 

after NVIDIA release the source code, I'll update it.


### logs
training logs

536:	-3.692588091 	1.84s/it
537:	-3.829968929 	1.84s/it
538:	-3.654531240 	1.88s/it
539:	-3.801097393 	1.88s/it
540:	-3.093513012 	1.91s/it
541:	-3.053615093 	1.87s/it
542:	-2.874729395 	1.85s/it
543:	-3.210449696 	1.87s/it
544:	-3.125212431 	1.87s/it
1050.56s/epoch 
Epoch: 1
545:	-3.140144587 	1.94s/it
546:	-3.212766171 	1.90s/it
547:	-3.457610607 	1.85s/it
548:	-3.353491783 	1.87s/it


### utilization

with 8 GPUs
```
Tue Nov  6 21:34:03 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.57                 Driver Version: 410.57                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
| N/A   65C    P0   196W / 300W |  14712MiB / 16130MiB |     91%      Default |
| N/A   68C    P0   228W / 300W |  14712MiB / 16130MiB |     91%      Default |
| N/A   72C    P0   234W / 300W |  14712MiB / 16130MiB |     91%      Default |
| N/A   65C    P0   200W / 300W |  14712MiB / 16130MiB |     92%      Default |
| N/A   66C    P0   195W / 300W |  14712MiB / 16130MiB |     90%      Default |
| N/A   71C    P0   233W / 300W |  14712MiB / 16130MiB |     88%      Default |
| N/A   74C    P0   239W / 300W |  14712MiB / 16130MiB |     84%      Default |
| N/A   65C    P0   230W / 300W |  14712MiB / 16130MiB |     86%      Default |
+-------------------------------+----------------------+----------------------+
```

with 16 GPUs
```
Tue Nov  6 12:01:28 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.33                 Driver Version: 410.33                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
| N/A   54C    P0   348W / 350W |  27296MiB / 32510MiB |     81%      Default |
| N/A   48C    P0    90W / 350W |  27194MiB / 32510MiB |     73%      Default |
| N/A   63C    P0   160W / 350W |  27194MiB / 32510MiB |     96%      Default |
| N/A   65C    P0   343W / 350W |  27194MiB / 32510MiB |     85%      Default |
| N/A   52C    P0   348W / 350W |  27194MiB / 32510MiB |     99%      Default |
| N/A   70C    P0   351W / 350W |  27194MiB / 32510MiB |     99%      Default |
| N/A   51C    P0   242W / 350W |  27194MiB / 32510MiB |     93%      Default |
| N/A   67C    P0   294W / 350W |  27194MiB / 32510MiB |     55%      Default |
| N/A   51C    P0   337W / 350W |  27194MiB / 32510MiB |     68%      Default |
| N/A   51C    P0    89W / 350W |  27194MiB / 32510MiB |     69%      Default |
| N/A   68C    P0   352W / 350W |  27194MiB / 32510MiB |     78%      Default |
| N/A   65C    P0   102W / 350W |  27194MiB / 32510MiB |     63%      Default |
| N/A   49C    P0   124W / 350W |  27194MiB / 32510MiB |    100%      Default |
| N/A   46C    P0   198W / 350W |  27194MiB / 32510MiB |     82%      Default |
| N/A   67C    P0   342W / 350W |  27194MiB / 32510MiB |     98%      Default |
| N/A   67C    P0   355W / 350W |  27194MiB / 32510MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
```
