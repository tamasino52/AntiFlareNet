# AntiFlareNet
Flare removal models for LG dacon competition.

# Directory
```
${POSE_ROOT}
|-- data
    |-- train
    |   |-- input
    |   |   |-- inp1.png
    |   |   |-- inp2.png
    |   |   |-- ...
    |   |-- target
    |   |   |-- tar1.png
    |   |   |-- tar2.png
    |   |   |-- ...
    |-- test
    |   |-- input
    |   |   |-- inp1.png
    |   |   |-- inp2.png
    |   |   |-- ...
    |   |-- target  #option
    |   |   |-- tar1.png
    |   |   |-- tar2.png
    |   |   |-- ...

```

# run
## Train
This repository use 2 stream models. You have to train each model seperately.
```
python run/train.py
python run/train_merge.py
```
## Test
```
python run/test_merge.py
```

# Figure
Input | Prediction | Target | 1.0 scale attention | 0.5 scale attention | 0.25 scale attention | 0.125 scale attention |
<img src="/figure/figure (1).jpg"><br>
<img src="/figure/figure (2).jpg"><br>
<img src="/figure/figure (3).jpg"><br>
<img src="/figure/figure (4).jpg"><br>
<img src="/figure/figure (5).jpg"><br>
<img src="/figure/figure (6).jpg"><br>
<img src="/figure/figure (7).jpg"><br>
<img src="/figure/figure (8).jpg"><br>
<img src="/figure/figure (9).jpg"><br>
<img src="/figure/figure (10).jpg"><br>
<img src="/figure/figure (11).jpg"><br>
<img src="/figure/figure (12).jpg">
