# gtav-vision

## Dependencies
- [win32gui](https://github.com/mhammond/pywin32)
- opencv-contrib-python
- keyboard

```sh
conda create -n gtav python==3.7
pip install pywin32
conda install -c conda-forge opencv
pip install keyboard
```

```sh
conda activate gtav
python bot.py
```

## Convert VoTT to YOLO
- pandas
- requests
- progressbar2
- tensorflow-gpu
- keras
- matplotlib

## Yolov3 weights
```sh
(gtav) D:\dev\gtav-vision\yolo\keras_yolo3>python convert.py yolov3.cfg yolov3.weights yolo.h5
```

## Pytorch
```sh
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
