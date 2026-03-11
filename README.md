# mel_filterbank_torch
PyTorch implementation of log-mel filterbanks with audio classification using group convolution

## Installation
```uv sync```

## Download dataset
```curl -L -o ./data/speech_commands_v0.02.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz```
  
```tar -xzf ./data/speech_commands_v0.02.tar.gz -C ./data/SpeechCommands/speech_commands_v0.02```

## Usage:
```python train.py --n_mels 80 --groups 1 --epochs 20```
  
## TensorBoard:
```tensorboard --logdir runs/```