# Transfer Learning using PyTorch's Torchvision pretrained Model

This repo contains code that demonstrates **Transfer Learning**. A pre-trained VGG-16 (with Batch Normalization) is downloaded using TorchVision first. Then the feature layer parameters are freezed during transfer learning and the classifier layers are replaced with custom layers. The classifier layers are then trained using our own custom datasets and the classifier parameters are saved separately. 

During evaluation, note that **both** feature layer parameters (from the downloaded pre-trained model) and the classifier layer parameters (from the transfer learning training) are needed since Torch only saves the trainable paramaters.

### Usage
1. Place your own custom dataset in a folder called **dataset** in the root directory.
2. If you wish to change any detail in the model's architecture or use a different model entirely, modify the codes in `model.py`. 
3. Start the training with `python main.py --mode train`.
4. Once the training is over, you can evaluate your images with `python main.py --mode eval` and then supplying the path to your image file that you wish to be evaluated when prompted.

#### License
___

MIT

