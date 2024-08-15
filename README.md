# VGGR
<p align="center">
  <img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/icon.png' height="150">
</p>
Have you ever seen gameplay footage and wondered what kind of video game it is from? No? Well, do not not wonder anymore.
<br /><br />
VGGR (Video Game Genre Recognition) is a Deep-Learning Image Classification project, answering questions nobody is asking.



## Requirements
1. Install **Python 3.10** or newer.

2. Clone the repository with
   >```
   >git clone https://github.com/m4cit/VGGR.git
   >```
   
   or download the latest [source code](https://github.com/m4cit/VGGR/releases).

3. Download the latest train, test, and validation img zip-files in [*releases*](https://github.com/m4cit/VGGR/releases).

4. Unzip the train, test, and validation img files inside their respective folders located in **./data/**.

5. Install [PyTorch](https://pytorch.org/get-started/locally/)

   5.1 Either with CUDA
      - Windows:
         >```
         > pip3 install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
         >```
      - Linux:
         >```
         > pip3 install torch==2.2.2 torchvision==0.17.2
         >```
         
   5.2 Or without CUDA
      - Windows:
         >```
         > pip3 install torch==2.2.2 torchvision==0.17.2
         >```
      - Linux:
         >```
         > pip3 install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
         >```
   
6. Navigate to the VGGR main directory
   >```
   > cd VGGR
   >```

7. Install dependencies
   >```
   > pip install -r requirements.txt
   >```


**Note:** The provided train dataset does not contain augmentations.


## Genres
- Football / Soccer
- First Person Shooter (FPS)
- 2D Platformer
- Racing

## Games
### Train Set
- FIFA 06
- Call of Duty Black Ops
- Call of Duty Modern Warfare 3
- DuckTales Remastered
- Project CARS

### Test Set
- PES 2012
- FIFA 10
- Counter Strike 1.6
- Counter Strike 2
- Ori and the Blind Forest
- Dirt 3

### Validation Set
- Left 4 Dead 2
- Oddworld Abe's Oddysee
- FlatOut 2

## Usage
### Commands
**--demo** | Demo predictions with the test set

**--augment** | Data Augmentation

**--train** | Train mode

**--predict** | Predict / inference mode

**--input (-i)** | File input for predict mode

**--model (-m)** | Model selection
  * cnn_v1 (default)
  * cnn_v2
  * cnn_v3

**--device (-d)** | Device selection
  * cpu (default)
  * cuda
  * ipu
  * xpu
  * mkldnn
  * opengl
  * opencl
  * ideep
  * hip
  * ve
  * fpga
  * ort
  * xla
  * lazy
  * vulkan
  * mps
  * meta
  * hpu
  * mtia

### Examples
#### Demo with Test Set
>```
>python VGGR.py --demo
>```
or
>```
>python VGGR.py --demo -m cnn_v1 -d cpu
>```
or
>```
>python VGGR.py --demo --model cnn_v1 --device cpu
>```

#### Predict with Custom Input
>```
>python VGGR.py --predict -i path/to/img.png
>```
or
>```
>python VGGR.py --predict -m cnn_v1 -d cpu -i path/to/img.png
>```

#### Training
>```
>python VGGR.py --train -m cnn_v1 -d cpu
>```

<br />

Delete the existing model to train from scratch.

## Results
The --demo mode creates html files with the predictions and corresponding images inside the _**results**_ folder.

## Performance
There are three Convolutional Neural Network (CNN) models available:

1. *cnn_v1* | F-score of **75 %**
2. *cnn_v2* | F-score of **58.33 %**
3. *cnn_v3* | F-score of **64.58 %**


### cnn_v1 --demo result examples
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_football.png' width="900">
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_fps.png' width="900">
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_racing.png' width="900">


## Data
Most of the images are from my own gameplay footage.
The PES 2012 and FIFA 10 images are from videos by [No Commentary Gameplays](https://www.youtube.com/@NCGameplays), and the FIFA 95 images are from a video by [10min Gameplay](https://www.youtube.com/@10minGameplay1) (YouTube).

The train dataset also contained augmentations (not in the provided zip-file).

### Augmentation
To augment the train data with *jittering*, *inversion*, and *5 part cropping*, copy-paste the metadata of the images into the _**augment.csv**_ file located in _**./data/train/metadata/**_.

Then run
>```
>python VGGR.py --augment
>```

The metadata of the resulting images are subsequently added to the _**metadata.csv**_ file.


### Preprocessing
All images are originally 2560x1440p, and get resized to 1280x720p before training, validation, and inference. 4:3 images are stretched to 16:9 to avoid black bars.


## Libraries
* [PyTorch](https://pytorch.org/) and its dependencies
* [tqdm](https://tqdm.github.io/)
* [pandas](https://pandas.pydata.org/)

