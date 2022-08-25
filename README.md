# KV6003-RC-Car

This project contains the code for a remote control car that is controlled by using a neural network. The car used for this project uses a raspberry pi, pi camera and distance sensors in it's usage.

Table of contents
=================

<!--ts-->
   * [Setup](#Setup)
   * [Installation](#Installation)
   * [Data](#Data)
   * [Project Tabs](#tabs)
      * [Predict](#Predict)
      * [Train](#Train)
<!--te-->

## Setup

Setup the virtual enviroment and install packages

```
python -m venv --system-site-packages .\CarEnv\
```

Activate the enviroment

[Note] Depending on system security settings, activation of the enviroment may not be possible. You can allow execution of the scripts
with:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```


```
.\CarEnv\Scripts\activate
```

ensure pip is the latest

```
pip install --upgrade pip
```

## Installation

Grab the work from git

```
git clone https://github.com/Amzo/KV6003-RC-Car.git
```

```
cd .\KV6003-RC-Car\
```

Ensure these are the latest to avoid opencv-python failing

```
pip install --upgrade setuptools wheel
```

Finally install the necessary packages

```
pip install -r requirements.txt
```

Finally compile the cython code

```
python.exe .\setup.py .\build.sh --inplace
```

## Data

```
git clone https://github.com/Amzo/KV6003-RC-Car-Data.git
```

[NOTE] When training a model in the train tab, browse to the downloaded dataset. The setup of cuda if required is down to the user.

 execute the program with the following:
 
 ```
 python .\client\main.py
 ```
 
 The server will fail to start unless a raspberry pi camera is detected
 
 ```
 python .\car.py --controller ai
 ```
 
 
## Project Tabs
### Predict

This tab will allow connecting to a remote server and sending the predicted commands as 4bytes containing the pridicted character and the newline characters '\n'. e.G if A is sent to the server the server will receive 'A\n'. If there is no server to connect to this tab will not work, as it requires an active connection before making a prediction.

### Train

This tab allows fitting the convolutional neural network. Browse to the necessary data and select a model output folder. The model consists of an Xception model as a base for transfer learning. Training will be painfully slow without GPU accelleration and is advised to avoid training the CNN if no GPU is available.
