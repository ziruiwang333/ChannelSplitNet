# **Computer Science Project - Plant Diseases Classification using Convolutional Neural Network** 

<font size=6>By ***Zirui Wang - 1935080***, Supervised by ***Dr. Kashif Rajpoot***</font>

&emsp;

Table of Contents

1. [GitHub Page](#1-GitHub-Page)
2. [Dataset](#2-dataset)
4. [Project Code Content](#3-project-code-contents)
5. [Run the Codes](#4-run-the-codes)



# 1. GitHub Page

GitHub page for this project is: https://github.com/ziruiwang333/ChannelSplitNet

# 2. Dataset

The dataset is on Kaggle: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset

# 3. Project Code Contents

* **0\. Import Libraries**
* **1\. Data Processing**
  * 1.1 Randomly allocate dataset
    * 1.1.1 Allocate training set
    * 1.1.2 Allocate validation set
    * 1.1.3 Allocate test set
  * 1.2 Data augmentation
    * 1.2.1 Data augmentation on training set
    * 1.2.2 Data augmentation on validation set
    * 1.3 Put dataset together
    * 1.3.1 Put training set together
    * 1.3.2 Put validation set together
* **2\. Load data**
  * 2.1 Load training data
    * 2.1.1 Load training data from directory
    * 2.1.2 Save training data as pickle file
    * 2.1.3 Load training data from pickle file
  * 2.2 Load validation data
    * 2.2.1 Load validation data from directory
    * 2.2.2 Save validation data as pickle file
    * 2.2.3 Load validation data from pickle file
  * 2.3 Load Test Data
    * 2.3.1 Load test data from directory
    * 2.3.2 Save test data as pickle file
    * 2.3.3 Function - Load test data from pickle file
* **3\. Training with Different CNNs**
  * 3.1 ChannelSplitNet
    * 3.1.1 Build the network
    * 3.1.2 ChannelSplitNet Dense output training
      * 3.1.2.1 First training
      * 3.1.2.2 Second training
      * 3.1.2.3 Third training
      * 3.1.2.4 Fourth training
      * 3.1.2.5 Save the best model
    * 3.1.3 Conv-Flatten output training
      * 3.1.3.1 Conv-Flatten output first training
      * 3.1.3.2 Conv-Flatten output second training
      * 3.1.3.3 Save the best model
  * 3.2 GoogLeNet
    * 3.2.1 Build the network
    * 3.2.2 GoogLeNet Training
    * 3.2.3 Save the best model
  * 3.3 AlexNet
    * 3.3.1 Build the network
    * 3.3.2 AlexNet dense output training
      * 3.3.2.1 Save the best model
    * 3.3.3 AlexNet Conv-Flatten output training
      * 3.3.3.1 Save the best model
  * 3.4 VGG16
    * 3.4.1 Build the network
    * 3.4.2 VGG16 trained from scratch
      * 3.4.2.1 First training
      * 3.4.2.2 Second training
      * 3.4.2.3 Third training
      * 3.4.2.4 Fourth training
      * 3.4.1.5 Save the best model
    * 3.4.3 Pretrained VGG16 with freezing Conv layer
      * 3.4.3.1 Build the pretrained VGG16 network
      * 3.4.3.2 Pretrained VGG16 dense output training
        * 3.4.3.2.1 Save the best model
      * 3.4.3.3 Pretrained VGG16 Conv-Flatten output training
        * 3.4.3.3.1 First training
        * 3.4.3.3.2 Second training
        * 3.4.3.3.3 Save the best model
  * 3.5 Pretrained ResNet50
    * 3.5.1 Build the network
    * 3.5.2 Pretrained ResNet50 dense output training
      * 3.5.2.1 First training
      * 3.5.2.2 Second training
      * 3.5.2.3 Save the best model
    * 3.5.3 Pretrained ResNet50 Conv-Flatten output training
      * 3.5.3.1 First training
      * 3.5.3.2 Second training
      * 3.5.3.3 Save the best model
  * 3.6 Inception V3
    * 3.6.1 Build the network
    * 3.6.2 Pretrained Inception V3 dense output training
      * 3.6.2.1 Save the best model
    * 3.6.3 Pretrained Inception V3 Conv-Flatten output training
      * 3.6.3.1 Save the best model
* **4\. Experiments**
  * 4.1 Conv layer experiments
    * 4.1.1 No Conv layer
      * 4.1.1.1 Save the best model
    * 4.1.2 One Conv layer
      * 4.1.2.1 Save the best model
    * 4.1.3 Two Conv layer
      * 4.1.3.1 Save the best model
    * 4.1.4 Four Conv layer
      * 4.1.4.1 Save the best model
  * 4.2 Conv-Flatten output VS. Dense output experiments
    * 4.2.1 One layer
      * 4.2.1.1 One dense layer output
        * 4.2.1.1.1 Save the best model
      * 4.2.1.2 One Conv-Flatten layer (7x1, 1x7) output
        * 4.2.1.2.1 Save the best model
    * 4.2.2 Two layers
      * 4.2.2.1 Two dense layers output
        * 4.2.2.1.1 Save the best model
      * 4.2.2.2 Two Conv-Flatten layers (7x7) output
        * 4.2.2.2.1 Save the best model
      * 4.2.2.3 Two Conv-Flatten layers (7x1, 1x7) output
        * 4.2.2.3.1 Save the best model
* **5\. Model Evaluation**
  * 5.1 Load test data
  * 5.2 ChannelSplitNet evaluation
    * 5.2.1 Dense output test
    * 5.2.2 Conv-Flatten test
  * 5.3 GoogLeNet evaluation
  * 5.4 AlexNet evaluation
    * 5.4.1 Dense output test
    * 5.4.2 Conv-Flatten output test
  * 5.5 VGG16 evaluation
    * 5.5.1 "Training from scratch" test
    * 5.5.2 Pretrained dense output test
    * 5.5.3 Pretrained Conv-Flatten output test
  * 5.6 ResNet50 evaluation
    * 5.6.1 Dense output test
    * 5.6.2 Conv-Flatten output test
  * 5.7 Inception V3 evaluation
    * 5.7.1 Dense output test
    * 5.7.2 Conv-Flatten output test
  * 5.8 Conv layer experiments evaluation
    * 5.8.1 No Conv layer test
    * 5.8.2 One Conv layer test
    * 5.8.3 Two Conv layers test
    * 5.8.4 Four Conv layers test
  * 5.9 Conv-Flatten output vs Dense output evaluation
    * 5.9.1 One dense layer output test
    * 5.9.2 One Conv-Flatten layer (7x7) output test
    * 5.9.3 One Conv-Flatten layer (7x1, 1x7) output test
    * 5.9.4 Two dense layer output test
    * 5.9.5 Two Conv-Flatten layer (7x7) output test
    * 5.9.6 Two Conv-Flatten layers (7x1, 1x7) output test
* **6\. Ensemble Model**

# 4. Run the Codes

**Files**:

* `CS_Project.ipynb`
* `ModelAnalysis.py`
* `Prediction.py`
* `ProcessData.py`

**Environments**: 

* `Python 3.7.9`
* `Keras 2.4.3`
* `GPU: NVDIA GeForce GTX 1050 Ti, Cuda: V10.1.243 `

*To run the codes, mask sure all **Files** listed above are in the same directory. `CS_Project.ipynb` is the main file to run the codes, other three files are imported to it. And make sure the dataset is already downloaded from the link provided in [2. Dataset](#2-dataset).

1. ***For data process***
   * Run *`0. Import Libraries`* to import the necessary libraries.
   * Run all the code in *`1. Data Processing`* for data re-allocation and data augmentation.
   * Run the code in `2.1.1, 2.1.2, 2.2.1, 2.2.2, 2.3.1, 2.3.2 ` (in  *`2. Load Data`*) to resize the data and save as pickle files, get prepared for training, validation and test.
2. ***For training the models***
   * Run *`0. Import Libraries`* to import the necessary libraries.
   * Run `2.1.3, 2.2.3` (in *`2. Load Data`*) to load training and validation data from pickle files to memory.
   * In *`3. Training with Different CNNs`*:
     * To train the **ChannelSplitNet with Dense output** model: Run `3.1.1, 3.1.2, 3.1.2.1` (in *`3.1 ChannelSplitNet`*) , for training, and run `3.1.2.5 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **ChannelSplitNet with Conv-Flatten output** model: Run `3.1.1, 3.1.3, 3.1.3.1` (in *`3.1 ChannelSplitNet`*) for training, and run `3.1.3.3 Save the best model`  to save the best model based on the best validation accuracy during training.
     * Run all the code in *`3.2 GoogLeNet`* to train the **GoogLeNet** model and save the best based on the best validation accuracy during training.
     * To train the **AlexNet with Dense output** model: Run `3.3.1, 3.3.2` (in *`3.3 AlexNet`*) for training, and run `3.3.2.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **AlexNet with Conv-Flatten output** model: Run `3.3.1, 3.3.3` (in *`3.3 AlexNet`*) for training, and run `3.3.3.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **VGG16 (trained from scratch)** model: Run `3.4.1, 3.4.2, 3.4.2.1` (in *`3.4 VGG16`*) for training, and run `3.4.2.5 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Pretrained VGG16 with Dense output** model: Run `3.4.3.1, 3.4.3.2` (in *`3.4 VGG16`*) for training, and run `3.4.3.2.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Pretrained VGG16 with Conv-Flatten output** model: Run `3.4.3.1, 3.4.3.3, 3.4.3.3.1` (in *`3.4 VGG16`*) for training, and run `3.4.3.3.3 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Pretrained ResNet50 with Dense output** model: Run `3.5.1, 3.5.2, 3.5.2.1` (in *`3.5 Pretrained ResNet50`*) for training, and run `3.5.2.3 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Pretrained ResNet50 with Conv-Flatten output** model: Run `3.5.1, 3.5.3, 3.5.3.1` (in *`3.5 Pretrained ResNet50`*) for training, and run `3.5.3.3` to save the best model based on the best validation accuracy during training.
     * To train the **Pretrained Inception V3 with Dense output** model: Run `3.6.1, 3.6.2` (in *`3.6 Pretrained Inception V3`*) for training, and run `3.6.2.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Pretrained Inception V3 with Conv-Flatten output** model: Run `3.6.1, 3.6.3` (in *`3.6 Pretrained Inception V3`*) for training, and run `3.6.3.1 Save the best model` to save the best model based on the best validation accuracy during training.
3. ***For experiments***
   * Run *`0. Import Libraries`* to import the necessary libraries.
   
   * Run `2.1.3, 2.2.3` (in *`2. Load Data`*) to load training and validation data from pickle files to memory.
   
   * in *`4. Experiments`*:
   
     * To train the **No Conv layer** model: Run `4.1.1` (in *`4.1 Conv layer experiments`*) for training, and run `4.1.1.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **One Conv layer** model: Run `4.1.2` (in *`4.1 Conv layer experiments`*) for training, and run `4.1.2.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Two Conv layer** model: Run `4.1.3` (in *`4.1 Conv layer experiments`*) for training, and run `4.1.3.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Four Conv layer** model: Run `4.1.4` (in *`4.1 Conv layer experiments`*) for training, and run `4.1.4.1 Save the best model` to save the best model based on the best validation accuracy during training.
   
     * To train the **One dense layer output** model: Run `4.2.1.1` (in *`4.2 Conv-Flatten output VS. Dense output experiments`*) for training, and run `4.2.1.1.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **One Conv-Flatten (7x1, 1x7) output** model: Run `4.2.1.2` (in *`4.2 Conv-Flatten output VS. Dense output experiments`*) for training, and run `4.2.1.2.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Two dense layers output** model: Run `4.2.2.1` (in *`4.2 Conv-Flatten output VS. Dense output experiments`*) for training, and run `4.2.2.1.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Two Conv-Flatten layers (7x7) output** model: Run `4.2.2.2` (in *`4.2 Conv-Flatten output VS. Dense output experiments`*) for training, and run `4.2.2.2.1 Save the best model` to save the best model based on the best validation accuracy during training.
     * To train the **Two Conv-Flatten layers (7x1, 1x7) output** model: Run `4.2.2.3` (in *`4.2 Conv-Flatten output VS. Dense output experiments`*) for training, and run `4.2.2.3.1 Save the best model` to save the best model based on the best validation accuracy during training.
4. ***For evaluation:***
   * Run *`0. Import Libraries`* to import the necessary libraries.
   * Run `2.3.3, 5.1` (in *`2. Load Data`* and *`5. Model Evaluation`*) to load test data from pickle files to memory.
   * In *`5. Model Evaluation`*:
     * To evaluate the **ChannelSplitNet with Dense output** model: Run `5.2.1` (in *`5.2 ChannelSplitNet evaluation`*) .
     * To evaluate the **ChannelSplitNet with Conv-Flatten output** model: Run `5.2.2` (in *`5.2 ChannelSplitNet evaluation`*).
     * To evaluate the **GoogLeNet** model: Run *`5.3 GoogLeNet evaluation`*.
     * To evaluate the **AlexNet with Dense output** model: Run `5.4.1` (in *`5.4 AlexNet evaluation`*).
     * To evaluate the **AlexNet with Conv-Flatten output** model: Run `5.4.2` (in *`5.4 AlexNet evaluation`*).
     * To evaluate the **VGG16 (Training from scratch)** model: Run `5.5.1` (in *`5.5 VGG16 evaluation`*).
     * To evaluate the **Pretrained VGG16 with Dense output** model: Run `5.5.2` (in *`5.5 VGG16 evaluation`*).
     * To evaluate the **Pretrained VGG16 with Conv-Flatten output** model: Run `5.5.3` (in *`5.5 VGG16 evaluation`*).
     * To evaluate the **Pretrained ResNet50 with Dense output** model: Run `5.6.1` (in *`5.6 ResNet50 evaluation`*).
     * To evaluate the **Pretrained ResNet50 with Conv-Flatten output** model: Run `5.6.2` (in *`5.6 ResNet50 evaluation`*).
     * To evaluate the **Pretrained Inception V3 with Dense output** model: Run `5.7.1` (in *`5.7 Inception V3 evaluation`*).
     * To evaluate the **Pretrained Inception V3 with Conv-Flatten output** model: Run `5.7.2` (in *`5.7 Inception V3 evaluation`*).
     * To evaluate the **Experiment - No Conv layer** model: Run `5.8.1` (in *`5.8 Conv layer experiments evaluation`*).
     * To evaluate the **Experiment - One Conv layer** model: Run `5.8.2` (in *`5.8 Conv layer experiments evaluation`*).
     * To evaluate the **Experiment - Two Conv layers** model: Run `5.8.3` (in *`5.8 Conv layer experiments evaluation`*).
     * To evaluate the **Experiment - Four Conv layers** model: Run `5.8.4` (in *`5.8 Conv layer experiments evaluation`*).
     * To evaluate the **Experiment - One dense layer output** model: Run `5.9.1` (in *`5.9 Conv-Flatten output vs Dense output evaluation`*).
     * To evaluate the **Experiment - One Conv-Flatten layer (7x7) output** model: Run `5.9.2` (in *`5.9 Conv-Flatten output vs Dense output evaluation`*).
     * To evaluate the **Experiment - One Conv-Flatten layer (7x1, 1x7) output** model: Run `5.9.3` (in *`5.9 Conv-Flatten output vs Dense output evaluation`*).
     * To evaluate the **Experiment - Two dense layer output** model: Run `5.9.4` (in *`5.9 Conv-Flatten output vs Dense output evaluation`*).
     * To evaluate the **Experiment - Two Conv-Flatten layer (7x7) output** model: Run `5.9.5` (in *`5.9 Conv-Flatten output vs Dense output evaluation`*).
     * To evaluate the **Experiment - Two Conv-Flatten layers (7x1, 1x7) output** model: Run `5.9.6` (in *`5.9 Conv-Flatten output vs Dense output evaluation`*).
5. ***For ensemble model:***
   * Run *`0. Import Libraries`* to import the necessary libraries.
   * Run `2.3.3, 5.1` (in *`2. Load Data`* and *`5. Model Evaluation`*) to load test data from pickle files to memory.
   * Run *`6. Ensemble Model`* to make prediction with ensemble model.
