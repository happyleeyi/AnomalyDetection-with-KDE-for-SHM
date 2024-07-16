# AnomalyDetection using KDE (Kernel Density Estimation) for SHM

## Introduction

The research to suggest a method to find whether and where the defect happens in the building on real-time.

We use the KDE method to find the location of the defect in this section.

## Dataset

3-story shake table test data with an undamaged case and several damaged cases

![image](https://github.com/happyleeyi/DeepSVDD/assets/173021832/f5b6d5f1-b504-4c4a-bdbe-be64fa2c6588)

(open-source by Engineering Institute at Los Alamos National Laboratory)

## Method

### What is KDE?

![KDE](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/2b25b0cc-cc22-42bb-948d-19ec51c0755b)

Method to estimate the probability density function from point distribution

![KDE - 복사본](https://github.com/user-attachments/assets/6fddb088-b737-40a9-b763-1c2cd7e1f251)


We can use KDE in anomaly detection to determine whether the corresponding data is normal or abnormal.

First of all, we can make a probability density function with KDE using normal data.

And then we can obtain the probability of the data that we want to know about from the calculated density function.

If the obtained probability of the data from density function is smaller than the threshold probability, the corresponding data is determined as abnormal.

### 1. data processing
1. prepare dataset (we use 3F shake table test dataset, 8 accelerometers per floor, a total of 24 accelerometers)
2. process dataset (we downsample and cut the training dataset and concatenate 8 acc data per each floor -> training dataset : (8, 512) for one data)

### 2. Training
![KDE pic](https://github.com/user-attachments/assets/12592825-ac29-429b-9898-c40a6bb94e9f)

1. training autoencoder
2. copy the encoder part of the autoencoder and use it as kernel to the hyperspace (we need to place the data in a multi-dimensional space to use KDE)
3. map the training data using the encoder to hyperspace (you can choose the representation dimension of hyperspace as you want)
4. train 3 different KDE models for every 3 floors, using the same kernel function

### 3. Test
1. cut the test data for 3 floors (test dataset : (24, 512) -> (8,512) * 3 for one data)
2. put the data into the encoder and map to hyperspace for every floor. (total 3 datas per one data in this dataset)
3. if every data is determined as normal by KDE algorithm, the building is on normal state.
4. however, if several datas are determined as abnormal by KDE algorithm, the floor that the calculated probability by KDE is smallest is determined to be damaged.

 ## Result

### 1. Accuracy of AD using KDE depending on bandwidth (parameter to determine how sharp the probability density function is)
![image](https://github.com/user-attachments/assets/1d5375d8-c3fc-4ad7-9ea9-d3ad3dabf265)

The graph shows that the accuracy increases as the representation dimension of the latent space of hyperspace increases.

Also, the accuracy increases as the bandwidth of KDE decreases.

### 2. Accuracy of AD using KDE depending on representation dimension (maximum accuracy)
![image](https://github.com/user-attachments/assets/12056663-4ce5-4ed1-9f99-414533928d36)


![image](https://github.com/user-attachments/assets/7b37d21b-8570-48fb-8309-22fed4287700)

The accuracy increases as the representation dimension of the latent space of hyperspace increases.
