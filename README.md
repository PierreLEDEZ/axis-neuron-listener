# Axis Neuron Listener

## Table of contents
* [General info](#general-info)
* [How To Use](#how-to-use)
* [Axis Neuron Software](#axis-neuron-software)

## General info

This is TCP listener in Python used to receive motion capture data from the axis neuron software. Once received, the motion data can be used in real-time by another module.

## How To Use

```console
# clone the repo
$ git clone URL REPO

# change the working directory to axis_neuron_listener
$ cd axis_neuron_listener

# install the requirements
$ pip install -r requirements.txt

# run the listener
$ python main.py
```


## Axis Neuron Software

### Check current configuration
This listener use the local server of the Axis Neuron software. To check the server configuration, please follow these few instructions.
1. Open **Axis Neuron** software
2. Check in `File>Settings>Output Format` if Rotation order is **YXZ** and if **Displacement** is checked.
3. Check in `File>Settings>Broadcasting` if BVH section is **enabled** and if the Format is **Binary**
The default port is **7001** but you can change it.
4. Check in `File>Settings>Broadcasting` if **TCP** is selected in TCP/UDP section

### Request format

When a client is connected to the server, the requests it receives have the following format:  

```bash
+------------------+-----------------------------+
|      HEADER      |         MOTION PART         |           
| 64 bytes length  |      1416 bytes length      |
+------------------+-----------------------------+
```

#### Header

The Header always starts with 0xDDFF and ends with 0xEEFF. Between this 2 markers, there is a lot of informations:
 - Data Version
 - Data Count
 - With Displacement
 - With Reference
 - Avatar Index
 - Avatar Name
 - Frame Index
 - Reserved
 - Reserved1
 - Reserved2

#### Motion Part

The BVH data obtained from the **Perception Neuron 32 v2** equipment includes 59 joints.  

If the displacement is enabled, a BVH file will contain 6 coordinates (Tx, Ty, Tz, Ry, Rx, Rz) for each joint, otherwise 3 coordinates (Ry, Rx, Rz).  

For one frame, a file with displacement counts 59 joints * 6 coordinates -> 354 floating numbers (59 joints * 3 coordinates -> 177 floating numbers without displacement).  

In the motion part, each floating number is encoded in 4 bytes. (354 * 4 = 1416 bytes or 177 * 4 = 708 bytes without displacement)


