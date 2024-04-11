# Projektarbeit
*"Realtime communication using 5G to enable deep learning on the cloud for robotic applications."*

## Motivation
- Machine Learning models often require substantial computational resources. In robotics, robust hardware typically equates to increased volume, weight, and power consumption.
- Off-board execution of deep learning models represents a significant advancement in robotics. It facilitates enhancements and customization of the control system while also boosting its autonomous decision-making capabilities.
- A crucial requirement for these technologies is the development of a system capable of managing input, processing, and decision-making in real-time, as perceived by humans. This system must have latency low enough to meet the specific requirements of the application.

## Project (Main) Dependencies
* Ubuntu 22.04
* Python 3.10
* DepthAI
> Follow the installation instructions on the official website: https://docs.luxonis.com/projects/api/en/latest/install/

## Installing Requirements
Create an environment using:
```
$ conda create --name <env> --file requirements_conda.txt
```
Alternatively, using pip:
```
$ conda create --name <env> python=3.10 pip
$ pip install -r requirements.txt
```
> Note: Both requirements files are available in this repository.

# Project Initialization
### 1. Navigate to the project repository
```
$ cd /path/to/projectarbeite
```
### 2. Activate the virtual environment
```
$ conda activate <env>
```
> Replace `<env>` with your virtual environment's actual name.

### 3. Setup the address
For debugging purposes, the server should be hosted and accessed through "localhost" (0.0.0.0) on port 8080.

```
# host_device.py:
await connection.connect("ws://localhost:8080")
```
```
# remote_device.py:
async with websockets.connect("ws://localhost:8080") as ws:
```
```
# signaling_server.py:
start_server = websockets.serve(handler, "localhost", 8080)
```

#### Note: When working with two different devices, the host machine's IP address needs to be provided

```
("ws://localhost:8080")        -> ("ws://192.168.X.X:8080")
(handler, "localhost", 8080)   -> (handler, "ws://192.168.X.X", 8080)
```
> Check your own IP address by running `$ ifconfig` or similar commands.

### 4. Run the scripts 
a) Host device
```
$ python signaling_server.py
$ python host_device.py
```

b) Remote device

With the Oak camera connected to the device, run:
```
$ python remote_device.py
```
> The order of script execution should be followed as shown above, even in setups involving only one device.


# Project Planning and Milestones *(coming soon)*
