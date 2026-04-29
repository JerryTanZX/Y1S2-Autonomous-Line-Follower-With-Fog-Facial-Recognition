# Autonomous Line-Following Robot with Fog-Based Facial Recognition

This repository contains the source code for the EEEE1027 Applied Electrical and Electronic Engineering Construction Project. 

The system features a multi-processing Python architecture running on a Raspberry Pi 4 for real-time PID motor control and multi-modal symbol recognition (ORB, contours, and HSV color tracking). It also includes a fog computing server application designed to run on a local Mac, which executes low-latency biometric facial recognition over a TCP socket.

## Hardware Requirements
* Raspberry Pi 4 Model B
* Raspberry Pi Camera Module 2
* L298N Dual H-Bridge Motor Driver
* Differential-drive robot chassis

## Repository Structure
* `pi_robot_driver.py`: The main multiprocessing script running on the Raspberry Pi. It handles PID line following, symbol detection, and the state machine.
* `mac_face_gate_server_v1.py`: The fog server script running on the local network. It receives compressed JPEG frames from the Pi and performs HOG-based face detection.
* `mac_face_test_v1.py`: A local utility script for testing the facial recognition pipeline using a built-in webcam.
* `known_faces/`: Directory for storing reference images for enrolled subjects. 
* `*.png`: Target symbols used by the ORB and Hu Moments detection algorithms.

## Setup and Installation

### 1. Fog Server (Mac)
Ensure Python 3 is installed, then install the required biometric libraries using your terminal:

`pip install opencv-python numpy dlib face_recognition`

To enroll a user, create a folder inside the `known_faces` directory with the user's name (e.g., `known_faces/Jerry/`) and place a clear photograph inside. *Note: Do not commit actual biometric photographs or the generated .face_cache.pkl file to version control.*

Run the server:

`python3 mac_face_gate_server_v1.py`

### 2. Raspberry Pi
Ensure the Raspberry Pi is running a modern OS with the picamera2 stack and gpiozero installed:

`pip install opencv-python numpy gpiozero`

Run the main driver:

`python3 pi_robot_driver.py`

## Security Note
For privacy and compliance with ethical data standards, all biometric reference images and generated `.pkl` encoding caches have been excluded from this public repository via `.gitignore`.
