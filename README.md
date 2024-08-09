# Demonstration Instructions

For this demonstration, the execution sequence is similar to previous examples, with one crucial addition: **You must open a browser and navigate to `http://localhost:8000` before executing `remote_device_demo.py`.**

### Step-by-Step Guide

1. **Start the Web Interface:**
   - Open your preferred web browser.
   - Navigate to `http://localhost:8000`.
   
   This page will display:
   - The path that the algorithm should track.
   - The current camera view.
   - The layers added by the AI model.

   **Note:** This page is optimized for use with two monitors stacked vertically in a landscape orientation. To view the full content, extend the browser window from the top monitor to the bottom one.

   *Example of the page layout:*
   ![Example Page Layout](/etc/localhost_view_example.png)
   
2. **Run the Signaling Server Script:**
   - In the terminal, execute the following command to start the signaling server:
     ```bash
     python signaling_server.py
     ```

3. **Run the Host Device Script:**
   - In the terminal, execute the following command to start the host device:
     ```bash
     python host_device_execution.py
     ```

   - You should see output similar to the following:
     ```bash
     YOLOv5 🚀 2024-4-17 Python-3.10.14 torch-2.2.2+cu121 CUDA:0 (NVIDIA GeForce RTX 2060, 5918MiB)
     
     Fusing layers...
     Model summary: 325 layers, 37853264 parameters, 0 gradients, 141.9 GFLOPs
     Model Loaded Successfully
     HTTP server running on http://localhost:8000
     Connecting to WebSocket at ws://172.21.1.173:8080
     ```

   - Pay particular attention to the line:
     ```plaintext
     HTTP server running on http://localhost:8000
     ```
     This confirms that the web interface is active.

4. **Position the Camera:**
   - Position the camera in front of the bottom monitor. This monitor should simulate the robot's view (represented by a green rectangle).
   - The top monitor will display the crack path that the robot needs to navigate.

5. **Run the Remote Device Script:**
   - Now, execute the `remote_device_demo.py` script to begin the demonstration.
