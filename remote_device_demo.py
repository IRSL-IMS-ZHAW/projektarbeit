import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
#from transformators import DepthAIVideoTransformTrack, OptionsWrapper
import time


import traceback
import numpy as np
from aiortc import VideoStreamTrack
import cv2
from av import VideoFrame
import depthai as dai
import blobconverter


start_0 = None
frame_sent = 0
interval = []
interval2 = []
last_frame = 0
msg_received = 0
frame_counter = 0
frame_counter2 = 0
frame_counter3 = 0
n_frame = 0
n_print = 0

white_frame = cv2.imread("/home/rafael2ms/Dev/oakmax_webrtc/final_pa/white_frame.jpg")
white_bool = False
white_time = 0
FPS = 20
white_interval = 2*FPS
t_zero = time.perf_counter()
recv_time = 0

class OptionsWrapper:
    def __init__(self,
                 #camera_type='rgb',
                 width=3840,
                 height=2160,
                 #nn='',
                 #mono_camera_resolution='4_K',
                 #median_filter='KERNEL_7x7',
                 #subpixel='',
                 #extended_disparity=''
                 ):
        #self.camera_type = camera_type
        self.width = width
        self.height = height

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, options, fps=10): # application, pc_id
        super().__init__(fps=fps)  # don't forget this!
        self.dummy = False
        #self.application = application
        #self.pc_id = pc_id
        self.options = options
        self.frame = None

    
    async def get_frame(self):
        raise NotImplementedError()

    async def return_frame(self, frame):
        global frame_counter2
        frame_counter2 += 1
        start_frame = time.perf_counter()
        pts, time_base = await self.next_timestamp()
        mid_frame = time.perf_counter()
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        end_frame = time.perf_counter()    
        return new_frame

    async def dummy_recv(self):
        print(self.options)
        frame = np.zeros((self.options.height, self.options.width, 3), np.uint8)
        y, x = frame.shape[0] / 2, frame.shape[1] / 2
        left, top, right, bottom = int(x - 50), int(y - 30), int(x + 50), int(y + 30)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "ERROR", (left, int((bottom + top) / 2 + 10)), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (255, 255, 255), 1)
        return await self.return_frame(frame)

    async def recv(self):
        global frame_counter3
        global recv_time
        global last_frame
        global white_time
        global white_bool
        global FPS

        if self.dummy:
            return await self.dummy_recv()
        try:  
            #start_frame = time.perf_counter()
            #print(".")
            frame = await self.get_frame()
            #mid_frame = time.perf_counter()
            r = await self.return_frame(frame)
            #end_frame = time.perf_counter()
            if frame_counter3 == 0:
                shape = frame.shape
                frameshape = f"{shape[0]},{shape[1]}"
                #print(f'frame_type = {type(frame)}')
                #print(f'r_type = {type(r)}')
                print(f"FPS: {FPS} || frameshape = {frameshape}")
                frame_counter3 += 1

            if white_bool is False:
                white_time = time.perf_counter()
                white_bool = True

            #print(f"Last frame = {time.perf_counter() - last_frame}")
            #last_frame = time.perf_counter() 

            return r
        except:
            print(traceback.format_exc())
            print('Switching to dummy mode...')
            self.dummy = True
            return await self.dummy_recv()


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


class DepthAIVideoTransformTrack(VideoTransformTrack):
    def __init__(self,  options, fps=10): # application, pc_id,
        global white_frame
        super().__init__(options, fps=fps) # application, pc_id
       
        num = 1
        den = 5

        self.options.height = int((num/den)*self.options.height)
        self.options.width  = int((num/den)*self.options.width)

        white_frame = cv2.resize(white_frame, (self.options.height, self.options.width))

        self.frame = np.zeros((self.options.height, self.options.width, 3), np.uint8)
        self.frame[:] = (0, 0, 0)
        #self.detections = []
        self.pipeline = dai.Pipeline()
        self.pipeline.setXLinkChunkSize(0)
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)

        controlIn = self.pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName("control")
        controlIn.out.link(self.camRgb.inputControl)

        self.xoutRgb.setStreamName("rgb")

        # Properties
        self.camRgb.setPreviewSize(self.options.width, self.options.height)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        print(f"FPS before: {self.camRgb.getFps()}")
        self.camRgb.setFps(FPS)
        print(f"FPS after: {self.camRgb.getFps()}")
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        self.camRgb.setIspScale(num, den)  # Scale down from 4K to 720P

        # Linking
        self.camRgb.preview.link(self.xoutRgb.input)
        self.device = dai.Device(self.pipeline)
        controlQueue = self.device.getInputQueue(name='control')
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(200)#0 to 255#
        controlQueue.send(ctrl)

        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        #if self.nn is not None:
        #    self.qDet = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.device.startPipeline()
    
    async def get_frame(self):
        global white_frame
        global white_bool
        global white_time
        global t_zero

        #global start_0
        #global frame_sent
        global last_frame
        global frame_counter
        global white_interval
        global n_frame

        frame = self.qRgb.tryGet()
        #self.frame = white_frame

        if frame is not None:
            self.frame = frame.getCvFrame()

            # OBS: THIS WAS COMMENTED OUT ON THE 26TH OF JUNE        
            # ========================================
            #      Functional White Frame Control
            # ========================================
            #self.frame = white_frame
            #n_frame += 1
            #if n_frame > white_interval:
            #    white_bool = False
            #    self.frame = frame.getCvFrame()
            #    n_frame = 0
            #    print("frame not none\n")
            #else:
            #    print("white frame\n")
            
        return self.frame

    def stop(self):
        super().stop()
        del self.device


async def setup_data_channel(pc):
    
    data_channel = pc.createDataChannel("chat")
    @data_channel.on("open")
    def on_data_channel_open():
        print("Data channel is open")
        # Test sending a message
        data_channel.send("Request frame info")

    @data_channel.on("message")
    def on_data_channel_message(message):
        global interval
        global msg_received
        global white_bool
        global white_time
        global recv_time
        global n_print
        
        #print(f"Message from receiver: {message}")

        if message == 'test': #!=

            print(f"Test message from receiver: {message}")
            #print(f'Elapsed Time (mm) = {(end_0-start_0)}')
        
        if message != '(0, 0)':
            t = time.perf_counter()
            #print(f"Rec: {message} \t @ {t} \t E2E = {t- white_time} \t dif_recv = {recv_time - white_time}")
            # =============== 26june2024 ==========================
            #print(f"{n_print} @ T0 = {white_time:5f} \t TF = {t:5f} \t E2E = {(t- white_time):5f}")
            n_print += 1

        # Process the received frame information
        #if (message) == "":
        #     data_channel.send("Stop current task!")
        
    @data_channel.on("closing")
    def on_data_channel_closing():
        print("Data channel is closing")

    @data_channel.on("close")
    def on_data_channel_close():
        print("Data channel is closed")
    
    return data_channel

async def signaling():
    global FPS
    #async with websockets.connect("ws://localhost:8080") as ws:
    #async with websockets.connect("ws://192.168.169.191:8080") as ws:172.21.1.61
    async with websockets.connect("ws://172.21.1.173:8080") as ws:
        pc = RTCPeerConnection()
        # Initialize options directly with default values
        options = OptionsWrapper()
        pc.addTrack(DepthAIVideoTransformTrack(options,FPS))

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            print(f"Connection State has changed to {pc.connectionState}")
            if pc.connectionState == "failed":
                print("Connection failed.")

        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            print(f"ICE Connection State has changed to {pc.iceConnectionState}")

        # Handle ICE candidates
        @pc.on("icecandidate")
        async def on_icecandidate(event):
            candidate = event.candidate
            if candidate:
                print("Sending ICE candidate")
                await ws.send(json.dumps({"candidate": event.candidate}))
            else:
                print("All local ICE candidates have been sent")
        
        # Setup data channel and start sending messages
        data_channel = await setup_data_channel(pc)

        # Create offer, set local description, and send offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        print("Sending offer")
        await ws.send(json.dumps({"offer": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}}))

        # Handle answer
        answer = json.loads(await ws.recv())
        if "answer" in answer:
            print("Receiving answer")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["answer"]["sdp"], type=answer["answer"]["type"]))

        await asyncio.Event().wait()  # Keep the event loop running indefinitely

asyncio.run(signaling())
