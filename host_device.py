import subprocess
import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import cv2
import sys
import time

sys.path.insert(0, '/home/rafael2ms/Dev/crack_seg_yolov7/yolov7/seg/segment')

from frameparameter_E_fast import load_model, run_seg

#============================================================================

import aiohttp
from aiohttp import web
import aiohttp_session
import socketio
import base64
import cv2

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)
n_print = 0

async def index(request):
    return web.FileResponse('index.html')

app.router.add_get('/', index)

frame_info_channel = None
#display_frame_interval = 20
frame_counter = 0

model, stride, names, device = load_model(weights='/home/rafael2ms/Dev/crack_seg_yolov7/yolov7/pretrained/best.pt')
save_image = False
frame_mask = None
frame_pca = None


class WebRTCConnection:
    def __init__(self, websocket_url):
        self.pc = RTCPeerConnection()
        self.websocket_url = websocket_url  # Store the WebSocket URL
        self.ws = None  # Initialize WebSocket client attribute

    async def connect(self):#, websocket_url
        async with websockets.connect(self.websocket_url) as ws:
            print(f"Connecting to WebSocket at {self.websocket_url}")
            self.ws = ws  # Store the WebSocket client
            #await self.handle_websocket_communication()
            self.pc.on("connectionstatechange", self.on_connection_state_change)
            self.pc.on("iceconnectionstatechange", self.on_ice_connection_state_change)
            self.pc.on("datachannel", self.on_data_channel)
            self.pc.on("track", self.on_track)
            await self.handle_messages()

    async def handle_offer(self, offer):
        print("[RC] Setting remote description (offer)")
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self.ws.send(json.dumps({"answer": {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}}))

    async def on_connection_state_change(self):
        print(f"[RC] Connection State has changed to {self.pc.connectionState}")
        if self.pc.connectionState == "failed":
            print("[RC] Connection failed.")

    async def on_ice_connection_state_change(self):
        print(f"[RC] ICE Connection State has changed to {self.pc.iceConnectionState}")

    async def handle_messages(self):
        async for message in self.ws:
            data = json.loads(message)
            if "candidate" in data:
                print("[RC] Adding ICE candidate")
                await self.pc.addIceCandidate(RTCIceCandidate(**data["candidate"]))
            elif "offer" in data:
                await self.handle_offer(data["offer"])

    def on_data_channel(self, channel):
        global frame_info_channel

        if channel.label == "chat":
            frame_info_channel = channel

        print("[RC] Data channel received:", channel.label)
        channel.on("open", lambda: print("[RC] Data channel opened"))
        channel.on("message", lambda message: self.on_message(channel, message))

    def on_message(self, channel, message):
        print(f"[RC] Received message: {message}")
        if message == "Request frame info" and channel.readyState == "open":
            frame_info = "test"  # Example, replace with actual frame processing
            channel.send(frame_info)
        elif channel.readyState == "open":
            channel.send(f"Echoing: {message}")

    async def on_track(self, track):
        print("[RC] Track received:", track.kind)
        asyncio.create_task(self.receive_frames(track))

    async def receive_frames(self, track):
        global frame_counter
        global n_print

        global save_image
        global frame_mask
        global frame_pca


        try:
            frame_data = 1
            web_img_delay = time.perf_counter()


            while True:
                frame = await track.recv()
                
                if (time.perf_counter() - web_img_delay) < 1:
                    continue

                web_img_delay = time.perf_counter()

                if (frame):
                    nframe = video_frame_to_ndarray(frame)
                    frame_counter += 1

                    frame_data = await process_video_frame(nframe)
                    
                    if frame_info_channel and frame_info_channel.readyState == "open":
                            frame_info_channel.send(f'{frame_data}')
                            ###print(frame_data)
                    else:
                        print(f"[RC] Error: {frame_info_channel}") 

                    if f'{frame_data}' != '(0, 0)':
                        n_print += 1


        except Exception as e:
            print(f"[RC] Error receiving frame: {e}")

def video_frame_to_ndarray(video_frame):
    # Convert VideoFrame to RGB NumPy array
    img = video_frame.to_ndarray(format='bgr24')
    return img

def hard_frame_process(nframe):
    global model
    global stride
    global names
    global device 
    global save_image
    global frame_mask
    global frame_pca
    
    save_image = True

    out = run_seg(model=model, stride=stride, names=names, device=device, im0=nframe,save_img=save_image)
    command = (out[0], out[1])

    save_image = False

    if out[2] is not None:
        frame_mask = out[2]
        cv2.imwrite(f'/home/rafael2ms/Dev/oakmax_webrtc/final_pa/frame_mask_local.jpg', frame_mask)

    if out[3] is not None:
        frame_pca = out[3]
        cv2.imwrite(f'/home/rafael2ms/Dev/oakmax_webrtc/final_pa/frame_pca_local.jpg', frame_pca)
    
    return command

async def process_video_frame(frame):
    # Offload the blocking operation to a separate thread
    processed_frame = await asyncio.to_thread(hard_frame_process, frame)
    return processed_frame


async def main():

    # Setup and connect WebRTC in a non-blocking way
    #connection = WebRTCConnection("ws://localhost:8080")
    connection = WebRTCConnection("ws://172.21.1.122:8080")
    #connection = WebRTCConnection("ws://192.168.169.191:8080")
    #connection = WebRTCConnection("ws://160.85.114.134:8080")
    webrtc_task = asyncio.create_task(connection.connect())

    await asyncio.gather(webrtc_task)

if __name__ == "__main__":
    asyncio.run(main())
