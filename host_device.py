import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import cv2

# Data channel
frame_info_channel = None
display_frame_interval = 30 
frame_counter = 0

class WebRTCConnection:
    def __init__(self):
        self.pc = RTCPeerConnection()
        self.ws = None  # Initialize WebSocket client attribute

    async def connect(self, websocket_url):
        async with websockets.connect(websocket_url) as ws:
            self.ws = ws  # Store the WebSocket client
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
        try:
            while True:
                frame = await track.recv()
                frame_counter += 1
                if frame_counter % display_frame_interval == 0:
                    print(frame_counter)
                    cv2.imwrite(f"./frames/frame_{frame_counter}.jpg", video_frame_to_ndarray(frame))

                frameshape = await process_video_frame(frame)
                if frame_info_channel and frame_info_channel.readyState == "open":
                    frame_info_channel.send(frameshape)
                    print(frameshape)
                else:
                    print(frame_info_channel)
                    #print("[RC] Error: frame_info_channel") 

        except Exception as e:
            print(f"[RC] Error receiving frame: {e}")

def video_frame_to_ndarray(video_frame):
    # Convert VideoFrame to RGB NumPy array
    img = video_frame.to_ndarray(format='bgr24')
    return img

def hard_frame_process(frame):
    nframe = video_frame_to_ndarray(frame)
    gray_frame = cv2.cvtColor(nframe, cv2.COLOR_BGR2GRAY)
    shape = gray_frame.shape
    frameshape = f"{shape[0]},{shape[1]}"
    return frameshape

async def process_video_frame(frame):
    # Offload the blocking operation to a separate thread
    processed_frame = await asyncio.to_thread(hard_frame_process, frame)
    return processed_frame

# Running the signaling process
async def main():
    connection = WebRTCConnection()
    await connection.connect("ws://localhost:8080")

asyncio.run(main())