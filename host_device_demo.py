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
#=============== 26june2024 ==========================
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

        #=============== 26june2024 ==========================
        global save_image
        global frame_mask
        global frame_pca

        

        try:
            crack_pathway_jpg = 'crack_path.jpg'  # Path to your image
            crack_pathway = cv2.imread(crack_pathway_jpg)
            height, width = crack_pathway.shape[:2]

            # Initial rectangle coordinates
            # x1, y1, x2, y2 
            #rect = (0, 930, 96, 1002) #(72 x 96)
            rect = (80, 850, 176, 922) #(72 x 96)
            side = True
            frame_data = 1
            web_img_delay = time.perf_counter()
            initial_delay = time.perf_counter()

            my_buffer = []
            #while (time.perf_counter() - initial_delay) < 5:
            #    pass

            while True:
                frame = await track.recv()
                if (frame):
                    #print(f'type: {type(frame)}')
                    nframe = video_frame_to_ndarray(frame)
                    frame_counter += 1
                    start_time = time.perf_counter()

                    start_ml = time.perf_counter()
                    frame_data = await process_video_frame(nframe)
                    end_ml = time.perf_counter()

                    #=============== 26june2024 ==========================
                    
                    if (time.perf_counter() - web_img_delay) > 0.05:
                        
                        #=========================================================
                        # Crop the magnified area
                        #=========================================================
                        x1, y1, x2, y2 = rect

                        #print(f'x1, y1, x2, y2 = {x1}, {y1}, {x2}, {y2}')
                        magnified_area = crack_pathway[y1:y2, x1:x2]

                        # Create a copy of the image and draw a rectangle
                        pathway_copy = crack_pathway.copy()
                        cv2.rectangle(pathway_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #=========================================================

                        #print("nframe type:" + str(type(nframe)))
                        #emit_image(nframe, 'image')

                        _, buffer = cv2.imencode('.jpg', pathway_copy)
                        encoded_frame = base64.b64encode(buffer).decode('utf-8')
                        await sio.emit('new_frame', {'crack_path': 'data:image/jpeg;base64,' + encoded_frame})
                        #cv2.imwrite(f'pathway_copy.jpg', pathway_copy)

                        _, buffer = cv2.imencode('.jpg', magnified_area)
                        encoded_frame = base64.b64encode(buffer).decode('utf-8')
                        await sio.emit('new_frame', {'image': 'data:image/jpeg;base64,' + encoded_frame})

                        #if (time.perf_counter() - initial_delay) > 5:
                            #my_buffer.append(encoded_frame)
                            #if len(my_buffer) > 100:
                            #await sio.emit('new_frame', {'image': 'data:image/jpeg;base64,' + my_buffer[0]})
                            #   my_buffer.pop(0)
                            #cv2.imwrite(f'magnified_area.jpg', magnified_area)
                            
                            #cmd = 'right'
                            #print(f'frame_data = {frame_data}')
                            #cmd=(0.1, 0.1)
                        
                           
                        rect = update_position(frame_data, rect, width, height)

                            #my_buffer.append(update_position(frame_data, rect, width, height))
                            #if len(my_buffer) > 2:
                            #    rect = my_buffer[0]
                            #    my_buffer.pop(0)
                            

                        #=============== 27june2024 ==========================
                        if frame_mask is not None:
                            #emit_image(frame_mask, 'frame_mask')

                            #print("frame_mask type:" + str(type(frame_mask)))
                            # Encode and send the mask frame
                            _, buffer = cv2.imencode('.jpg', frame_mask)
                            encoded_frame = base64.b64encode(buffer).decode('utf-8')
                            await sio.emit('new_frame', {'frame_mask': 'data:image/jpeg;base64,' + encoded_frame})
                        
                        if frame_pca is not None:
                            #emit_image(frame_pca, 'frame_pca')

                            #print("frame_mask type:" + str(type(frame_mask)))
                            # Encode and send the mask frame
                            _, buffer = cv2.imencode('.jpg', frame_pca)
                            encoded_frame = base64.b64encode(buffer).decode('utf-8')
                            await sio.emit('new_frame', {'frame_pca': 'data:image/jpeg;base64,' + encoded_frame})

                            #print(f'frame_pca.shape[:2] = {frame_pca.shape[:2]}')

                        #_, buffer = cv2.imencode('.jpg', magnified_area)
                        #encoded_frame = base64.b64encode(buffer).decode('utf-8')
                        #await sio.emit('new_frame', {'image': 'data:image/jpeg;base64,' + encoded_frame})

                        #_, buffer = cv2.imencode('.jpg', pathway_copy)
                        #encoded_frame = base64.b64encode(buffer).decode('utf-8')
                        #await sio.emit('new_frame', {'crack_path': 'data:image/jpeg;base64,' + encoded_frame})
                        #=========================================================


                        #await sio.emit('new_frame', buffer.tobytes())  # Sending binary data directly
                        web_img_delay = time.perf_counter()
                        save_image = True
                        #=========================================================

                    if frame_info_channel and frame_info_channel.readyState == "open":
                            frame_info_channel.send(f'{frame_data}')
                            ###print(frame_data)
                    else:
                        #print(frame_info_channel)
                        print(f"[RC] Error: {frame_info_channel}") 

                    if f'{frame_data}' != '(0, 0)':
                        end_time = time.perf_counter()
                        #print(f"Rec: {message} \t @ {t} \t E2E = {t- white_time} \t dif_recv = {recv_time - white_time}")
                        #=============== 26june2024 ==========================
                        # print(f"{n_print} @ ST:{start_time:5f} \t ET:{end_time:5f} \t ML = {(end_ml - start_ml):5f} \t TOT = {(end_time - start_time):5f}") # WEB = {(end_web - start_time):5f}
                        n_print += 1


        except Exception as e:
            print(f"[RC] Error receiving frame: {e}")

# Function to emit the image to the webserver
#async def emit_image(image, img_str):
#    _, buffer = cv2.imencode('.jpg', image)
#    encoded_frame = base64.b64encode(buffer).decode('utf-8')
#    await sio.emit('new_frame', {img_str: 'data:image/jpeg;base64,' + encoded_frame})

# Update position based on command
def update_position(cmd, rect, max_width, max_height, step=30):
    x1, y1, x2, y2 = rect
    step_x, step_y = cmd

    ## this went to frameparameter_E_fast
    ## Makes more sense to work on the command only there
    step_x *= 4
    step_y *= 4

    x1 += step_x
    x2 += step_x
    
    y1 += step_y
    y2 += step_y
    '''
    if cmd == 'left' and x1 > step:
        x1 -= step
        x2 -= step

    elif cmd == 'right' and x2 < max_width - step:
        x1 += step
        x2 += step

    elif cmd == 'up' and y1 > step:
        y1 -= step
    elif cmd == 'down' and y2 < max_height - step:
        y1 += step
 
    
    # Ensure the rectangle does not move out of image bounds
    x1 = max(0, min(x1, max_width - (x2 - x1)))
    y1 = max(0, min(y1, max_height - (y2 - y1)))
    x2 = x1 + (x2 - x1)
    y2 = y1 + (y2 - y1)
       '''

    if (x1 < 0):
        x1 = 0
        x2 = 96
        
    if (x2 > max_width):
        x2 = max_width
        x1 = max_width - 96

    if (y1 < 0):
        y1 = 0
        y2 = 72
        
    if (y2 > max_height):
        y2 = max_height
        y1 = max_height - 72

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    return (x1, y1, x2, y2)


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
    
    #=============== 27june2024 ==========================
    save_image = True

    # out = command_x,command_y, im0, img
    out = run_seg(model=model, stride=stride, names=names, device=device, im0=nframe,save_img=save_image)
    command = (out[0], out[1])
    #print(f'hard_frame_process = {command},{out[0]},{out[1]}')

    save_image = False

    #=============== 27june2024 ==========================
    if out[2] is not None:
        frame_mask = out[2]
        #cv2.imwrite(f'/home/rafael2ms/Dev/oakmax_webrtc/final_pa/frame_mask_local.jpg', frame_mask)

    if out[3] is not None:
        frame_pca = out[3]
        #cv2.imwrite(f'/home/rafael2ms/Dev/oakmax_webrtc/final_pa/frame_mask_local.jpg', frame_mask)
    
    return command

async def process_video_frame(frame):
    # Offload the blocking operation to a separate thread
    processed_frame = await asyncio.to_thread(hard_frame_process, frame)
    return processed_frame

async def start_web_server():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8000)
    await site.start()
    print("HTTP server running on http://localhost:8000")


async def main():
    # Run web server
    web_server = asyncio.create_task(start_web_server())

    # Setup and connect WebRTC in a non-blocking way
    #connection = WebRTCConnection("ws://localhost:8080")
    connection = WebRTCConnection("ws://172.21.1.173:8080")
    #connection = WebRTCConnection("ws://192.168.169.191:8080")
    #connection = WebRTCConnection("ws://160.85.114.134:8080")
    webrtc_task = asyncio.create_task(connection.connect())

    # Wait for both tasks to run indefinitely
    await asyncio.gather(web_server, webrtc_task)

if __name__ == "__main__":
    asyncio.run(main())
