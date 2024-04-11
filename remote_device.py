import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from transformators import DepthAIVideoTransformTrack, OptionsWrapper

async def setup_data_channel(pc):
    data_channel = pc.createDataChannel("chat")

    @data_channel.on("open")
    def on_data_channel_open():
        print("Data channel is open")
        # Test sending a message
        data_channel.send("Request frame info")

    @data_channel.on("message")
    def on_data_channel_message(message):
        print(f"Received from receiver: {message}")
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
    async with websockets.connect("ws://localhost:8080") as ws:
        pc = RTCPeerConnection()

        # Initialize options directly with default values
        options = OptionsWrapper()

        pc.addTrack(DepthAIVideoTransformTrack(options))

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
