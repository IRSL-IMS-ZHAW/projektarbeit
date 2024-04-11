import asyncio
import websockets

connected = set()

async def handler(websocket, path):
    global connected
    connected.add(websocket)
    print(f"[Server] New connection: {len(connected)} total connections.")

    async for message in websocket:
        print(f"[Server] Received message: {message}")
        for conn in connected:
            if conn != websocket:
                await conn.send(message)
                print("[Server] Message forwarded.")

    connected.remove(websocket)
    print("[Server] Connection closed.")

start_server = websockets.serve(handler, "localhost", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
