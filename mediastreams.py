#/home/rafael2ms/anaconda3/envs/oakmax_web_py310/lib/python3.10/site-packages/aiortc/mediastreams.py
#
# Updated the file to meet specific streaming requirements.
#
import asyncio
import fractions
import time
import uuid
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

from av import AudioFrame, VideoFrame
from av.frame import Frame
from av.packet import Packet
from pyee.asyncio import AsyncIOEventEmitter

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
#VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
#video_fps = 30

def convert_timebase(
    pts: int, from_base: fractions.Fraction, to_base: fractions.Fraction
) -> int:
    if from_base != to_base:
        pts = int(pts * from_base / to_base)
    return pts


class MediaStreamError(Exception):
    pass


class MediaStreamTrack(AsyncIOEventEmitter, metaclass=ABCMeta):
    """
    A single media track within a stream.
    """

    kind = "unknown"

    def __init__(self) -> None:
        super().__init__()
        self.__ended = False
        self._id = str(uuid.uuid4())

    @property
    def id(self) -> str:
        """
        An automatically generated globally unique ID.
        """
        return self._id

    @property
    def readyState(self) -> str:
        return "ended" if self.__ended else "live"

    @abstractmethod
    async def recv(self) -> Union[Frame, Packet]:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`,
        :class:`~av.video.frame.VideoFrame` or :class:`~av.packet.Packet`
        """

    def stop(self) -> None:
        if not self.__ended:
            self.__ended = True
            self.emit("ended")

            # no more events will be emitted, so remove all event listeners
            # to facilitate garbage collection.
            self.remove_all_listeners()


class AudioStreamTrack(MediaStreamTrack):
    """
    A dummy audio track which reads silence.
    """

    kind = "audio"

    _start: float
    _timestamp: int

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.

        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise MediaStreamError

        sample_rate = 8000
        samples = int(AUDIO_PTIME * sample_rate)

        if hasattr(self, "_timestamp"):
            self._timestamp += samples
            wait = self._start + (self._timestamp / sample_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = self._timestamp
        frame.sample_rate = sample_rate
        frame.time_base = fractions.Fraction(1, sample_rate)
        return frame


class VideoStreamTrack(MediaStreamTrack):
    """
    A dummy video track which reads green frames.
    """
    kind = "video"

    def __init__(self, fps=10):
        super().__init__()
        self.video_fps = fps
        #self._start = None
        #self._timestamp = 0    
    
    _start: float
    _timestamp: int


    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        video_ptime = 1/self.video_fps

        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int(video_ptime * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.perf_counter()
            #print(f'awaiting = {self._start:.4f} + {(self._timestamp / VIDEO_CLOCK_RATE):.4f} - {time.perf_counter():.4f} = {wait:.4f}')
            await asyncio.sleep(wait)
        else:
            self._start = time.perf_counter()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.video.frame.VideoFrame`.

        The base implementation just reads a 640x480 green frame at 30fps,
        subclass :class:`VideoStreamTrack` to provide a useful implementation.
        """
        pts, time_base = await self.next_timestamp()

        #frame = VideoFrame(width=640, height=480)
        frame = VideoFrame(width=720, height=1280)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        return frame
