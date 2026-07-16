import logging
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem
from speech_to_speech.VAD.aec import AecProcessor

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: threading.Event,
        list_play_chunk_size: int = 512,
        enable_aec: bool = False,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen

        # Construct eagerly so missing library fails loudly at startup, not
        # deep in the realtime callback.
        self._aec: AecProcessor | None = AecProcessor(sample_rate=16000) if enable_aec else None

    def run(self) -> None:
        # Pre-generate a static dither buffer (±1 LSB, -96 dB) to keep the
        # audio sink active without calling numpy inside the real-time callback.
        dither = np.random.randint(-1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16)

        def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time: float, status: str) -> None:
            # Always capture near-end audio unconditionally so it is never
            # silently dropped regardless of what the output queue contains.
            near_bytes = np.ascontiguousarray(indata, dtype=np.int16).tobytes()

            # During shutdown, output silence and discard the near-end capture.
            if self.stop_event.is_set():
                outdata.fill(0)
                return

            if self.output_queue.empty():
                self.input_queue.put(near_bytes)
                outdata[:] = dither
            else:
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, np.ndarray):
                        far_flat = audio_chunk.flatten().astype(np.int16)
                        if self._aec is not None:
                            near_bytes = self._aec.process(near_bytes, far_flat.tobytes())
                        self.input_queue.put(near_bytes)
                        outdata[:] = audio_chunk[:, np.newaxis]
                    elif audio_chunk == AUDIO_RESPONSE_DONE:
                        self.should_listen.set()
                        logger.debug("Response complete, listening re-enabled")
                        self.input_queue.put(near_bytes)
                        outdata.fill(0)
                    else:
                        self.input_queue.put(near_bytes)
                        outdata.fill(0)
                except Exception:
                    logger.exception("Error in audio callback output branch; outputting silence")
                    self.input_queue.put(near_bytes)
                    outdata.fill(0)

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
