# Dependencies
import sys
import queue
from datetime import datetime
from pathlib import Path
import tkinter as tk

try:
	import sounddevice as sd # sounddevice provides the live input stream from your microphone
	import soundfile as sf # soundfile handles writing audio data to .wav files
except ImportError:
	print(
		"Missing dependencies. Install with: pip install sounddevice soundfile",
		file=sys.stderr,
	)
	raise


# Thread-safe buffer holding audio chunks coming from the callback
# Type hint note: the annotation is in quotes so it doesn't need to be imported at runtime.
_audio_queue: "queue.Queue" = queue.Queue()


def _audio_callback(indata, frames, time_info, status):
	""" 
	    Frames will affect latency and perrformance, Smaller frames is lower latency, 
		but more callback so higher CPU overhead.
	"""
	if status:
		print(f"Audio status: {status}", file=sys.stderr)
	# .copy() ensures the buffer isn't reused/overwritten after the callback returns
	_audio_queue.put(indata.copy())


class RecorderApp:
	"""
        Sample rate describes the quality
	"""

	SAMPLE_RATE = 16000 # 16kHz 
	CHANNELS = 1 # 1 = mono, 2 = Stereo
	SUBTYPE = "PCM_16" # 16-bit PCM MAV


	def __init__(self):
		""" 
	        Init is a dunder method, which will called automatically after a new instance is created. 
            Essentialy initializing the object.
	    """
		# Create the main window and basic UI elements
		self.root = tk.Tk()
		self.root.title("Recording...")
		self.root.protocol("WM_DELETE_WINDOW", self.stop)

		self.status_var = tk.StringVar(value="Recording at 16 kHz mono. Click Stop to save.")
		self.label = tk.Label(self.root, textvariable=self.status_var, padx=16, pady=12)
		self.label.pack()

		self.stop_btn = tk.Button(self.root, text="Stop and Save", width=20, command=self.stop)
		self.stop_btn.pack(pady=(0, 12))

		# Where we store recordings; create the folder if missing
		self.out_dir = Path(__file__).resolve().parent / "recorded_Audio" # CHANGE PATH HERE
		self.out_dir.mkdir(parents=True, exist_ok=True)
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.out_path = self.out_dir / f"recording_{ts}.wav"

		self.wav_file = None
		self.stream = None
		self.running = True

	def start(self):
		# Open the WAV file first. We request 16-bit PCM on disk (subtype="PCM_16").
		# Even though the live stream provides float32 samples, soundfile will convert
		# to the requested disk format when we write.
		try:
			self.wav_file = sf.SoundFile(
				str(self.out_path), mode="x", samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, subtype=self.SUBTYPE
			)
			# Create the input stream from the default microphone.
			# dtype="float32" is efficient for processing; we'll convert on write.
			self.stream = sd.InputStream(
				samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype="float32", callback=_audio_callback
			)
			self.stream.start()
			# Start polling the Queue from the GUI thread to write audio without blocking the UI
			self._poll_queue()
			self.root.mainloop()
		except Exception as e:
			self.status_var.set(f"Failed to start recording: {e}")
			self.stop_btn.configure(state=tk.DISABLED)

	def _poll_queue(self):
		"""Periodically drain the audio Queue and write to disk.

		We use Tkinter's .after(...) to schedule the next check ~50 times/sec,
		which keeps the window responsive and the audio writing steady.
		"""
		if self.running and self.wav_file is not None:
			# Write all available chunks without blocking the UI
			while True:
				try:
					chunk = _audio_queue.get_nowait()
					self.wav_file.write(chunk)
				except queue.Empty:
					break
			self.root.after(20, self._poll_queue)
		else:
			self._finalize_save()

	def stop(self):
		# Stop can be triggered by the button or window close (idempotent)
		if not self.running:
			return
		self.running = False
		self.stop_btn.configure(state=tk.DISABLED)
		self.status_var.set("Stopping, finalizing file...")
		try:
			if self.stream is not None:
				self.stream.stop()
				self.stream.close()
		except Exception:
			pass

	def _finalize_save(self):
		# Drain any remaining audio data in the Queue before closing the file
		if self.wav_file is not None:
			while not _audio_queue.empty():
				try:
					self.wav_file.write(_audio_queue.get_nowait())
				except queue.Empty:
					break
			self.wav_file.close()
			self.wav_file = None
		self.status_var.set(f"Saved to: {self.out_path}")
		# Give a brief visual confirmation, then close the window
		self.root.after(1200, self.root.destroy)


def main():
	# Typical Python entry point guard lives below; keeping main() tiny is a best practice
	app = RecorderApp()
	app.start()


if __name__ == "__main__":
	# This condition is True when you run `python audio_capture.py` directly.
	# It prevents main() from running if the file is imported as a module.
	main()

