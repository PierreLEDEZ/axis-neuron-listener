import socket
import numpy as np
import binascii
import struct
import threading
import queue
from .predicter import Predicter


class Client(threading.Thread):

    def __init__(self, host, port):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.predicting_thread = Predicter()
        self.frames = queue.Queue()
        self.cooldown = 0 
        self.max_frames = 300
        self.killed = False

    def kill(self):
        self.killed = True

    def ieee_754_conversion(self, hex_string):
        return struct.unpack("<f", binascii.unhexlify(hex_string.replace(" ", "")))[0]
    
    def update_frames(self, frame):
        if self.frames.qsize() == self.max_frames:
            self.frames.get()
            self.frames.put(np.array(frame))
            if self.cooldown == 0:
                self.predicting_thread.update_frames(list(self.frames.queue))
                self.predicting_thread.run()
                self.cooldown = 150
            else:
                self.cooldown -= 1
        else:
            self.frames.put(frame)

    def run(self):
        print("[+] - Connecting to Axis Neuron")
        self.socket.connect((self.host, self.port))
        try:
            while not self.killed:
                data = self.socket.recv(1480)
                header = data[:64].hex().upper()
                nb_frame = int(header[94:96]+header[92:94], 16)
                content = data[64:].hex().upper()
                frame = []
                for i in range(0, len(content), 8):
                    frame.append(self.ieee_754_conversion(content[i:i+8]))
                self.update_frames(frame)
            self.socket.close()
        except KeyboardInterrupt:
            self.socket.close()