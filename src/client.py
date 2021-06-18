import socket
import numpy as np
import binascii
import struct
import threading
import queue
from .predicter import Predicter


class Client(threading.Thread):
    """
        This class is the client of the Axis-Neuron software. It is in charge of receive and decode frames.
        When the window is complete, it sends frames to predicter

        Run in its own thread
    """

    def __init__(self, host, port, mode):
        """ Initialize the Client object with its socket, predicting thread """
        
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mode = mode
        self.predicting_thread = Predicter(self.mode)
        self.predicting_thread.make_first_prediction()
        self.frames = queue.Queue()
        self.cooldown = 0 
        self.max_frames = 150
        self.killed = False

    def kill(self):
        """ Break the listening loop of the client when called """
        
        self.killed = True
        self.predicting_thread.kill()

    def ieee_754_conversion(self, hex_string):
        """
            Convert HEX string in IEEE754 format to float

            :param hex_string: HEX string to convert
            :type hex_string: string
        """

        return struct.unpack("<f", binascii.unhexlify(hex_string.replace(" ", "")))[0]
    
    def update_frames(self, frame):
        """
            Update queue of frames
            
            When queue is full (size == self.max_frames), remove one and add the new frame
            If the prediction cooldown is at zero, create a predicting_thread
            Else decrease cooldown

            :param frame: new frame to add to the queue
            :type frame: numpy array
        """
        if self.frames.qsize() == self.max_frames:
            self.frames.get()
            self.frames.put(np.array(frame))
            if self.cooldown == 0:
                self.predicting_thread.update_frames(list(self.frames.queue))
                self.predicting_thread.run()
                self.cooldown = 75
            else:
                self.cooldown -= 1
        else:
            self.frames.put(frame)

    def run(self):
        """
            Main function of this class

            Connect previously created socket to Axis-Neuron server and listen to it
        """

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