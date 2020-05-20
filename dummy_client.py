import socket
import struct
import pickle
import time

HOST = '127.0.0.1'
PORT = 12345
print(1)

def recvall(sock, n):
    """Helper function to recv n bytes or return None if EOF is hit"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def send_msg(sock, msg):
    """Prefix each message with a 4-byte length (network byte order)"""
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print('connected')
        data = ''
        while True:
            msg = 'Hello, World' + data
            msg = pickle.dumps(msg)
            send_msg(s, msg)
            time.sleep(5)
            data = recv_msg(s)
            data = pickle.loads(data)
            print('Received', repr(data))


