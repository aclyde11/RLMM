import socket
import pickle
import struct
import time
import threading


class MockRLMMEnv:
    def __init__(self):
        pass
    def act(self, action):
        print("env received action \'{}\'".format(action))
        return "observations for action \'{}\'".format(action)


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


class TcpEnvWrapper:
    def __init__(self, env, host, port):
        self.env = env
        self.host = host
        self.port = port
        self.server_thread = threading.Thread(name='nondaemonworker', target=self.server_worker)  ## should we use processes instead of threads? -Marius
        self.server_thread.start()
        self.client = None

    def stop(self):
        """stop everything"""
        self.client.send("break")
        self.client.close()
        print("server and client shut off")

    def server_worker(self):
        """server that will pass commands that it receives to actual env object"""
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serv.bind((self.host, self.port)) ###create a list of server workers and name them? -Marius
        serv.listen(5)
        keeprunning = True
        print("running server")
        while keeprunning:
            conn, addr = serv.accept()
            while True:
                msg = recv_msg(serv)
                received_action = pickle.loads(msg)
                if not received_action: break
                if received_action == 'break':
                    keeprunning = False
                    break
                print("server received action: {}".format(received_action))
                msg = pickle.dumps(self.env.act(received_action))  ## Need Looping here for on some threshold for local policy -Marius
                send_msg(serv, msg)
                print('Observation sent to client')
            conn.close()
            print('client disconnected')
        print("server closed")

    def act(self, action):
        """send action to the env server that's listening on self.host:self.port"""
        if not self.client:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.host, self.port))
        # Pickle the object and send it to the server
        msg = pickle.dumps(action)
        send_msg(self.client, msg)
        print('Data sent to server')
        # Receive object and unpickle it
        msg = recv_msg(self.client)
        obs = pickle.loads(msg)
        print('Data received from server')
        return obs ### add some sort of naming to append to local directories by server wroker? -Marius



testenv = TcpEnvWrapper(MockRLMMEnv(), 'localhost', 12345)

for i in range(4):
    time.sleep(2)
    observations = testenv.act("action_"+str(i))
    print("got observations \'{}\'".format(observations))
testenv.stop()
