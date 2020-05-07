import socket
import time
import threading


class MockRLMMEnv:
    def __init__(self):
        pass
    def act(self, action):
        print("env received action \'{}\'".format(action))
        return "observations for action \'{}\'".format(action)


class TcpEnvWrapper:
    def __init__(self, env, host, port):
        self.env = env
        self.host = host
        self.port = port
        self.server_thread = threading.Thread(name='nondaemonworker', target=self.server_worker)
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
        serv.bind((self.host, self.port))
        serv.listen(5)
        keeprunning = True
        print("running server")
        while keeprunning:
            conn, addr = serv.accept()
            while True:
                received_action = conn.recv(4096)
                if not received_action: break
                if received_action == 'break':
                    keeprunning = False
                    break
                print("server received action: {}".format(received_action))
                conn.send(self.env.act(received_action))
            conn.close()
            print('client disconnected')
        print("server closed")


    def act(self, action):
        """send action to the env server tahts listening on self.host:self.port"""
        if not self.client:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.host, self.port))
        self.client.send(str(action))
        obs = self.client.recv(4096)
        return obs



testenv = TcpEnvWrapper(MockRLMMEnv(), 'localhost', 12345)

for i in range(4):
    time.sleep(2)
    observations = testenv.act("action_"+str(i))
    print("got observations \'{}\'".format(observations))
testenv.stop()
