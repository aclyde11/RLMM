import socket
import os
import sys
import shutil
import pickle
import struct
from datetime import datetime
import threading
from typing import TypeVar, Generic
import logging
import warnings
import shutil

from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.rl.Expert import ExpertPolicy
from rlmm.utils.config import Config

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
    print(raw_msglen)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


class TcpWrapper:
    def __init__(self, worker: bool = True, worker_id: int = 1, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.client = None
        self.is_worker = worker
        if self.is_worker:
            self.id = worker_id
        else:
            self.id = 0

    def stop(self):
        """stop everything"""
        self.client.send("break")
        self.client.close()
        print("server and client shut off")

    def setup_temp_files(self, config):
        current_temp_dir = config.configs['tempdir'] + '_' + str(self.id) + '/'
        try:
            os.mkdir(current_temp_dir)
        except FileExistsError:
            pass
        if not config.configs['overwrite_static']:
            current_temp_dir = current_temp_dir + "{}/".format(
                datetime.now().strftime("rlmm_%d_%m_%YT%H%M%S"))
            try:
                os.mkdir(current_temp_dir)
            except FileExistsError:
                print("Somehow the directory already exists... exiting")
                exit()
        else:
            try:
                shutil.rmtree(current_temp_dir)
                os.mkdir(current_temp_dir)
            except FileExistsError:
                print("Somehow the directory already exists... exiting")
                exit()

        for k, v in config.configs.items():
            if k in ['actions', 'systemloader', 'openmmWrapper', 'obsmethods']:
                for k_, v_ in config.configs.items():
                    if k_ != k:
                        v.update(k_, v_)

    def server_worker(self, local_policy_steps: int = 0):
        """server that will pass commands that it receives to actual env object"""
        # copy from test_load_system
        from openeye import oechem
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logging.getLogger('openforcefield').setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        conf_file = 'examples/example1_config.yaml'
        config = Config.load_yaml(conf_file)
        self.setup_temp_files(config)
        shutil.copy('rlmm/tests/test_config.yaml', config.configs['tempdir'] + '_' + str(self.id) + '/' + "config.yaml")
        env = OpenMMEnv(OpenMMEnv.Config(config.configs))

        received_action = env.reset()
        energies = []

        # Bind and do work
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            print("running worker")
            while True:
                for i in range(100):
                    # run simulation
                    obs, reward, done, data = env.step(received_action)
                    energies.append(data['energies'])
                    with open("rundata.pkl", 'wb') as f:
                        pickle.dump(env.data, f)
                    # send observation and receive action
                    print('Message to be sent', obs)
                    msg = pickle.dumps(obs)  ## Need Looping here for on some threshold for local policy
                    send_msg(s, msg)
                    print('Observation sent to master')
                    msg = recv_msg(s)
                    print('message received')
                    received_action = pickle.loads(msg)
                    print('message successfully unpickled')
                    if received_action == 'break':
                        break
                    print("server received action: {}".format(received_action))
                print('Work is finished!')
                msg = pickle.dumps('Work is finished!')
                send_msg(s, msg)
                break
            s.close()
            print('master disconnected')
            print("worker closed")

    def policy_master(self):
        import logging
        import warnings
        import shutil
        from openeye import oechem
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logging.getLogger('openforcefield').setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        conf_file = 'examples/example1_config.yaml'
        config = Config.load_yaml(conf_file)
        self.setup_temp_files(config)
        shutil.copy(conf_file, config.configs['tempdir'] + '_' + str(self.id) + '/' + "config.yaml")
        env = OpenMMEnv(OpenMMEnv.Config(config.configs))
        policy = ExpertPolicy(env, num_returns=-1, sort='dscores',
                              orig_pdb=config.configs['systemloader'].pdb_file_name)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(5)
            while True:
                print('running master')
                conn, addr = s.accept()
                print('connected to', addr)
                while True:
                    obs = recv_msg(s)
                    obs = pickle.loads(obs)
                    print('Received', repr(obs))
                    if obs == 'Work is Finished!':
                        break
                    msg = policy.choose_action(obs)
                    print('Sending action to client')
                    msg = pickle.dumps(msg)
                    send_msg(s, msg)
                print('Worker finished: closed connection')
                s.close()


if __name__ == '__main__':
    if len(sys.argv)-1 == 0:
        print('Please specify parameters:'
              'python.py worker worker_id_int or python.py master')
        sys.exit()

    if sys.argv[1] != 'worker' and sys.argv[1] != 'master':
        print(sys.argv[1])
        print('Please specify: worker or master for parameter 1')
        sys.exit()

    if sys.argv[1] == 'worker':
        if sys.argv[2] is None:
            print('Please specify a worker id (int) for parameter 2')
            sys.exit()

    # High level setup #

    if sys.argv[1] == 'worker':
        TcpWrapper(True, int(sys.argv[2]), '127.0.0.1', 12345).server_worker()

    if sys.argv[1] == 'master':
        TcpWrapper(False, 0, 'localhost', 12345).policy_master()
