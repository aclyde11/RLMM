class TCPCommunicator:

    def __init__(self, env, num_envs, **env_kwargs):

        self.host = '127.0.0.1'
        self.env = env
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs


    def start_envs(self):
        """ spawn a thread for each env with a server each, store threads in self.server_threads
            start up environments in each server using env_kwargs
            start servers on open ports, this can be done using the following commands:
                your_socket.bind(('', 0)) # picks up a free port and binds to it
                your_socket.getsockname()[1] # this has the value of the picked port, should be returned ot the communicator class
            create clients for each server and store in self.clients
        """
        pass


    def send_action_msg(self, action, sim_steps):
        """ for each action, if action != None
            encode actions using struct module
            send encoded actions to each server using respective clients in self.clients
        """
        pass


    def send_reset_msg(self, env_ids):
        """ send reset message to each env in env_ids """
        pass


    def stop(self):
        """ send a stop message to all servers, so that servers stop environments and stor serving
            close all clients
            join all threads
        """
        pass


class DistributedEnv:

    def __init__(self, env, num_envs, comm_type='tcp', vector_obs=False, **env_kwargs):

        self.num_envs = envs
        self.env = env
        self.comm_type = comm_type
        self.vector_obs = vector_obs

        if comm_type == 'tcp':
            self.communicator = TCPCommunicator(self.env, self.num_envs, env_kwargs)
        elif comm_type == 'mpi':
            self.communicator = MPICommunicator(self.env, self.num_envs, env_kwargs)

        self.communicator.start_envs()


    def step(self, action, sim_steps):

        if not self.vector_obs:
            action = [action for a in range(self.num_envs)]

        data = self.communicator.send_action_msg(action, sim_steps)
        return data


    def reset(self, env_ids=None):

        if not env_ids:
            env_ids = [x for x in num_envs]

        data = self.communicator.send_reset_msg(env_ids)
        return data


    def stop(self):
        self.communicator.stop()
