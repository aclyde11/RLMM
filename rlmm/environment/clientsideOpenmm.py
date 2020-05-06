import socket
import tqdm
import os

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step


host = "192.168.1.101"
port = 5001


# create the client socket
s = socket.socket()
print(f"[+] Connecting to {host}:{port}")
s.connect((host, port))
print("[+] Connected.")

## Send Protein ##

filename = "protein.pdb"
filesize = os.path.getsize(filename)
s.send(f"{filename}{SEPARATOR}{filesize}".encode())


# start sending the file
progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
with open(filename, "rb") as f:
    for _ in progress:
        # read the bytes from the file
        bytes_read = f.read(BUFFER_SIZE)
        if not bytes_read:
            # file transmitting is done
            break
        # we use sendall to assure transimission in
        # busy networks
        s.sendall(bytes_read)
        # update the progress bar
        progress.update(len(bytes_read))

while True:

## Send Ligand ##

## Recieve Score ##

##Run RL -> Get new Ligand##

## Run RL -> Get Stop##
## Send Stop ##

# close the socket
s.close()