import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("", 9998))
print("Waiting for STWIN.box logs...")
while True:
    data, addr = s.recvfrom(2048)
    print(addr[0], data.decode(errors="replace"), end="")