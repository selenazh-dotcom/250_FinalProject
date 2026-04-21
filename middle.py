#rpi middle test
import socket

LAP_A_PORT = 5005
LAP_B_IP = "172.20.10.9" # Replace with Laptop B's IP
LAP_B_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', LAP_A_PORT))

print("RPi relay started...")
while True:
    data, addr = sock.recvfrom(65536)
    sock.sendto(data, (LAP_B_IP, LAP_B_PORT))