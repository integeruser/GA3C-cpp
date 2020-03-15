#!/usr/bin/env python


def recv_exactly(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        data.extend(packet)
        if len(packet) == 0: break
    return bytes(data)


def recv_message(sock, cls):
    message_len = int.from_bytes(recv_exactly(sock, 4), byteorder='little')
    message_bytes = recv_exactly(sock, message_len)
    message = cls()
    message.ParseFromString(message_bytes)
    return message


def send_message(sock, message):
    message_bytes = message.SerializeToString()
    message_len = len(message_bytes).to_bytes(4, byteorder='little')
    sock.sendall(message_len)
    sock.sendall(message_bytes)
