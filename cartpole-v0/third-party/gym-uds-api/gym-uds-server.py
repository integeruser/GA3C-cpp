#!/usr/bin/env python
import argparse
import os
import socket

import gym
import numpy as np

import gym_uds_pb2
import utils


class Environment:
    def __init__(self, env_id, sock):
        self.env = gym.make(env_id)
        self.sock = sock
        self.sock.settimeout(1)

    def run(self):
        while True:
            request = utils.recv_message(self.sock, gym_uds_pb2.Request)
            if request.type == gym_uds_pb2.Request.DONE: break
            elif request.type == gym_uds_pb2.Request.RESET: self.reset()
            elif request.type == gym_uds_pb2.Request.STEP: self.step()
            elif request.type == gym_uds_pb2.Request.SAMPLE: self.sample()

    def reset(self):
        observation = self.env.reset()
        observation_pb = gym_uds_pb2.Observation(data=observation.ravel(), shape=observation.shape)
        utils.send_message(self.sock,
                           gym_uds_pb2.State(observation=observation_pb, reward=0.0, done=False))

    def step(self):
        action = utils.recv_message(self.sock, gym_uds_pb2.Action)
        observation, reward, done, _ = self.env.step(action.value)
        assert type(observation) is np.ndarray

        observation_pb = gym_uds_pb2.Observation(data=observation.ravel(), shape=observation.shape)
        utils.send_message(self.sock,
                           gym_uds_pb2.State(observation=observation_pb, reward=reward, done=done))

    def sample(self):
        action = self.env.action_space.sample()
        utils.send_message(self.sock, gym_uds_pb2.Action(value=action))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', help='the id of the gym environment to simulate')
    parser.add_argument(
        'filepath',
        nargs='?',
        default='/tmp/gym-uds-socket',
        help='a unique filepath where the socket will bind')
    args = parser.parse_args()

    try:
        os.remove(args.filepath)
    except FileNotFoundError:
        pass

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(args.filepath)
    sock.listen()

    while True:
        try:
            conn, _ = sock.accept()
            env = Environment(args.id, conn)
            env.run()
        except BrokenPipeError:
            pass
        except socket.timeout:
            print('socket.timeout!')
            pass
        finally:
            try:
                del env
            except NameError:
                pass
