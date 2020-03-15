#!/usr/bin/env python
import argparse
import socket

import numpy as np

import gym_uds_pb2
import utils


class Environment:
    def __init__(self, sock_filepath):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_filepath)
        self.action_space = lambda: None
        self.action_space.sample = self.sample

    def reset(self):
        utils.send_message(self.sock, gym_uds_pb2.Request(type=gym_uds_pb2.Request.RESET))
        state_pb = utils.recv_message(self.sock, gym_uds_pb2.State)
        observation = np.asarray(state_pb.observation.data).reshape(state_pb.observation.shape)
        return observation

    def step(self, action):
        utils.send_message(self.sock, gym_uds_pb2.Request(type=gym_uds_pb2.Request.STEP))
        utils.send_message(self.sock, gym_uds_pb2.Action(value=action))
        state_pb = utils.recv_message(self.sock, gym_uds_pb2.State)
        observation = np.asarray(state_pb.observation.data).reshape(state_pb.observation.shape)
        return observation, state_pb.reward, state_pb.done

    def sample(self):
        utils.send_message(self.sock, gym_uds_pb2.Request(type=gym_uds_pb2.Request.SAMPLE))
        action_pb = utils.recv_message(self.sock, gym_uds_pb2.Action)
        return action_pb.value

    def done(self):
        utils.send_message(self.sock, gym_uds_pb2.Request(type=gym_uds_pb2.Request.DONE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filepath',
        nargs='?',
        default='/tmp/gym-uds-socket',
        help='a unique filepath where the socket will connect')
    args = parser.parse_args()

    env = Environment(args.filepath)

    num_episodes = 3
    for episode in range(1, num_episodes + 1):
        observation = env.reset()

        episode_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done = env.step(action)
            episode_reward += reward
        print('Ep. %d: %.2f' % (episode, episode_reward))
