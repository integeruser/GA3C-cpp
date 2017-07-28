#!/usr/bin/env python3
import argparse

import tensorflow as tf

ENV_ID = 'CartPole-v0'
GRAPH_FILEPATH = 'models/graph'

NUM_OBSERVATIONS = 4
NUM_ACTIONS = 2

LEARNING_RATE = 0.005
DECAY = 0.99

LOSS_V = 0.50
LOSS_ENTROPY = 0.01


class Model:
    def __init__(self):
        self.x_states = tf.placeholder(tf.float32, shape=(None, NUM_OBSERVATIONS), name='x_states')
        self.y_policies = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name='y_policies')
        self.y_values = tf.placeholder(tf.float32, shape=(None, 1), name='y_values')

        hidden = tf.layers.dense(inputs=self.x_states, units=16, activation=tf.nn.relu)
        self.out_policies = tf.layers.dense(
            inputs=hidden, units=NUM_ACTIONS, activation=tf.nn.softmax, name='out_policies')
        self.out_values = tf.layers.dense(inputs=hidden, units=1, name='out_values')

        log_prob = tf.log(
            tf.reduce_sum(self.y_policies * self.out_policies, axis=1, keep_dims=True) + 1e-10)
        advantage = self.y_values - self.out_values

        policy_loss = -log_prob * tf.stop_gradient(advantage)
        value_loss = LOSS_V * tf.square(advantage)
        entropy = LOSS_ENTROPY * tf.reduce_sum(
            self.y_policies * tf.log(self.y_policies + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(policy_loss + value_loss + entropy)
        minimize_op = tf.train.RMSPropOptimizer(
            LEARNING_RATE, decay=DECAY).minimize(
                loss_total, name='minimize')

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def predict_policy(self, state):
        return self.session.run(self.out_policies, feed_dict={self.x_states: [state]})[0]

    def save(self, path):
        tf.train.Saver().save(self.session, path)

    def restore(self, path):
        tf.train.Saver().restore(self.session, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'test'])
    args = parser.parse_args()

    model = Model()
    if args.action == 'generate':
        model.save(GRAPH_FILEPATH)
    elif args.action == 'test':
        import gym
        import numpy as np

        model.restore(GRAPH_FILEPATH)

        env = gym.make(ENV_ID)
        for _ in range(5):
            curr_state = env.reset()
            env.render()

            episode_reward = 0
            done = False
            while not done:
                action = np.argmax(model.predict_policy(curr_state))
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                env.render()

                curr_state = next_state
            print(episode_reward)
