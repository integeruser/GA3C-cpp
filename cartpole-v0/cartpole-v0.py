#!/usr/bin/env python
import argparse

import dlib
import tensorflow as tf

ENV_ID = 'CartPole-v0'
NUM_OBSERVATIONS = 4
NUM_ACTIONS = 2

LOSS_VALUE_COEFF = 0.50
LOSS_ENTROPY_COEFF = 0.01

LEARNING_RATE = 0.005
DECAY = 0.99


class Model:
    GRAPH_FILEPATH = 'models/graph'

    def __init__(self):
        self.x_states = tf.placeholder(tf.float32, shape=(None, NUM_OBSERVATIONS), name='x_states')
        self.y_policies = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name='y_policies')
        self.y_values = tf.placeholder(tf.float32, shape=(None, 1), name='y_values')

        hidden_layer = tf.layers.dense(inputs=self.x_states, units=16, activation=tf.nn.relu)

        self.out_policies = tf.layers.dense(
            inputs=hidden_layer, units=NUM_ACTIONS, activation=tf.nn.softmax, name='out_policies')
        self.out_values = tf.layers.dense(inputs=hidden_layer, units=1, name='out_values')

        EPSILON = 1e-10

        log_prob = tf.log(
            tf.reduce_sum(self.y_policies * self.out_policies, axis=1, keep_dims=True) + EPSILON)
        advantage = self.y_values - self.out_values

        policy_loss = -log_prob * tf.stop_gradient(advantage)
        value_loss = LOSS_VALUE_COEFF * tf.square(advantage)
        entropy = LOSS_ENTROPY_COEFF * tf.reduce_sum(
            self.y_policies * tf.log(self.y_policies + EPSILON), axis=1, keep_dims=True)

        total_loss = tf.reduce_mean(policy_loss + value_loss + entropy)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY)
        optimizer.minimize(total_loss, name='minimize')

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def predict_policy(self, state):
        return self.session.run(self.out_policies, feed_dict={self.x_states: [state]})[0]

    def save(self, path=GRAPH_FILEPATH):
        tf.train.Saver().save(self.session, path)

    def restore(self, path=GRAPH_FILEPATH):
        tf.train.Saver().restore(self.session, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'test'])
    args = parser.parse_args()

    model = Model()
    if args.action == 'generate':
        model.save()
    elif args.action == 'test':
        import gym
        import numpy as np

        model.restore()

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
