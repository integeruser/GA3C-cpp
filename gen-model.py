#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

OBSERVATION_SPACE_SHAPE = (4, )
NUM_ACTIONS = 2

LEARNING_RATE = 0.005
DECAY = 0.99

LOSS_V = 0.50
LOSS_ENTROPY = 0.01

x_states = tf.placeholder(tf.float32, shape=(None, *OBSERVATION_SPACE_SHAPE), name='x_states')
y_policies = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name='y_policies')
y_values = tf.placeholder(tf.float32, shape=(None, 1), name='y_values')

hidden = tf.layers.dense(inputs=x_states, units=16, activation=tf.nn.relu)
out_policies = tf.layers.dense(
    inputs=hidden, units=NUM_ACTIONS, activation=tf.nn.softmax, name='out_policies')
out_values = tf.layers.dense(inputs=hidden, units=1, name='out_values')

log_prob = tf.log(tf.reduce_sum(y_policies * out_policies, axis=1, keep_dims=True) + 1e-10)
advantage = y_values - out_values

policy_loss = -log_prob * tf.stop_gradient(advantage)
value_loss = LOSS_V * tf.square(advantage)
entropy = LOSS_ENTROPY * tf.reduce_sum(
    y_policies * tf.log(y_policies + 1e-10), axis=1, keep_dims=True)

loss_total = tf.reduce_mean(policy_loss + value_loss + entropy)
minimize_op = tf.train.RMSPropOptimizer(
    LEARNING_RATE, decay=DECAY).minimize(
        loss_total, name='minimize')

session = tf.Session()
session.run(tf.global_variables_initializer())

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

if not args.test:
    saver = tf.train.Saver()
    saver.save(session, 'models/graph')
else:
    import gym
    saver = tf.train.Saver()
    saver.restore(session, 'models/graph')

    def predict_policy(state):
        return session.run(out_policies, feed_dict={x_states: [state]})[0]

    def predict_value(state):
        return session.run(out_values, feed_dict={x_states: [state]})[0]

    env = gym.make('CartPole-v0')
    for _ in range(10):
        curr_state = env.reset()
        env.render()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(session.run(out_policies, feed_dict={x_states: [curr_state]})[0])
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            curr_state = next_state
        print(total_reward)
