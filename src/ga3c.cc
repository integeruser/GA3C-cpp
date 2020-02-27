#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "environment.h"
#include "model.h"
#include "queue.h"

using experience_t = std::tuple<observation_t, action_t, float, observation_t, bool>;
using training_experience_t = std::tuple<observation_t, action_t, float>;
using memory_t = std::deque<experience_t>;


extern uint32_t NUM_AGENTS;
extern uint32_t MAX_NUM_TRAINING_EXPERIENCES;

const uint32_t BATCH_SIZE = 32;

const uint32_t N_STEP_RETURN = 8;
const float GAMMA = 0.99f;

const float EPSILON_START = 1.00f;
const float EPSILON_END   = 0.15f;


std::shared_ptr<Model> shared_model;
Queue<training_experience_t> training_experiences_queue;
uint32_t num_training_experiences = 0;

std::random_device random_device;
std::default_random_engine random_engine(random_device());

/******************************************************************************/

action_t select_action(std::shared_ptr<Environment> env, float epsilon, const std::vector<float>& policy)
{
    const bool should_explore = std::uniform_real_distribution<float>(0.0f, 1.0f)(random_engine) < epsilon;
    if (should_explore) {
        // perform a random action
        return env->sample();
    }
    else {
        // perform an action according to the current policy
        return std::discrete_distribution<action_t>(policy.cbegin(), policy.cend())(random_engine);
    }
}

training_experience_t extract_accumulated_experience(memory_t& memory, float next_state_value)
{
    assert(not memory.empty());
    const auto n = std::min<uint32_t>(N_STEP_RETURN, memory.size());

    // pick experience at time step t
    uint32_t t = 0;
    const auto curr_state = std::get<0>(memory[t]);
    const auto action = std::get<1>(memory[t]);

    // and calculate its n-step return (until time step t+n)
    float n_step_return = 0.0f;
    for (; t < n; ++t) {
        const auto reward = std::get<2>(memory[t]);
        n_step_return += std::pow(GAMMA, t)*reward;
    }
    const auto done = std::get<4>(memory[n-1]);
    if (not done) { n_step_return += std::pow(GAMMA, n)*next_state_value; }

    memory.pop_front();
    return training_experience_t{curr_state, action, n_step_return};
}

void agent(uint32_t id, std::shared_ptr<Environment> env)
{
    for (uint32_t episode = 1; true; ++episode) {
        const float t = (float)num_training_experiences / (MAX_NUM_TRAINING_EXPERIENCES-1);
        const float epsilon = (1-t)*EPSILON_START + t*EPSILON_END;

        // reset the environment starting a new episode
        observation_t curr_state, next_state;
        curr_state = env->reset();

        memory_t memory;
        float reward = 0.0f;
        float episode_reward = 0.0f;

        bool done = false;
        while (not done) {
            // stop training after reaching a fixed number of training experiences
            if (num_training_experiences >= MAX_NUM_TRAINING_EXPERIENCES) { return; }

            // predict policy and value
            std::vector<float> policy;
            float value;
            std::tie(policy, value) = shared_model->predict_policy_and_value(curr_state);

            // select and perform an action
            const auto action = select_action(env, epsilon, policy);
            std::tie(next_state, reward, done) = env->step(action);
            episode_reward += reward;

            experience_t experience = experience_t{curr_state, action, reward, next_state, done};
            memory.push_back(experience);

            // add experiences to the training queue when:
            // - enough of them are accumulated in memory and the n-step return can be computed
            // - the episode is over and the memory is not empty
            while ((memory.size() >= N_STEP_RETURN) or (done and not memory.empty())) {
                const auto training_experience = extract_accumulated_experience(memory, value);
                training_experiences_queue.push(training_experience);
            }

            curr_state = next_state;
        }

        if (episode % 10 == 0) {
            std::cout << "[" << std::setfill('0') << std::setw(2) << id << "] "
                      << "Ep. " << std::setfill('0') << std::setw(5) << episode << " "
                      << "reward: " << episode_reward << std::endl;
        }
    }
}

/******************************************************************************/

void fit(const std::vector<training_experience_t>& batch)
{
    std::vector<observation_t> states;
    std::vector<action_t> actions;
    std::vector<float> rewards;

    observation_t state;
    action_t action;
    float reward;
    for (const auto& experience: batch) {
        std::tie(state, action, reward) = experience;
        states.push_back(state);
        actions.push_back(action);
        rewards.push_back(reward);
    }
    shared_model->fit(states, actions, rewards);
}

void trainer()
{
    while (num_training_experiences < MAX_NUM_TRAINING_EXPERIENCES) {
        const auto batch_size = std::min(BATCH_SIZE, MAX_NUM_TRAINING_EXPERIENCES-num_training_experiences);

        auto batch = std::vector<training_experience_t>(batch_size);
        for (uint32_t i = 0; i < batch.size(); ++i) {
            batch[i] = training_experiences_queue.pop();
        }
        fit(batch);
        num_training_experiences += batch_size;
    }
}

/******************************************************************************/

void train(std::shared_ptr<Model> model, const std::vector<std::shared_ptr<Environment>>& environments)
{
    assert(BATCH_SIZE >= 0);

    assert(N_STEP_RETURN >= 1);
    assert(GAMMA >= 0.0f and GAMMA <= 1.0f);

    assert(EPSILON_START >= 0.0f and EPSILON_START <= 1.0f);
    assert(EPSILON_END   >= 0.0f and EPSILON_END   <= 1.0f and EPSILON_END <= EPSILON_START);

    assert(NUM_AGENTS >= 0);
    assert(MAX_NUM_TRAINING_EXPERIENCES >= 0);

    shared_model = model;
    assert(environments.size() == NUM_AGENTS);

    // start the trainer and the agents
    auto trainer_thread = std::thread(trainer);
    auto agents_threads = std::vector<std::thread>(NUM_AGENTS);
    for (uint32_t id = 0; id < NUM_AGENTS; ++id) {
        agents_threads[id] = std::thread(agent, id, environments[id]);
    }

    // and wait for them to finish
    for (uint32_t id = 0; id < NUM_AGENTS; ++id) {
        agents_threads[id].join();
    }
    trainer_thread.join();
    assert(num_training_experiences == MAX_NUM_TRAINING_EXPERIENCES);
}
