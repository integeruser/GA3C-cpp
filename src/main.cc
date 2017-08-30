#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "gym-uds.h"
#include "model.h"
#include "queue.h"

using experience_t = std::tuple<gym_uds::observation_t, gym_uds::action_t, float, gym_uds::observation_t, bool>;
using memory_t = std::deque<experience_t>;


const uint32_t NUM_AGENTS = 5;
const uint32_t MAX_NUM_TRAINING_EXPERIENCES = 75000;

const uint32_t BATCH_SIZE = 32;

const uint32_t N_STEP_RETURN = 8;
const float GAMMA = 0.99f;

const float EPSILON_START = 1.00f;
const float EPSILON_END   = 0.15f;


Model model;
Queue<experience_t> training_experiences_queue;
uint32_t num_training_experiences = 0;

std::random_device random_device;
std::default_random_engine random_engine(random_device());

/******************************************************************************/

gym_uds::action_t select_action(gym_uds::Environment& env, const gym_uds::observation_t& state, float epsilon)
{
    const bool should_explore = std::uniform_real_distribution<float>(0.0f, 1.0f)(random_engine) < epsilon;
    if (should_explore) {
        // perform a random action
        return env.sample();
    }
    else {
        // perform an action according to the current policy
        const auto policy = model.predict_policy(state);
        return std::discrete_distribution<float>(policy.cbegin(), policy.cend())(random_engine);
    }
}

experience_t extract_accumulated_experience(memory_t& memory)
{
    assert(not memory.empty());
    const auto n = std::min<uint32_t>(N_STEP_RETURN, memory.size());

    // pick experience at time step t
    auto t = 0;
    const auto curr_state = std::get<0>(memory[t]);
    const auto action = std::get<1>(memory[t]);

    // and calculate its n-step return (until time step t+n)
    float n_step_return = 0.0f;
    for (; t < n; ++t) {
        const auto reward = std::get<2>(memory[t]);
        n_step_return += std::pow(GAMMA, t)*reward;
    }
    const auto next_state = std::get<3>(memory[n-1]);
    const auto done = std::get<4>(memory[n-1]);
    if (not done) { n_step_return += std::pow(GAMMA, n)*model.predict_value(next_state); }

    memory.pop_front();
    return {curr_state, action, n_step_return, next_state, done};
}

void agent(uint32_t id)
{
    auto env = gym_uds::Environment("/tmp/gym-uds-socket-GA3C-" + std::to_string(id));

    for (uint32_t episode = 1; true; ++episode) {
        const float t = (float)num_training_experiences / (MAX_NUM_TRAINING_EXPERIENCES-1);
        const float epsilon = (1-t)*EPSILON_START + t*EPSILON_END;

        // reset the environment starting a new episode
        gym_uds::observation_t curr_state, next_state;
        curr_state = env.reset();

        memory_t memory;
        float reward = 0.0f;
        float episode_reward = 0.0f;

        bool done = false;
        while (not done) {
            // stop training after reaching a fixed number of training experiences
            if (num_training_experiences >= MAX_NUM_TRAINING_EXPERIENCES) { return; }

            const auto action = select_action(env, curr_state, epsilon);
            std::tie(next_state, reward, done) = env.step(action);
            episode_reward += reward;

            experience_t experience = {curr_state, action, reward, next_state, done};
            memory.push_back(experience);

            // add experiences to the training queue when:
            // - enough of them are accumulated in memory and the n-step return can be computed
            // - the episode is over and the memory is not empty
            while ((memory.size() >= N_STEP_RETURN) or (done and not memory.empty())) {
                const auto experience = extract_accumulated_experience(memory);
                training_experiences_queue.push(experience);
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

void fit(const std::vector<experience_t>& batch)
{
    std::vector<gym_uds::observation_t> states;
    std::vector<gym_uds::action_t> actions;
    std::vector<float> rewards;

    for (const auto experience: batch) {
        const auto curr_state = std::get<0>(experience);
        const auto action = std::get<1>(experience);
        const auto reward = std::get<2>(experience);

        states.push_back(curr_state);
        actions.push_back(action);
        rewards.push_back(reward);
    }
    model.fit(states, actions, rewards);
}

void trainer()
{
    while (num_training_experiences < MAX_NUM_TRAINING_EXPERIENCES) {
        const auto batch_size = std::min(BATCH_SIZE, MAX_NUM_TRAINING_EXPERIENCES-num_training_experiences);

        auto batch = std::vector<experience_t>(batch_size);
        for (auto i = 0; i < batch.size(); ++i) {
            batch[i] = training_experiences_queue.pop();
        }
        fit(batch);
        num_training_experiences += batch_size;
    }
}

/******************************************************************************/

int main(int argc, char const *argv[])
{
    assert(NUM_AGENTS >= 0);
    assert(MAX_NUM_TRAINING_EXPERIENCES >= 0);

    assert(BATCH_SIZE >= 0);

    assert(N_STEP_RETURN >= 1);
    assert(GAMMA >= 0.0f and GAMMA <= 1.0f);

    assert(EPSILON_START >= 0.0f and EPSILON_START <= 1.0f);
    assert(EPSILON_END   >= 0.0f and EPSILON_END   <= 1.0f and EPSILON_END <= EPSILON_START);


    const auto start = std::chrono::steady_clock::now();

    // start the trainer and the agents
    auto trainer_thread = std::thread(trainer);
    auto agents_threads = std::vector<std::thread>(NUM_AGENTS);
    for (auto i = 0; i < NUM_AGENTS; ++i) {
        agents_threads[i] = std::thread(agent, i);
    }

    // and wait for them to finish
    for (auto i = 0; i < NUM_AGENTS; ++i) {
        agents_threads[i].join();
    }
    trainer_thread.join();
    assert(num_training_experiences == MAX_NUM_TRAINING_EXPERIENCES);

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<float> duration_s = end-start;
    std::cout << "Training finished in " << duration_s.count() << " seconds" << std::endl;

    // save the trained model
    model.save();
}
