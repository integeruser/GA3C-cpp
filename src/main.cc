#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
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
const uint32_t NUM_TRAINING_STEPS = 75000;

const uint32_t BATCH_SIZE = 32;

const uint32_t N_STEP_RETURN = 8;
const float DISCOUNT   = 0.99f;
const float DISCOUNT_N = std::pow(DISCOUNT, N_STEP_RETURN);

const float EPSILON_START = 1.00f;
const float EPSILON_END   = 0.15f;

Model model;
Queue<experience_t> experiences_queue;
uint32_t training_steps_done = 0;
std::mt19937 rand_engine(1337);

/******************************************************************************/

experience_t get_sample(const memory_t& memory, float R, int n)
{
    const auto curr_state = std::get<0>(memory[0]);
    const auto action = std::get<1>(memory[0]);
    const auto next_state = std::get<3>(memory[n-1]);
    const auto done = std::get<4>(memory[n-1]);
    return {curr_state, action, R, next_state, done};
}

void update(memory_t& memory, float& R, const experience_t& experience)
{
    gym_uds::observation_t curr_state, next_state;
    gym_uds::action_t action;
    float reward;
    bool done;
    std::tie(curr_state, action, reward, next_state, done) = experience;

    memory.push_back({curr_state, action, reward, next_state, done});

    R = (R + reward*DISCOUNT_N) / DISCOUNT;

    if (done) {
        while (memory.size() > 0) {
            const auto n = memory.size();
            const auto experience = get_sample(memory, R, n);
            experiences_queue.push(experience);
            ++training_steps_done;

            const auto reward = std::get<2>(memory[0]);
            R = (R - reward) / DISCOUNT;
            memory.pop_front();
        }
        R = 0.0f;
    }

    if (memory.size() >= N_STEP_RETURN) {
        const auto experience = get_sample(memory, R, N_STEP_RETURN);
        experiences_queue.push(experience);
        ++training_steps_done;

        const auto reward = std::get<2>(memory[0]);
        R = (R - reward);
        memory.pop_front();
    }

    // possible edge case - if an episode ends in <N steps, the computation is incorrect
}

gym_uds::action_t pick_action(gym_uds::Environment& env, const gym_uds::observation_t& state, float epsilon)
{
    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rand_engine) < epsilon) {
        return env.sample();
    }
    else {
        const auto actions_probs = model.predict_policy(state);
        return std::discrete_distribution<float>(actions_probs.cbegin(), actions_probs.cend())(rand_engine);
    }
}

void agent(uint32_t i)
{
    auto env = gym_uds::Environment("/tmp/gym-uds-socket-GA3C-" + std::to_string(i));

    memory_t memory;
    float R = 0.0f;

    for (uint32_t episode = 1; true; ++episode) {
        gym_uds::observation_t curr_state, next_state;
        curr_state = env.reset();

        float t = (float)training_steps_done / (NUM_TRAINING_STEPS-1);
        float epsilon = (1-t)*EPSILON_START + t*EPSILON_END;

        float reward = 0.0f;
        float episode_reward = 0.0f;

        bool done = false;
        while (!done) {
            if (training_steps_done > NUM_TRAINING_STEPS) { return; }

            const auto action = pick_action(env, curr_state, epsilon);
            std::tie(next_state, reward, done) = env.step(action);
            episode_reward += reward;

            experience_t experience = {curr_state, action, reward, next_state, done};
            update(memory, R, experience);

            curr_state = next_state;
        }
        std::cout << episode << ": " << episode_reward << std::endl;
    }
}

/******************************************************************************/

void fit(const std::vector<experience_t>& batch)
{
    std::vector<gym_uds::observation_t> states;
    std::vector<gym_uds::action_t> actions;
    std::vector<float> rewards;

    for (const auto& experience: batch) {
        gym_uds::observation_t curr_state, next_state;
        gym_uds::action_t action;
        float reward;
        bool done;
        std::tie(curr_state, action, reward, next_state, done) = experience;

        states.push_back(curr_state);
        actions.push_back(action);
        if (!done) { reward += model.predict_reward(next_state)*DISCOUNT_N; }
        rewards.push_back(reward);
    }

    model.fit(states, actions, rewards);
}

void trainer()
{
    while (training_steps_done < NUM_TRAINING_STEPS) {
        const auto batch_size = std::min(BATCH_SIZE, NUM_TRAINING_STEPS-training_steps_done);

        auto batch = std::vector<experience_t>(batch_size);
        for (auto i = 0; i < batch.size(); ++i) {
            batch[i] = experiences_queue.pop();
        }
        fit(batch);
    }
}

/******************************************************************************/

int main(int argc, char const *argv[])
{
    // start the trainer
    auto trainer_thread = std::thread(trainer);

    // start the agents
    auto agents_threads = std::vector<std::thread>(NUM_AGENTS);
    for (auto i = 0; i < NUM_AGENTS; ++i) {
        agents_threads[i] = std::thread(agent, i);
    }

    // and wait for them to finish
    for (auto i = 0; i < NUM_AGENTS; ++i) {
        agents_threads[i].join();
    }
    trainer_thread.join();

    // save the trained model
    model.save();
}
