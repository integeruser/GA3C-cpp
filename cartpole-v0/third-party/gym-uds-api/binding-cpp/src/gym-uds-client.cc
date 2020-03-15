#include <iostream>

#include "gym-uds.h"
#include "gym-uds.pb.h"


int main(int argc, char const *argv[])
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto env = gym_uds::Environment("/tmp/gym-uds-socket");

    const int num_episodes = 3;
    for (int episode = 1; episode <= num_episodes; ++episode) {
        gym_uds::observation_t observation = env.reset();

        float reward, episode_reward = 0.0f;
        bool done = false;
        while (!done) {
            gym_uds::action_t action = env.sample();
            std::tie(observation, reward, done) = env.step(action);
            episode_reward += reward;
        }
        std::cout << "Ep. " << episode << ": " << episode_reward << std::endl;
    }

    return 0;
}
