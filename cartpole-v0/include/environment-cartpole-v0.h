#ifndef CARTPOLE_ENVIRONMENT_H
#define CARTPOLE_ENVIRONMENT_H

#include <cstdint>
#include <string>

#include "environment.h"
#include "gym-uds.h"


class CartPoleEnvironment : public Environment
{
    private:
        gym_uds::Environment gym_uds_env;

    public:
        CartPoleEnvironment(uint64_t id) : gym_uds_env("/tmp/gym-uds-socket-GA3C-" + std::to_string(id)) {}

        observation_t reset() { return gym_uds_env.reset(); }
        state_t step(const action_t& action) { return gym_uds_env.step(action); }

        action_t sample() { return gym_uds_env.sample(); }
};


#endif // CARTPOLE_ENVIRONMENT_H
