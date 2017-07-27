#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <tuple>
#include <vector>

using action_t = int;
using observation_t = std::vector<float>;
using state_t = std::tuple<observation_t, float, bool>;


class Environment
{
    public:
        virtual ~Environment() {}

        virtual observation_t reset()=0;
        virtual state_t step(const action_t&)=0;

        virtual action_t sample()=0;
};


class DummyEnvironment : public Environment
{
    public:
        observation_t reset() { return observation_t(); }
        state_t step(const action_t&) { return state_t(); }

        action_t sample() { return action_t(); }
};

#endif
