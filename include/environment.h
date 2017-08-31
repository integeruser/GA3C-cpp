#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using action_t = int64_t;
using observation_t = std::vector<float>;
using state_t = std::tuple<observation_t, float, bool>;


class Environment
{
    public:
        Environment() {}

        virtual observation_t reset() { return {}; }
        virtual state_t step(const action_t&) { return {}; }

        virtual action_t sample() { return {}; }
};


#endif // ENVIRONMENT_H
