#ifndef GYM_H
#define GYM_H

#include <string>
#include <tuple>
#include <vector>

namespace gym_uds
{
using action_t = int;
using observation_t = std::vector<float>;
using state_t = std::tuple<observation_t, float, bool>;


class Environment
{
    private:
        int sock;

        template<typename T>
        T recv_message();

        template<typename T>
        void send_message(const T&);

    public:
        Environment(const std::string&);

        observation_t reset();
        state_t step(const action_t&);

        action_t sample();
};
}


#endif
