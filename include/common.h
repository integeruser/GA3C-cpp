#ifndef COMMON_H
#define COMMON_H

#include <deque>
#include <tuple>
#include <vector>

using observation_t = std::vector<float>;
using action_t = std::vector<float>;

using experience_t = std::tuple<observation_t, action_t, float, observation_t, bool>;
using memory_t = std::deque<experience_t>;


#endif
