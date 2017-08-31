#ifndef MODEL_H
#define MODEL_H

#include <utility>
#include <vector>

#include "environment.h"


class Model
{
    public:
        Model() {}

        virtual void fit(const std::vector<observation_t>&,
                         const std::vector<action_t>&,
                         const std::vector<float>&) {}

        virtual std::pair<std::vector<float>, float> predict_policy_and_value(const observation_t&) { return {}; }

        virtual void save() {}
};


#endif // MODEL_H
