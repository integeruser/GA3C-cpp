#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <string>
#include <vector>

#include "gym.h"

#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;


class Model
{
    private:
        const std::string graph_filepath = "models/graph";
        const std::string metagraph_filepath = "models/graph.meta";
        std::unique_ptr<tf::Session> session;

    public:
        Model();

        void fit(const std::vector<gym::observation_t>&,
                 const std::vector<std::vector<float>>&,
                 const std::vector<gym::reward_t>&);

        std::vector<float> predict_policy(const gym::observation_t&);
        gym::reward_t predict_reward(const gym::observation_t&);

        void save();
};


#endif
