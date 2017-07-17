#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <string>
#include <vector>

#include "common.h"

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

        void fit(const std::vector<observation_t>&,
                 const std::vector<action_t>&,
                 const std::vector<float>&);

        action_t predict_policy(const observation_t&);
        float predict_reward(const observation_t&);

        void save();
};


#endif
