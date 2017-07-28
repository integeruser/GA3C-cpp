#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <string>
#include <vector>

#include "gym-uds.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;


class Model
{
    private:
        const std::string GRAPH_FILEPATH = "models/graph";
        const std::string META_GRAPH_FILEPATH = "models/graph.meta";

        tf::MetaGraphDef meta_graph_def;
        std::unique_ptr<tf::Session> session;

    public:
        Model();

        void fit(const std::vector<gym_uds::observation_t>&,
                 const std::vector<gym_uds::action_t>&,
                 const std::vector<float>&);

        std::vector<float> predict_policy(const gym_uds::observation_t&);
        float predict_reward(const gym_uds::observation_t&);

        void save();
};


#endif
