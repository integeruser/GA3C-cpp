#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <string>
#include <vector>

#include "gym.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;


class Model
{
    private:
        const std::string graph_filepath = "models/graph";
        const std::string meta_graph_filepath = "models/graph.meta";
        tf::MetaGraphDef meta_graph_def;
        std::unique_ptr<tf::Session> session;

    public:
        Model();

        void fit(const std::vector<gym::observation_t>&,
                 const std::vector<gym::action_t>&,
                 const std::vector<float>&);

        std::vector<float> predict_policy(const gym::observation_t&);
        float predict_reward(const gym::observation_t&);

        void save();
        void restore();
};


#endif
