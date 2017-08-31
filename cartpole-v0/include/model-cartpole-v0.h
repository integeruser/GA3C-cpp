#ifndef CARTPOLE_MODEL_H
#define CARTPOLE_MODEL_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

#include "model.h"

namespace tf = tensorflow;


class CartPoleModel : public Model
{
    private:
        const std::string GRAPH_FILEPATH = "models/graph";
        const std::string META_GRAPH_FILEPATH = "models/graph.meta";

        tf::MetaGraphDef meta_graph_def;
        std::unique_ptr<tf::Session> session;

    public:
        CartPoleModel();

        void fit(const std::vector<observation_t>&,
                 const std::vector<action_t>&,
                 const std::vector<float>&);

        std::pair<std::vector<float>, float> predict_policy_and_value(const observation_t&);

        void save();
};


#endif // CARTPOLE_MODEL_H
