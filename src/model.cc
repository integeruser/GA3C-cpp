#include "model.h"

#include <algorithm>
#include <cassert>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"


const auto NUM_STATES  = 4;
const auto NUM_ACTIONS = 2;


Model::Model():
    session(tf::NewSession( {}))
{
    // create the graph
    tf::MetaGraphDef metagraph_def;
    TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), metagraph_filepath, &metagraph_def));
    TF_CHECK_OK(session->Create(metagraph_def.graph_def()));

    // and restore the values of its variables
    auto graph_filepath_tensor = tf::Tensor(tf::DT_STRING, {});
    graph_filepath_tensor.scalar<tf::string>()() = graph_filepath;

    std::vector<std::pair<tf::string, tf::Tensor>> inputs = {
        {metagraph_def.saver_def().filename_tensor_name(), graph_filepath_tensor}
    };
    TF_CHECK_OK(session->Run(inputs, {}, {metagraph_def.saver_def().restore_op_name()}, nullptr));
}


void Model::fit(const std::vector<gym::observation_t>& states,
                const std::vector<std::vector<float>>& actions,
                const std::vector<float>& rewards)
{
    assert(states.size() == actions.size() and actions.size() == rewards.size());

    // fill the states tensor
    auto states_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(states.size()), NUM_STATES});
    auto states_eigenmatrix = states_tensor.matrix<float>();
    for (auto i = 0; i < states.size(); ++i) {
        const auto state = states[i];
        for (auto j = 0; j < state.size(); ++j) {
            states_eigenmatrix(i, j) = state[j];
        }
    }

    // fill the actions tensor
    auto actions_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(actions.size()), NUM_ACTIONS});
    auto actions_eigenmatrix = actions_tensor.matrix<float>();
    for (auto i = 0; i < actions.size(); ++i) {
        const auto action = actions[i];
        for (auto j = 0; j < action.size(); ++j) {
            actions_eigenmatrix(i, j) = action[j];
        }
    }

    // fill the rewards tensor
    auto rewards_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(rewards.size()), 1});
    auto rewards_eigenmatrix = rewards_tensor.matrix<float>();
    for (auto i = 0; i < rewards.size(); ++i) {
        rewards_eigenmatrix(i, 0) = rewards[i];
    }

    // train the model
    std::vector<std::pair<tf::string, tf::Tensor>> inputs = {
        {"x_states", states_tensor}, {"y_policies", actions_tensor}, {"y_values", rewards_tensor}
    };
    TF_CHECK_OK(session->Run(inputs, {}, {"minimize:0"}, nullptr));
}


std::vector<float> Model::predict_policy(const gym::observation_t& state)
{
    // fill the state tensor
    auto state_tensor = tf::Tensor(tf::DT_FLOAT, {1, NUM_STATES});
    std::copy_n(state.cbegin(), state.size(), state_tensor.flat<float>().data());

    // predict output
    std::vector<std::pair<tf::string, tf::Tensor>> inputs = {
        {"x_states", {state_tensor}}
    };
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(session->Run(inputs, {"out_policies/Softmax:0"}, {}, &outputs));

    const auto out_policies_eigentensor = outputs[0].flat<float>();
    std::vector<float> out_policies(out_policies_eigentensor.size());
    for (auto i = 0; i < out_policies.size(); ++i) {
        out_policies[i] = out_policies_eigentensor(i);
    }
    const auto sum = std::accumulate(out_policies.cbegin(), out_policies.cend(), 0.0f);
    assert(sum > 0.99f and sum < 1.01f);
    return out_policies;
}

float Model::predict_reward(const gym::observation_t& state)
{
    // fill the state tensor
    auto state_tensor = tf::Tensor(tf::DT_FLOAT, {1, NUM_STATES});
    std::copy_n(state.cbegin(), state.size(), state_tensor.flat<float>().data());

    // predict output
    std::vector<std::pair<tf::string, tf::Tensor>> inputs = {
        {"x_states", {state_tensor}}
    };
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(session->Run(inputs, {"out_values/BiasAdd:0"}, {}, &outputs));

    const float reward = outputs[0].scalar<float>()();
    return reward;
}


void Model::save()
{
    auto graph_filepath_tensor = tf::Tensor(tf::DT_STRING, {});
    graph_filepath_tensor.scalar<tf::string>()() = graph_filepath;
    TF_CHECK_OK(session->Run({{"save/Const:0", graph_filepath_tensor}}, {}, {"save/control_dependency:0"}, nullptr));
}
