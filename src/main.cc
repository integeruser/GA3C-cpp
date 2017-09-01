#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "environment.h"
#include "model.h"


extern void train(std::shared_ptr<Model>, const std::vector<std::shared_ptr<Environment>>&);

uint32_t NUM_AGENTS = 5;
uint32_t MAX_NUM_TRAINING_EXPERIENCES = 75000;


int main(int argc, char const *argv[])
{
    const auto start = std::chrono::steady_clock::now();

    // train the model
    auto model = std::make_shared<Model>();
    std::vector<std::shared_ptr<Environment>> environments;
    for (uint32_t id = 0; id < NUM_AGENTS; ++id) {
        auto env = std::make_shared<Environment>();
        environments.push_back(env);
    }
    train(model, environments);

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<float> duration_s = end-start;
    std::cout << "Training finished in " << duration_s.count() << " seconds" << std::endl;

    // save the trained model
    model->save();
}
