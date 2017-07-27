#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <string>
#include <vector>

#include "environment.h"


class Model
{
    public:
        virtual ~Model() {}

        virtual void fit(const std::vector<observation_t>&,
                         const std::vector<action_t>&,
                         const std::vector<float>&)=0;

        virtual std::vector<float> predict_policy(const observation_t&)=0;
        virtual float predict_reward(const observation_t&)=0;

        virtual void save()=0;
        virtual void restore()=0;
};


class DummyModel : public Model
{
    public:
        void fit(const std::vector<observation_t>&,
                 const std::vector<action_t>&,
                 const std::vector<float>&) {}

        std::vector<float> predict_policy(const observation_t&) { return std::vector<float>(); }
        float predict_reward(const observation_t&) { return float(); }

        void save() {}
        void restore() {}
};


#endif
