#include "value.cpp";

class Module
{
    void zeroGrad()
    {
        for (auto &p : parameters())
        {
            p->setGrad(0.0);
        }
    }

    std::vector<Value *> parameters()
    {
        return std::vector<Value *>();
    }
};
