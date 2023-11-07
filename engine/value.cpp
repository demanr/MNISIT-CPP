#include <set>
#include <functional>
#include <string>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <math.h> // for power

class Value
{
public:
    double data;
    double grad;
    std::function<void()> backwards;
    std::set<Value *> prev;
    std::string op;

    // base constructor
    Value(double data)
    {
        this->data = data;
        this->grad = 0.0;
    }

    // constructor with children and op
    Value(double data, std::set<Value *> children, std::string op)
    {
        this->data = data;
        this->grad = 0.0;
        this->prev = children;
        this->op = op;
    }

    // get and set data
    double getData()
    {
        return this->data;
    }

    void setData(double data)
    {
        this->data = data;
    }

    // get and set grad
    double getGrad()
    {
        return this->grad;
    }

    void setGrad(double grad)
    {
        this->grad = grad;
    }

    // get and set op
    std::set<Value *> getPrev()
    {
        return this->prev;
    }

    void setPrev(std::set<Value *> prev)
    {
        this->prev = prev;
    }
    // operator overloading
    Value operator+(Value &other)
    {
        std::set<Value *> outChildren = {this, &other};
        Value out = Value((this->data + other.data), outChildren, "+");

        return out;
    }
    // multiplication
    Value operator*(Value &other)
    {
        std::set<Value *> outChildren = {this, &other};
        Value out = Value((this->data * other.data), outChildren, "*");

        return out;
    }
    // division
    Value operator/(Value &other)
    {
        std::set<Value *> outChildren = {this, &other};
        Value out = Value((this->data / other.data), outChildren, "*");

        return out;
    }
    // subtraction
    Value operator-(Value &other)
    {
        std::set<Value *> outChildren = {this, &other};
        Value out = Value((this->data - other.data), outChildren, "-");

        return out;
    }
    // negation
    Value operator-()
    {
        return Value(-(this->data));
    }
    // power
    Value operator^(Value &other)
    {
        std::set<Value *> outChildren = {this, &other};
        double outData = pow(this->data, other.data);

        Value out = Value(outData, outChildren, "^");

        return out;
    }
};

int main()
{
    Value a = Value(1.0);
    Value b = Value(-1.0);
    Value c = Value(3.0);
    Value e = c + a;
    Value d = c ^ b;

    printf("%f\n", b.getData());
    printf("%f\n", c.getData());
    printf("%f\n", d.getData());
    for (Value *p : d.getPrev())
    {
        printf("children:%f\n", (*p).getData());
    }

    return 0;
}