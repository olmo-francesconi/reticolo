#include <any>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/core.hpp"

template <class T>
class BaseTempl {
  public:
    virtual ~BaseTempl(){};
    virtual void print() = 0;
    virtual void set(T) = 0;
    virtual T    get() = 0;
};

template <class T>
class Deriv : public BaseTempl<T> {
  private:
    T _Val;

  public:
    void print() override { std::cout << _Val << '\n'; };
    void set(T val) override { _Val = val; };
    T    get() override { return _Val; };
};

template <class T>
auto factory() -> std::unique_ptr<BaseTempl<T>> {
    return std::make_unique<Deriv<T>>();
}

int main(int argc, char* argv[]) {
    auto Obj = factory<int>();

    Obj->print();
    return EXIT_SUCCESS;
}