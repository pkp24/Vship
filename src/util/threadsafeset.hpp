#ifndef THREADSAFESETHPP
#define THREADSAFESETHPP
#include "preprocessor.hpp"

template<typename T>
class threadSet{
    std::condition_variable_any com;
    std::mutex lock;
    std::set<T> data;
public:
    threadSet(const std::set<T>& in){
        data = in;
    }
    void insert(const T& a){
        lock.lock();
        com.notify_one();
        data.insert(a);
        lock.unlock();
    }
    bool empty(){
        lock.lock();
        bool ret = data.empty();
        lock.unlock();
        return ret;
    }
    T pop(){
        //return the lowest element
        lock.lock();
        while (data.empty()){
            com.wait(lock);
        }
        T ret = *data.begin();
        data.erase(ret);
        lock.unlock();
        return ret;
    }
};

#endif