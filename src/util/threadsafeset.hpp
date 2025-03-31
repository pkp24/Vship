#ifndef THREADSAFESETHPP
#define THREADSAFESETHPP
#include "preprocessor.hpp"

class threadSet{
    std::condition_variable_any com;
    std::mutex lock;
    std::set<int> data;
public:
    threadSet(std::set<int> in){
        data = in;
    }
    void insert(int a){
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
    int pop(){
        //return the lowest element
        lock.lock();
        while (data.empty()){
            com.wait(lock);
        }
        int ret = *data.begin();
        data.erase(ret);
        lock.unlock();
        return ret;
    }
};

#endif