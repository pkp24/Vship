#ifndef THREADSAFESETHPP
#define THREADSAFESETHPP
#include "preprocessor.hpp"

#include <optional>

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

template<typename T>
class ClosableThreadSet{
    std::condition_variable_any com;
    std::mutex lock;
    std::set<T> data;
    bool closed = false;
public:
    ClosableThreadSet(const std::set<T>& in){
        data = in;
    }
    void insert(const T& a){
        lock.lock();
        if (closed) std::cerr << "Error: something inserted into a closed threadset. That likely means something is wrong" << std::endl;
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
    std::optional<T> pop(){
        //return the lowest element
        lock.lock();
        if (closed) return std::nullopt;
        while (data.empty()){
            com.wait(lock);
            if (closed) return std::nullopt;
        }
        T ret = *data.begin();
        data.erase(ret);
        lock.unlock();
        return ret;
    }
    void close(){
        lock.lock();
        closed = true;
        com.notify_all();
        lock.unlock();
    }
};

#endif