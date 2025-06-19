#pragma once

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>

#ifndef ASSERT_WITH_MESSAGE
#define ASSERT_WITH_MESSAGE(condition, message)                                \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr,                                               \
                         "Assertion failed!\n"                                 \
                         "  Expression : %s\n"                                 \
                         "  File       : %s\n"                                 \
                         "  Line       : %d\n"                                 \
                         "  Message    : %s\n",                                \
                         #condition, __FILE__, __LINE__, message);             \
            std::abort();                                                      \
        }                                                                      \
    } while (false)

#endif

template <typename ElementType> class ThreadSafeQueue {
  private:
    std::queue<ElementType> internal_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_cv_;
    std::condition_variable queue_not_full_cv_;
    const size_t maximum_capacity_;
    bool is_queue_closed_ = false;

  public:
    explicit ThreadSafeQueue(size_t max_capacity)
        : maximum_capacity_(max_capacity) {
        ASSERT_WITH_MESSAGE(maximum_capacity_ > 0,
                            "Queue capacity must be greater than 0");
    }

    ThreadSafeQueue(const ThreadSafeQueue &) = delete;
    ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;

    bool push(ElementType element) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_not_full_cv_.wait(lock, [this]() {
            return internal_queue_.size() < maximum_capacity_ ||
                   is_queue_closed_;
        });

        ASSERT_WITH_MESSAGE(!is_queue_closed_,
                            "Attempt to push on a closed queue");

        ASSERT_WITH_MESSAGE(internal_queue_.size() < maximum_capacity_,
                            "Queue full on push - unexpected state");

        internal_queue_.push(std::move(element));
        lock.unlock();
        queue_not_empty_cv_.notify_one();
        return true;
    }

    std::optional<ElementType> pop() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_not_empty_cv_.wait(lock, [this]() {
            return !internal_queue_.empty() || is_queue_closed_;
        });

        if (internal_queue_.empty()) {
            return std::nullopt;
        }

        ElementType element = std::move(internal_queue_.front());
        internal_queue_.pop();
        lock.unlock();
        queue_not_full_cv_.notify_one();
        return element;
    }

    void close() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        is_queue_closed_ = true;
        queue_not_empty_cv_.notify_all();
        queue_not_full_cv_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return is_queue_closed_;
    }
    size_t capacity() const noexcept { return maximum_capacity_; }
};

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
