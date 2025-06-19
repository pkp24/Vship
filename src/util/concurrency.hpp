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

template <typename ElementType> class ThreadSafePoolVecWrapper {
  private:
    std::vector<ElementType> pool_elements;
    std::vector<size_t> free_indices;
    mutable std::mutex pool_mutex;

    static constexpr bool is_element_pointer =
        std::is_pointer<ElementType>::value;

    using AcquiredPointerType =
        std::conditional_t<is_element_pointer,
                           std::remove_pointer_t<ElementType> *, ElementType *>;

  public:
    explicit ThreadSafePoolVecWrapper(std::vector<ElementType> initial_elements)
        : pool_elements(std::move(initial_elements)) {
        free_indices.reserve(pool_elements.size());
        for (size_t i = 0; i < pool_elements.size(); ++i) {
            free_indices.push_back(i);
        }
    }

    ThreadSafePoolVecWrapper(const ThreadSafePoolVecWrapper &) = delete;
    ThreadSafePoolVecWrapper &
    operator=(const ThreadSafePoolVecWrapper &) = delete;

    AcquiredPointerType acquire() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (free_indices.empty()) {
            return nullptr;
        }

        const size_t index = free_indices.back();
        free_indices.pop_back();

        if constexpr (is_element_pointer) {
            return pool_elements[index];
        } else {
            return &pool_elements[index];
        }
    }

    void release(AcquiredPointerType object_pointer) {
        ASSERT_WITH_MESSAGE(object_pointer != nullptr,
                            "Attempted to release null object.");

        std::lock_guard<std::mutex> lock(pool_mutex);

        size_t index = static_cast<size_t>(-1);

        if constexpr (is_element_pointer) {
            for (size_t i = 0; i < pool_elements.size(); ++i) {
                if (pool_elements[i] == object_pointer) {
                    index = i;
                    break;
                }
            }

            ASSERT_WITH_MESSAGE(index != static_cast<size_t>(-1),
                                "Pointer not found in pool (release failure).");

        } else {
            ASSERT_WITH_MESSAGE(object_pointer >= &pool_elements[0] &&
                                    object_pointer < &pool_elements[0] +
                                                         pool_elements.size(),
                                "Pointer out of bounds for release.");

            index = static_cast<size_t>(object_pointer - &pool_elements[0]);
        }

        ASSERT_WITH_MESSAGE(index < pool_elements.size(),
                            "Invalid pool index during release.");

        free_indices.push_back(index);
    }

    size_t capacity() const noexcept { return pool_elements.size(); }

    size_t free_count() const noexcept {
        std::lock_guard<std::mutex> lock(pool_mutex);
        return free_indices.size();
    }
};

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
