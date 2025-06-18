#pragma once

#include<mutex>
#include<iostream>

//this progressBar will be owned by multiple threads so it must be threadSafe
//the maximum is not known at creation
template<int BarSize>
class ProgressBar{
    std::mutex lock;
    int counter = 0;
    int max_counter = 0;
    int refresh_rate = 0;
    double value_sum = 0; //is used to print real time average
public:
    void set_max(int new_max){
        lock.lock();
        max_counter = new_max;
        lock.unlock();
    }
    ProgressBar(int refresh_rate, int max_counter = 0) : refresh_rate(refresh_rate), max_counter(max_counter){

    }
};