#include "tasksys.h"

#include <mutex>
#include <atomic>
#include <thread>
#include <vector>



IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    num_threads_ = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // atomic ver.
    std::atomic<int> current_task_id(0);
    auto worker_logic = [&](){
        while (true) {
            int my_task_id = current_task_id.fetch_add(1);
            if (my_task_id >= num_total_tasks) {
                break;
            }
            runnable->runTask(my_task_id, num_total_tasks);
        }
    };

    /* lock ver.
    std::mutex mutex_;
    int current_task_id = 0;
    
    auto worker_logic = [&](){
        while (true) {
            int my_task_id = -1;

            mutex_.lock();
            if (current_task_id >= num_total_tasks) {
                mutex_.unlock();
                break;
            }
            my_task_id = current_task_id;
            current_task_id++;
            mutex_.unlock();

            runnable->runTask(my_task_id, num_total_tasks);
        }
    };
    */

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads_; i++) {
        threads.push_back(std::thread(worker_logic));
    }

    for (int i = 0; i < num_threads_; i++) {
        threads[i].join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    stop_pool_.store(false);
    current_runnable_ = nullptr;

    num_total_tasks_.store(-1);  // avoid workers start working before all setting are done
    current_task_id_.store(0);
    completed_task_.store(0);

    auto worker_logic = [&](){
        while (!stop_pool_) {
            if (current_task_id_.load() < num_total_tasks_) {      // condition to start working
                int my_task_id = current_task_id_.fetch_add(1);
                if (my_task_id < num_total_tasks_) {
                    current_runnable_->runTask(my_task_id, num_total_tasks_);
                    completed_task_.fetch_add(1);
                }
            }
        }
    };

    for (int i = 0; i < num_threads-1; i++) {
        workers_.push_back(std::thread(worker_logic));
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    stop_pool_.store(true);
    for (int i = 0; i < (int)workers_.size(); i++) {
        workers_[i].join();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    
    num_total_tasks_.store(-1);  // avoid workers start working before all setting are done

    current_runnable_ = runnable;
    current_task_id_.store(0);
    completed_task_.store(0);

    num_total_tasks_.store(num_total_tasks);  // workers can start working now

    while (completed_task_.load() < num_total_tasks) {
        if (current_task_id_.load() < num_total_tasks) {
            int my_task_id = current_task_id_.fetch_add(1);
            if (my_task_id < num_total_tasks) {
                current_runnable_->runTask(my_task_id, num_total_tasks);
                completed_task_.fetch_add(1);
            }
        }
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    stop_pool_ = false;
    num_total_tasks_ = 0;  // let workers sleep until run() is called
    current_task_id_ = 0;
    completed_task_ = 0;

    auto worker_logic = [&](){
        while (!stop_pool_) {
            if (current_task_id_.load() >= num_total_tasks_) {      // condition to start working
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&](){ return current_task_id_.load() < num_total_tasks_ || stop_pool_; });
                lock.unlock();
            }
            int my_task_id = current_task_id_.fetch_add(1);
            if (my_task_id < num_total_tasks_) {
                current_runnable_->runTask(my_task_id, num_total_tasks_);
                completed_task_.fetch_add(1);
            }
        }
    };

    for (int i = 0; i < num_threads-1; i++) {
        workers_.push_back(std::thread(worker_logic));
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    {
        std::lock_guard<std::mutex> lock(mutex_);  // use lock to avoid workers read inconsistent state of params
        stop_pool_ = true;
    }
    cv_.notify_all();  // wake up sleeping threads to exit

    for (int i = 0; i < (int)workers_.size(); i++) {
        workers_[i].join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    std::unique_lock<std::mutex> lock(mutex_);  // use lock to avoid workers read inconsistent state of params
    
    current_runnable_ = runnable;
    num_total_tasks_ = num_total_tasks;
    current_task_id_ = 0;
    completed_task_ = 0;

    lock.unlock();  // may have worker just start work here

    cv_.notify_all();  // wake up sleeping threads to start working

    while (current_task_id_.load() < num_total_tasks) {
        int my_task_id = current_task_id_.fetch_add(1);
        if (my_task_id < num_total_tasks) {
            current_runnable_->runTask(my_task_id, num_total_tasks);
            completed_task_.fetch_add(1);
        }
    }
    while (completed_task_.load() < num_total_tasks_) {
        std::this_thread::yield();
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
