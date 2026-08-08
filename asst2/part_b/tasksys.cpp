#include "tasksys.h"

#include <algorithm>


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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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
    next_task_id_ = 0;
    active_bulk_tasks_ = 0;

    auto worker_logic = [&](){
        while (true) {
            BulkTask* current_bulk_task = nullptr;
            int my_task_id = -1;
            int total_tasks = -1;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                worker_cv_.wait(lock, [this]{
                    return !ready_queue_.empty() || stop_pool_;
                });

                if (ready_queue_.empty() && stop_pool_) {
                    return;
                }
                
                current_bulk_task = ready_queue_.front();
                my_task_id = current_bulk_task->current_task_id_++;
                total_tasks = current_bulk_task->num_total_tasks_;

                if (my_task_id == total_tasks - 1) {
                    ready_queue_.pop();
                }
            }

            current_bulk_task->runnable_->runTask(my_task_id, total_tasks);

            int threads_to_notify = 0;
            bool delete_bulk_task = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                current_bulk_task->tasks_remaining_--;
                if (current_bulk_task->tasks_remaining_ == 0) {
                    delete_bulk_task = true;
                    completed_tasks_.insert(current_bulk_task->id_);

                    for (TaskID dependent_task_id : dependency_graph_[current_bulk_task->id_]) {
                        dependency_count_[dependent_task_id]--;
                        if (dependency_count_[dependent_task_id] == 0) {
                            ready_queue_.push(waiting_tasks_[dependent_task_id]);
                            threads_to_notify += waiting_tasks_[dependent_task_id]->num_total_tasks_;
                            waiting_tasks_.erase(dependent_task_id);
                        }
                    }
                    dependency_graph_.erase(current_bulk_task->id_);

                    active_bulk_tasks_--;
                    if (active_bulk_tasks_ == 0) {
                        sync_cv_.notify_all();
                    }
                }
            }
            threads_to_notify = std::min(threads_to_notify, (int)workers_.size());
            for (int i = 0; i < threads_to_notify; i++) {
                worker_cv_.notify_one();
            }
            if (delete_bulk_task) {
                delete current_bulk_task;
            }
        }
    };

    for (int i = 0; i < num_threads - 1; i++) {
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
        std::lock_guard<std::mutex> lock(mutex_);
        stop_pool_ = true;
    }
    worker_cv_.notify_all();

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

    runAsyncWithDeps(runnable, num_total_tasks, {});
    sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    TaskID my_task_id;

    int threads_to_notify = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        my_task_id = next_task_id_++;
        BulkTask* new_bulk_task = new BulkTask{my_task_id, runnable, num_total_tasks, 0, num_total_tasks};
        active_bulk_tasks_++;

        std::vector<TaskID> current_deps;
        for (TaskID dep : deps) {
            if (completed_tasks_.find(dep) == completed_tasks_.end()) {
                current_deps.push_back(dep);
            }
        }
        
        if (current_deps.empty()) {
            ready_queue_.push(new_bulk_task);
            threads_to_notify = std::min(new_bulk_task->num_total_tasks_, (int)workers_.size());
        } else {
            waiting_tasks_[my_task_id] = new_bulk_task;
            dependency_count_[my_task_id] = current_deps.size();
            for (TaskID dep : current_deps) {
                dependency_graph_[dep].push_back(my_task_id);
            }
        }
    }
    for (int i = 0; i < threads_to_notify; i++) {
        worker_cv_.notify_one();
    }

    return my_task_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    std::unique_lock<std::mutex> lock(mutex_);
    sync_cv_.wait(lock, [this]{
        return active_bulk_tasks_ == 0;
    });
}
