#include <condition_variable>
#include <deque>
#include <mutex>

/* https://stackoverflow.com/questions/12805041/c-equivalent-to-javas-blockingqueue */

template <typename T>
class Queue
{
    private:
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<T> queue;

    public:
        void push(T const& value)
        {
            {
                std::unique_lock<std::mutex> lock(this->mutex);
                queue.push_front(value);
            }
            this->condition.notify_one();
        }
        T pop()
        {
            std::unique_lock<std::mutex> lock(this->mutex);
            this->condition.wait(lock, [=] { return !this->queue.empty(); });
            T rc(std::move(this->queue.back()));
            this->queue.pop_back();
            return rc;
        }
};
