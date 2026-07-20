#include <stdio.h>
#include <algorithm>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "CycleTimer.h"
#include "saxpy_ispc.h"

extern void saxpySerial(int N, float a, float* X, float* Y, float* result);

void saxpy_avx2_stream(int N, float scale, float* X, float* Y, float* result) {
    __m256 v_scale = _mm256_set1_ps(scale);
    int i = 0;
    
    for (; i <= N - 8; i += 8) {
        __m256 v_X = _mm256_load_ps(&X[i]);
        __m256 v_Y = _mm256_load_ps(&Y[i]);
        
        // FMA 指令：(scale * X) + Y
        __m256 v_res = _mm256_fmadd_ps(v_scale, v_X, v_Y);
        
        // Non-Temporal Store (純寫入，繞過 Cache 直接寫回主記憶體)
        _mm256_stream_ps(&result[i], v_res);
    }

    // 處理尾部未滿 8 個元素的部分
    for (; i < N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

// 1. AVX2 Worker 函數
void saxpy_avx2_stream_worker(int start, int end, float scale, float* X, float* Y, float* result) {
    __m256 v_scale = _mm256_set1_ps(scale);
    int i = start;
    
    // 因為 start 保證是 8 的倍數，可以直接安全使用 _mm256_load_ps
    for (; i <= end - 8; i += 8) {
        __m256 v_X = _mm256_load_ps(&X[i]);
        __m256 v_Y = _mm256_load_ps(&Y[i]);
        
        __m256 v_res = _mm256_fmadd_ps(v_scale, v_X, v_Y);
        _mm256_stream_ps(&result[i], v_res);
    }

    for (; i < end; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

// 2. AVX2 Task 分派函數
void saxpy_avx2_stream_withtasks(int N, float scale, float* X, float* Y, float* result) {
    // 配合 i5-1135G7 (4核8緒)，這裡開 8 個 task 可降低 OS 建立執行緒的 Overhead
    const int num_tasks = 8; 
    
    // 計算 span，並利用位元運算 (& ~7) 強制進位到 8 的倍數，確保記憶體對齊
    const int span = ((N + num_tasks - 1) / num_tasks + 7) & ~7; 

    std::vector<std::thread> threads;
    for (int i = 0; i < num_tasks; ++i) {
        int start = i * span;
        int end = std::min(N, start + span);
        if (start < N) {
            threads.emplace_back(saxpy_avx2_stream_worker, start, end, scale, X, Y, result);
        }
    }
    
    // 等待所有 Task 完成
    for (auto& t : threads) {
        t.join();
    }
}

// return GB/s
static float
toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float
toGFLOPS(int ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (result[i] != gold[i]) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

using namespace ispc;


int main() {

    const unsigned int N = 20 * 1000 * 1000; // 20 M element vectors (~80 MB)
    const unsigned int TOTAL_BYTES = 4 * N * sizeof(float);
    const unsigned int TOTAL_FLOPS = 2 * N;

    float scale = 2.f;

    float* arrayX = (float*)_mm_malloc(N * sizeof(float), 32);
    float* arrayY = (float*)_mm_malloc(N * sizeof(float), 32);
    float* resultSerial = new float[N];
    float* resultISPC = new float[N];
    float* resultTasks = new float[N];
    float* resultAVX2 = (float*)_mm_malloc(N * sizeof(float), 32);
    float* resultAVX2_Tasks = (float*)_mm_malloc(N * sizeof(float), 32);

    // initialize array values
    for (unsigned int i=0; i<N; i++)
    {
        arrayX[i] = i;
        arrayY[i] = i;
        resultSerial[i] = 0.f;
        resultISPC[i] = 0.f;
        resultTasks[i] = 0.f;
        resultAVX2[i] = 0.f;
        resultAVX2_Tasks[i] = 0.f;
    }

    //
    // Run the serial implementation. Repeat three times for robust
    // timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime =CycleTimer::currentSeconds();
        saxpySerial(N, scale, arrayX, arrayY, resultSerial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
            minSerial * 1000,
            toBW(TOTAL_BYTES, minSerial),
            toGFLOPS(TOTAL_FLOPS, minSerial));

    //
    // Run the ISPC (single core) implementation
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_ispc(N, scale, arrayX, arrayY, resultISPC);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    verifyResult(N, resultISPC, resultSerial);

    printf("[saxpy ispc]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minISPC * 1000,
           toBW(TOTAL_BYTES, minISPC),
           toGFLOPS(TOTAL_FLOPS, minISPC));

    //
    // Run the ISPC (multi-core) implementation
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_ispc_withtasks(N, scale, arrayX, arrayY, resultTasks);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    verifyResult(N, resultTasks, resultSerial);

    printf("[saxpy task ispc]:\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minTaskISPC * 1000,
           toBW(TOTAL_BYTES, minTaskISPC),
           toGFLOPS(TOTAL_FLOPS, minTaskISPC));

    //
    // Run the AVX2 Stream (single core) implementation
    //
    double minAVX2 = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_avx2_stream(N, scale, arrayX, arrayY, resultAVX2);
        double endTime = CycleTimer::currentSeconds();
        minAVX2 = std::min(minAVX2, endTime - startTime);
    }

    verifyResult(N, resultAVX2, resultSerial);

    printf("[saxpy avx2 stream]:\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minAVX2 * 1000,
           toBW(TOTAL_BYTES, minAVX2),
           toGFLOPS(TOTAL_FLOPS, minAVX2));

    //
    // Run the AVX2 Stream (multi-core) implementation
    //
    double minTaskAVX2 = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_avx2_stream_withtasks(N, scale, arrayX, arrayY, resultAVX2_Tasks);
        double endTime = CycleTimer::currentSeconds();
        minTaskAVX2 = std::min(minTaskAVX2, endTime - startTime);
    }

    verifyResult(N, resultAVX2_Tasks, resultSerial);

    printf("[saxpy avx2 task]:\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minTaskAVX2 * 1000,
           toBW(TOTAL_BYTES, minTaskAVX2),
           toGFLOPS(TOTAL_FLOPS, minTaskAVX2));
    
    printf("\t\t\t\t(%.2fx speedup from use of tasks)\n", minISPC/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from AVX2 Stream)\n", minSerial/minAVX2);
    printf("\t\t\t\t(%.2fx speedup from AVX2 task)\n", minSerial/minTaskAVX2);

    _mm_free(arrayX);
    _mm_free(arrayY);
    delete[] resultSerial;
    delete[] resultISPC;
    delete[] resultTasks;
    _mm_free(resultAVX2);
    _mm_free(resultAVX2_Tasks);

    return 0;
}
