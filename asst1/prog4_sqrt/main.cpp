#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <immintrin.h>
#include <math.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

static void sqrt_avx2(int N,
                             float initialGuess,
                             float values[],
                             float output[])
{
    const float kThreshold = 0.00001f; 

    __m256 v_three = _mm256_set1_ps(3.0f);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_threshold = _mm256_set1_ps(kThreshold);
    __m256 v_init_guess = _mm256_set1_ps(initialGuess);
    
    __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (int i = 0; i < N / 8 * 8; i += 8) {
        __m256 x = _mm256_loadu_ps(values + i);
        __m256 guess = v_init_guess;

        // err = abs(guess * guess * x - 1.f)
        __m256 err = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x), v_one), v_abs_mask);

        __m256 maskKeepIterate = _mm256_cmp_ps(err, v_threshold, _CMP_GT_OQ);

        while (_mm256_movemask_ps(maskKeepIterate) != 0) {

            // next_guess = (3.f * guess - x * guess * guess * guess) * 0.5f
            __m256 x_g_cubed = _mm256_mul_ps(x, _mm256_mul_ps(guess, _mm256_mul_ps(guess, guess)));
            __m256 next_guess = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(v_three, guess), x_g_cubed), v_half);

            // 處理 Control Flow Divergence：
            // 只有 mask 為 true 的 Lane 才會更新成 guess_next，已收斂的保留原 guess
            guess = _mm256_blendv_ps(guess, next_guess, maskKeepIterate);

            err = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x), v_one), v_abs_mask);

            maskKeepIterate = _mm256_cmp_ps(err, v_threshold, _CMP_GT_OQ);
        }

        __m256 result = _mm256_mul_ps(x, guess);
        _mm256_storeu_ps(output + i, result);
    }
    for (int i = N / 8 * 8; i < N; ++i) {
        float x = values[i];
        float guess = initialGuess;
        float err = std::abs(guess * guess * x - 1.0f);

        while (err > kThreshold) {
            guess = (3.0f * guess - x * guess * guess * guess) * 0.5f;
            err = std::abs(guess * guess * x - 1.0f);
        }
        output[i] = x * guess;
    } 
}


int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        // values[i] = 2.998f; // for higher speedup
        /*  if (i % 8 == 0) {
                values[i] = 2.998f; 
            } else {
                values[i] = 1.f;
            } */
    }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    // using sqrt_avx2
    double minAVX2 = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_avx2(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minAVX2 = std::min(minAVX2, endTime - startTime);
    }

    printf("[sqrt avx2]:\t\t[%.3f] ms\n", minAVX2 * 1000);

    verifyResult(N, output, gold);


    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from AVX2)\n", minSerial/minAVX2);

    delete [] values;
    delete [] output;
    delete [] gold;

    return 0;
}
