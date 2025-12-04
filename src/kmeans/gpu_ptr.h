#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

template <typename T>
class GpuPtr {
private:
    T* raw_ptr;
    size_t count;
    bool owning;

    void allocate(size_t n) {
        if (n == 0) return;
        cudaError_t err = cudaMalloc(&raw_ptr, n * sizeof(T));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        count = n;
    }

public:

    // -----------------------------
    // CONSTRUCTORS AND DESTRUCTOR
    // -----------------------------
    GpuPtr() : raw_ptr(nullptr), count(0), owning(false) {}

    explicit GpuPtr(size_t n) : raw_ptr(nullptr), count(0), owning(true) {
        allocate(n);
    }

    // Wrap existing pointer on creation (non-owning)
    GpuPtr(T* existing_ptr, size_t count)
        : raw_ptr(existing_ptr), count(count), owning(false) {}

    // Destructor: RAII
    ~GpuPtr() {
        if (raw_ptr && owning) {
            cudaFree(raw_ptr);
        }
    }

    // -----------------------------
    // MEMORY MANAGEMENT
    // -----------------------------

    void alloc(size_t n) {
        if (raw_ptr && owning) {
            cudaFree(raw_ptr);
        }
        count = n;
        allocate(n);
        owning = true;
    }

    // --- attach external memory ---
    void attach(T* external_ptr, size_t c) {
        if (raw_ptr && owning)
            cudaFree(raw_ptr);

        raw_ptr = external_ptr;
        count = c;
        owning = false;  // <--- very important
    }

    void realloc(size_t n) {
        if (owning && raw_ptr) {
            cudaError_t err = cudaFree(raw_ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaFree failed (%s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
        raw_ptr = nullptr;
        count = 0;
        allocate(n);
        owning = true;
    }

    // -----------------------------
    // DISABLE COPY (prevents double free)
    // -----------------------------
    GpuPtr(const GpuPtr&) = delete;
    GpuPtr& operator=(const GpuPtr&) = delete;

    // -----------------------------
    // MOVE CONSTRUCTOR
    // -----------------------------
    GpuPtr(GpuPtr&& other) noexcept
    : raw_ptr(other.raw_ptr),
      count(other.count),
      owning(other.owning)
    {
        other.raw_ptr = nullptr;
        other.count = 0;
        other.owning = false;
    }


    // -----------------------------
    // MOVE ASSIGNMENT
    // -----------------------------
    GpuPtr& operator=(GpuPtr&& other) noexcept {
        if (this != &other) {

            // libera somente se este GpuPtr realmente é dono
            if (owning && raw_ptr) {
                cudaFree(raw_ptr);
            }

            // transfere dados
            raw_ptr = other.raw_ptr;
            count = other.count;
            owning = other.owning;

            // limpa o objeto movido
            other.raw_ptr = nullptr;
            other.count = 0;
            other.owning = false;
        }
        return *this;
    }


    // -----------------------------
    // resize() — reallocates memory
    // -----------------------------
    void resize(size_t n) {
        if (raw_ptr) cudaFree(raw_ptr);
        raw_ptr = nullptr;
        count = 0;
        allocate(n);
    }

    // -----------------------------
    // Utility functions
    // -----------------------------
    T* ptr() const { return raw_ptr; }
    T* get() const { return raw_ptr; }
    size_t size() const { return count; }
    size_t bytes() const { return count * sizeof(T); }

    // -----------------------------
    // Zero memory (cudaMemset)
    // -----------------------------
    void zero() {
        if (raw_ptr) {
            cudaError_t err = cudaMemset(raw_ptr, 0, count * sizeof(T));
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaMemset failed (%s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
    }

    // -----------------------------
    // Copying Device -> Device
    // -----------------------------
    void copyFromDevice(const T* d_data) {
        cudaError_t err = cudaMemcpy(raw_ptr, d_data, count * sizeof(T), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy D2D failed (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // -----------------------------
    // Copying Host -> Device
    // -----------------------------
    void copyFromHost(const T* h_data) {
        cudaError_t err = cudaMemcpy(raw_ptr, h_data, count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy H2D failed (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // -----------------------------
    // Copying Device -> Host
    // -----------------------------
    void copyToHost(T* h_data) const {
        cudaError_t err = cudaMemcpy(h_data, raw_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy D2H failed (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    void free() {
        if (owning && raw_ptr != nullptr) {
            cudaError_t err = cudaFree(raw_ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }

        // Always leave in a safe state
        raw_ptr = nullptr;
        count = 0;
        owning = false;
    }
};
