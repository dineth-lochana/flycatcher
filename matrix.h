#include "std_includes.h"

#ifndef MATRIX_H
#define MATRIX_H

// Represents a matrix of data in row-major order
// Can optionally be a "view" into another matrix's data (owns_data = 0)
typedef struct Matrix_ {
    size_t rows;
    size_t cols;
    size_t stride;    // actual number of floats between row starts (for views)
    float* data;
    int owns_data;    // if 0, this is a view and should not free data
} Matrix;

// Represents training data - now uses contiguous storage
typedef struct DataSet_ {
    size_t rows;
    size_t cols;
    float* data;      // contiguous row-major storage
} DataSet;

// Memory pool for reducing allocations during training
typedef struct MatrixPool_ {
    float* buffer;
    size_t capacity;
    size_t used;
} MatrixPool;

// Create a memory pool with given capacity (in floats)
static MatrixPool* createMatrixPool(size_t capacity);

// Reset pool for reuse (doesn't free memory)
static void resetMatrixPool(MatrixPool* pool);

// Allocate from pool (returns NULL if insufficient space)
static float* poolAlloc(MatrixPool* pool, size_t count);

// Destroy pool
static void destroyMatrixPool(MatrixPool* pool);

// Create dataset with contiguous storage
static DataSet* createDataSet(size_t rows, size_t cols, float* data);

// Create empty dataset (allocates memory)
static DataSet* createDataSetEmpty(size_t rows, size_t cols);

// Get pointer to row i of dataset
static float* getDataSetRow(DataSet* dataset, size_t row);

// Get element from dataset
static float getDataSetElement(DataSet* dataset, size_t row, size_t col);

// Set element in dataset
static void setDataSetElement(DataSet* dataset, size_t row, size_t col, float val);

// Shuffle two datasets together maintaining alignment
static void shuffleTogether(DataSet* A, DataSet* B);

// Destroy dataset
static void destroyDataSet(DataSet* dataset);

// Convert dataset to matrix (creates a view, no copy)
static Matrix* dataSetToMatrixView(DataSet* dataset);

// Convert dataset to matrix (creates a copy)
static Matrix* dataSetToMatrix(DataSet* dataset);

// Creates a matrix given data (takes ownership)
static Matrix* createMatrix(size_t rows, size_t cols, float* data);

// Creates a matrix view (does not own data)
static Matrix* createMatrixView(size_t rows, size_t cols, size_t stride, float* data);

// Creates a view of a single row of a matrix
static Matrix* createRowView(Matrix* mat, size_t row);

// Creates a view of a single row of a dataset
static Matrix* createDataSetRowView(DataSet* dataset, size_t row);

// Creates a matrix zeroed out
static Matrix* createMatrixZeroes(size_t rows, size_t cols);

// Creates a matrix using pool memory (view, doesn't own data)
static Matrix* createMatrixFromPool(MatrixPool* pool, size_t rows, size_t cols);

// Get an element of a matrix
static float getMatrix(Matrix* mat, size_t row, size_t col);

// Set an element of a matrix
static void setMatrix(Matrix* mat, size_t row, size_t col, float val);

// Sets the values in $to equal to values in $from
static void copyValuesInto(Matrix* from, Matrix* to);

// Prints the entries of a matrix
static void printMatrix(Matrix* input);

// Sets each entry in matrix to 0
static void zeroMatrix(Matrix* orig);

// Returns transpose of matrix
static Matrix* transpose(Matrix* orig);

// Transposes matrix and places data into $origT
static void transposeInto(Matrix* orig, Matrix* origT);

// Adds two matrices and returns result
static Matrix* add(Matrix* A, Matrix* B);

// Adds $from to $to and places result in $to
static void addTo(Matrix* from, Matrix* to);

// Adds $B, a row vector, to each row of $A
static Matrix* addToEachRow(Matrix* A, Matrix* B);

// Adds $B, a row vector, to each row of $A, stores in $into
static void addToEachRowInto(Matrix* A, Matrix* B, Matrix* into);

// Multiplies every element of $orig by $C
static void scalarMultiply(Matrix* orig, float c);

// Multiplies $A and $B (ordering: AB) and returns product matrix
static Matrix* multiply(Matrix* A, Matrix* B);

// Multiplies $A and $B (ordering: AB) and places values into $into
static void multiplyInto(Matrix* A, Matrix* B, Matrix* into);

// Element-wise multiplication
static Matrix* hadamard(Matrix* A, Matrix* B);

// Places values of hadamard product of $A and $B into $into
static void hadamardInto(Matrix* A, Matrix* B, Matrix* into);

// Returns a deep copy of input matrix
static Matrix* copy(Matrix* orig);

// Returns 1 if matrices are equal, 0 otherwise
static int equals(Matrix* A, Matrix* B);

// Frees a matrix and its data (if owned)
static void destroyMatrix(Matrix* matrix);


/*
    Begin functions.
*/

MatrixPool* createMatrixPool(size_t capacity) {
    MatrixPool* pool = (MatrixPool*)malloc(sizeof(MatrixPool));
    pool->buffer = (float*)malloc(sizeof(float) * capacity);
    pool->capacity = capacity;
    pool->used = 0;
    return pool;
}

void resetMatrixPool(MatrixPool* pool) {
    pool->used = 0;
}

float* poolAlloc(MatrixPool* pool, size_t count) {
    if (pool->used + count > pool->capacity) {
        return NULL;
    }
    float* ptr = pool->buffer + pool->used;
    pool->used += count;
    return ptr;
}

void destroyMatrixPool(MatrixPool* pool) {
    free(pool->buffer);
    free(pool);
}

DataSet* createDataSet(size_t rows, size_t cols, float* data) {
    DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
    dataset->rows = rows;
    dataset->cols = cols;
    dataset->data = data;
    return dataset;
}

DataSet* createDataSetEmpty(size_t rows, size_t cols) {
    DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
    dataset->rows = rows;
    dataset->cols = cols;
    dataset->data = (float*)calloc(rows * cols, sizeof(float));
    return dataset;
}

float* getDataSetRow(DataSet* dataset, size_t row) {
    return dataset->data + row * dataset->cols;
}

float getDataSetElement(DataSet* dataset, size_t row, size_t col) {
    return dataset->data[row * dataset->cols + col];
}

void setDataSetElement(DataSet* dataset, size_t row, size_t col, float val) {
    dataset->data[row * dataset->cols + col] = val;
}

void shuffleTogether(DataSet* A, DataSet* B) {
    assert(A->rows == B->rows);
    size_t colsA = A->cols;
    size_t colsB = B->cols;
    
    // Temporary buffers for swapping rows
    float* tmpA = (float*)malloc(sizeof(float) * colsA);
    float* tmpB = (float*)malloc(sizeof(float) * colsB);
    
    size_t i;
    for (i = 0; i < A->rows - 1; i++) {
        size_t j = i + rand() / (RAND_MAX / (A->rows - i) + 1);
        if (i != j) {
            // Swap rows in A
            memcpy(tmpA, getDataSetRow(A, i), sizeof(float) * colsA);
            memcpy(getDataSetRow(A, i), getDataSetRow(A, j), sizeof(float) * colsA);
            memcpy(getDataSetRow(A, j), tmpA, sizeof(float) * colsA);
            
            // Swap rows in B
            memcpy(tmpB, getDataSetRow(B, i), sizeof(float) * colsB);
            memcpy(getDataSetRow(B, i), getDataSetRow(B, j), sizeof(float) * colsB);
            memcpy(getDataSetRow(B, j), tmpB, sizeof(float) * colsB);
        }
    }
    
    free(tmpA);
    free(tmpB);
}

void destroyDataSet(DataSet* dataset) {
    free(dataset->data);
    free(dataset);
}

Matrix* dataSetToMatrixView(DataSet* dataset) {
    return createMatrixView(dataset->rows, dataset->cols, dataset->cols, dataset->data);
}

Matrix* dataSetToMatrix(DataSet* dataset) {
    float* data = (float*)malloc(sizeof(float) * dataset->rows * dataset->cols);
    memcpy(data, dataset->data, sizeof(float) * dataset->rows * dataset->cols);
    return createMatrix(dataset->rows, dataset->cols, data);
}

Matrix* createMatrix(size_t rows, size_t cols, float* data) {
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->stride = cols;
    matrix->data = data;
    matrix->owns_data = 1;
    return matrix;
}

Matrix* createMatrixView(size_t rows, size_t cols, size_t stride, float* data) {
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->stride = stride;
    matrix->data = data;
    matrix->owns_data = 0;
    return matrix;
}

Matrix* createRowView(Matrix* mat, size_t row) {
    assert(row < mat->rows);
    return createMatrixView(1, mat->cols, mat->stride, mat->data + row * mat->stride);
}

Matrix* createDataSetRowView(DataSet* dataset, size_t row) {
    assert(row < dataset->rows);
    return createMatrixView(1, dataset->cols, dataset->cols, getDataSetRow(dataset, row));
}

Matrix* createMatrixZeroes(size_t rows, size_t cols) {
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->stride = cols;
    matrix->data = (float*)calloc(rows * cols, sizeof(float));
    matrix->owns_data = 1;
    return matrix;
}

Matrix* createMatrixFromPool(MatrixPool* pool, size_t rows, size_t cols) {
    float* data = poolAlloc(pool, rows * cols);
    if (data == NULL) return NULL;
    memset(data, 0, sizeof(float) * rows * cols);
    return createMatrixView(rows, cols, cols, data);
}

float getMatrix(Matrix* mat, size_t row, size_t col) {
    return mat->data[row * mat->stride + col];
}

void setMatrix(Matrix* mat, size_t row, size_t col, float val) {
    mat->data[row * mat->stride + col] = val;
}

void copyValuesInto(Matrix* from, Matrix* to) {
    assert(from->rows == to->rows && from->cols == to->cols);
    if (from->stride == from->cols && to->stride == to->cols) {
        // Both contiguous, use memcpy
        memcpy(to->data, from->data, sizeof(float) * to->rows * to->cols);
    } else {
        // Handle strided copies
        size_t i;
        for (i = 0; i < from->rows; i++) {
            memcpy(to->data + i * to->stride, 
                   from->data + i * from->stride, 
                   sizeof(float) * from->cols);
        }
    }
}

void printMatrix(Matrix* input) {
    size_t i, j;
    for (i = 0; i < input->rows; i++) {
        printf("\n");
        for (j = 0; j < input->cols; j++) {
            printf("%.2f ", getMatrix(input, i, j));
        }
    }
    printf("\n");
}

void zeroMatrix(Matrix* orig) {
    if (orig->stride == orig->cols) {
        memset(orig->data, 0, orig->rows * orig->cols * sizeof(float));
    } else {
        size_t i;
        for (i = 0; i < orig->rows; i++) {
            memset(orig->data + i * orig->stride, 0, orig->cols * sizeof(float));
        }
    }
}

Matrix* transpose(Matrix* orig) {
    float* data = (float*)malloc(sizeof(float) * orig->rows * orig->cols);
    Matrix* trans = createMatrix(orig->cols, orig->rows, data);
    size_t i, j;
    for (i = 0; i < orig->rows; i++) {
        for (j = 0; j < orig->cols; j++) {
            setMatrix(trans, j, i, getMatrix(orig, i, j));
        }
    }
    return trans;
}

void transposeInto(Matrix* orig, Matrix* origT) {
    assert(orig->rows == origT->cols && orig->cols == origT->rows);
    size_t i, j;
    for (i = 0; i < orig->rows; i++) {
        for (j = 0; j < orig->cols; j++) {
            setMatrix(origT, j, i, getMatrix(orig, i, j));
        }
    }
}

Matrix* add(Matrix* A, Matrix* B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    float* data = (float*)malloc(sizeof(float) * A->rows * A->cols);
    Matrix* result = createMatrix(A->rows, A->cols, data);
    size_t i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            setMatrix(result, i, j, getMatrix(B, i, j) + getMatrix(A, i, j));
        }
    }
    return result;
}

void addTo(Matrix* from, Matrix* to) {
    assert(from->rows == to->rows && from->cols == to->cols);
    if (from->stride == from->cols && to->stride == to->cols) {
        /* Contiguous memory - vectorizable loop */
        size_t total = from->rows * from->cols;
        size_t i;
        for (i = 0; i < total; i++) {
            to->data[i] = to->data[i] + from->data[i];
        }
    } else {
        /* Strided memory */
        size_t i, j;
        float* from_row;
        float* to_row;
        for (i = 0; i < from->rows; i++) {
            from_row = from->data + i * from->stride;
            to_row = to->data + i * to->stride;
            for (j = 0; j < from->cols; j++) {
                to_row[j] = to_row[j] + from_row[j];
            }
        }
    }
}

Matrix* addToEachRow(Matrix* A, Matrix* B) {
    assert(A->cols == B->cols && B->rows == 1);
    float* data = (float*)malloc(sizeof(float) * A->rows * A->cols);
    Matrix* result = createMatrix(A->rows, A->cols, data);
    size_t i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            setMatrix(result, i, j, getMatrix(A, i, j) + getMatrix(B, 0, j));
        }
    }
    return result;
}

void addToEachRowInto(Matrix* A, Matrix* B, Matrix* into) {
    assert(A->cols == B->cols && B->rows == 1);
    assert(A->rows == into->rows && A->cols == into->cols);
    size_t i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            setMatrix(into, i, j, getMatrix(A, i, j) + getMatrix(B, 0, j));
        }
    }
}

void scalarMultiply(Matrix* orig, float c) {
    if (orig->stride == orig->cols) {
        size_t total = orig->rows * orig->cols;
        size_t i;
        for (i = 0; i < total; i++) {
            orig->data[i] = orig->data[i] * c;
        }
    } else {
        size_t i, j;
        float* row;
        for (i = 0; i < orig->rows; i++) {
            row = orig->data + i * orig->stride;
            for (j = 0; j < orig->cols; j++) {
                row[j] = row[j] * c;
            }
        }
    }
}

Matrix* multiply(Matrix* A, Matrix* B) {
    assert(A->cols == B->rows);
    float* data = (float*)malloc(sizeof(float) * A->rows * B->cols);
    Matrix* result = createMatrix(A->rows, B->cols, data);
    size_t i, j, k;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < B->cols; j++) {
            float sum = 0;
            for (k = 0; k < B->rows; k++) {
                sum = sum + (getMatrix(A, i, k) * getMatrix(B, k, j));
            }
            setMatrix(result, i, j, sum);
        }
    }
    return result;
}

void multiplyInto(Matrix* A, Matrix* B, Matrix* into) {
    assert(A->cols == B->rows);
    assert(A->rows == into->rows && B->cols == into->cols);
    
    size_t i, j, k;
    float sum;
    float* a_row;
    float* b_col_base;
    float* into_row;
    
    /* Zero output first */
    if (into->stride == into->cols) {
        memset(into->data, 0, into->rows * into->cols * sizeof(float));
    } else {
        for (i = 0; i < into->rows; i++) {
            memset(into->data + i * into->stride, 0, into->cols * sizeof(float));
        }
    }
    /* Outer loop over A rows, inner over B cols */
    for (i = 0; i < A->rows; i++) {
        a_row = A->data + i * A->stride;
        into_row = into->data + i * into->stride;
        for (k = 0; k < A->cols; k++) {
            float a_val = a_row[k];
            float* b_row = B->data + k * B->stride;
            /* Vectorizable inner loop */
            for (j = 0; j < B->cols; j++) {
                into_row[j] = into_row[j] + (a_val * b_row[j]);
            }
        }
    }
}

Matrix* hadamard(Matrix* A, Matrix* B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    float* data = (float*)malloc(sizeof(float) * A->rows * A->cols);
    Matrix* result = createMatrix(A->rows, A->cols, data);
    size_t i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            setMatrix(result, i, j, getMatrix(A, i, j) * getMatrix(B, i, j));
        }
    }
    return result;
}

void hadamardInto(Matrix* A, Matrix* B, Matrix* into) {
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(A->rows == into->rows && A->cols == into->cols);
    size_t i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            setMatrix(into, i, j, getMatrix(A, i, j) * getMatrix(B, i, j));
        }
    }
}

Matrix* copy(Matrix* orig) {
    float* data = (float*)malloc(sizeof(float) * orig->rows * orig->cols);
    Matrix* result = createMatrix(orig->rows, orig->cols, data);
    copyValuesInto(orig, result);
    return result;
}

int equals(Matrix* A, Matrix* B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        return 0;
    }
    size_t i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            if (getMatrix(A, i, j) != getMatrix(B, i, j)) {
                return 0;
            }
        }
    }
    return 1;
}

void destroyMatrix(Matrix* matrix) {
    if (matrix->owns_data) {
        free(matrix->data);
    }
    free(matrix);
}

#endif
