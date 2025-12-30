#include "std_includes.h"
#include "matrix.h"

#ifndef FUNCTION_H
#define FUNCTION_H

typedef void (*Activation)(Matrix*);

#define MAX(a,b) (((a)>(b))?(a):(b))

// Raw activation functions
static float sigmoidFunc(float input);
static float sigmoidDeriv(float sigmoidInput);
static float reluFunc(float input);
static float reluDeriv(float reluInput);
static float tanHFunc(float input);
static float tanHDeriv(float tanhInput);
static float linearDeriv(float linearInput);

// Matrix activation functions
static void sigmoid(Matrix* input);
static void relu(Matrix* input);
static void tanH(Matrix* input);
static void softmax(Matrix* input);
static void linear(Matrix* input);

// Utility functions
static float box_muller(void);
static const char* getFunctionName(Activation func);
static Activation getFunctionByName(const char* name);
static float (*activationDerivative(Activation func))(float);


/*
    Begin functions.
*/

float sigmoidFunc(float input) {
    return 1.0f / (1.0f + expf(-input));
}

float sigmoidDeriv(float sigmoidInput) {
    return sigmoidInput * (1.0f - sigmoidInput);
}

float reluFunc(float input) {
    return MAX(0.0f, input);
}

float reluDeriv(float reluInput) {
    return reluInput > 0.0f ? 1.0f : 0.0f;
}

float tanHFunc(float input) {
    return tanhf(input);
}

float tanHDeriv(float tanhInput) {
    return 1.0f - (tanhInput * tanhInput);
}

float linearDeriv(float linearInput) {
    (void)linearInput;
    return 1.0f;
}

void sigmoid(Matrix* input) {
    size_t i, j;
    for (i = 0; i < input->rows; i++) {
        for (j = 0; j < input->cols; j++) {
            setMatrix(input, i, j, sigmoidFunc(getMatrix(input, i, j)));
        }
    }
}

void relu(Matrix* input) {
    size_t i, j;
    for (i = 0; i < input->rows; i++) {
        for (j = 0; j < input->cols; j++) {
            setMatrix(input, i, j, reluFunc(getMatrix(input, i, j)));
        }
    }
}

void tanH(Matrix* input) {
    size_t i, j;
    for (i = 0; i < input->rows; i++) {
        for (j = 0; j < input->cols; j++) {
            setMatrix(input, i, j, tanHFunc(getMatrix(input, i, j)));
        }
    }
}

void softmax(Matrix* input) {
    size_t i, j;
    for (i = 0; i < input->rows; i++) {
        // Find max for numerical stability
        float maxVal = getMatrix(input, i, 0);
        for (j = 1; j < input->cols; j++) {
            float val = getMatrix(input, i, j);
            if (val > maxVal) maxVal = val;
        }
        
        float summed = 0.0f;
        for (j = 0; j < input->cols; j++) {
            float expVal = expf(getMatrix(input, i, j) - maxVal);
            setMatrix(input, i, j, expVal);
            summed += expVal;
        }
        
        float invSum = 1.0f / summed;
        for (j = 0; j < input->cols; j++) {
            setMatrix(input, i, j, getMatrix(input, i, j) * invSum);
        }
    }
}

void linear(Matrix* input) {
    (void)input;
}

float box_muller(void) {
    static float z1;
    static int generate = 0;
    generate = !generate;
    
    if (!generate) {
        return z1;
    }
    
    const float two_pi = 2.0f * 3.14159265358979323846f;
    float u1, u2;
    
    do {
        u1 = (float)rand() / (float)RAND_MAX;
        u2 = (float)rand() / (float)RAND_MAX;
    } while (u1 <= FLT_MIN);
    
    float r = sqrtf(-2.0f * logf(u1));
    float theta = two_pi * u2;
    
    z1 = r * sinf(theta);
    return r * cosf(theta);
}

const char* getFunctionName(Activation func) {
    if (func == sigmoid) return "sigmoid";
    if (func == relu) return "relu";
    if (func == tanH) return "tanH";
    if (func == softmax) return "softmax";
    return "linear";
}

Activation getFunctionByName(const char* name) {
    if (strcmp(name, "sigmoid") == 0) return sigmoid;
    if (strcmp(name, "relu") == 0) return relu;
    if (strcmp(name, "tanH") == 0) return tanH;
    if (strcmp(name, "softmax") == 0) return softmax;
    return linear;
}

float (*activationDerivative(Activation func))(float) {
    if (func == sigmoid) return sigmoidDeriv;
    if (func == relu) return reluDeriv;
    if (func == tanH) return tanHDeriv;
    return linearDeriv;
}

#endif
