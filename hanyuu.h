#ifndef HANYUU_H
#define HANYUU_H

#include "tinn.h"

typedef enum {
    ACT_SIGMOID,
    ACT_TANH,
    ACT_RELU,
    ACT_LEAKY_RELU
} Activation;

typedef enum {
    INIT_RANDOM,
    INIT_XAVIER,
    INIT_HE
} WeightInit;

typedef enum {
    NORM_NONE,
    NORM_MINMAX,
    NORM_ZSCORE
} Normalization;

typedef struct {
    float* min_vals;
    float* max_vals;
    float* means;
    float* stddevs;
    int size;
    Normalization type;
} Normalizer;

typedef struct {
    float* velocity_w;
    float* velocity_b;
    float momentum;
    int nw;
    int nb;
} Momentum;

typedef struct {
    int batch_size;
    int current_batch;
    float** batch_inputs;
    float** batch_targets;
} MiniBatch;

typedef struct {
    float best_error;
    int patience;
    int wait_count;
    int enabled;
    float min_delta;
} EarlyStopping;

typedef struct {
    Tinn base;
    Normalizer* norm;
    Momentum* mom;
    MiniBatch* batch;
    EarlyStopping* early_stop;
    Activation hidden_act;
    Activation output_act;
    WeightInit init_type;
} Hanyuu;

typedef struct {
    float learning_rate;
    float momentum;
    int batch_size;
    Activation hidden_act;
    Activation output_act;
    WeightInit init_type;
    Normalization norm_type;
    int early_stop_patience;
    float early_stop_delta;
} HanyuuConfig;

Hanyuu hbuild(int nips, int nhid, int nops, HanyuuConfig config);

void hfree(Hanyuu h);

float htrain(Hanyuu h, const float* in, const float* tg, float rate);

float* hpredict(Hanyuu h, const float* in);

void hsave(Hanyuu h, const char* path);

Hanyuu hload(const char* path);

Normalizer* hnorm_create(int nips, Normalization type);

void hnorm_fit(Normalizer* norm, float** data, int rows);

void hnorm_transform(Normalizer* norm, float* data);

void hnorm_free(Normalizer* norm);

int hearly_stop_check(EarlyStopping* es, float error);

HanyuuConfig hconfig_default(void);

#endif
