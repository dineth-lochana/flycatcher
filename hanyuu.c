#include "hanyuu.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static float frand(void) {
    return rand() / (float)RAND_MAX;
}

static float act_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float act_tanh(float x) {
    return tanhf(x);
}

static float act_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static float act_leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

static float pd_sigmoid(float y) {
    return y * (1.0f - y);
}

static float pd_tanh(float y) {
    return 1.0f - y * y;
}

static float pd_relu(float y) {
    return y > 0.0f ? 1.0f : 0.0f;
}

static float pd_leaky_relu(float y) {
    return y > 0.0f ? 1.0f : 0.01f;
}

static float activate(float x, Activation act) {
    if(act == ACT_SIGMOID) return act_sigmoid(x);
    if(act == ACT_TANH) return act_tanh(x);
    if(act == ACT_RELU) return act_relu(x);
    return act_leaky_relu(x);
}

static float pd_activate(float y, Activation act) {
    if(act == ACT_SIGMOID) return pd_sigmoid(y);
    if(act == ACT_TANH) return pd_tanh(y);
    if(act == ACT_RELU) return pd_relu(y);
    return pd_leaky_relu(y);
}

static void init_weights_xavier(float* w, int nin, int nout, int count) {
    float limit = sqrtf(6.0f / (nin + nout));
    for(int i = 0; i < count; i++)
        w[i] = (frand() * 2.0f - 1.0f) * limit;
}

static void init_weights_he(float* w, int nin, int count) {
    float stddev = sqrtf(2.0f / nin);
    for(int i = 0; i < count; i++)
        w[i] = (frand() * 2.0f - 1.0f) * stddev;
}

HanyuuConfig hconfig_default(void) {
    HanyuuConfig c;
    c.learning_rate = 0.1f;
    c.momentum = 0.9f;
    c.batch_size = 1;
    c.hidden_act = ACT_TANH;
    c.output_act = ACT_SIGMOID;
    c.init_type = INIT_XAVIER;
    c.norm_type = NORM_ZSCORE;
    c.early_stop_patience = 10;
    c.early_stop_delta = 0.0001f;
    return c;
}

Normalizer* hnorm_create(int nips, Normalization type) {
    Normalizer* n = (Normalizer*)malloc(sizeof(Normalizer));
    n->size = nips;
    n->type = type;
    n->min_vals = (float*)calloc(nips, sizeof(float));
    n->max_vals = (float*)calloc(nips, sizeof(float));
    n->means = (float*)calloc(nips, sizeof(float));
    n->stddevs = (float*)calloc(nips, sizeof(float));
    return n;
}

void hnorm_fit(Normalizer* norm, float** data, int rows) {
    if(norm->type == NORM_NONE) return;
    
    for(int j = 0; j < norm->size; j++) {
        norm->min_vals[j] = data[0][j];
        norm->max_vals[j] = data[0][j];
        norm->means[j] = 0.0f;
    }
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < norm->size; j++) {
            float val = data[i][j];
            norm->means[j] = norm->means[j] + val;
            if(val < norm->min_vals[j]) norm->min_vals[j] = val;
            if(val > norm->max_vals[j]) norm->max_vals[j] = val;
        }
    }
    
    for(int j = 0; j < norm->size; j++)
        norm->means[j] = norm->means[j] / rows;
    
    if(norm->type == NORM_ZSCORE) {
        for(int j = 0; j < norm->size; j++)
            norm->stddevs[j] = 0.0f;
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < norm->size; j++) {
                float diff = data[i][j] - norm->means[j];
                norm->stddevs[j] = norm->stddevs[j] + diff * diff;
            }
        }
        
        for(int j = 0; j < norm->size; j++) {
            norm->stddevs[j] = sqrtf(norm->stddevs[j] / rows);
            if(norm->stddevs[j] < 1e-8f) norm->stddevs[j] = 1.0f;
        }
    }
}

void hnorm_transform(Normalizer* norm, float* data) {
    if(norm->type == NORM_MINMAX) {
        for(int i = 0; i < norm->size; i++) {
            float range = norm->max_vals[i] - norm->min_vals[i];
            if(range > 1e-8f)
                data[i] = (data[i] - norm->min_vals[i]) / range;
        }
    }
    else if(norm->type == NORM_ZSCORE) {
        for(int i = 0; i < norm->size; i++)
            data[i] = (data[i] - norm->means[i]) / norm->stddevs[i];
    }
}

void hnorm_free(Normalizer* norm) {
    if(norm == NULL) return;
    free(norm->min_vals);
    free(norm->max_vals);
    free(norm->means);
    free(norm->stddevs);
    free(norm);
}

int hearly_stop_check(EarlyStopping* es, float error) {
    if(!es->enabled) return 0;
    
    if(error < es->best_error - es->min_delta) {
        es->best_error = error;
        es->wait_count = 0;
        return 0;
    }
    
    es->wait_count++;
    if(es->wait_count >= es->patience) return 1;
    return 0;
}

Hanyuu hbuild(int nips, int nhid, int nops, HanyuuConfig config) {
    Hanyuu h;
    h.base = xtbuild(nips, nhid, nops);
    h.hidden_act = config.hidden_act;
    h.output_act = config.output_act;
    h.init_type = config.init_type;
    
    if(config.init_type == INIT_XAVIER) {
        init_weights_xavier(h.base.w, nips, nhid, nhid * nips);
        init_weights_xavier(h.base.x, nhid, nops, nhid * nops);
    }
    else if(config.init_type == INIT_HE) {
        init_weights_he(h.base.w, nips, nhid * nips);
        init_weights_he(h.base.x, nhid, nhid * nops);
    }
    
    if(config.norm_type != NORM_NONE) {
        h.norm = hnorm_create(nips, config.norm_type);
    } else {
        h.norm = NULL;
    }
    
    if(config.momentum > 0.0f) {
        h.mom = (Momentum*)malloc(sizeof(Momentum));
        h.mom->nw = h.base.nw;
        h.mom->nb = h.base.nb;
        h.mom->momentum = config.momentum;
        h.mom->velocity_w = (float*)calloc(h.base.nw, sizeof(float));
        h.mom->velocity_b = (float*)calloc(h.base.nb, sizeof(float));
    } else {
        h.mom = NULL;
    }
    
    if(config.batch_size > 1) {
        h.batch = (MiniBatch*)malloc(sizeof(MiniBatch));
        h.batch->batch_size = config.batch_size;
        h.batch->current_batch = 0;
        h.batch->batch_inputs = NULL;
        h.batch->batch_targets = NULL;
    } else {
        h.batch = NULL;
    }
    
    if(config.early_stop_patience > 0) {
        h.early_stop = (EarlyStopping*)malloc(sizeof(EarlyStopping));
        h.early_stop->best_error = 1e9f;
        h.early_stop->patience = config.early_stop_patience;
        h.early_stop->wait_count = 0;
        h.early_stop->enabled = 1;
        h.early_stop->min_delta = config.early_stop_delta;
    } else {
        h.early_stop = NULL;
    }
    
    return h;
}

void hfree(Hanyuu h) {
    xtfree(h.base);
    hnorm_free(h.norm);
    if(h.mom) {
        free(h.mom->velocity_w);
        free(h.mom->velocity_b);
        free(h.mom);
    }
    if(h.batch) {
        free(h.batch);
    }
    if(h.early_stop) {
        free(h.early_stop);
    }
}

static void hfprop(Hanyuu h, const float* in) {
    Tinn t = h.base;
    for(int i = 0; i < t.nhid; i++) {
        float sum = 0.0f;
        for(int j = 0; j < t.nips; j++)
            sum = sum + in[j] * t.w[i * t.nips + j];
        t.h[i] = activate(sum + t.b[0], h.hidden_act);
    }
    for(int i = 0; i < t.nops; i++) {
        float sum = 0.0f;
        for(int j = 0; j < t.nhid; j++)
            sum = sum + t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = activate(sum + t.b[1], h.output_act);
    }
}

static float hbprop(Hanyuu h, const float* in, const float* tg, float rate) {
    Tinn t = h.base;
    float* dw = (float*)calloc(t.nw, sizeof(float));
    float* db = (float*)calloc(t.nb, sizeof(float));
    
    for(int i = 0; i < t.nhid; i++) {
        float sum = 0.0f;
        for(int j = 0; j < t.nops; j++) {
            float a = t.o[j] - tg[j];
            float b = pd_activate(t.o[j], h.output_act);
            sum = sum + a * b * t.x[j * t.nhid + i];
            dw[t.nhid * t.nips + j * t.nhid + i] = a * b * t.h[i];
        }
        for(int j = 0; j < t.nips; j++) {
            float pd = pd_activate(t.h[i], h.hidden_act);
            dw[i * t.nips + j] = sum * pd * in[j];
        }
    }
    
    for(int i = 0; i < t.nops; i++) {
        float a = t.o[i] - tg[i];
        float b = pd_activate(t.o[i], h.output_act);
        db[1] = db[1] + a * b;
    }
    
    for(int i = 0; i < t.nhid; i++) {
        float sum = 0.0f;
        for(int j = 0; j < t.nops; j++) {
            float a = t.o[j] - tg[j];
            float b = pd_activate(t.o[j], h.output_act);
            sum = sum + a * b * t.x[j * t.nhid + i];
        }
        float pd = pd_activate(t.h[i], h.hidden_act);
        db[0] = db[0] + sum * pd;
    }
    
    if(h.mom) {
        for(int i = 0; i < t.nw; i++) {
            h.mom->velocity_w[i] = h.mom->momentum * h.mom->velocity_w[i] + rate * dw[i];
            t.w[i] = t.w[i] - h.mom->velocity_w[i];
        }
        for(int i = 0; i < t.nb; i++) {
            h.mom->velocity_b[i] = h.mom->momentum * h.mom->velocity_b[i] + rate * db[i];
            t.b[i] = t.b[i] - h.mom->velocity_b[i];
        }
    } else {
        for(int i = 0; i < t.nw; i++)
            t.w[i] = t.w[i] - rate * dw[i];
        for(int i = 0; i < t.nb; i++)
            t.b[i] = t.b[i] - rate * db[i];
    }
    
    float error = 0.0f;
    for(int i = 0; i < t.nops; i++) {
        float diff = tg[i] - t.o[i];
        error = error + 0.5f * diff * diff;
    }
    
    free(dw);
    free(db);
    return error;
}

float htrain(Hanyuu h, const float* in, const float* tg, float rate) {
    float* norm_in = (float*)malloc(h.base.nips * sizeof(float));
    memcpy(norm_in, in, h.base.nips * sizeof(float));
    
    if(h.norm)
        hnorm_transform(h.norm, norm_in);
    
    hfprop(h, norm_in);
    float error = hbprop(h, norm_in, tg, rate);
    
    free(norm_in);
    return error;
}

float* hpredict(Hanyuu h, const float* in) {
    float* norm_in = (float*)malloc(h.base.nips * sizeof(float));
    memcpy(norm_in, in, h.base.nips * sizeof(float));
    
    if(h.norm)
        hnorm_transform(h.norm, norm_in);
    
    hfprop(h, norm_in);
    free(norm_in);
    return h.base.o;
}

void hsave(Hanyuu h, const char* path) {
    FILE* f = fopen(path, "w");
    fprintf(f, "%d %d %d\n", h.base.nips, h.base.nhid, h.base.nops);
    fprintf(f, "%d %d %d\n", h.hidden_act, h.output_act, h.init_type);
    for(int i = 0; i < h.base.nb; i++) fprintf(f, "%f\n", (double)h.base.b[i]);
    for(int i = 0; i < h.base.nw; i++) fprintf(f, "%f\n", (double)h.base.w[i]);
    fclose(f);
}

Hanyuu hload(const char* path) {
    FILE* f = fopen(path, "r");
    int nips, nhid, nops, ha, oa, it;
    fscanf(f, "%d %d %d\n", &nips, &nhid, &nops);
    fscanf(f, "%d %d %d\n", &ha, &oa, &it);
    
    HanyuuConfig cfg = hconfig_default();
    cfg.hidden_act = (Activation)ha;
    cfg.output_act = (Activation)oa;
    cfg.init_type = (WeightInit)it;
    cfg.norm_type = NORM_NONE;
    cfg.momentum = 0.0f;
    
    Hanyuu h = hbuild(nips, nhid, nops, cfg);
    for(int i = 0; i < h.base.nb; i++) fscanf(f, "%f\n", &h.base.b[i]);
    for(int i = 0; i < h.base.nw; i++) fscanf(f, "%f\n", &h.base.w[i]);
    fclose(f);
    return h;
}
