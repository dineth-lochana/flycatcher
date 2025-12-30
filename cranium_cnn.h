#ifndef CRANIUM_CNN_H
#define CRANIUM_CNN_H

#include "std_includes.h"
#include "matrix.h"
#include "function.h"

typedef struct Conv1DLayer_ {
    size_t inputChannels;
    size_t outputChannels;
    size_t kernelSize;
    size_t stride;
    size_t padding;
    Matrix* weights;    
    Matrix* bias;       
    Matrix* inputCache;  
} Conv1DLayer;

static size_t kad_len(Matrix* mat) {
    return mat->rows * mat->cols;
}

static Conv1DLayer* createConv1DLayer(size_t inputChannels, size_t outputChannels, 
                                       size_t kernelSize, size_t stride, size_t padding) {
    Conv1DLayer* layer = (Conv1DLayer*)malloc(sizeof(Conv1DLayer));
    layer->inputChannels = inputChannels;
    layer->outputChannels = outputChannels;
    layer->kernelSize = kernelSize;
    layer->stride = stride;
    layer->padding = padding;
    
    size_t weightCols = inputChannels * kernelSize;
    layer->weights = createMatrixZeroes(outputChannels, weightCols);
    layer->bias = createMatrixZeroes(1, outputChannels);
    layer->inputCache = NULL;
    
    float scale = sqrtf(2.0f / (float)weightCols);
    size_t i, j;
    for (i = 0; i < layer->weights->rows; ++i) {
        for (j = 0; j < layer->weights->cols; ++j) {
            setMatrix(layer->weights, i, j, box_muller() * scale);
        }
    }
    
    return layer;
}

static void destroyConv1DLayer(Conv1DLayer* layer) {
    destroyMatrix(layer->weights);
    destroyMatrix(layer->bias);
    if (layer->inputCache) destroyMatrix(layer->inputCache);
    free(layer);
}

static void conv1DForward(Conv1DLayer* layer, Matrix* input, Matrix* output) {
    size_t batchSize = input->rows;
    size_t inputLen = input->cols / layer->inputChannels;
    size_t outputLen = (inputLen + 2 * layer->padding - layer->kernelSize) / layer->stride + 1;
    
    assert(output->rows == batchSize);
    assert(output->cols == outputLen * layer->outputChannels);
    
    if (layer->inputCache == NULL || layer->inputCache->rows != batchSize || 
        layer->inputCache->cols != input->cols) {
        if (layer->inputCache) destroyMatrix(layer->inputCache);
        layer->inputCache = createMatrixZeroes(batchSize, input->cols);
    }
    copyValuesInto(input, layer->inputCache);
    
    size_t b, outPos, outChan, k, c;
    float* paddedInput = NULL;
    size_t paddedLen = inputLen + 2 * layer->padding;
    
    if (layer->padding > 0) {
        paddedInput = (float*)calloc(paddedLen * layer->inputChannels, sizeof(float));
    }
    
    for (b = 0; b < batchSize; ++b) {
        float* inputRow = input->data + b * input->stride;
        
        if (layer->padding > 0) {
            memset(paddedInput, 0, paddedLen * layer->inputChannels * sizeof(float));
            for (c = 0; c < layer->inputChannels; ++c) {
                for (size_t i = 0; i < inputLen; ++i) {
                    paddedInput[(layer->padding + i) * layer->inputChannels + c] = 
                        inputRow[i * layer->inputChannels + c];
                }
            }
            inputRow = paddedInput;
        }
        
        for (outPos = 0; outPos < outputLen; ++outPos) {
            size_t inputStartPos = outPos * layer->stride;
            
            for (outChan = 0; outChan < layer->outputChannels; ++outChan) {
                float sum = getMatrix(layer->bias, 0, outChan);
                
                for (k = 0; k < layer->kernelSize; ++k) {
                    for (c = 0; c < layer->inputChannels; ++c) {
                        size_t inputIdx = (inputStartPos + k) * layer->inputChannels + c;
                        size_t weightIdx = k * layer->inputChannels + c;
                        sum += inputRow[inputIdx] * 
                               getMatrix(layer->weights, outChan, weightIdx);
                    }
                }
                
                size_t outputIdx = outPos * layer->outputChannels + outChan;
                setMatrix(output, b, outputIdx, sum);
            }
        }
    }
    
    if (paddedInput) free(paddedInput);
}

static void conv1DBackward(Conv1DLayer* layer, Matrix* outputGrad, 
                           Matrix* inputGrad, Matrix* weightGrad, Matrix* biasGrad) {
    size_t batchSize = outputGrad->rows;
    size_t inputLen = layer->inputCache->cols / layer->inputChannels;
    size_t outputLen = outputGrad->cols / layer->outputChannels;
    
    size_t b, outPos, outChan, k, c;
    float* paddedInputGrad = NULL;
    size_t paddedLen = inputLen + 2 * layer->padding;
    
    if (layer->padding > 0) {
        paddedInputGrad = (float*)calloc(paddedLen * layer->inputChannels, sizeof(float));
    }
    
    if (inputGrad) zeroMatrix(inputGrad);
    zeroMatrix(weightGrad);
    zeroMatrix(biasGrad);
    
    for (b = 0; b < batchSize; ++b) {
        float* inputRow = layer->inputCache->data + b * layer->inputCache->stride;
        float* inputGradRow = inputGrad ? inputGrad->data + b * inputGrad->stride : NULL;
        
        if (layer->padding > 0 && inputGradRow) {
            memset(paddedInputGrad, 0, paddedLen * layer->inputChannels * sizeof(float));
        }
        
        for (outPos = 0; outPos < outputLen; ++outPos) {
            size_t inputStartPos = outPos * layer->stride;
            
            for (outChan = 0; outChan < layer->outputChannels; ++outChan) {
                size_t outputIdx = outPos * layer->outputChannels + outChan;
                float grad = getMatrix(outputGrad, b, outputIdx);
                
                setMatrix(biasGrad, 0, outChan, 
                         getMatrix(biasGrad, 0, outChan) + grad);
                
                for (k = 0; k < layer->kernelSize; ++k) {
                    for (c = 0; c < layer->inputChannels; ++c) {
                        size_t inputIdx = (inputStartPos + k) * layer->inputChannels + c;
                        size_t weightIdx = k * layer->inputChannels + c;
                        
                        float inputVal = layer->padding > 0 ? 
                            (inputStartPos + k >= layer->padding && 
                             inputStartPos + k < inputLen + layer->padding ?
                             inputRow[(inputStartPos + k - layer->padding) * 
                                     layer->inputChannels + c] : 0.0f) :
                            inputRow[inputIdx];
                        
                        setMatrix(weightGrad, outChan, weightIdx,
                                 getMatrix(weightGrad, outChan, weightIdx) + 
                                 grad * inputVal);
                        
                        if (inputGradRow) {
                            float weight = getMatrix(layer->weights, outChan, weightIdx);
                            if (layer->padding > 0) {
                                paddedInputGrad[inputIdx] += grad * weight;
                            } else {
                                inputGradRow[inputIdx] += grad * weight;
                            }
                        }
                    }
                }
            }
        }
        
        if (layer->padding > 0 && inputGradRow) {
            for (c = 0; c < layer->inputChannels; ++c) {
                for (size_t i = 0; i < inputLen; ++i) {
                    inputGradRow[i * layer->inputChannels + c] += 
                        paddedInputGrad[(layer->padding + i) * layer->inputChannels + c];
                }
            }
        }
    }
    
    if (paddedInputGrad) free(paddedInputGrad);
}

typedef struct MaxPool1D_ {
    size_t poolSize;
    size_t stride;
    size_t channels;
    int* indices;  
    size_t indicesSize;
} MaxPool1D;

static MaxPool1D* createMaxPool1D(size_t poolSize, size_t stride, size_t channels) {
    MaxPool1D* pool = (MaxPool1D*)malloc(sizeof(MaxPool1D));
    pool->poolSize = poolSize;
    pool->stride = stride;
    pool->channels = channels;
    pool->indices = NULL;
    pool->indicesSize = 0;
    return pool;
}

static void destroyMaxPool1D(MaxPool1D* pool) {
    if (pool->indices) free(pool->indices);
    free(pool);
}

static void maxPool1DForward(MaxPool1D* pool, Matrix* input, Matrix* output) {
    size_t batchSize = input->rows;
    size_t inputLen = input->cols / pool->channels;
    size_t outputLen = (inputLen - pool->poolSize) / pool->stride + 1;
    
    assert(output->rows == batchSize);
    assert(output->cols == outputLen * pool->channels);
    
    size_t neededSize = batchSize * outputLen * pool->channels;
    if (pool->indicesSize < neededSize) {
        pool->indices = (int*)realloc(pool->indices, neededSize * sizeof(int));
        pool->indicesSize = neededSize;
    }
    
    size_t b, outPos, c, k;
    for (b = 0; b < batchSize; ++b) {
        for (outPos = 0; outPos < outputLen; ++outPos) {
            size_t inputStartPos = outPos * pool->stride;
            
            for (c = 0; c < pool->channels; ++c) {
                float maxVal = -FLT_MAX;
                int maxIdx = -1;
                
                for (k = 0; k < pool->poolSize; ++k) {
                    size_t inputIdx = (inputStartPos + k) * pool->channels + c;
                    float val = getMatrix(input, b, inputIdx);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = (int)inputIdx;
                    }
                }
                
                size_t outputIdx = outPos * pool->channels + c;
                setMatrix(output, b, outputIdx, maxVal);
                pool->indices[b * outputLen * pool->channels + outputIdx] = maxIdx;
            }
        }
    }
}

static void maxPool1DBackward(MaxPool1D* pool, Matrix* outputGrad, Matrix* inputGrad) {
    size_t batchSize = outputGrad->rows;
    size_t outputLen = outputGrad->cols / pool->channels;
    
    zeroMatrix(inputGrad);
    
    size_t b, outPos, c;
    for (b = 0; b < batchSize; ++b) {
        for (outPos = 0; outPos < outputLen; ++outPos) {
            for (c = 0; c < pool->channels; ++c) {
                size_t outputIdx = outPos * pool->channels + c;
                int inputIdx = pool->indices[b * outputLen * pool->channels + outputIdx];
                float grad = getMatrix(outputGrad, b, outputIdx);
                
                setMatrix(inputGrad, b, inputIdx, 
                         getMatrix(inputGrad, b, inputIdx) + grad);
            }
        }
    }
}

typedef struct Dropout_ {
    float rate;
    uint8_t* mask;
    size_t maskSize;
} Dropout;

static Dropout* createDropout(float rate) {
    Dropout* dropout = (Dropout*)malloc(sizeof(Dropout));
    dropout->rate = rate;
    dropout->mask = NULL;
    dropout->maskSize = 0;
    return dropout;
}

static void destroyDropout(Dropout* dropout) {
    if (dropout->mask) free(dropout->mask);
    free(dropout);
}

static void dropoutForward(Dropout* dropout, Matrix* input, Matrix* output, int training) {
    size_t len = kad_len(input);
    
    if (!training) {
        copyValuesInto(input, output);
        return;
    }
    
    if (dropout->maskSize < len) {
        dropout->mask = (uint8_t*)realloc(dropout->mask, len * sizeof(uint8_t));
        dropout->maskSize = len;
    }
    
    float keepProb = 1.0f - dropout->rate;
    float scale = 1.0f / keepProb;
    
    size_t i;
    for (i = 0; i < len; ++i) {
        dropout->mask[i] = ((float)rand() / (float)RAND_MAX) >= dropout->rate;
        float val = input->data[i];
        output->data[i] = dropout->mask[i] ? val * scale : 0.0f;
    }
}

static void dropoutBackward(Dropout* dropout, Matrix* outputGrad, Matrix* inputGrad) {
    size_t len = kad_len(outputGrad);
    float keepProb = 1.0f - dropout->rate;
    float scale = 1.0f / keepProb;
    
    size_t i;
    for (i = 0; i < len; ++i) {
        inputGrad->data[i] += dropout->mask[i] ? outputGrad->data[i] * scale : 0.0f;
    }
}

typedef struct LayerNorm_ {
    size_t size;
    Matrix* gamma;  
    Matrix* beta;   
    float* savedMean;
    float* savedInvStd;
    size_t cacheSize;
} LayerNorm;

static LayerNorm* createLayerNorm(size_t size) {
    LayerNorm* norm = (LayerNorm*)malloc(sizeof(LayerNorm));
    norm->size = size;
    norm->gamma = createMatrixZeroes(1, size);
    norm->beta = createMatrixZeroes(1, size);
    
    size_t i;
    for (i = 0; i < size; ++i) {
        setMatrix(norm->gamma, 0, i, 1.0f);
    }
    
    norm->savedMean = NULL;
    norm->savedInvStd = NULL;
    norm->cacheSize = 0;
    return norm;
}

static void destroyLayerNorm(LayerNorm* norm) {
    destroyMatrix(norm->gamma);
    destroyMatrix(norm->beta);
    if (norm->savedMean) free(norm->savedMean);
    if (norm->savedInvStd) free(norm->savedInvStd);
    free(norm);
}

static void layerNormForward(LayerNorm* norm, Matrix* input, Matrix* output) {
    size_t batchSize = input->rows;
    size_t n = norm->size;
    
    assert(input->cols == n);
    assert(output->rows == batchSize && output->cols == n);
    
    if (norm->cacheSize < batchSize) {
        norm->savedMean = (float*)realloc(norm->savedMean, batchSize * sizeof(float));
        norm->savedInvStd = (float*)realloc(norm->savedInvStd, batchSize * sizeof(float));
        norm->cacheSize = batchSize;
    }
    
    size_t b, i;
    const float epsilon = 1e-5f;
    
    for (b = 0; b < batchSize; ++b) {
        float* x = input->data + b * input->stride;
        float* y = output->data + b * output->stride;
        
        double sum = 0.0;
        for (i = 0; i < n; ++i) {
            sum += x[i];
        }
        float mean = (float)(sum / n);
        norm->savedMean[b] = mean;
        
        sum = 0.0;
        for (i = 0; i < n; ++i) {
            float diff = x[i] - mean;
            sum += diff * diff;
        }
        float variance = (float)(sum / n);
        float invStd = 1.0f / sqrtf(variance + epsilon);
        norm->savedInvStd[b] = invStd;
        
        for (i = 0; i < n; ++i) {
            float normalized = (x[i] - mean) * invStd;
            y[i] = normalized * getMatrix(norm->gamma, 0, i) + getMatrix(norm->beta, 0, i);
        }
    }
}

static void layerNormBackward(LayerNorm* norm, Matrix* outputGrad, Matrix* inputGrad,
                              Matrix* gammaGrad, Matrix* betaGrad) {
    size_t batchSize = outputGrad->rows;
    size_t n = norm->size;
    
    zeroMatrix(gammaGrad);
    zeroMatrix(betaGrad);
    if (inputGrad) zeroMatrix(inputGrad);
    
    size_t b, i;
    
    for (b = 0; b < batchSize; ++b) {
        float* dy = outputGrad->data + b * outputGrad->stride;
        float* dx = inputGrad ? inputGrad->data + b * inputGrad->stride : NULL;
        
        float mean = norm->savedMean[b];
        float invStd = norm->savedInvStd[b];
        
        for (i = 0; i < n; ++i) {
            float normalized = 0.0f;  
            setMatrix(gammaGrad, 0, i, getMatrix(gammaGrad, 0, i) + dy[i] * normalized);
            setMatrix(betaGrad, 0, i, getMatrix(betaGrad, 0, i) + dy[i]);
        }
        
        if (dx) {
            double dgamma_sum = 0.0, dbeta_sum = 0.0;
            for (i = 0; i < n; ++i) {
                float gamma = getMatrix(norm->gamma, 0, i);
                dgamma_sum += dy[i] * gamma;
            }
            
            for (i = 0; i < n; ++i) {
                float gamma = getMatrix(norm->gamma, 0, i);
                dx[i] = (dy[i] * gamma - (float)dgamma_sum / n) * invStd;
            }
        }
    }
}

typedef struct RMSprop_ {
    size_t nParams;
    float* r;        
    float learningRate;
    float decay;
    float epsilon;
} RMSprop;

static RMSprop* createRMSprop(size_t nParams, float learningRate, float decay) {
    RMSprop* opt = (RMSprop*)malloc(sizeof(RMSprop));
    opt->nParams = nParams;
    opt->r = (float*)calloc(nParams, sizeof(float));
    opt->learningRate = learningRate;
    opt->decay = decay;
    opt->epsilon = 1e-8f;
    return opt;
}

static void destroyRMSprop(RMSprop* opt) {
    free(opt->r);
    free(opt);
}

static void rmspropUpdate(RMSprop* opt, float* params, float* grads) {
    size_t i;
    for (i = 0; i < opt->nParams; ++i) {
        opt->r[i] = opt->decay * opt->r[i] + (1.0f - opt->decay) * grads[i] * grads[i];
        params[i] -= opt->learningRate * grads[i] / sqrtf(opt->r[i] + opt->epsilon);
    }
}

#endif
