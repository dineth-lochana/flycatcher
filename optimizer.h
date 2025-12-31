#include "std_includes.h"
#include "matrix.h"
#include "function.h"
#include "layer.h"
#include "network.h"

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef enum LOSS_FUNCTION_ {
    CROSS_ENTROPY_LOSS, 
    MEAN_SQUARED_ERROR
} LOSS_FUNCTION;

typedef struct ParameterSet_ {
    Network* network;
    DataSet* data;
    DataSet* classes;
    LOSS_FUNCTION lossFunction;
    size_t batchSize;
    float learningRate;
    float searchTime;
    float regularizationStrength;
    float momentumFactor;
    int maxIters; // Represents number of Epochs
    int shuffle;
    int verbose;
} ParameterSet;

typedef struct TrainingBuffers_ {
    Matrix** errors;
    Matrix** dW;
    Matrix** db;
    Matrix** dW_avg;
    Matrix** db_avg;
    Matrix** dW_momentum;
    Matrix** db_momentum;
    Matrix** reg;
    Matrix** weightsT;
    Matrix** inputT;
    Matrix** errorTemp;
    Matrix** fprime;
    Matrix* beforeOutputT;
    size_t numLayers;
    size_t numConnections;
    size_t numHidden;
} TrainingBuffers;

static TrainingBuffers* createTrainingBuffers(Network* network);
static void destroyTrainingBuffers(TrainingBuffers* buffers);
static void zeroGradients(TrainingBuffers* buffers);
static void zeroAverages(TrainingBuffers* buffers);

static void batchGradientDescent(Network* network, DataSet* data, DataSet* classes, 
                                  LOSS_FUNCTION lossFunction, size_t batchSize, 
                                  float learningRate, float searchTime, 
                                  float regularizationStrength, float momentumFactor, 
                                  int maxIters, int shuffle, int verbose);

static void optimize(ParameterSet params) {
    batchGradientDescent(params.network, params.data, params.classes, 
                         params.lossFunction, params.batchSize, 
                         params.learningRate, params.searchTime, 
                         params.regularizationStrength, params.momentumFactor, 
                         params.maxIters, params.shuffle, params.verbose);
}


/*
    Begin functions.
*/

TrainingBuffers* createTrainingBuffers(Network* network) {
    TrainingBuffers* buf = (TrainingBuffers*)malloc(sizeof(TrainingBuffers));
    
    size_t numConn = network->numConnections;
    size_t numLayers = network->numLayers;
    size_t numHidden = numLayers - 2;
    
    buf->numLayers = numLayers;
    buf->numConnections = numConn;
    buf->numHidden = numHidden;
    
    buf->errors = (Matrix**)malloc(sizeof(Matrix*) * numLayers);
    size_t i;
    for (i = 0; i < numLayers; i++) {
        buf->errors[i] = createMatrixZeroes(1, network->layers[i]->size);
    }
    
    buf->dW = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    buf->db = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    buf->dW_avg = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    buf->db_avg = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    buf->dW_momentum = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    buf->db_momentum = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    buf->reg = (Matrix**)malloc(sizeof(Matrix*) * numConn);
    
    for (i = 0; i < numConn; i++) {
        size_t wRows = network->connections[i]->weights->rows;
        size_t wCols = network->connections[i]->weights->cols;
        size_t bCols = network->connections[i]->bias->cols;
        
        buf->dW[i] = createMatrixZeroes(wRows, wCols);
        buf->db[i] = createMatrixZeroes(1, bCols);
        buf->dW_avg[i] = createMatrixZeroes(wRows, wCols);
        buf->db_avg[i] = createMatrixZeroes(1, bCols);
        buf->dW_momentum[i] = createMatrixZeroes(wRows, wCols);
        buf->db_momentum[i] = createMatrixZeroes(1, bCols);
        buf->reg[i] = createMatrixZeroes(wRows, wCols);
    }
    
    buf->beforeOutputT = createMatrixZeroes(network->layers[numLayers - 2]->size, 1);
    
    if (numHidden > 0) {
        buf->weightsT = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        buf->inputT = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        buf->errorTemp = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        buf->fprime = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        
        for (i = 0; i < numHidden; i++) {
            Connection* conn = network->connections[i + 1];
            buf->weightsT[i] = createMatrixZeroes(conn->weights->cols, conn->weights->rows);
            buf->inputT[i] = createMatrixZeroes(network->connections[i]->from->size, 1);
            buf->errorTemp[i] = createMatrixZeroes(1, buf->weightsT[i]->cols);
            buf->fprime[i] = createMatrixZeroes(1, network->connections[i]->to->size);
        }
    } else {
        buf->weightsT = NULL;
        buf->inputT = NULL;
        buf->errorTemp = NULL;
        buf->fprime = NULL;
    }
    
    return buf;
}

void destroyTrainingBuffers(TrainingBuffers* buf) {
    size_t i;
    
    for (i = 0; i < buf->numLayers; i++) {
        destroyMatrix(buf->errors[i]);
    }
    free(buf->errors);
    
    for (i = 0; i < buf->numConnections; i++) {
        destroyMatrix(buf->dW[i]);
        destroyMatrix(buf->db[i]);
        destroyMatrix(buf->dW_avg[i]);
        destroyMatrix(buf->db_avg[i]);
        destroyMatrix(buf->dW_momentum[i]);
        destroyMatrix(buf->db_momentum[i]);
        destroyMatrix(buf->reg[i]);
    }
    free(buf->dW);
    free(buf->db);
    free(buf->dW_avg);
    free(buf->db_avg);
    free(buf->dW_momentum);
    free(buf->db_momentum);
    free(buf->reg);
    
    destroyMatrix(buf->beforeOutputT);
    
    if (buf->numHidden > 0) {
        for (i = 0; i < buf->numHidden; i++) {
            destroyMatrix(buf->weightsT[i]);
            destroyMatrix(buf->inputT[i]);
            destroyMatrix(buf->errorTemp[i]);
            destroyMatrix(buf->fprime[i]);
        }
        free(buf->weightsT);
        free(buf->inputT);
        free(buf->errorTemp);
        free(buf->fprime);
    }
    
    free(buf);
}

void zeroGradients(TrainingBuffers* buf) {
    size_t i;
    for (i = 0; i < buf->numLayers; i++) {
        zeroMatrix(buf->errors[i]);
    }
    for (i = 0; i < buf->numConnections; i++) {
        zeroMatrix(buf->dW[i]);
        zeroMatrix(buf->db[i]);
    }
    zeroMatrix(buf->beforeOutputT);
    if (buf->numHidden > 0) {
        for (i = 0; i < buf->numHidden; i++) {
            zeroMatrix(buf->weightsT[i]);
            zeroMatrix(buf->inputT[i]);
            zeroMatrix(buf->errorTemp[i]);
            zeroMatrix(buf->fprime[i]);
        }
    }
}

void zeroAverages(TrainingBuffers* buf) {
    size_t i;
    for (i = 0; i < buf->numConnections; i++) {
        zeroMatrix(buf->dW_avg[i]);
        zeroMatrix(buf->db_avg[i]);
        zeroMatrix(buf->reg[i]);
    }
}

void batchGradientDescent(Network* network, DataSet* data, DataSet* classes, 
                          LOSS_FUNCTION lossFunction, size_t batchSize, 
                          float learningRate, float searchTime, 
                          float regularizationStrength, float momentumFactor, 
                          int maxIters, int shuffle, int verbose) {
    
    assert(network->layers[0]->size == data->cols);
    assert(data->rows == classes->rows);
    assert(network->layers[network->numLayers - 1]->size == classes->cols);
    assert(batchSize <= data->rows);
    assert(maxIters >= 1);
    
    TrainingBuffers* buf = createTrainingBuffers(network);
    ensureBatchCapacity(network, 1);
    
    Matrix* exampleView = createMatrixView(1, data->cols, data->cols, NULL);
    Matrix* targetView = createMatrixView(1, classes->cols, classes->cols, NULL);
    
    size_t numBatches = (data->rows + batchSize - 1) / batchSize;
    int epoch;
    
    for (epoch = 1; epoch <= maxIters; epoch++) {
        if (shuffle != 0) {
            shuffleTogether(data, classes);
        }
        
        float currentLearningRate = learningRate;
        if (searchTime > 0.0f) {
            currentLearningRate = learningRate / (1.0f + ((float)epoch / searchTime));
        }
        
        size_t batch;
        for (batch = 0; batch < numBatches; batch++) {
            size_t batchStart = batch * batchSize;
            size_t curBatchSize = batchSize;
            if (batchStart + batchSize > data->rows) {
                curBatchSize = data->rows - batchStart;
            }
            
            size_t t;
            for (t = 0; t < curBatchSize; t++) {
                size_t exampleIdx = batchStart + t;
                
                exampleView->data = getDataSetRow(data, exampleIdx);
                targetView->data = getDataSetRow(classes, exampleIdx);
                
                forwardPass(network, exampleView);
                
                int layer;
                size_t j, k;
                for (layer = (int)network->numLayers - 1; layer > 0; layer--) {
                    Connection* conn = network->connections[layer - 1];
                    float* layerOutput = network->layerBuffers[layer]->data;
                    float* layerError = buf->errors[layer]->data;
                    size_t errorSize = buf->errors[layer]->cols;
                    
                    if (layer == (int)network->numLayers - 1) {
                        /* Output layer - direct array access */
                        for (j = 0; j < errorSize; j++) {
                            float output = layerOutput[j];
                            float target = targetView->data[j];
                            float errorTerm = output - target;
                            
                            if (lossFunction == MEAN_SQUARED_ERROR) {
                                if (network->layers[layer]->activation != softmax) {
                                    float (*deriv)(float) = activationDerivative(network->layers[layer]->activation);
                                    errorTerm = errorTerm * deriv(output);
                                }
                            }
                            
                            layerError[j] = errorTerm;
                        }
                        
                        /* Gradient accumulation - optimized */
                        float* prevOutput = network->layerBuffers[layer - 1]->data;
                        size_t prevSize = network->layerBuffers[layer - 1]->cols;
                        float* dW_data = buf->dW[layer - 1]->data;
                        size_t dW_stride = buf->dW[layer - 1]->stride;
                        
                        /* Outer product: prevOutput^T * layerError */
                        for (j = 0; j < prevSize; j++) {
                            float prev_val = prevOutput[j];
                            float* dW_row = dW_data + j * dW_stride;
                            for (k = 0; k < errorSize; k++) {
                                dW_row[k] = prev_val * layerError[k];
                            }
                        }
                        
                        /* Bias gradient */
                        memcpy(buf->db[layer - 1]->data, layerError, errorSize * sizeof(float));
                        
                    } else {
                        /* Hidden layer */
                        size_t hiddenIdx = (size_t)(layer - 1);
                        Connection* nextConn = network->connections[layer];
                        
                        /* Error backprop: error = nextError * weights^T */
                        float* nextError = buf->errors[layer + 1]->data;
                        size_t nextErrorSize = buf->errors[layer + 1]->cols;
                        float* weights = nextConn->weights->data;
                        size_t weights_stride = nextConn->weights->stride;
                        
                        memset(layerError, 0, errorSize * sizeof(float));
                        
                        for (j = 0; j < errorSize; j++) {
                            float sum = 0.0f;
                            float* weight_col = weights + j;
                            for (k = 0; k < nextErrorSize; k++) {
                                sum = sum + (nextError[k] * weight_col[k * weights_stride]);
                            }
                            layerError[j] = sum;
                        }
                        
                        /* Apply activation derivative */
                        float (*deriv)(float) = activationDerivative(conn->to->activation);
                        for (j = 0; j < errorSize; j++) {
                            layerError[j] = layerError[j] * deriv(layerOutput[j]);
                        }
                        
                        /* Gradient computation - optimized */
                        float* prevOutput = network->layerBuffers[layer - 1]->data;
                        size_t prevSize = network->layerBuffers[layer - 1]->cols;
                        float* dW_data = buf->dW[layer - 1]->data;
                        size_t dW_stride = buf->dW[layer - 1]->stride;
                        
                        for (j = 0; j < prevSize; j++) {
                            float prev_val = prevOutput[j];
                            float* dW_row = dW_data + j * dW_stride;
                            for (k = 0; k < errorSize; k++) {
                                dW_row[k] = prev_val * layerError[k];
                            }
                        }
                        
                        memcpy(buf->db[layer - 1]->data, layerError, errorSize * sizeof(float));
                    }
                }
                
                /* Accumulate gradients */
                size_t i;
                for (i = 0; i < network->numConnections; i++) {
                    addTo(buf->dW[i], buf->dW_avg[i]);
                    addTo(buf->db[i], buf->db_avg[i]);
                }
                
                zeroGradients(buf);
            }
            
            /* Apply updates at end of batch */
            size_t i;
            float scale = 1.0f / (float)curBatchSize;
            
            for (i = 0; i < network->numConnections; i++) {
                Matrix* gradW = buf->dW_avg[i];
                Matrix* gradB = buf->db_avg[i];
                Matrix* weights = network->connections[i]->weights;
                Matrix* bias = network->connections[i]->bias;
                
                scalarMultiply(gradW, scale);
                scalarMultiply(gradB, scale);

                if (regularizationStrength > 0.0f) {
                     copyValuesInto(weights, buf->reg[i]);
                     scalarMultiply(buf->reg[i], regularizationStrength);
                     addTo(buf->reg[i], gradW);
                }
                
                scalarMultiply(gradW, currentLearningRate);
                scalarMultiply(gradB, currentLearningRate);

                if (momentumFactor > 0.0f) {
                    Matrix* velocityW = buf->dW_momentum[i];
                    Matrix* velocityB = buf->db_momentum[i];
                    
                    scalarMultiply(velocityW, momentumFactor);
                    scalarMultiply(velocityB, momentumFactor);
                    
                    scalarMultiply(gradW, -1.0f);
                    scalarMultiply(gradB, -1.0f);
                    addTo(gradW, velocityW);
                    addTo(gradB, velocityB);
                    
                    addTo(velocityW, weights);
                    addTo(velocityB, bias);
                } else {
                    scalarMultiply(gradW, -1.0f);
                    scalarMultiply(gradB, -1.0f);
                    
                    addTo(gradW, weights);
                    addTo(gradB, bias);
                }
            }
            
            zeroAverages(buf);
        }
        
        if (verbose != 0) {
            if (epoch % 10 == 0 || epoch == 1 || epoch == maxIters) {
                forwardPassDataSet(network, data);
                float loss;
                if (lossFunction == CROSS_ENTROPY_LOSS) {
                    loss = crossEntropyLoss(network, getOutput(network), classes, 
                                           regularizationStrength);
                } else {
                    loss = meanSquaredError(network, getOutput(network), classes, 
                                           regularizationStrength);
                }
                printf("EPOCH %d: loss is %f\n", epoch, loss);
            }
        }
    }
    
    destroyMatrix(exampleView);
    destroyMatrix(targetView);
    destroyTrainingBuffers(buf);
}

#endif
