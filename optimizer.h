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
    
    size_t numBatches = (data->rows + batchSize - 1) / batchSize; // Ceiling division
    int epoch;
    
    // Corrected loop structure: maxIters represents Epochs
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
            // Accumulate gradients over the batch
            for (t = 0; t < curBatchSize; t++) {
                size_t exampleIdx = batchStart + t;
                
                exampleView->data = getDataSetRow(data, exampleIdx);
                targetView->data = getDataSetRow(classes, exampleIdx);
                
                forwardPass(network, exampleView);
                
                int layer;
                for (layer = (int)network->numLayers - 1; layer > 0; layer--) {
                    Connection* conn = network->connections[layer - 1];
                    Matrix* layerOutput = network->layerBuffers[layer];
                    Matrix* layerError = buf->errors[layer];
                    
                    if (layer == (int)network->numLayers - 1) {
                        size_t j;
                        for (j = 0; j < layerError->cols; j++) {
                            float output = getMatrix(layerOutput, 0, j);
                            float target = targetView->data[j];
                            
                            // FIX: Calculate derivative of cost function correctly
                            float errorTerm = output - target;
                            
                            if (lossFunction == MEAN_SQUARED_ERROR) {
                                // For MSE, delta = (a - y) * f'(z)
                                // Exception: If activation is Linear, f'(z)=1, so (a-y) is fine.
                                // If activation is Softmax, we generally assume CE, but if forced to MSE:
                                // Softmax derivative is complex (Jacobian), usually not supported well in 
                                // simple element-wise libs. We skip deriv for Softmax to keep it simple 
                                // (or rely on user not to mix MSE+Softmax).
                                // But for Sigmoid/Tanh/Relu + MSE, we MUST multiply by deriv.
                                if (network->layers[layer]->activation != softmax) {
                                    float (*deriv)(float) = activationDerivative(network->layers[layer]->activation);
                                    errorTerm = errorTerm * deriv(output);
                                }
                            }
                            // For CrossEntropy + Softmax, delta = (a - y). 
                            // This code assumes the "Canonical Link" simplification.
                            
                            layerError->data[j] = errorTerm;
                        }
                        
                        Matrix* prevOutput = network->layerBuffers[layer - 1];
                        transposeInto(prevOutput, buf->beforeOutputT);
                        multiplyInto(buf->beforeOutputT, layerError, buf->dW[layer - 1]);
                        copyValuesInto(layerError, buf->db[layer - 1]);
                        
                    } else {
                        size_t hiddenIdx = (size_t)(layer - 1);
                        Connection* nextConn = network->connections[layer];
                        
                        transposeInto(nextConn->weights, buf->weightsT[hiddenIdx]);
                        multiplyInto(buf->errors[layer + 1], buf->weightsT[hiddenIdx], 
                                    buf->errorTemp[hiddenIdx]);
                        
                        float (*deriv)(float) = activationDerivative(conn->to->activation);
                        size_t j;
                        for (j = 0; j < buf->fprime[hiddenIdx]->cols; j++) {
                            buf->fprime[hiddenIdx]->data[j] = deriv(getMatrix(layerOutput, 0, j));
                        }
                        
                        hadamardInto(buf->errorTemp[hiddenIdx], buf->fprime[hiddenIdx], layerError);
                        
                        Matrix* prevOutput = network->layerBuffers[layer - 1];
                        transposeInto(prevOutput, buf->inputT[hiddenIdx]);
                        multiplyInto(buf->inputT[hiddenIdx], layerError, buf->dW[layer - 1]);
                        copyValuesInto(layerError, buf->db[layer - 1]);
                    }
                }
                
                size_t i;
                for (i = 0; i < network->numConnections; i++) {
                    addTo(buf->dW[i], buf->dW_avg[i]);
                    addTo(buf->db[i], buf->db_avg[i]);
                }
                
                zeroGradients(buf);
            }
            
            // Apply updates at end of batch
            size_t i;
            // Scale gradients by batch size (Average Gradient)
            float scale = 1.0f / (float)curBatchSize;
            
            for (i = 0; i < network->numConnections; i++) {
                Matrix* gradW = buf->dW_avg[i];
                Matrix* gradB = buf->db_avg[i];
                Matrix* weights = network->connections[i]->weights;
                Matrix* bias = network->connections[i]->bias;
                
                scalarMultiply(gradW, scale);
                scalarMultiply(gradB, scale);

                // Add regularization term to gradient: w = w - lr * (grad + lambda * w)
                if (regularizationStrength > 0.0f) {
                     copyValuesInto(weights, buf->reg[i]);
                     scalarMultiply(buf->reg[i], regularizationStrength);
                     addTo(buf->reg[i], gradW);
                }
                
                // Apply Learning Rate to the total gradient direction
                scalarMultiply(gradW, currentLearningRate);
                scalarMultiply(gradB, currentLearningRate);
                // Now gradW is (lr * gradient_total)

                if (momentumFactor > 0.0f) {
                    Matrix* velocityW = buf->dW_momentum[i];
                    Matrix* velocityB = buf->db_momentum[i];

                    // v = mu * v - lr * grad
                    // Here 'gradW' contains (lr * grad). We want to SUBTRACT it.
                    // So: v = mu * v + (-gradW)
                    
                    // 1. Decay velocity
                    scalarMultiply(velocityW, momentumFactor);
                    scalarMultiply(velocityB, momentumFactor);
                    
                    // 2. Subtract scaled gradient
                    scalarMultiply(gradW, -1.0f);
                    scalarMultiply(gradB, -1.0f);
                    addTo(gradW, velocityW);
                    addTo(gradB, velocityB);
                    
                    // 3. Update weights: w = w + v
                    addTo(velocityW, weights);
                    addTo(velocityB, bias);
                } else {
                    // Standard SGD: w = w - lr * grad
                    // gradW is already (lr * grad).
                    scalarMultiply(gradW, -1.0f);
                    scalarMultiply(gradB, -1.0f);
                    
                    addTo(gradW, weights);
                    addTo(gradB, bias);
                }
            }
            
            zeroAverages(buf);
        }
        
        // Verbose logging check per Epoch (not per batch)
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
