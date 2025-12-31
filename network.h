#include "std_includes.h"
#include "matrix.h"
#include "function.h"
#include "layer.h"

#ifndef NETWORK_H
#define NETWORK_H

typedef struct Network_ {
    size_t numLayers;
    Layer** layers;
    size_t numConnections;
    Connection** connections;
    // Pre-allocated buffers for batch forward pass
    Matrix** layerBuffers;
    Matrix* tempBuffer;
    size_t maxBatchSize;
} Network;

static Network* createNetwork(size_t numFeatures, size_t numHiddenLayers, 
                              size_t* hiddenSizes, Activation* hiddenActivations, 
                              size_t numOutputs, Activation outputActivation);

static void ensureBatchCapacity(Network* network, size_t batchSize);

static void forwardPass(Network* network, Matrix* input);

static void forwardPassDataSet(Network* network, DataSet* input);

static float crossEntropyLoss(Network* network, Matrix* prediction, DataSet* actual, 
                              float regularizationStrength);

static float meanSquaredError(Network* network, Matrix* prediction, DataSet* actual, 
                              float regularizationStrength);

static Matrix* getOutput(Network* network);

static int* predict(Network* network);

static float accuracy(Network* network, DataSet* data, DataSet* classes);

static void destroyNetwork(Network* network);

static void saveNetwork(Network* network, char* path);

static Network* readNetwork(char* path);


/*
    Begin functions.
*/

static void freeLayerBuffers(Network* network) {
    if (network->layerBuffers != NULL) {
        size_t i;
        for (i = 0; i < network->numLayers; i++) {
            if (network->layerBuffers[i] != NULL) {
                destroyMatrix(network->layerBuffers[i]);
            }
        }
        free(network->layerBuffers);
        network->layerBuffers = NULL;
    }
    if (network->tempBuffer != NULL) {
        destroyMatrix(network->tempBuffer);
        network->tempBuffer = NULL;
    }
}

void ensureBatchCapacity(Network* network, size_t batchSize) {
    if (batchSize <= network->maxBatchSize) {
        return;
    }
    
    // Free old buffers
    freeLayerBuffers(network);
    
    // Allocate new buffers
    network->layerBuffers = (Matrix**)malloc(sizeof(Matrix*) * network->numLayers);
    size_t maxLayerSize = 0;
    size_t i;
    
    for (i = 0; i < network->numLayers; i++) {
        network->layerBuffers[i] = createMatrixZeroes(batchSize, network->layers[i]->size);
        if (network->layers[i]->size > maxLayerSize) {
            maxLayerSize = network->layers[i]->size;
        }
    }
    
    // Temp buffer for intermediate calculations
    network->tempBuffer = createMatrixZeroes(batchSize, maxLayerSize);
    network->maxBatchSize = batchSize;
}

Network* createNetwork(size_t numFeatures, size_t numHiddenLayers, 
                       size_t* hiddenSizes, Activation* hiddenActivations, 
                       size_t numOutputs, Activation outputActivation) {
    assert(numFeatures > 0 && numOutputs > 0);
    
    Network* network = (Network*)malloc(sizeof(Network));
    network->numLayers = 2 + numHiddenLayers;
    network->layers = (Layer**)malloc(sizeof(Layer*) * network->numLayers);
    
    size_t i;
    for (i = 0; i < network->numLayers; i++) {
        if (i == 0) {
            network->layers[i] = createLayer(INPUT, numFeatures, NULL);
        } else if (i == network->numLayers - 1) {
            network->layers[i] = createLayer(OUTPUT, numOutputs, outputActivation);
        } else {
            network->layers[i] = createLayer(HIDDEN, hiddenSizes[i - 1], hiddenActivations[i - 1]);
        }
    }
    
    network->numConnections = network->numLayers - 1;
    network->connections = (Connection**)malloc(sizeof(Connection*) * network->numConnections);
    
    for (i = 0; i < network->numConnections; i++) {
        network->connections[i] = createConnection(network->layers[i], network->layers[i + 1]);
        initializeConnection(network->connections[i]);
    }
    
    // Initialize buffer pointers
    network->layerBuffers = NULL;
    network->tempBuffer = NULL;
    network->maxBatchSize = 0;
    
    return network;
}

void forwardPass(Network* network, Matrix* input) {
    assert(input->cols == network->layers[0]->size);
    
    size_t batchSize = input->rows;
    ensureBatchCapacity(network, batchSize);

    // Update row count of buffers to match current batch size
    size_t l;
    for(l = 0; l < network->numLayers; l++) {
        network->layerBuffers[l]->rows = batchSize;
    }
    
    // Copy input to first layer buffer
    copyValuesInto(input, network->layerBuffers[0]);
    
    size_t i, j, k;
    for (i = 0; i < network->numConnections; i++) {
        Matrix* fromBuffer = network->layerBuffers[i];
        Matrix* toBuffer = network->layerBuffers[i + 1];
        Connection* conn = network->connections[i];
        
        // toBuffer = fromBuffer * weights
        multiplyInto(fromBuffer, conn->weights, toBuffer);
        
        // Add bias to each row
        for (j = 0; j < batchSize; j++) {
            for (k = 0; k < conn->bias->cols; k++) {
                setMatrix(toBuffer, j, k, 
                         getMatrix(toBuffer, j, k) + getMatrix(conn->bias, 0, k));
            }
        }
        
        // Apply activation
        if (network->layers[i + 1]->activation != NULL) {
            network->layers[i + 1]->activation(toBuffer);
        }
    }
    
    // For single examples, copy back to layer->input for legacy access
    if (batchSize == 1) {
        for (i = 0; i < network->numLayers; i++) {
            copyValuesInto(network->layerBuffers[i], network->layers[i]->input);
        }
    }
}

void forwardPassDataSet(Network* network, DataSet* input) {
    Matrix* inputMatrix = dataSetToMatrixView(input);
    forwardPass(network, inputMatrix);
    destroyMatrix(inputMatrix);
}

float crossEntropyLoss(Network* network, Matrix* prediction, DataSet* actual, 
                       float regularizationStrength) {
    assert(prediction->rows == actual->rows);
    assert(prediction->cols == actual->cols);
    
    float total_err = 0.0f;
    size_t i, j, k;
    
    for (i = 0; i < prediction->rows; i++) {
        for (j = 0; j < prediction->cols; j++) {
            float target = getDataSetElement(actual, i, j);
            float pred = getMatrix(prediction, i, j);
            total_err = total_err + (target * logf(MAX(FLT_MIN, pred)));
        }
    }
    
    float reg_err = 0.0f;
    if (network != NULL && regularizationStrength > 0.0f) {
        for (i = 0; i < network->numConnections; i++) {
            Matrix* weights = network->connections[i]->weights;
            for (j = 0; j < weights->rows; j++) {
                for (k = 0; k < weights->cols; k++) {
                    float w = getMatrix(weights, j, k);
                    reg_err = reg_err + (w * w);
                }
            }
        }
    }
    
    return ((-1.0f / actual->rows) * total_err) + (regularizationStrength * 0.5f * reg_err);
}

float meanSquaredError(Network* network, Matrix* prediction, DataSet* actual, 
                       float regularizationStrength) {
    assert(prediction->rows == actual->rows);
    assert(prediction->cols == actual->cols);
    
    float total_err = 0.0f;
    size_t i, j, k;
    
    for (i = 0; i < prediction->rows; i++) {
        for (j = 0; j < prediction->cols; j++) {
            float diff = getDataSetElement(actual, i, j) - getMatrix(prediction, i, j);
            total_err = total_err + (diff * diff);
        }
    }
    
    float reg_err = 0.0f;
    if (network != NULL && regularizationStrength > 0.0f) {
        for (i = 0; i < network->numConnections; i++) {
            Matrix* weights = network->connections[i]->weights;
            for (j = 0; j < weights->rows; j++) {
                for (k = 0; k < weights->cols; k++) {
                    float w = getMatrix(weights, j, k);
                    reg_err = reg_err + (w * w);
                }
            }
        }
    }
    
    return ((0.5f / actual->rows) * total_err) + (regularizationStrength * 0.5f * reg_err);
}

Matrix* getOutput(Network* network) {
    if (network->maxBatchSize > 0) {
        return network->layerBuffers[network->numLayers - 1];
    }
    return network->layers[network->numLayers - 1]->input;
}

int* predict(Network* network) {
    Matrix* output = getOutput(network);
    int* predictions = (int*)malloc(sizeof(int) * output->rows);
    
    size_t i, j;
    for (i = 0; i < output->rows; i++) {
        // Binary classification logic (1 output neuron)
        if (output->cols == 1) {
            float val = getMatrix(output, i, 0);
            predictions[i] = (val > 0.5f) ? 1 : 0;
        } 
        else {
            int maxIdx = 0;
            float maxVal = getMatrix(output, i, 0);
            for (j = 1; j < output->cols; j++) {
                float val = getMatrix(output, i, j);
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = (int)j;
                }
            }
            predictions[i] = maxIdx;
        }
    }
    
    return predictions;
}

float accuracy(Network* network, DataSet* data, DataSet* classes) {
    assert(data->rows == classes->rows);
    
    forwardPassDataSet(network, data);
    int* predictions = predict(network);
    
    float numCorrect = 0.0f;
    size_t i;
    for (i = 0; i < data->rows; i++) {
        if (classes->cols == 1) {
            // Binary case
            int actual = (int)getDataSetElement(classes, i, 0);
            if (actual == predictions[i]) {
                numCorrect = numCorrect + 1.0f;
            }
        } else {
            // One-hot case
             if (getDataSetElement(classes, i, predictions[i]) == 1.0f) {
                numCorrect = numCorrect + 1.0f;
            }
        }
    }
    
    free(predictions);
    return numCorrect / (float)classes->rows;
}

void destroyNetwork(Network* network) {
    size_t i;
    
    freeLayerBuffers(network);
    
    for (i = 0; i < network->numLayers; i++) {
        destroyLayer(network->layers[i]);
    }
    for (i = 0; i < network->numConnections; i++) {
        destroyConnection(network->connections[i]);
    }
    
    free(network->layers);
    free(network->connections);
    free(network);
}

void saveNetwork(Network* network, char* path) {
    FILE* fp = fopen(path, "w");
    if (fp == NULL) return;
    
    size_t i, j, k;
    
    fprintf(fp, "%zu\n", network->numLayers);
    
    for (i = 0; i < network->numLayers; i++) {
        fprintf(fp, "%zu\n", network->layers[i]->size);
    }
    
    for (i = 0; i < network->numLayers - 1; i++) {
        fprintf(fp, "%s\n", getFunctionName(network->layers[i + 1]->activation));
    }
    
    // Save Weights using high-precision scientific notation
    for (k = 0; k < network->numConnections; k++) {
        Connection* con = network->connections[k];
        for (i = 0; i < con->weights->rows; i++) {
            for (j = 0; j < con->weights->cols; j++) {
                fprintf(fp, "%.20e\n", getMatrix(con->weights, i, j));
            }
        }
    }
    
    // Save Biases using high-precision scientific notation
    for (k = 0; k < network->numConnections; k++) {
        Connection* con = network->connections[k];
        for (i = 0; i < con->bias->cols; i++) {
            fprintf(fp, "%.20e\n", getMatrix(con->bias, 0, i));
        }
    }
    
    fclose(fp);
}

Network* readNetwork(char* path) {
    FILE* fp = fopen(path, "r");
    if (fp == NULL) return NULL;
    
    size_t i, j, k;
    char buf[128];
    
    size_t numLayers;
    if (fgets(buf, 128, fp) == NULL) { fclose(fp); return NULL; }
    sscanf(buf, "%zu", &numLayers);
    
    size_t* layerSizes = (size_t*)malloc(sizeof(size_t) * numLayers);
    for (i = 0; i < numLayers; i++) {
        if (fgets(buf, 128, fp) == NULL) { free(layerSizes); fclose(fp); return NULL; }
        sscanf(buf, "%zu", &layerSizes[i]);
    }
    
    Activation* funcs = (Activation*)malloc(sizeof(Activation) * (numLayers - 1));
    char funcString[50];
    for (i = 0; i < numLayers - 1; i++) {
        if (fgets(buf, 128, fp) == NULL) { free(layerSizes); free(funcs); fclose(fp); return NULL; }
        sscanf(buf, "%49s", funcString);
        funcs[i] = getFunctionByName(funcString);
    }
    
    Network* network;
    size_t inputSize = layerSizes[0];
    size_t outputSize = layerSizes[numLayers - 1];
    size_t numHiddenLayers = numLayers - 2;
    Activation outputFunc = funcs[numLayers - 2];
    
    if (numHiddenLayers > 0) {
        size_t* hiddenSizes = (size_t*)malloc(sizeof(size_t) * numHiddenLayers);
        Activation* hiddenFuncs = (Activation*)malloc(sizeof(Activation) * numHiddenLayers);
        
        for (i = 0; i < numHiddenLayers; i++) {
            hiddenSizes[i] = layerSizes[1 + i];
            hiddenFuncs[i] = funcs[i];
        }
        
        network = createNetwork(inputSize, numHiddenLayers, hiddenSizes, hiddenFuncs, 
                               outputSize, outputFunc);
        
        free(hiddenSizes);
        free(hiddenFuncs);
    } else {
        network = createNetwork(inputSize, 0, NULL, NULL, outputSize, outputFunc);
    }
    
    free(layerSizes);
    free(funcs);
    
    // Read Weights
    for (k = 0; k < network->numConnections; k++) {
        Connection* con = network->connections[k];
        for (i = 0; i < con->weights->rows; i++) {
            for (j = 0; j < con->weights->cols; j++) {
                if (fgets(buf, 128, fp) == NULL) { destroyNetwork(network); fclose(fp); return NULL; }
                float val = 0.0f;
                // Use %f which supports both standard and scientific notation (1.23e-4)
                sscanf(buf, "%f", &val);
                setMatrix(con->weights, i, j, val);
            }
        }
    }
    
    // Read Biases
    for (k = 0; k < network->numConnections; k++) {
        Connection* con = network->connections[k];
        for (i = 0; i < con->bias->cols; i++) {
            if (fgets(buf, 128, fp) == NULL) { destroyNetwork(network); fclose(fp); return NULL; }
            float val = 0.0f;
            sscanf(buf, "%f", &val);
            setMatrix(con->bias, 0, i, val);
        }
    }
    
    fclose(fp);
    return network;
}

#endif
