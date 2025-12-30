#include "std_includes.h"
#include "matrix.h"
#include "function.h"

#ifndef LAYER_H
#define LAYER_H

typedef enum LAYER_TYPE_ {
    INPUT,
    HIDDEN,
    OUTPUT
} LAYER_TYPE;

typedef struct Layer_ {
    LAYER_TYPE type;
    size_t size;
    Activation activation;
    Matrix* input;
} Layer;

typedef struct Connection_ {
    Layer* from;
    Layer* to;
    Matrix* weights;
    Matrix* bias;
} Connection;

static Layer* createLayer(LAYER_TYPE type, size_t size, Activation activation);
static Connection* createConnection(Layer* from, Layer* to);
static void initializeConnection(Connection* connection);
static void activateLayer(Layer* layer);
static void destroyLayer(Layer* layer);
static void destroyConnection(Connection* connection);


/*
    Begin functions.
*/

Layer* createLayer(LAYER_TYPE type, size_t size, Activation activation) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->type = type;
    layer->size = size;
    layer->activation = activation;
    layer->input = createMatrixZeroes(1, size);
    return layer;
}

Connection* createConnection(Layer* from, Layer* to) {
    Connection* connection = (Connection*)malloc(sizeof(Connection));
    connection->from = from;
    connection->to = to;
    connection->weights = createMatrixZeroes(from->size, to->size);
    connection->bias = createMatrixZeroes(1, to->size);
    return connection;
}

void initializeConnection(Connection* connection) {
    size_t i, j;
    
    // Zero biases
    for (i = 0; i < connection->bias->cols; i++) {
        setMatrix(connection->bias, 0, i, 0.0f);
    }
    
    // Gaussian random weights using fan-in method
    float scale = 1.0f / sqrtf((float)connection->weights->rows);
    for (i = 0; i < connection->weights->rows; i++) {
        for (j = 0; j < connection->weights->cols; j++) {
            setMatrix(connection->weights, i, j, box_muller() * scale);
        }
    }
}

void activateLayer(Layer* layer) {
    if (layer->activation != NULL) {
        layer->activation(layer->input);
    }
}

void destroyLayer(Layer* layer) {
    destroyMatrix(layer->input);
    free(layer);
}

void destroyConnection(Connection* connection) {
    destroyMatrix(connection->weights);
    destroyMatrix(connection->bias);
    free(connection);
}

#endif
