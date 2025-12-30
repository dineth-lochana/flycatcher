#ifndef GRADUAL_LOADER_H
#define GRADUAL_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cranium.h" 

// Structure to hold a mini-batch of data
typedef struct BatchPair_ {
    DataSet* inputs;
    DataSet* targets;
    int endOfFile; // Flag: 1 if EOF reached, 0 otherwise
} BatchPair;

// Helper to parse a comma-separated line into a float array
static void parse_csv_line(char* line, float* dest, size_t count) {
    char* token;
    size_t i = 0;
    
    // Make a copy to avoid modifying the original buffer if needed, 
    // though strsep/strtok modifies in place.
    token = strtok(line, ",");
    while(token != NULL && i < count) {
        dest[i++] = strtof(token, NULL);
        token = strtok(NULL, ",");
    }
}

// Loads a chunk of data from an open CSV file
// file: Open file pointer (cursor must be at start of a line)
// batchSize: Number of rows to read
// numFeatures: Number of input columns (assumed first N columns)
// numOutputs: Number of target columns (assumed last M columns)
static BatchPair loadNextBatch(FILE* file, size_t batchSize, size_t numFeatures, size_t numOutputs) {
    BatchPair batch;
    batch.endOfFile = 0;
    
    // 1. Allocate maximum possible memory for this batch
    // We might resize later or just report smaller rows if we hit EOF.
    float* inputData = (float*)malloc(sizeof(float) * batchSize * numFeatures);
    float* targetData = (float*)malloc(sizeof(float) * batchSize * numOutputs);
    
    size_t rowsRead = 0;
    char buffer[4096]; // Buffer for a single line
    size_t totalCols = numFeatures + numOutputs;
    float* tempRow = (float*)malloc(sizeof(float) * totalCols);

    while(rowsRead < batchSize && fgets(buffer, sizeof(buffer), file)) {
        // Skip empty lines
        if(strlen(buffer) < 2) continue;

        parse_csv_line(buffer, tempRow, totalCols);

        // Split into input and target arrays
        memcpy(inputData + (rowsRead * numFeatures), tempRow, sizeof(float) * numFeatures);
        memcpy(targetData + (rowsRead * numOutputs), tempRow + numFeatures, sizeof(float) * numOutputs);

        rowsRead++;
    }

    free(tempRow);

    if (rowsRead < batchSize) {
        batch.endOfFile = 1;
        if (rowsRead == 0) {
            // EOF reached immediately
            free(inputData);
            free(targetData);
            batch.inputs = NULL;
            batch.targets = NULL;
            return batch;
        }
        // Resize datasets to actual rows read would be ideal, 
        // but creating DataSet with `rowsRead` is sufficient 
        // (extra allocated memory is just wasted until free).
    }

    batch.inputs = createDataSet(rowsRead, numFeatures, inputData);
    batch.targets = createDataSet(rowsRead, numOutputs, targetData);

    return batch;
}

static void destroyBatch(BatchPair* batch) {
    if (batch->inputs) destroyDataSet(batch->inputs);
    if (batch->targets) destroyDataSet(batch->targets);
}

#endif
