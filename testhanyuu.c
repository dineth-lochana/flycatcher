// set SMLRASM=n2f.exe
// set SMLRC=C:\Users\DinethLochana\Downloads\C\smlrc
// smlrcc -Wall -win -SI include -SL lib tinn.c hanyuu.c testhanyuu.c -o testhanyuu.exe

#include "hanyuu.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

typedef struct
{
    float** in;
    float** tg;
    int nips;
    int nops;
    int rows;
}
Data;

typedef struct
{
    Normalization norm;
    WeightInit init;
    Activation hidden_act;
    Activation output_act;
    float momentum;
    int early_stop;
    float learning_rate;
    int nhid;
    float final_error;
    float accuracy;
}
GridResult;

static int lns(FILE* const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

static Data ndata(const int nips, const int nops, const int rows)
{
    Data data;
    data.in = new2d(rows, nips);
    data.tg = new2d(rows, nops);
    data.nips = nips;
    data.nops = nops;
    data.rows = rows;
    return data;
}

static void parse(const Data data, char* line, const int row)
{
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

static void parse_csv(const Data data, char* line, const int row)
{
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, ","));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float* ot = d.tg[a];
        float* it = d.in[a];
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

static Data build(const char* path, const int nips, const int nops, int use_csv)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(nips, nops, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        if(use_csv)
            parse_csv(data, line, row);
        else
            parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

static Data split_data(const Data full, int start, int end)
{
    int size = end - start;
    Data subset = ndata(full.nips, full.nops, size);
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < full.nips; j++)
            subset.in[i][j] = full.in[start + i][j];
        for(int j = 0; j < full.nops; j++)
            subset.tg[i][j] = full.tg[start + i][j];
    }
    return subset;
}

static float calc_accuracy(Hanyuu h, const Data test)
{
    int correct = 0;
    for(int i = 0; i < test.rows; i++)
    {
        const float* pred = hpredict(h, test.in[i]);
        int pred_class = 0;
        int true_class = 0;
        float max_pred = pred[0];
        float max_true = test.tg[i][0];
        
        for(int j = 1; j < test.nops; j++)
        {
            if(pred[j] > max_pred)
            {
                max_pred = pred[j];
                pred_class = j;
            }
            if(test.tg[i][j] > max_true)
            {
                max_true = test.tg[i][j];
                true_class = j;
            }
        }
        
        if(pred_class == true_class)
            correct++;
    }
    return (float)correct / test.rows;
}

static const char* norm_name(Normalization n)
{
    if(n == NORM_NONE) return "None";
    if(n == NORM_MINMAX) return "MinMax";
    return "ZScore";
}

static const char* init_name(WeightInit i)
{
    if(i == INIT_RANDOM) return "Random";
    if(i == INIT_XAVIER) return "Xavier";
    return "He";
}

static const char* act_name(Activation a)
{
    if(a == ACT_SIGMOID) return "Sigmoid";
    if(a == ACT_TANH) return "Tanh";
    if(a == ACT_RELU) return "ReLU";
    return "LeakyReLU";
}

static GridResult run_config(const Data train, const Data test, 
                             Normalization norm, WeightInit init,
                             Activation hidden_act, Activation output_act,
                             float momentum_val, int early_stop_enabled,
                             float lr, int nhid, int iterations)
{
    GridResult result;
    result.norm = norm;
    result.init = init;
    result.hidden_act = hidden_act;
    result.output_act = output_act;
    result.momentum = momentum_val;
    result.early_stop = early_stop_enabled;
    result.learning_rate = lr;
    result.nhid = nhid;
    
    HanyuuConfig cfg = hconfig_default();
    cfg.learning_rate = lr;
    cfg.momentum = momentum_val;
    cfg.batch_size = 1;
    cfg.hidden_act = hidden_act;
    cfg.output_act = output_act;
    cfg.init_type = init;
    cfg.norm_type = norm;
    cfg.early_stop_patience = early_stop_enabled ? 15 : 0;
    cfg.early_stop_delta = 0.0001f;
    
    Hanyuu h = hbuild(train.nips, nhid, train.nops, cfg);
    
    if(norm != NORM_NONE)
        hnorm_fit(h.norm, train.in, train.rows);
    
    float rate = lr;
    const float anneal = 0.99f;
    float final_error = 0.0f;
    
    for(int iter = 0; iter < iterations; iter++)
    {
        float error = 0.0f;
        for(int j = 0; j < train.rows; j++)
        {
            const float* const in = train.in[j];
            const float* const tg = train.tg[j];
            error = error + htrain(h, in, tg, rate);
        }
        final_error = error / train.rows;
        
        if(h.early_stop && hearly_stop_check(h.early_stop, final_error))
            break;
        
        rate = rate * anneal;
    }
    
    result.final_error = final_error;
    result.accuracy = calc_accuracy(h, test);
    
    hfree(h);
    return result;
}

static void print_result(GridResult r, int config_num)
{
    printf("Config %d: Norm=%s Init=%s HAct=%s OAct=%s Mom=%.2f ES=%d LR=%.3f Hid=%d -> Err=%.6f Acc=%.4f\n",
           config_num,
           norm_name(r.norm),
           init_name(r.init),
           act_name(r.hidden_act),
           act_name(r.output_act),
           (double)r.momentum,
           r.early_stop,
           (double)r.learning_rate,
           r.nhid,
           (double)r.final_error,
           (double)r.accuracy);
}

static int rand_choice(int max_val)
{
    return rand() % max_val;
}

static float rand_float_range(float min_val, float max_val)
{
    float range = max_val - min_val;
    return min_val + ((float)rand() / (float)RAND_MAX) * range;
}

static void test_dataset(const char* name, const char* path, 
                        int nips, int nops, int use_csv, int iterations)
{
    printf("\nTesting %s dataset\n", name);
    printf("Inputs: %d, Outputs: %d\n\n", nips, nops);
    
    const Data full_data = build(path, nips, nops, use_csv);
    
    shuffle(full_data);

    const int train_size = (int)(full_data.rows * 0.8f);
    const Data train = split_data(full_data, 0, train_size);
    const Data test = split_data(full_data, train_size, full_data.rows);
    
    printf("Train samples: %d, Test samples: %d\n\n", train.rows, test.rows);
    
    /* Define hyperparameter choices */
    Normalization norms[] = {NORM_NONE, NORM_MINMAX, NORM_ZSCORE};
    WeightInit inits[] = {INIT_XAVIER, INIT_HE};
    Activation hidden_acts[] = {ACT_TANH, ACT_RELU, ACT_LEAKY_RELU};
    Activation output_acts[] = {ACT_SIGMOID, ACT_TANH};
    
    int norm_count = 3;
    int init_count = 2;
    int hact_count = 3;
    int oact_count = 2;
    
    /* Random search parameters */
    int n_random_trials = 50;
    
    printf("Running Random Search with %d trials\n\n", n_random_trials);
    
    GridResult best_result;
    best_result.accuracy = 0.0f;
    best_result.final_error = FLT_MAX;
    
    for(int trial = 0; trial < n_random_trials; trial++)
    {
        /* Randomly select hyperparameters */
        Normalization norm = norms[rand_choice(norm_count)];
        WeightInit init = inits[rand_choice(init_count)];
        Activation hidden_act = hidden_acts[rand_choice(hact_count)];
        Activation output_act = output_acts[rand_choice(oact_count)];
        
        /* Random continuous values */
        float momentum_val = rand_float_range(0.5f, 0.95f);
        float lr = rand_float_range(0.01f, 0.3f);
        
        /* Random discrete choices */
        int early_stop_enabled = rand_choice(2);
        
        /* Hidden layer size: random between nips/2 and nips*2 */
        int min_hid = nips / 2;
        if(min_hid < 4) min_hid = 4;
        int max_hid = nips * 2;
        int nhid = min_hid + rand_choice(max_hid - min_hid + 1);
        
        GridResult r = run_config(
            train, test,
            norm, init,
            hidden_act, output_act,
            momentum_val, early_stop_enabled,
            lr, nhid, iterations
        );
        
        if((trial + 1) % 5 == 0 || r.accuracy > best_result.accuracy)
            print_result(r, trial + 1);
        
        if(r.accuracy > best_result.accuracy)
            best_result = r;
    }
    
    printf("\n");
    printf("Best configuration found:\n");
    print_result(best_result, -1);
    
    /* Test a few sensible defaults for comparison */
    printf("\nTesting defaults\n");
    
    GridResult default1 = run_config(train, test, NORM_ZSCORE, INIT_XAVIER, 
                                     ACT_TANH, ACT_SIGMOID, 0.9f, 1, 0.1f, nips, iterations);
    printf("Default 1 (ZScore, Xavier, Tanh/Sigmoid, mom=0.9, ES, lr=0.1, hid=nips): ");
    printf("Err=%.6f Acc=%.4f\n", (double)default1.final_error, (double)default1.accuracy);
    
    GridResult default2 = run_config(train, test, NORM_MINMAX, INIT_HE, 
                                     ACT_RELU, ACT_SIGMOID, 0.8f, 1, 0.05f, nips * 2, iterations);
    printf("Default 2 (MinMax, He, ReLU/Sigmoid, mom=0.8, ES, lr=0.05, hid=nips*2): ");
    printf("Err=%.6f Acc=%.4f\n", (double)default2.final_error, (double)default2.accuracy);
    
    dfree(train);
    dfree(test);
    dfree(full_data);
}

int main(void)
{
    srand(time(0));
    printf("Hanyuu Random Search\n");
    test_dataset("Semeion", "semeion.data", 256, 10, 0, 64);
    test_dataset("Wine", "wine.data", 13, 3, 1, 128);
    printf("\nComplete!\n");
    return 0;
}
