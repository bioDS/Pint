#include "../../src/liblasso.h"
#include <gsl/gsl_vector.h>
#include <stdlib.h>

enum Output_Mode { quit,
    file,
    terminal };

int main(int argc, char** argv)
{
    if (argc != 11) {
        fprintf(stderr, "usage: ./lasso_exe X.csv Y.csv [depth] verbose=T/F [max lambda] N P [max nz] [q/t/filename] [log_level [i]ter/[l]ambda/[n]one]\n");
        printf("actual args(%d): '", argc);
        for (long i = 0; i < argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        return 1;
    }

    int depth = atoi(argv[4]);
    printf("using depth: %d (%s)\n", depth, argv[4]);
    if (depth < 1 || depth > 3) {
      printf("depth must be between 1 and 3 inclusive.\n");
      exit(EXIT_FAILURE);
    }
    char* verbose = argv[5];
    printf("verbose: %s\n", verbose);
    char* output_filename = argv[10];
    FILE* output_file = NULL;

    enum Output_Mode output_mode = terminal;
    if (strcmp(output_filename, "t") == 0)
        ;
    else if (strcmp(output_filename, "q") == 0)
        output_mode = quit;
    else {
        output_mode = file;
        output_file = fopen(output_filename, "w");
        if (output_file == NULL) {
            perror("opening output file failed");
        }
    }

    VERBOSE = 0;
    if (strcmp(verbose, "T") == 0) {
        printf("verbose\n");
        VERBOSE = 1;
    }

    float lambda;

    if ((lambda = strtod(argv[5], NULL)) == 0)
        lambda = 3.604;
    printf("using lambda = %f\n", lambda);

    long N = atoi(argv[6]);
    long P = atoi(argv[7]);
    printf("using N = %ld, P = %ld\n", N, P);

    long max_nz = atoi(argv[8]);
    printf("using max nz beta: %ld\n", max_nz);

    enum LOG_LEVEL log_level = NONE;
    if (strcmp(argv[10], "i") == 0) {
        log_level = ITER;
    } else if (strcmp(argv[10], "l") == 0) {
        log_level = LAMBDA;
    } else if (strcmp(argv[10], "n") != 0) {
        printf("using 'log_level = NONE', no valid argument given");
    }

    initialise_static_resources();

    // testing: wip
    XMatrix xmatrix = read_x_csv(argv[1], N, P);
    float* Y = read_y_csv(argv[2], N);

    long** X2;
    long nbeta;
    nbeta = xmatrix.actual_cols;
    X2 = xmatrix.X;
    printf("using nbeta = %ld\n", nbeta);

    if (xmatrix.X == NULL) {
        fprintf(stderr, "failed to read X\n");
        return 1;
    }
    if (Y == NULL) {
        fprintf(stderr, "failed to read Y\n");
        return 1;
    }

    printf("begginning coordinate descent\n");
    const char* log_file = "exe.log";
    auto beta_sets = simple_coordinate_descent_lasso(xmatrix, Y, N, nbeta, -1,
        5000, lambda, 300, VERBOSE, -1, 1.0001, log_level, argv, argc, FALSE, max_nz, log_file, depth);
    int nbeta_int = nbeta;
    auto beta = beta_sets.beta3;
    //nbeta_int = get_p_int(nbeta, max_interaction_distance);
    //if (beta == NULL) {
    //	fprintf(stderr, "failed to estimate beta values\n");
    //	return 1;
    //}
    //for (long i = 0; i < nbeta_int; i++) {
    //	printf("%f ", beta[i]);
    //}
    //printf("\n");

    printf("indices non-zero (|x| != 0):\n");
    long printed = 0;
    long sig_beta_count = 0;
    //TODO: remove hack to avoid printing too much for the terminal

    printf("\n\n");

    for (long i = 0; i < xmatrix.actual_cols; i++)
        free(xmatrix.X[i]);
    free(xmatrix.X);
    free(Y);
    printf("freeing X/Y\n");
    switch (output_mode) {
    case terminal:
        printf("main:\n");
        for (auto it = beta_sets.beta1.begin(); it != beta_sets.beta1.end(); it++) {
            long val = it->first;
            float coef = it->second;
            printf("%ld: %f\n", val, coef);
        }
        printf("int:\n");
        for (auto it = beta_sets.beta2.begin(); it != beta_sets.beta2.end(); it++) {
            long val = it->first;
            auto ij = val_to_pair(val, nbeta);
            float coef = it->second;
            printf("%ld,%ld: %f\n", std::get<0>(ij), std::get<1>(ij), coef);
        }
        printf("trip:\n");
        for (auto it = beta_sets.beta3.begin(); it != beta_sets.beta3.end(); it++) {
            long val = it->first;
            auto abc = val_to_triplet(val, nbeta);
            float coef = it->second;
            printf("%ld,%ld,%ld: %f\n", std::get<0>(abc), std::get<1>(abc), std::get<2>(abc), coef);
        }
        //for (long i = 0; i < nbeta_int && printed < 100; i++) {
        //    if (fabs((beta)[i]) > 0) {
        //        printed++;
        //        sig_beta_count++;
        //        int_pair ip = get_num(i, nbeta);
        //        if (ip.i == ip.j)
        //            printf("main: %ld (%ld):     %f\n", i, ip.i + 1, (beta)[i]);
        //        else
        //            printf("int: %ld  (%ld, %ld): %f\n", i, ip.i + 1, ip.j + 1, (beta)[i]);
        //    }
        //}
        break;
    case file:
        for (long i = 0; i < nbeta_int; i++) {
            if (beta[i] != 0.0) {
                printed++;
                sig_beta_count++;
                int_pair ip = get_num(i, nbeta);
                if (ip.i == ip.j)
                    fprintf(output_file, "main: %ld (%ld):     %f\n", i, ip.i + 1, (beta)[i]);
                else
                    fprintf(output_file, "int: %ld  (%ld, %ld): %f\n", i, ip.i + 1, ip.j + 1, (beta)[i]);
            }
        }
        fclose(output_file);
        break;
    case quit:
        break;
    }
    if (output_mode == terminal) {
    }
    //endwin();
    // free(beta);
    beta.clear();
    free_static_resources();
    return 0;
}
