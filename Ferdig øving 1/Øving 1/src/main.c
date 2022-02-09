#include <crypt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include "util.h"
// mpirun crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt
const int MAX_STRING = 100;
int comm_sz;
int my_rank;

static bool run_master(int argc, char **argv);
static bool init_master(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv);
static void master_cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict);
static void master_crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry);
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict);
static void replica_crack(Options *options);
static void crack_job(CrackResult *result, CrackJob *job);
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file);
static void handle_overview_result(Options *options, OverviewCrackResult *overview_result);

/*
 * Main entrypoint.
 */

int main(int argc, char **argv) {
    char greeting[MAX_STRING];
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    bool success = run_master(argc, argv);
    MPI_Finalize();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*
 * Entrypoint for master.
 */
static bool run_master(int argc, char **argv) {
    Options options = {};
    Dictionary dict = {};
    FILE *shadow_file = NULL;
    FILE *result_file = NULL;

    bool init_success = init_master(&options, &dict, &shadow_file, &result_file, argc, argv);

    // If init successful, try to crack all shadow entries
    if (!options.quiet && !my_rank) {
        printf("\nEntries:\n");
    }
    OverviewCrackResult overview_result = {};
    if (init_success) {
        ShadowEntry shadow_entry;
        while (get_next_shadow_entry(&shadow_entry, shadow_file)) {
            ExtendedCrackResult result;
            master_crack(&result, &options, &dict, &shadow_entry);
            handle_result(&options, &result, &overview_result, result_file);
        }
        
    }

    // Handle overall result
    if (my_rank==0){
    handle_overview_result(&options, &overview_result);
    master_cleanup(shadow_file, result_file, &dict);
    }
    return true;
}

/*
 * Initialize master stuff.
 */
static bool init_master(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv) {
    // Parse CLI args
    if (!parse_cli_args(options, argc, argv)) {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }
    if (my_rank == 0){
    // Print some useful info
    if (!options->quiet) {
        // TODO
        
        MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
        printf("Workers: %d\n", comm_sz);
        printf("Max symbols: %ld\n", options->max_length);
        printf("Symbol separator: \"%s\"\n", options->separator);
    }
    }
    

    // Open shadow file
    if (!options->quiet) {
        if (!my_rank){
        printf("Shadow file: %s\n", options->shadow_file);
    }
    }
    if (!open_file(shadow_file, options->shadow_file, "r")) {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }
    // Open output file if provided
    if (options->result_file[0] != 0) {
        if (!options->quiet && !my_rank) {
            printf("Output file: %s\n", options->result_file);
        }
        if (!open_file(result_file, options->result_file, "w")) {
            master_cleanup(*shadow_file, *result_file, dict);
            return false;
        }
    }
    // Read full directory
    if (!options->quiet && !my_rank) {
        printf("Dictionary file: %s\n", options->dict_file);
    }
    if (!read_dictionary(dict, options, options->dict_file)) {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }

    return true;
}

/*
 * Cleanup master stuff.
 */
static void master_cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict) {
    if (shadow_file) {
        fclose(shadow_file);
    }
    if (result_file) {
        fclose(result_file);
    }
    if (dict->elements) {
        free(dict->elements);
    }
}

/*
 * Crack a shadow password entry as master.
 */
static void master_crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry) {
    // Initialize result
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    memset(result, 0, sizeof(ExtendedCrackResult));
    strncpy(result->user, entry->user, MAX_SHADOW_USER_LENGTH);
    strncpy(result->passfield, entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    result->alg = entry->alg;

    // Accept only known algs
    if (entry->alg == ALG_UNKNOWN) {
        result->status = STATUS_SKIP;
        return;
    }

    // Setup vars for cracking
    ProbeConfig config = {};
    config.dict_positions = calloc(options->max_length, sizeof(size_t));
    config.symbols = calloc(options->max_length, MAX_DICT_ELEMENT_LENGTH + 1);

    //Allocate a buffer to send jobs to in scatter
    CrackJob *this_job = calloc(1,sizeof(CrackJob));
    CrackResult *this_result = calloc(1, sizeof(CrackJob));

    //Allocate space for jobs and results, with number of cores CrackJobs
    CrackJob *jobs = calloc(comm_sz, sizeof(CrackJob));
    CrackResult *results = calloc(comm_sz, sizeof(CrackResult));

    if (my_rank==0){
        strncpy(jobs[0].passfield, entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    }
    // TODO
    double start_time = (double)MPI_Wtime();

    // Try probes until the status changes (when a match is found or the search space is exhausted)
    while(result->status == STATUS_PENDING) {
        MPI_Barrier(MPI_COMM_WORLD);
        //Broadcast current result to all ranks
        MPI_Bcast(result, sizeof(result), MPI_BYTE, 0, MPI_COMM_WORLD);
        // Make jobs with new probes
        if (my_rank==0){
            for (int i= 0; i<comm_sz; i++){
                //Assign values and different probes for each process
                memset(&jobs[i], 0, sizeof(CrackJob));
                bool more_probes = get_next_probe(&config, options, dict);
                if (!more_probes) {
                    jobs[i].action = ACTION_WAIT;
                    result->status = STATUS_SKIP;
                }else{
                jobs[i].action = ACTION_WORK;
                strncpy(jobs[i].passfield, entry->passfield, MAX_PASSWORD_LENGTH);
                jobs[i].alg = entry->alg;
                jobs[i].salt_end = entry->salt_end;
                strncpy(jobs[i].probe, config.probe, MAX_PASSWORD_LENGTH);
                }
                if (options->verbose) {
                    printf("this is the probe of process %d : %s\n",i, jobs[i].probe);
                }
            }
            
            MPI_Scatter(jobs, sizeof(CrackJob), MPI_BYTE, this_job, sizeof(CrackJob), MPI_BYTE, 0, MPI_COMM_WORLD);
            
        }else{
            MPI_Scatter(jobs, sizeof(CrackJob), MPI_BYTE, this_job, sizeof(CrackJob), MPI_BYTE, 0, MPI_COMM_WORLD);
        }

        // Process jobs 
        crack_job(&this_result[0], &this_job[0]);
        
        //Gather results from all ranks
        if (my_rank==0){
            MPI_Gather(this_result, sizeof(CrackResult), MPI_BYTE,results,sizeof(CrackResult),MPI_BYTE,0,MPI_COMM_WORLD);
            
        }else{
            MPI_Gather(this_result, sizeof(CrackResult), MPI_BYTE,results,sizeof(CrackResult),MPI_BYTE,0,MPI_COMM_WORLD);
        }
        
        
        if (my_rank==0){
        //Count attempts and check for correct passwords
        for (int q = 0; q<comm_sz; q++){
            if (results[q].status == STATUS_SUCCESS) {
                result->status = STATUS_SUCCESS;
                strncpy(result->password, results[q].password, MAX_PASSWORD_LENGTH);
                result->attempts++;
                break;
            }else if(results[q].status == STATUS_PENDING){
                result->attempts++;
                }
        }
        }
        //Broadcast result to all ranks
        MPI_Bcast(result, sizeof(result), MPI_BYTE, 0, MPI_COMM_WORLD);
        //If you don't have a barrier here, you disrupt the end print.
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    double end_time = (double)MPI_Wtime();
    result->duration = end_time - start_time;
    free(config.dict_positions);
    free(config.symbols);
    free(this_result);
    free(this_job);
    free(jobs);
    free(results);
}

/*
 * Build the next probe. Returns false with an empty probe when the search space is exhausted.
 */
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict) {
    // Check if dict is empty
    if (dict->length == 0) {
        return false;
    }

    // Find last symbol which can be replaced with the next one, if any exists
    ssize_t last_replaceable_pos = -1;
    for (size_t i = 0; i < config->size; i++) {
        if (config->dict_positions[i] < dict->length - 1) {
            last_replaceable_pos = i;
        }
    }

    // A symbol can be replaced, replace last one and reset all behind it
    if (last_replaceable_pos >= 0) {
        size_t new_dict_pos = config->dict_positions[last_replaceable_pos] + 1;
        config->dict_positions[last_replaceable_pos] = new_dict_pos;
        strncpy(config->symbols[last_replaceable_pos], dict->elements[new_dict_pos], MAX_DICT_ELEMENT_LENGTH);
        for (size_t i = last_replaceable_pos + 1; i < config->size; i++) {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }
    // No symbols can be replaced and no more symbols are allowed, return error
    else if (config->size == options->max_length) {
        config->probe[0] = 0;
        return false;
    }
    // New symbol can be added, reset all previous positions and add it
    else {
        config->size++;
        for (size_t i = 0; i < config->size; i++) {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }

    // Build probe
    config->probe[0] = 0;
    for (size_t i = 0; i < config->size; i++) {
        if (i > 0) {
            strncat(config->probe, options->separator, MAX_PASSWORD_LENGTH);
        }
        strncat(config->probe, config->symbols[i], MAX_PASSWORD_LENGTH);
    }

    return true;
}

/*
 * Handle result from trying to crack a single password.
 */
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file) {
    // Make representations
    char const *alg_str = cryptalg_to_string(result->alg);
    char const *status_str = crack_result_status_to_string(result->status);
    double attempts_per_second = result->attempts / result->duration;

    // Format and print
    
    size_t const static max_output_length = 1023;
    char *output = malloc(max_output_length + 1);
    if (!my_rank){
    snprintf(output, max_output_length + 1, "user=\"%s\" alg=\"%s\" status=\"%s\" duration=\"%fs\" attempts=\"%ld\" attempts_per_second=\"%f\" password=\"%s\"",
            result->user, alg_str, status_str, result->duration, result->attempts, attempts_per_second, result->password);
    if (!options->quiet) {
        printf("%s\n", output);
    }
    if (result_file) {
        fprintf(result_file, "%s\n", output);
        fflush(result_file);
    }
    }
    free(output);

    // Update overview
    overview_result->statuses[result->status]++;
    overview_result->duration += result->duration;
    overview_result->attempts += result->attempts;
}

/*
 * Handle result from trying to crack all passwords.
 */
static void handle_overview_result(Options *options, OverviewCrackResult *result) {
    if (!options->quiet && !my_rank) {
        printf("\nOverview:\n");
        printf("Total duration: %.3fs\n", result->duration);
        printf("Total attempts: %ld\n", result->attempts);
        printf("Total attempts per second: %.3f\n", result->attempts / result->duration);
        printf("Skipped: %ld\n", result->statuses[STATUS_SKIP]);
        printf("Successful: %ld\n", result->statuses[STATUS_SUCCESS]);
        printf("Failed: %ld\n", result->statuses[STATUS_FAIL]);
    }
}

/*
 * Hash probe and compare.
 */
static void crack_job(CrackResult *result, CrackJob *job) {
    memset(result, 0, sizeof(CrackResult));
    // Only accept known (redundant check)
    if (job->alg == ALG_UNKNOWN) {
        result->status = STATUS_SKIP;
        return;
    }
    if (job->probe != 0){
    char const *new_passfield = crypt(job->probe, job->passfield);
    if (new_passfield != NULL && strncmp(job->passfield, new_passfield, MAX_SHADOW_PASSFIELD_LENGTH) == 0) {
        // Match found, abort search
        result->status = STATUS_SUCCESS;
        strncpy(result->password, job->probe, MAX_PASSWORD_LENGTH);
        result->password[MAX_PASSWORD_LENGTH] = 0;
    }
    }else{
        job->action = ACTION_STOP;
        result->status = STATUS_SKIP;
    }
}
