/*
 * blbn.h
 *
 *  Created on: Sep 2, 2010
 *      Author: mokogobo
 */

// Functions are named roughly according to the following, but not always:
//
// blbn_<verb1>_<adjective1>_<noun1>_[not]_<adjective2>_<position>

#ifndef BLBN_H_
#define BLBN_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h>
#include "../netica/Netica.h"
#include "../netica/NeticaEx.h"

// Indicates whether or not to print output to stdout
#define BLBN_STDOUT 0

#define LICENSE_STRING "INSERT/LICENSE/STRING/HERE"

#define BLBN_METADATA_FLAG_TARGET    0x01
#define BLBN_METADATA_FLAG_PURCHASED 0x02
#define BLBN_METADATA_FLAG_LEARNED   0x04

#define BLBN_POLICY_ROUND_ROBIN  0 // Round Robin
#define BLBN_POLICY_BIASED_ROBIN 1 // Biased Robin
#define BLBN_POLICY_SFL          2 // Single-Feature Lookahead
#define BLBN_POLICY_GSFL         20 // Generalized Single-Feature Lookahead (for Bayesian networks)
#define BLBN_POLICY_RSFL         3 // Randomized Single-Feature Lookahead
#define BLBN_POLICY_GRSFL        30 // Generalized Randomized Single-Feature Lookahead (for Bayesian networks)
#define BLBN_POLICY_EMPG         4 // "Tell Me What I Want to Hear" // TODO: Rename this algorithm
#define BLBN_POLICY_CHEATING     5 // Cheating algorithm
// TODO: Exploration vs. expectation

// The global Netica environment structure
environ_ns* env;

FILE *graph_fp;
FILE *log_fp;

typedef struct blbn_select_action {
	unsigned int node_index; // j; // node (column)
	unsigned int case_index; // i; // case (row)

	struct blbn_select_action *prev; // previous selection
	struct blbn_select_action *next; // next selection

} blbn_select_action_t;

typedef struct blbn_state {
	unsigned int node_count; // number of nodes columns (i.e., variable n in a matrix)
	unsigned int case_count; // number of cases rows (i.e., variable m in a matrix)

	char **nodes; // Character string array of node names (this is the static ordering used in BLBN library)
	int **state; // 2D array of states
	unsigned int **cost; // 2D array of costs for each (node, case) pair

	unsigned int budget; // Current budget

	int target; // Target node index

	unsigned int **flags; // 2D array of status flags (used to indicate whether or not a node is (1) purchased and (2) learned)

	blbn_select_action_t *sel_action_seq; // Pointer to head of linked list of select actions (in order of selection)

	double last_log_loss;
	double curr_log_loss;

	// TODO: <experiment name> (used for output directory)

	// Netica-related data structures (encapsulated by this data structure)
	environ_ns *env; // Netica environment
	net_bn *orig_net; // Original network with parameters read from the DNE or NETA file (not modified during execution of program)
	net_bn *prior_net; // Network parameterized according to specified prior distribution (e.g., uniform, noisy).  The structure of this network is identical to that of the original network.
	net_bn *work_net; // Network parameterized using available data (i.e., target values and purchased non-target node values) selected by a selection policy (e.g., round robin, biased robin, etc.)
	nodelist_bn *nodelist;

	caseset_cs* validation_caseset;

} blbn_state_t;

// Function prototypes
int blbn_init ();
blbn_state_t* blbn_init_state (char *experiment_name, char *data_filepath, char *validation_data_filepath, char *model_filepath, char *target_node_name, unsigned int budget, char *output_folder, int k, int f);
void blbn_free_state (blbn_state_t *state);

char* blbn_get_node_name (blbn_state_t *state, unsigned int node_index);

void blbn_learn_all_v0 (stream_ns *casefile, net_bn *net, nodelist_bn *nodes, caseposn_bn *case_posn);
void blbn_learn_baseline (blbn_state_t *state);
void blbn_learn (blbn_state_t *state, int policy);

void error (environ_ns* env);

int blbn_count_findings_in_node_not_purchased (blbn_state_t *state, int node_index);
int blbn_count_findings_in_case_not_purchased (blbn_state_t *state, int case_index);
int blbn_count_actions (blbn_state_t *state);
void blbn_learn_targets (blbn_state_t *state, double ess);
void blbn_revise_by_case_findings_v1 (blbn_state_t *state, int case_index);
void blbn_learn_baseline (blbn_state_t *state);
void blbn_learn_case_v1 (blbn_state_t *state, int case_index);
void blbn_learn_case_v2 (blbn_state_t *state, int case_index);
void blbn_unlearn_case_v1 (blbn_state_t *state, int case_index);
void blbn_unlearn_case_v2 (blbn_state_t *state, int case_index);
void blbn_retract_findings (blbn_state_t *state);
void blbn_retract_findings_not_target (blbn_state_t *state);
void blbn_set_net_findings_available (blbn_state_t *state, int case_index);
void blbn_set_net_findings_learned (blbn_state_t *state, int case_index);
void blbn_set_net_findings (blbn_state_t *state, int case_index);
void blbn_set_node_finding_if_available (blbn_state_t *state, int node_index, int case_index);
void blbn_set_finding_not_learned (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_learned (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_not_purchased (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_purchased (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_not_target (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_target (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_prior_belief_state (blbn_state_t *state);
void blbn_set_uniform_prior (blbn_state_t *state, double experience);
blbn_select_action_t* blbn_get_action_head (blbn_state_t *state);
blbn_select_action_t* blbn_get_action (blbn_state_t *state, unsigned int index);
blbn_select_action_t* blbn_get_action_tail (blbn_state_t *state);
int blbn_get_random_finding_not_purchased_in_node (blbn_state_t *state, int node_index);
int blbn_get_random_finding_not_purchased_in_node_with_label (blbn_state_t *mdata, int node_index, int target_state);
char* blbn_get_node_name (blbn_state_t *state, unsigned int node_index);
int blbn_get_node_by_name (blbn_state_t *state, char *name);
int blbn_get_node_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
int blbn_get_node_index (blbn_state_t *state, char* node_name);
double blbn_get_error_rate (blbn_state_t *state);
double blbn_get_log_loss (blbn_state_t *state);
int blbn_get_minimum_cost (blbn_state_t *state);
int blbn_get_minimum_cost_in_node (blbn_state_t *state, unsigned int node_index);
int blbn_get_minimum_cost_in_case (blbn_state_t *state, unsigned int case_index);
int blbn_get_findings_not_purchased_for_node (blbn_state_t *state, int node_index, int **result);
int blbn_get_findings_not_purchased_in_case (blbn_state_t *state, int case_index, int **result);
int blbn_has_finding_set (blbn_state_t *state, unsigned node_index);
char blbn_has_findings_available (blbn_state_t *state);
char blbn_has_findings_available_not_learned (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_not_available (blbn_state_t *state);
char blbn_has_findings_not_learned (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_learned (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_purchased (blbn_state_t *state);
char blbn_has_findings_not_purchased (blbn_state_t *state);
char blbn_has_findings_purchased_in_case (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_not_purchased_in_case (blbn_state_t *state, unsigned int case_index);
char blbn_has_cases_not_learned (blbn_state_t *state, unsigned int node_index);
char blbn_has_cases_learned (blbn_state_t *state, unsigned int node_index);
char blbn_has_cases_not_purchased (blbn_state_t *state, unsigned int node_index);
char blbn_has_cases_purchased (blbn_state_t *state, unsigned int node_index);
char blbn_has_findings_available_in_case (blbn_state_t *state, unsigned int case_index);
char blbn_has_cases_available (blbn_state_t *state, unsigned int node_index);
char blbn_has_parents_with_findings (blbn_state_t *state, int node_index, int case_index);
char blbn_is_learned_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_available_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_purchased_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_target_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_valid_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_valid_case (blbn_state_t *state, unsigned int case_index);
char blbn_is_valid_node (blbn_state_t *state, unsigned int node_index);
void blbn_restore_prior_network (blbn_state_t *state);
int blbn_is_target_node (blbn_state_t *state, unsigned int node_index);
int blbn_is_non_target_node (blbn_state_t *state, unsigned int node_index);
void blbn_assert_node_finding (blbn_state_t *state, int node_index, int state_index);
void blbn_assert_node_finding_for_case (blbn_state_t *state, int node_index, int case_index, int state_index);

int blbn_get_d_separated_nodes (blbn_state_t *state, unsigned int node_index, int **d_separated_node_indices);
int blbn_get_d_separated_nodes_and_separating_nodes (blbn_state_t *state, unsigned int node_index, int **d_separated_node_indices);

blbn_select_action_t* blbn_select_next_rr       (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_br       (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_sfl      (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_gsfl     (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_rsfl     (blbn_state_t *state, int K, double tao);
blbn_select_action_t* blbn_select_next_grsfl    (blbn_state_t *state, int K, double tao);
blbn_select_action_t* blbn_select_next_empg     (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_cheating (blbn_state_t *state);

double** blbn_util_sfl     (blbn_state_t *state);
double*  blbn_util_sfl_row (blbn_state_t *state, int case_index);
double** blbn_util_empg    (blbn_state_t *state);
double** blbn_util_cheat   (blbn_state_t *state);

#endif /* BLBN_H_ */
