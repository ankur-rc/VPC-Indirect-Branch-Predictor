// my_predictor.h
// Author: Ankur Roy Chowdhury
// Conditional Predictor: Merging Path & GShare Perceptron + Indirect Predictor: VPC

#include <cmath>
#include <cstddef>
#include <cstring>
#include <bitset>

#define H 6				// Weights per perceptron (excluding bias)
#define HIST_LEN 64		// History length
#define NUM_WTS 4096	// Number of weights per table
#define MASK 0x000003FF // Masking bit for segmenting the ghr
#define MASK_BITS 10	// Number of mask bits set
#define MAX_WEIGHT 127  // Max value of bias/weight
#define MIN_WEIGHT -128 // Min value of bias/weight
#define THETA 25		// floor(1.93*H+14); Perceptron optimum value

#define NUM_TARGETS 32768	 // Size of the BTB
#define MAX_VPC_ITERS 20	  // Max number of VPC iterations
#define NUM_LFU_COUNTERS 1640 // Size of LFU counter array

static const unsigned int VPC_HASH[20] = {
	0xbbc346ad, 0x129f47dd, 0xa63dcb5a, 0x3253f058,
	0xe2186929, 0x891f0f2e, 0x9d891952, 0xbe98e380,
	0x9d2071a3, 0xcf52069f, 0xd304bb44, 0x85f89e7d,
	0x4db71167, 0xa6ac37d6, 0x3f135331, 0xe8737721,
	0x86727eb1, 0xbaa58cc9, 0x4053e7f0, 0x7dc47f0f};

class my_update : public branch_update
{
  public:
	// conditional predictor
	unsigned int weight_index[H + 1];
	int perceptron_output;

	// indirect predictor
	unsigned int predicted_iter;
	bool btb_miss;
	unsigned int iter_weight_indices[MAX_VPC_ITERS][H + 1];
	int iter_perceptron_outputs[MAX_VPC_ITERS];
	bool iter_predicted_directions[MAX_VPC_ITERS];

	my_update(void)
	{
		memset(weight_index, 0, sizeof(weight_index));
		perceptron_output = 0;

		predicted_iter = 0;
		btb_miss = false;
		memset(iter_weight_indices, 0, sizeof(iter_weight_indices));
		memset(iter_perceptron_outputs, 0, sizeof(iter_perceptron_outputs));
		memset(iter_predicted_directions, 0, sizeof(iter_predicted_directions));
	}
};

class my_predictor : public branch_predictor
{
  public:
	my_update u;
	branch_info bi;

	std::bitset<HIST_LEN> history;		// global history register
	std::bitset<HIST_LEN> path;			// path register
	char weight_tables[H + 1][NUM_WTS]; // perceptron weight matrix

	unsigned int targets[NUM_TARGETS];						// BTB
	unsigned char lfu_ctr[NUM_LFU_COUNTERS][MAX_VPC_ITERS]; // LFU Counter matrix

	my_predictor(void)
	{
		memset(weight_tables, 0, sizeof(weight_tables));
		memset(targets, 0, sizeof(targets));
		memset(lfu_ctr, 0, sizeof(lfu_ctr));
	}

	branch_update *predict(branch_info &b)
	{
		bi = b;
		u = my_update(); //reinit temp variables

		if (b.br_flags & BR_CONDITIONAL) // For conditional branches
		{
			bool taken = predict_direction(b.address, history, path, u.perceptron_output);
			u.direction_prediction(taken);
		}
		else
		{
			u.direction_prediction(true);
		}

		if (b.br_flags & BR_INDIRECT) // For indirect branches
		{
			// Initialize vpca, vghr, vpath and predicted_target
			unsigned int vpca = bi.address;
			std::bitset<HIST_LEN> vghr = history;
			std::bitset<HIST_LEN> vpath = path;
			unsigned int predicted_target = 0;

			vpath = (vpath << 4).to_ulong() | (vpca & 0xF); // set virtual path
			vpca = bi.address ^ VPC_HASH[0];				// hash next virtual pc
			vghr = vghr << 1;								// last virtual branch not taken

			int iter = 0;
			unsigned int target = 0;
			bool done = false;
			while (done == false)
			{
				target = targets[vpca % NUM_TARGETS];
				int perceptron_output = 0;
				bool predicted_direction = predict_direction(vpca, vghr, vpath, perceptron_output);
				u.iter_predicted_directions[iter] = predicted_direction;
				u.iter_perceptron_outputs[iter] = perceptron_output;
				for (int i = 0; i < H + 1; i++)
					u.iter_weight_indices[iter][i] = u.weight_index[i];

				// case 1: A hit!
				if ((target != 0) && (predicted_direction == true))
				{
					predicted_target = target; // store the target
					u.predicted_iter = iter;   // store predicted iteration
					done = true;
				}
				//case 2 : A miss!
				else if ((target == 0) || (iter >= MAX_VPC_ITERS - 1))
				{
					u.btb_miss = true;
					u.predicted_iter = iter; // store predicted iteration
					done = true;
				}
				//case 3: Predicted as not taken!
				vpath = (vpath << 4).to_ulong() | (vpca & 0xF); // set virtual path
				vpca = bi.address ^ VPC_HASH[iter];				// hash next virtual pc
				vghr = vghr << 1;								// last virtual branch not taken
				iter++;
			}

			u.target_prediction(predicted_target);
			printf("\npredicted  iter: %d\n", u.predicted_iter);
		}

		return &u;
	}

	/* Direction prediction Algorithm
	*/
	bool predict_direction(const unsigned int &address, const std::bitset<HIST_LEN> &history, const std::bitset<HIST_LEN> &path, int &perceptron_op)
	{

		bool taken = false;

		u.weight_index[0] = ((address) % (NUM_WTS));			   // Bias is obtained by address
																   // lower order bits
		u.perceptron_output = weight_tables[0][u.weight_index[0]]; // Add bias to perceptron output

		unsigned int segment;			// Each segment = History length/Masking bit length
		for (int i = 1; i < H + 1; i++) // Get the weights of the perceptron
		{
			segment = ((history ^ path).to_ulong() & (MASK << (i - 1) * MASK_BITS)) >> (i - 1) * MASK_BITS; // segment is the hash of history and path

			u.weight_index[i] = ((segment) ^ (address)) % (NUM_WTS); // weight is obtained by the hash of each segment and the address
			u.perceptron_output += weight_tables[i][u.weight_index[i]];
		}

		if (u.perceptron_output >= 0)
		{
			taken = true; // Predict true if perceptron output is greater than 0
		}
		else
		{
			taken = false; // else predict false
		}

		perceptron_op = u.perceptron_output;

		return taken;
	}

	void update(branch_update *u, bool taken, unsigned int target)
	{
		if (bi.br_flags & BR_CONDITIONAL) // for conditional branches
		{
			train_predictor(u->direction_prediction(), taken, ((my_update *)u)->weight_index, ((my_update *)u)->perceptron_output);

			history <<= 1;
			history |= taken;

			path = path << 4;
			path = path.to_ulong() | (bi.address & 0xF);
		}

		if (bi.br_flags & BR_INDIRECT)
		{

			my_update *mu = (my_update *)u;

			// case 1: when prediction is correct
			if (target == u->target_prediction())
			{
				unsigned int iter = 0;
				while (iter <= mu->predicted_iter)
				{
					unsigned int weight_indices[H + 1] = {0}; // store the weight indices
					for (int i = 0; i < H + 1; i++)			  // of current iter in temp array
					{
						weight_indices[i] = mu->iter_weight_indices[iter][i];
					}

					if (iter == mu->predicted_iter)
					{
						train_predictor(mu->iter_predicted_directions[iter], true, weight_indices, mu->iter_perceptron_outputs[iter]); // train bp on taken
						char lfu_val = lfu_ctr[bi.address % NUM_LFU_COUNTERS][iter];
						lfu_ctr[bi.address % NUM_LFU_COUNTERS][iter] = ((lfu_val < 127) ? lfu_val++ : 127); // update replacement policy bit
					}
					else
					{
						train_predictor(mu->iter_predicted_directions[iter], false, weight_indices, mu->iter_perceptron_outputs[iter]); // train bp on not taken
					}

					iter++;
				}
			}
			// case 2: when prediction is incorrect
			else
			{
				// Initialize vpca
				unsigned int vpca = bi.address;
				bool found_correct_target = false;

				int iter = 0;
				while ((iter < MAX_VPC_ITERS) && (found_correct_target == false))
				{
					unsigned int weight_indices[H + 1] = {0}; // store the weight indices
					for (int i = 0; i < H + 1; i++)			  // of current iter in temp array
					{
						weight_indices[i] = mu->iter_weight_indices[iter][i];
					}

					unsigned int predicted_target = targets[vpca % NUM_TARGETS];
					if (predicted_target == target)
					{
						train_predictor(false, true, weight_indices, mu->iter_perceptron_outputs[iter]); // train bp on taken
						char lfu_val = lfu_ctr[bi.address % NUM_LFU_COUNTERS][iter];
						lfu_ctr[bi.address % NUM_LFU_COUNTERS][iter] = ((lfu_val < 127) ? lfu_val++ : 127); // update replacement policy bit
						found_correct_target = true;
					}
					else if (predicted_target && iter <= mu->predicted_iter)
					{
						train_predictor(mu->iter_predicted_directions[iter], false, weight_indices, mu->iter_perceptron_outputs[iter]); // train bp on not taken
					}
					vpca = bi.address ^ VPC_HASH[iter];

					iter++;
				}

				if (!found_correct_target)
				{
					int iter = 0;
					if (mu->btb_miss) //get the iter from predicted iter if there's a btb miss
						iter = mu->predicted_iter;
					else // get the iter from least frequently used value
					{
						int minIdx = 0;
						for (int i = 0; i < MAX_VPC_ITERS; i++)
						{
							if (lfu_ctr[bi.address % NUM_LFU_COUNTERS][i] < lfu_ctr[bi.address % NUM_LFU_COUNTERS][minIdx])
								minIdx = i;
						}
						iter = minIdx;
					}

					unsigned int vpca = bi.address ^ VPC_HASH[iter];
					targets[vpca % NUM_TARGETS] = target;
					lfu_ctr[bi.address % NUM_LFU_COUNTERS][iter] = 1;

					unsigned int weight_indices[H + 1] = {0}; // store the weight indices
					for (int i = 0; i < H + 1; i++)			  // of current iter in temp array
					{
						weight_indices[i] = mu->iter_weight_indices[iter][i];
					}
					train_predictor(mu->iter_predicted_directions[iter], true, weight_indices, mu->iter_perceptron_outputs[iter]);
				}
			}
		}
	}

	/* Training algorithm
	*/
	void train_predictor(const bool &predicted_taken, const bool &taken,
						 const unsigned int weight_indices[], const int &perceptron_output)
	{
		bool direction_prediction = predicted_taken;
		int prediction_output = perceptron_output;

		if ((direction_prediction != taken) || (abs(prediction_output) <= THETA))
		{
			for (int i = 0; i < H + 1; i++)
			{
				unsigned int weight_index = weight_indices[i];
				char *c = &weight_tables[i][weight_index];

				if (taken)
				{
					*c < MAX_WEIGHT ? (*c)++ : MAX_WEIGHT;
				}
				else
				{
					*c > MIN_WEIGHT ? (*c)-- : MIN_WEIGHT;
				}
			}
		}
	}
};
