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

#define NUM_TARGETS 65536	 // Size of the BTB
#define MAX_VPC_ITERS 20	  // Max number of VPC iterations
#define NUM_LFU_COUNTERS 1024 // Size of LFU counter array

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

	my_update(void)
	{
		memset(weight_index, 0, sizeof(weight_index));
		perceptron_output = 0;

		predicted_iter = 0;
		btb_miss = false;
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

	unsigned int targets[NUM_TARGETS];			   // BTB
	char lfu_ctr[NUM_LFU_COUNTERS][MAX_VPC_ITERS]; // LFU Counter matrix

	my_predictor(void)
	{
		memset(weight_tables, 0, sizeof(weight_tables));
		memset(targets, 0, sizeof(targets));
		memset(lfu_ctr, 0, sizeof(lfu_ctr));
	}

	branch_update *predict(branch_info &b)
	{
		bi = b;

		if (b.br_flags & BR_CONDITIONAL) // For conditional branches
		{
			bool taken = predict_direction(b.address, history, path);
			u.direction_prediction(taken);
		}
		else
		{
			u.direction_prediction(true);
		}

		if (b.br_flags & BR_INDIRECT) // For indirect branches
		{
			// Initialize vpca, vghr, vpath and predicted_target
			unsigned int vpca = b.address;
			std::bitset<HIST_LEN> vghr = history;
			std::bitset<HIST_LEN> vpath = path;
			unsigned int predicted_target = 0;

			int iter = 0;

			while (iter < MAX_VPC_ITERS)
			{
				predicted_target = targets[vpca % NUM_TARGETS];
				bool predicted_direction = predict_direction(vpca, vghr, vpath);

				// case 1: A hit!
				if ((predicted_target != 0) && (predicted_direction == true))
				{
					predicted_target = targets[vpca % NUM_TARGETS]; // rewriting for clarity
					u.predicted_iter = iter;						// store predicted iteration
					break;
				}
				//case 2 : A miss!
				else if ((predicted_target == 0) || (iter >= MAX_VPC_ITERS))
				{
					u.btb_miss = true;
					break;
				}
				//case 3: Predicted not taken!
				vpca = b.address ^ VPC_HASH[iter];				// hash next virtual pc
				vghr = vghr << 1;								// last virtual branch not taken
				vpath = (vpath << 4).to_ulong() | (vpca & 0xF); // set virtual path
				iter++;
			}

			u.target_prediction(predicted_target);
		}

		return &u;
	}

	/* Direction prediction Algorithm
	*/
	bool predict_direction(unsigned int &address, std::bitset<HIST_LEN> &history, std::bitset<HIST_LEN> &path)
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

		return taken;
	}

	void update(branch_update *u, bool taken, unsigned int target)
	{
		if (bi.br_flags & BR_CONDITIONAL) // for conditional branches
		{
			train_predictor(u, taken, target);

			history <<= 1;
			history |= taken;

			path = path << 4;
			path = path.to_ulong() | (bi.address & 0xF);
		}

		if (bi.br_flags & BR_INDIRECT)
		{
			targets[bi.address & (NUM_TARGETS - 1)] = target;
		}
	}

	/* Training algorithm
	*/
	void train_predictor(branch_update *u, bool taken, unsigned int target)
	{
		bool direction_prediction = ((my_update *)u)->direction_prediction();
		int prediction_output = ((my_update *)u)->perceptron_output;

		if ((direction_prediction != taken) || (abs(prediction_output) <= THETA))
		{
			for (int i = 0; i < H + 1; i++)
			{
				unsigned int weight_index = ((my_update *)u)->weight_index[i];
				char *c = &weight_tables[i][weight_index];

				if (taken)
				{
					if (*c < MAX_WEIGHT)
						(*c)++;
				}
				else
				{
					if (*c > MIN_WEIGHT)
						(*c)--;
				}
			}
		}
	}
};
