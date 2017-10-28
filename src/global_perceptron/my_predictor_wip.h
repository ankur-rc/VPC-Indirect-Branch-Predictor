// my_predictor.h
// Author: Ankur Roy Chowdhury
// Conditional Predictor: Hashed Perceptron + Indirect Predictor: VPC

#include <cmath>
#include <cstddef>
#include <cstring>
#include <bitset>

#define H 59			// History length or weights per perceptron
#define NUM_WTS 1024	// Number of weights per table
#define MAX_WEIGHT 127  // Max value of bias/weight
#define MIN_WEIGHT -128 // Min value of bias/weight
#define THETA 127		// floor(1.93*H+14); Perceptron optimum value

#define NUM_TARGETS 32768	 // Size of the BTB
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
	unsigned int weight_index;
	int perceptron_output;

	// indirect predictor
	unsigned int predicted_iter;
	bool btb_miss;

	my_update(void)
	{
		weight_index = 0;
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

	std::bitset<H> history;
	char weight_tables[H + 1][NUM_WTS];

	unsigned int targets[NUM_TARGETS];
	unsigned char lfu_ctr[NUM_LFU_COUNTERS][MAX_VPC_ITERS];

	my_predictor(void) : history(0)
	{
		memset(weight_tables, 0, sizeof(weight_tables));
		memset(targets, 0, sizeof(targets));
		memset(lfu_ctr, 0, sizeof(lfu_ctr));
	}

	/* Prediction Algorithm
	*/
	branch_update *predict(branch_info &b)
	{
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) // For conditional branches
		{
			bool taken = predict_direction(b);
			u.direction_prediction(taken);
		}
		else
		{
			u.direction_prediction(true);
		}

		if (b.br_flags & BR_INDIRECT) // For indirect branches
		{
			unsigned int vpca = b.address;
			std::bitset<H> vghr = history;
			int iter = 0;

			// while (iter < MAX_VPC_ITERS)
			// {
			// }
			u.target_prediction(targets[b.address & (NUM_TARGETS - 1)]);
		}

		return &u;
	}

	bool predict_direction(branch_info &b)
	{

		bool taken = false;
		u.weight_index = (history.to_ulong() ^ b.address) % NUM_WTS; // Hash global history with address
																	 // and take lower order bits

		u.perceptron_output = weight_tables[0][u.weight_index]; // Add bias to perceptron output

		for (int i = 1; i < H + 1; i++) // Get the weights of the perceptron
		{
			unsigned int mask = 1 << (i - 1); // add or subtract based on history bit
			if (history.to_ulong() & mask)	// if history bit is 1
				u.perceptron_output += weight_tables[i][u.weight_index];
			else // if history bit is 0 treat as -1 (bipolar conversion)
				u.perceptron_output -= weight_tables[i][u.weight_index];
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

	/* Training algorithm
	*/
	void update(branch_update *u, bool taken, unsigned int target)
	{
		if (bi.br_flags & BR_CONDITIONAL) // for conditional branches
		{
			train_predictor(u, taken, target);
			history <<= 1;
			history |= taken;
		}

		if (bi.br_flags & BR_INDIRECT)
		{
			targets[bi.address & (NUM_TARGETS - 1)] = target;
		}
	}

	void train_predictor(branch_update *u, bool taken, unsigned int target)
	{
		bool direction_prediction = ((my_update *)u)->direction_prediction();
		int prediction_output = ((my_update *)u)->perceptron_output;

		if ((direction_prediction != taken) || (abs(prediction_output) <= THETA))
		{
			// Update the bias
			char *bias = &weight_tables[0][((my_update *)u)->weight_index];
			if (direction_prediction == true) // increment the bias if prediction is true
			{
				if (*bias < MAX_WEIGHT)
					(*bias)++;
			}
			else // decrement the bias if prediction is false
			{
				if (*bias > MIN_WEIGHT)
					(*bias)--;
			}

			// Update the weights
			for (int i = 1; i < H + 1; i++)
			{
				bool history_bit = history.to_ulong() & (1 << (i - 1));
				char *weight = &weight_tables[i][((my_update *)u)->weight_index];

				if (history_bit == taken) // increment the weight if history_bit matches the prediction
				{
					if (*weight < MAX_WEIGHT)
						(*weight)++;
				}
				else // else decrement the weight
				{
					if (*weight > MIN_WEIGHT)
						(*weight)--;
				}
			}
		}
	}
};
