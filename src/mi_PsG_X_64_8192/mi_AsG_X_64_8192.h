// my_predictor.h
// This file contains a sample my_predictor class.
// Merging Gshare and Address indexing in perceptron predictor

#include <cmath>
#include <cstddef>
#include <cstring>

#define H 64		  //Number of weight tables or pipeline stages
#define NUM_WTS 8192  //Number of weights per table
#define HIST_PER_WT 2 //Number of history bits per weight
#define TARGET_BITS 15
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

class my_update : public branch_update
{
  public:
	unsigned int weight_index[H];
	int perceptron_output;

	my_update(void)
	{
		memset(weight_index, 0, sizeof(weight_index));
		perceptron_output = 0;
	}
};

class my_predictor : public branch_predictor
{
  public:
	static const unsigned int theta = 1.93 * H + 14;
	my_update u;
	branch_info bi;

	unsigned int history;

	char weight_tables[H][NUM_WTS];
	unsigned int targets[1 << TARGET_BITS];

	my_predictor(void) : history(0)
	{
		memset(weight_tables, 0, sizeof(weight_tables));
		memset(targets, 0, sizeof(targets));
	}

	branch_update *predict(branch_info &b)
	{
		bi = b;
		if (b.br_flags & BR_CONDITIONAL)
		{
			u.weight_index[0] = ((b.address) % (NUM_WTS));
			u.perceptron_output = weight_tables[0][u.weight_index[0]];

			for (int i = 1; i < H; i++)
			{
				u.weight_index[i] = ((history >> i * HIST_PER_WT) ^ b.address) % NUM_WTS;
				u.perceptron_output += weight_tables[i][u.weight_index[i]];
			}

			if (u.perceptron_output >= 0)
			{
				u.direction_prediction(true);
			}
			else
			{
				u.direction_prediction(false);
			}
		}
		else
		{
			u.direction_prediction(true);
		}
		if (b.br_flags & BR_INDIRECT)
		{
			u.target_prediction(targets[b.address & ((1 << TARGET_BITS) - 1)]);
		}
		return &u;
	}

	void update(branch_update *u, bool taken, unsigned int target)
	{
		if (bi.br_flags & BR_CONDITIONAL)
		{
			for (int i = 0; i < H; i++)
			{
				char *c = &weight_tables[i][((my_update *)u)->weight_index[i]];
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

		history <<= 1;
		history |= taken;

		if (bi.br_flags & BR_INDIRECT)
		{
			targets[bi.address & ((1 << TARGET_BITS) - 1)] = target;
		}
	}
};
