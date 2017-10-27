// my_predictor.h
// This file contains a sample my_predictor class.
// Global Perceptron Predictor

#include <cmath>
#include <cstddef>
#include <cstring>

#define H 59		 //History length or weights per perceptron
#define NUM_WTS 1024 //Number of weights per table
#define TARGET_BITS 15
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

class my_update : public branch_update
{
  public:
	unsigned int weight_index;
	int perceptron_output;

	my_update(void)
	{
		weight_index = 0;
		perceptron_output = 0;
	}
};

class my_predictor : public branch_predictor
{
  public:
	static const unsigned int THETA = 127; // 1.93*H+14
	my_update u;
	branch_info bi;

	unsigned int history;

	char weight_tables[H + 1][NUM_WTS];
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
			unsigned int history_lob = history % NUM_WTS;
			unsigned int address_lob = b.address % NUM_WTS;
			u.weight_index = history_lob ^ address_lob;

			u.perceptron_output = weight_tables[0][u.weight_index];

			for (int i = 1; i < H + 1; i++)
			{
				unsigned int mask = 1 << (i - 1);
				if (history & mask) // if history bit is 1
					u.perceptron_output += weight_tables[i][u.weight_index];
				else // if history bit is 0 treat as -1 (bipolar conversion)
					u.perceptron_output -= weight_tables[i][u.weight_index];
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
			bool direction_prediction = ((my_update *)u)->direction_prediction();
			int prediction_output = ((my_update *)u)->perceptron_output;

			if ((direction_prediction != taken) || (abs(prediction_output) <= THETA))
			{
				// update the bias
				char *bias = &weight_tables[0][((my_update *)u)->weight_index];
				if (direction_prediction == true)
				{
					if (*bias < MAX_WEIGHT)
						(*bias)++;
				}
				else
				{
					if (*bias > MIN_WEIGHT)
						(*bias)--;
				}

				// update the weights
				for (int i = 1; i < H + 1; i++)
				{
					bool history_bit = history & (1 << (i - 1));
					char *weight = &weight_tables[i][((my_update *)u)->weight_index];

					if (history_bit == taken)
					{
						if (*weight < MAX_WEIGHT)
							(*weight)++;
					}
					else
					{
						if (*weight > MIN_WEIGHT)
							(*weight)--;
					}
				}
			}

			history <<= 1;
			history |= taken;
		}

		if (bi.br_flags & BR_INDIRECT)
		{
			targets[bi.address & ((1 << TARGET_BITS) - 1)] = target;
		}
	}
};
