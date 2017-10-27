// my_predictor.h
// This file contains a sample my_predictor class.
// It has a simple 32,768-entry gshare with a history length of 15 and a
// simple direct-mapped branch target buffer for indirect branch prediction.
#include <math.h>
#include <cstddef>
#include <cstring>
#include <bitset>

#define H 8
#define NUM_WTS 4096
#define HIST_LEN 128
#define PATH_LEN 16
#define MAX_WT 127
#define MIN_WT -128
#define TARGET_BITS 15

class my_update : public branch_update
{
  public:
    unsigned int index[H]; // index to the table
    int perc_out;          // table output
    my_update(void)
    {
        for (int i = 0; i < H; i++)
            index[i] = 0;
        perc_out = 0;
    }
};

class my_predictor : public branch_predictor
{
  public:
    char wt_tab[H][NUM_WTS]; // table of weights
    unsigned int targets[1 << TARGET_BITS];
    std::bitset<HIST_LEN> hist_reg; // hist reg
    std::bitset<HIST_LEN> path_reg; // path reg

    my_predictor(void)
    {
        // initialize the weight table to 0
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < NUM_WTS; j++)
            {
                wt_tab[i][j] = 0;
            }
        }

        memset(targets, 0, sizeof(targets));
    }
    my_update u;
    branch_info bi;

    // branch prediction
    branch_update *predict(branch_info &b)
    {
        bi = b;
        if (b.br_flags & BR_CONDITIONAL)
        {
            u.index[0] = (b.address) % (NUM_WTS);
            u.perc_out = wt_tab[0][u.index[0]];

            const unsigned int seg_size = HIST_LEN / H;
            std::bitset<seg_size> seg;
            std::bitset<HIST_LEN> mask;
            std::bitset<HIST_LEN> hash = hist_reg ^ path_reg;

            for (int i = 1; i < H; i++)
            {
                // create segments starting from the most recent history bits and then
                // moving left
                int start = i * H;
                int end = start + H;

                for (int k = start; k < end; k++)
                {
                    seg <<= hash[k];
                }

                u.index[i] = (seg.to_ulong() ^ (b.address << 1)) % (NUM_WTS);
                u.perc_out += wt_tab[i][u.index[i]];
            }
            if (u.perc_out >= 0)
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
            u.direction_prediction(true); // unconditional branch
        }
        if (b.br_flags & BR_INDIRECT)
        {
            u.target_prediction(targets[b.address & ((1 << TARGET_BITS) - 1)]);
        }
        return &u;
    }

    // training algorithm
    void update(branch_update *u, bool taken, unsigned int target)
    {
        float theta = (int)(1.93 * H + 14);
        if (bi.br_flags & BR_CONDITIONAL)
        {
            if (u->direction_prediction() != taken || abs(((my_update *)u)->perc_out) < theta)
            {
                for (int i = 0; i < H; i++)
                {
                    char *c = &wt_tab[i][((my_update *)u)->index[i]];
                    if (taken) // agree
                    {
                        if (*c < MAX_WT)
                            (*c)++;
                    }
                    else // disagree
                    {
                        if (*c > MIN_WT)
                            (*c)--;
                    }
                }
            }

            // update the hist reg
            hist_reg <<= 1;
            hist_reg |= taken;
            // update the path reg
            // take the last 4 bits of branch addr for path hist reg
            path_reg = bi.address & 0xF;
            path_reg <<= 1;
        }

        if (bi.br_flags & BR_INDIRECT)
        {
            targets[bi.address & ((1 << TARGET_BITS) - 1)] = target;
        }
    }
};
