#include "misc.h"

void qn_maps::set_block_to_lm()
{
    for (auto& it : this->lm_to_block)
    {
        this->block_to_lm[it.second] = it.first;
    }
}