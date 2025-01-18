#pragma once 

#include <map>

class qn_maps 
{  
    public: 
    std::map<std::pair<int,int>,int> lm_to_block;
    std::map<int,std::pair<int,int>> block_to_lm;
    void set_block_to_lm();
};