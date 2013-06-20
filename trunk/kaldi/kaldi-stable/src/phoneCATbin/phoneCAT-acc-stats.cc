#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "phoneCAT/am-phoneCAT.h"
#include "hmm/transition-model.h"
#include "phoneCAT/estimate-am-phoneCAT.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate stats for Phone CAT training.\n"
        "Usage: phoneCAT-acc-stats [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <stats-out>\n"
        "e.g.: phoneCAT-acc-stats 1.mdl 1.ali scp:train.scp 'ark:ali-to-post 1.ali ark:-|' 1.acc\n";
    
    ParseOptions po(usage);
    
    bool binary = true;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    std::string update_flags_str = "vMwSt";
    BaseFloat rand_prune = 1.0e-05;
    
    SgmmGselectConfig sgmm_opts;
