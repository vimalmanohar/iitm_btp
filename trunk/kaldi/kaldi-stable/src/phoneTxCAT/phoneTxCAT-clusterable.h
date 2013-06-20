// phoneTxCAT/phoneTxCAT-clusterable.h

#ifndef KALDI_PHONETXCAT_PHONETXCAT_CLUSTERABLE_H_
#define KALDI_PHONETXCAT_PHONETXCAT_CLUSTERABLE_H_

#include <vector>
#include <queue>

#include "phoneTxCAT/am-phoneTxCAT.h"
#include "hmm/transition-model.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

/// This header defines an object that can be used to create decision
/// trees using a form of PhoneTxCAT statistics.  It is analogous to the
/// GaussClusterable object, but uses the PhoneTxCAT.  The auxiliary function
/// it uses is related to the normal PhoneTxCAT auxiliary function, but for
/// efficiency it uses a simpler model on the weights, which is equivalent
/// to assuming the weights w_{ji} [there no index m since we assume one
/// mixture per state!] are directly estimated using ML.

class PhoneTxCATClusterable: public Clusterable {
  public:
    PhoneTxCATClusterable(const AmPhoneTxCAT &phoneTxCAT,
        const std::vector< SpMatrix<double> > &H): // H can be empty vector
      // at initialization. Used to cache something from the model.
      
      phoneTxCAT_(phoneTxCAT),
      H_(H),
      gamma_(phoneTxCAT.NumGauss()),

