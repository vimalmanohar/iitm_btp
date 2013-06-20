#ifndef KALDI_PHONECAT_AM_PHONECAT_H_
#define KALDI_PHONECAT_AM_PHONECAT_H_

#include <vector>
#include <queue>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "util/parse-options.h"
#include "util/table-types.h"

namespace kaldi {

  struct PhoneCATGselectConfig {
    /// Number of highest-scoring diagonal-covariance Gaussians per frame.
    int32 diag_gmm_nbest;

    PhoneCATGselectConfig() {
      diag_gmm_nbest = 50;
    }

    void Register(ParseOptions *po) {
      po->Register("diag-gmm-nbest", &diag_gmm_nbest, "Number of highest-scoring"
          " diagonal-covariance Gaussians selected per frame.");
    }
  };

  /** \struct PhoneCATPerFrameDerivedVars
   *  Holds the per-frame precomputed quantities x(t), z_{i}(t), and
   *  n_{i}(t) for the PhoneCAT, as well as the cached Gaussian
   *  selection records.
   */
  struct PhoneCATPerFrameDerivedVars {
    std::vector<int32> gselect;
    Vector<BaseFloat> xt;   ///< x(t), dim = [D]
                            /// Just the observation vector.
                            /// In future, FMLLR can be added
    Matrix<BaseFloat> zti;  ///< z_{i}(t), dim = [I][S]
    Vector<BaseFloat> nti;  ///< n_{i}(t), dim = [I]

    PhoneCATPerFrameDerivedVars() : xt(0), zti(0,0), nti(0) {}
    void Resize(int32 ngauss, int32 feat_dim, int32 phn_dim) {
      xt.Resize(feat_dim);
      zti.Resize(ngauss, phn_dim);
      nti.Resize(nguass);
    }

    bool IsEmpty() const {
      return (xt.Dim() == 0 || zti.NumRows() == 0 || nti.Dim() == 0);
    }

    bool NeedsResizing(int32 ngauss, int32 feat_dim, int32 phn_dim) const {
      return (xt.Dim() != feat_dim 
          || zti.NumRows() != ngauss || zti.NumCols() != phn_dim
          || nti.Dim() != nguass);
    }
  };
}
