// phoneTxCAT/decodable-am-phoneTxCAT.cc

#include <vector>
using std::vector;

#include "phoneTxCAT/decodable-am-phoneTxCAT.h"

namespace kaldi {

  BaseFloat DecodableAmPhoneTxCAT::LogLikelihoodZeroBased(int32 frame, 
      int32 pdf_id) {
    KALDI_ASSERT(frame >= 0 && frame < NumFrames());
    KALDI_ASSERT(pdf_id >= 0 && pdf_id < NumIndices());

    if (log_like_cache_[pdf_id].hit_time == frame) {
      return log_like_cache_[pdf_id].log_like;  
      // return cached value, if found
    }

    const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);
    // check if everything is in order
    if (acoustic_model_.FeatureDim() != data.Dim()) {
      KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << "vs. model dim = " << acoustic_model_.FeatureDim();
    }

    if (frame != previous_frame_) {  // Per-frame precomputation for phoneTxCAT
      if (gselect_all_.empty())
        acoustic_model_.GaussianSelection(phoneTxCAT_config_, data, &gselect_);
      else {
        KALDI_ASSERT(frame < gselect_all_.size());
        gselect_ = gselect_all_[frame];
      }
      acoustic_model_.ComputePerFrameVars(data, gselect_, &per_frame_vars_);
      previous_frame_ = frame;
    }

    BaseFloat loglike = acoustic_model_.LogLikelihood(per_frame_vars_, pdf_id,
        log_prune_);
    if (KALDI_ISNAN(loglike) || KALDI_ISINF(loglike))
      KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
    log_like_cache_[pdf_id].log_like = loglike;
    log_like_cache_[pdf_id].hit_time = frame;
    return loglike;
  }

  void DecodableAmPhoneTxCAT::ResetLogLikeCache() {
    if (log_like_cache_.size() != acoustic_model_.NumPdfs()) {
      log_like_cache_.resize(acoustic_model_.NumPdfs());
    }
    vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
    for (; it != end; ++it) { it->hit_time = -1; }
  }
    
} // namespace kaldi


