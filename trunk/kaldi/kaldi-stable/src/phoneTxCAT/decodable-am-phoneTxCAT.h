// phoneTxCAT/decodable-am-phoneTxCAT.h

#ifndef KALDI_DECODER_DECODABLE_AM_PHONETXCAT_H_
#define KALDI_DECODER_DECODABLE_AM_PHONETXCAT_H_

#include <vector>

#include "base/kaldi-common.h"
#include "phoneTxCAT/am-phoneTxCAT.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {

  class DecodableAmPhoneTxCAT : public DecodableInterface {
    public:
      DecodableAmPhoneTxCAT(const PhoneTxCATGselectConfig &opts,
          const AmPhoneTxCAT &am,
          const TransitionModel &tm,
          const Matrix<BaseFloat> &feats,
          const std::vector<std::vector<int32> > &gselect_all,
          BaseFloat log_prune):  // gselect_all may be empty
        acoustic_model_(am), phoneTxCAT_config_(opts), 
        trans_model_(tm), feature_matrix_(feats),
        gselect_all_(gselect_all), previous_frame_(-1),
        log_prune_(log_prune) {
          ResetLogLikeCache();
        }

      // Note, frames are numbered from zero, but transition indices are 1-based!
      // This is for compatibility with OpenFST
      virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
        return LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid));
      }

      int32 NumFrames() { return feature_matrix_.NumRows(); }
      virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

      virtual bool IsLastFrame(int32 frame) {
        KALDI_ASSERT(frame < NumFrames());
        return (frame == NumFrames() - 1);
      }

    protected:
      void ResetLogLikeCache();
      virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 pdf_id);

      const AmPhoneTxCAT &acoustic_model_;
      const PhoneTxCATGselectConfig &phoneTxCAT_config_;
      const TransitionModel &trans_model_;  ///< for tid to pdf mapping
      const Matrix<BaseFloat> &feature_matrix_;
      const std::vector< std::vector<int32> > gselect_all_; ///< if nonempty,
      ///< precomputed gaussian indices.
      int32 previous_frame_;
      BaseFloat log_prune_;

      /// Defines a cache record for a state
      struct LikelihoodCacheRecord {
        BaseFloat log_like;   ///< Cache value
        int32 hit_time;       ///< Frame for which this value is relevant
      };

      /// Cached per-frame quantities used in PhoneTxCAT likelihood computation
      std::vector<LikelihoodCacheRecord> log_like_cache_;
      std::vector<int32> gselect_;
      PhoneTxCATPerFrameDerivedVars per_frame_vars_;

    private:
      KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmPhoneTxCAT);
  };

  class DecodableAmPhoneTxCATScaled : public DecodableAmPhoneTxCAT {
    public:
      DecodableAmPhoneTxCATScaled(const PhoneTxCATGselectConfig &opts,
          const AmPhoneTxCAT &am,
          const TransitionModel &tm,
          const Matrix<BaseFloat> &feats,
          const std::vector<std::vector<int32> > &gselect_all,
          // gselect_all may be empty
          BaseFloat log_prune,
          BaseFloat scale)
        : DecodableAmPhoneTxCAT(opts, am, tm, feats, gselect_all, log_prune),
        scale_(scale) {}

      // Note, frames are numbered from zero but transition-ids from one.
      virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
        return LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid)) * scale_;
      }

    private:
      BaseFloat scale_;
      KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmPhoneTxCATScaled);
  };

} // namespace kaldi

#endif // KALDI_DECODER_DECODABLE_AM_PHONETXCAT_H_
