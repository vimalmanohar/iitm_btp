#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>
using std::vector;

#include "phoneCAT/am-phoneCAT.h"
#include "thread/kaldi-thread.h"

namespace kaldi {

void AmPhoneCAT::Read(std::istream &in_stream, bool binary) {
  int32 num_states, feat_dim, num_gauss;
  std::string token;

  ExpectToken(in_stream, binary, "<PhoneCAT>");
  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &num_states);
  ExpectToken(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &feat_dim);
  KALDI_ASSERT(num_states > 0 && feat_dim > 0);
  
  ReadToken(in_stream, binary, &token);
  
  while (token != "</PhoneCAT>") {
    if (token == "<DIAG_UBM>") {
      diag_ubm_.Read(in_stream, binary);
    } else if (token == "<FULL_UBM>") {
      full_ubm_.Read(in_stream, binary);
    } else if (token == "<SigmaInv>") {
      ExpectToken(in_stream, binary, "<NUMGaussians>");
      ReadBasicType(in_stream, binary, &num_gauss);
      SigmaInv_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        SigmaInv_[i].Read(in_stream, binary);
      }
    } else if (token == "<M>") {
      ExpectToken(in_stream, binary, "<NUMGaussians>");
      ReadBasicType(in_stream, binary, &num_gauss);
      M_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        M_[i].Read(in_stream, binary);
      }
    } else if (token == "<w>") {
      w_.Read(in_stream, binary);
    } else if (token == "<v>") {
      v_.resize(num_states);
      for (int32 j = 0; j < num_states; j++) {
        v_[j].Read(in_stream, binary);
      }
    } else if (token == "<n>") {
      n_.resize(num_states);
      for (int32 j = 0; j < num_states; j++) {
        n_[j].Read(in_stream, binary);
      }
    } else {
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadToken(in_stream, binary, &token);
  }

  if (n_.empty()) {
    ComputeNormalizers();
  }
}

void AmPhoneCAT::Write(std::ostream &out_stream, bool binary,
                   PhoneCATWriteFlagsType write_params) const {
  int32 num_states = NumPdfs(),
      feat_dim = FeatureDim(),
      num_gauss = NumGauss();

  WriteToken(out_stream, binary, "<PhoneCAT>");
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<NUMSTATES>");
  WriteBasicType(out_stream, binary, num_states);
  WriteToken(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, feat_dim);
  if (!binary) out_stream << "\n";

  if (write_params & kPhoneCATBackgroundGmms) {
    WriteToken(out_stream, binary, "<DIAG_UBM>");
    diag_ubm_.Write(out_stream, binary);
  }

  if (write_params & kPhoneCATGlobalParams) {
    WriteToken(out_stream, binary, "<SigmaInv>");
    WriteToken(out_stream, binary, "<NUMGaussians>");
    WriteBasicType(out_stream, binary, num_gauss);
    if (!binary) out_stream << "\n";
    for (int32 i = 0; i < num_gauss; i++) {
      SigmaInv_[i].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<M>");
    WriteToken(out_stream, binary, "<NUMGaussians>");
    WriteBasicType(out_stream, binary, num_gauss);
    if (!binary) out_stream << "\n";
    for (int32 i = 0; i < num_gauss; i++) {
      M_[i].Write(out_stream, binary);
    }
  }

  if (write_params & kPhoneCATStateParams) {
    WriteToken(out_stream, binary, "<v>");
    for (int32 j = 0; j < num_states; j++) {
      v_[j].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<w>");
    for (int32 j = 0; j < num_states, j++) {
      w_[j].Write(out_stream, binary);
    }
  }

  if (write_params & kPhoneCATNormalizers) {
    WriteToken(out_stream, binary, "<n>");
    if (n_.empty())
      KALDI_WARN << "Not writing normalizers since they are not present.";
    else
      for (int32 j = 0; j < num_states; j++)
        n_[j].Write(out_stream, binary);
  }
  
  WriteToken(out_stream, binary, "</PhoneCAT>");
}

void AmPhoneCAT::Check(bool show_properties) {
  int32 num_states = NumPdfs(),
      num_gauss = NumGauss(),
      feat_dim = FeatureDim(),
      phn_dim = PhoneSpaceDim(),

  if (show_properties)
    KALDI_LOG << "AmPhoneCAT: #states = " << num_states << ", #Gaussians = "
              << num_gauss << ", feature dim = " << feat_dim
              << ", phone-space dim =" << phn_dim;
  KALDI_ASSERT(num_states > 0 && num_gauss > 0 && feat_dim > 0 && phn_dim > 0);

  std::ostringstream debug_str;
  
  // First check the diagonal-covariance UBM.
  KALDI_ASSERT(diag_ubm_.NumGauss() == num_gauss);
  KALDI_ASSERT(diag_ubm_.Dim() == feat_dim);

  // Check the globally-shared covariance matrices.
  KALDI_ASSERT(SigmaInv_.size() == static_cast<size_t>(num_gauss));
  for (int32 i = 0; i < num_gauss; i++) {
    KALDI_ASSERT(SigmaInv_[i].NumRows() == feat_dim &&
                 SigmaInv_[i](0, 0) > 0.0);  // or it wouldn't be +ve definite.
  }

  KALDI_ASSERT(M_.size() == static_cast<size_t>(num_gauss));
  for (int32 i = 0; i < num_gauss; i++) {
    KALDI_ASSERT(M_[i].NumRows() == feat_dim && M_[i].NumCols() == phn_dim);
  }
  
  {  // check v, w.
    KALDI_ASSERT(v_.size() == static_cast<size_t>(num_states) &&
                 w_.size() == static_cast<size_t>(num_states));
    for (int32 j = 0; j < num_states; j++) {
      KALDI_ASSERT(v_[j].size() == phn_dim && 
          w_[j].size() == num_gauss);
    }
  }

  // check n.
  if (n_.size() == 0) {
    debug_str << "Normalizers: no.  ";
  } else {
    debug_str << "Normalizers: yes.  ";
    KALDI_ASSERT(n_.size() == static_cast<size_t>(num_states));
    for (int32 j = 0; j < num_states; j++) {
      KALDI_ASSERT(n_[j].size() == num_gauss);
    }
  }
  
  if (show_properties)
    KALDI_LOG << "Phone CAT model properties: " << debug_str.str();
}

void AmPhoneCAT::InitializeFromDiagGmm(const DiagGmm &diag_gmm,
                                   int32 num_states,
                                   int32 phn_subspace_dim) {
  diag_ubm_.CopyFromDiagGmm(diag_gmm);

  if (phn_subspace_dim < 1 || phn_subspace_dim > diag_gmm.Dim()) {
    KALDI_WARN << "Initial phone-subspace dimension must be in [1, "
               << diag_gmm.Dim() << "]. Changing from " << phn_subspace_dim
               << " to " << diag_gmm.Dim();
    phn_subspace_dim = diag_gmm.Dim() + 1;
  }

  v_.clear();
  w_clear();
  SigmaInv_.clear();
  
  KALDI_LOG << "Initializing model";
  
  InitializeM();
  InitializeVecs(num_states);
  KALDI_LOG << "Initializing variances";
  InitializeCovars();
}

void AmPhoneCAT::CopyFromPhoneCAT(const AmPhoneCAT &other,
                                  bool copy_normalizers) {

  KALDI_LOG << "Copying AmPhoneCAT";
  
  // Copy background GMMs
  diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
  
  // Copy global params
  SigmaInv_ = other.SigmaInv_;
  M_ = other.M_;

  // Copy state-specific params, but only copy normalizers if requested.
  v_ = other.v_;
  w_ = other.w_;
  if (copy_normalizers) n_ = other.n_;

  KALDI_LOG << "Done.";
}

// Copy global vectors from another model but initialize
// the state vectors to zero
void AmPhoneCAT::CopyGlobalsInitVecs(const AmPhoneCAT &other,
                                 int32 phn_subspace_dim,
                                 int32 num_pdfs) {
  if (phn_subspace_dim < 1 || phn_subspace_dim > other.PhoneSpaceDim()) {
    KALDI_WARN << "Initial phone-subspace dimension must be in [1, "
        << other.PhoneSpaceDim() << "]. Changing from " << phn_subspace_dim
        << " to " << other.PhoneSpaceDim();
    phn_subspace_dim = other.PhoneSpaceDim();
  }
  
  KALDI_LOG << "Initializing model";

  // Copy background GMMs
  diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
  
  // Copy global params
  SigmaInv_ = other.SigmaInv_;
  int32 num_gauss = diag_ubm_.NumGauss(),
      data_dim = other.FeatureDim();
  M_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    M_[i].Resize(data_dim, phn_subspace_dim);
    M_[i].CopyFromMat(other.M_[i].Range(0, data_dim, 0, phn_subspace_dim),
                      kNoTrans);
  }
  
  InitializeVecs(num_pdfs);
}


void AmPhoneCAT::ComputePerFrameVars(const VectorBase<BaseFloat> &data,
                                 const std::vector<int32> &gselect,
                                 BaseFloat logdet_s,
                                 PhoneCATPerFrameDerivedVars *per_frame_vars) const {
  KALDI_ASSERT(!n_.empty() && "ComputeNormalizers() must be called.");
  
  if (per_frame_vars->NeedsResizing(gselect.size(),
                                    FeatureDim(),
                                    PhoneSpaceDim()))
    per_frame_vars->Resize(gselect.size(), FeatureDim(), PhoneSpaceDim());

  per_frame_vars->gselect = gselect;
  per_frame_vars->xt.CopyFromVec(data);

  Vector<BaseFloat> SigmaInv_xt(FeatureDim());
  for (int32 ki = 0, last = gselect.size(); ki < last; ki++) {
    int32 i = gselect[ki];
    SigmaInv_xt.AddSpVec(1.0, SigmaInv_[i], per_frame_vars->xt, 0.0);
    
    // z_{i}(t) = M_{i}^T \SigmaInv{i} x(t)
    per_frame_vars->zti.Row(ki).AddMatVec(1.0, M_[i], kTrans, SigmaInv_xt,0.0);

    // n_{i}(t) = -0.5 x(t) ^T \SigmaInv{i} x(t)
    per_frame_vars->nti(ki) = -0.5 * VecVec(per_frame_vars->xt, SigmaInv_xt);
  }
}


BaseFloat AmPhoneCAT::LogLikelihood(const AmPhoneCATPerFrameDerivedVars &per_frame_vars, int32 j, BaseFloat log_prune) const {
  KALDI_ASSERT(j < NumPdfs());
  const vector<int32> &gselect = per_frame_vars.gselect;

  // log p( x(t),i | j ) [indexed by j, ki]
  // Although the extra memory allocation of storing this as a
  // vector might seem unnecessary, we save time in the LogSumExp()
  // via more effective pruning.
  vector<BaseFloat> logp_x(gselect.size());

  for (int32 ki = 0, last = gselect.size();  ki < last; ki++) {
    int32 i = gselect[ki];
    // Compute z_{i}^T v_{j}
    logp_x[ki] = VecVec(per_frame_vars.zti.Row(ki), v_[j]);
    logp_x[ki] += n_[j][i];
    logp_x[ki] += nti(ki);
  }

  // log p(x(t)/j) = log \sum_{i} p( x(t),i | j )
  return logp_x.LogSumExp(log_prune);
}

BaseFloat AmPhoneCAT::ComponentPosteriors(
    const PhoneCATPerFrameDerivedVars &per_frame_vars,
    int32 j,
    Vector<BaseFloat> *post) const {
  KALDI_ASSERT(j < NumPdfs());
  if (post == NULL) KALDI_ERR << "NULL pointer passed as return argument.";

  const vector<int32> &gselect = per_frame_vars.gselect;
  int32 num_gselect = gselect.size();
  post->Resize(num_gselect);

  // log p( x(t),i | j ) = z_{i}^T v_{j} // for the gselect-ed gaussians
  post.AddVecVec(1.0, per_frame_vars.zti, kNoTrans, v_[j], 0.0);
  
  for (int32 ki = 0; ki < num_gselect; ki++) {
    int32 i = gselect[ki];
    // log p( x(t),i | j ) += n_{ji} + n_{i}(t)
    post[ki] += n_[j][i];
    post[ki] += nti(ki);
  }

  // log p(x(t)|j) = log \sum_{i} p(x(t), i|j)
  return post->ApplySoftMax();
}

void AmPhoneCAT::ComputeDerivedVars() {
  if (n_.empty()) {
    ComputeNormalizers();
  }
}

class ComputeNormalizersClass: public MultiThreadable { // For multi-threaded.
 public:
  ComputeNormalizersClass(AmPhoneCAT *am_phoneCAT,
                          int32 *entropy_count_ptr,
                          double *entropy_sum_ptr):
      am_phoneCAT_(am_phoneCAT), entropy_count_ptr_(entropy_count_ptr),
      entropy_sum_ptr_(entropy_sum_ptr), entropy_count_(0),
      entropy_sum_(0.0) { }

  ~ComputeNormalizersClass() {
    *entropy_count_ptr_ += entropy_count_;
    *entropy_sum_ptr_ += entropy_sum_;
  }
  
  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to original pointer in the destructor.
    am_sgmm_->ComputeNormalizersInternal(num_threads_, thread_id_,
                                         &entropy_count_,
                                         &entropy_sum_);
  }
 private:
  ComputeNormalizersClass() { } // Disallow empty constructor.
  AmPhoneCAT *am_phoneCAT_;
  int32 *entropy_count_ptr_;
  double *entropy_sum_ptr_;
  int32 entropy_count_;
  double entropy_sum_;

};

void AmPhoneCAT::ComputeNormalizers() {
  KALDI_LOG << "Computing normalizers";
  n_.resize(NumPdfs());         // NumPdfs == Num_TiedStates
  int32 entropy_count = 0;
  double entropy_sum = 0.0;
  
  ComputeNormalizersClass c(this, &entropy_count, &entropy_sum);
  RunMultiThreaded(c);
  KALDI_LOG << "Entropy of weights in substates is "
            << (entropy_sum / entropy_count) << " over " << entropy_count
            << " substates, equivalent to perplexity of "
            << (exp(entropy_sum /entropy_count));
  KALDI_LOG << "Done computing normalizers";
}

///////////////////////////////////////////////////////////////////////////////

template<class Real>
void AmPhoneCAT::ComputeH(std::vector< SpMatrix<Real> > *H) const {
  KALDI_ASSERT(NumGauss() != 0);
  (*H).resize(NumGauss());
  SpMatrix<BaseFloat> H_tmp(PhoneSpaceDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    (*H)[i].Resize(PhoneSpaceDim());
    H_tmp.AddMat2Sp(1.0, M_[i], kTrans, SigmaInv_[i], 0.0);
    (*H)[i].CopyFromSp(H_tmp);
  }
}

// Instantiate the template.
template
void AmPhoneCAT::ComputeH(std::vector< SpMatrix<float> > *H) const;
template
void AmPhoneCAT::ComputeH(std::vector< SpMatrix<double> > *H) const;

