#include <algorithm>
#include <string>
#include <sstream>
using std::string;
#include <vector>
using std::vector;

#include "phoneTxCAT/am-phoneTxCAT.h"
#include "phoneTxCAT/estimate-am-phoneTxCAT.h"
#include "thread/kaldi-thread.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

  void MleAmPhoneTxCATAccs::Write(
      std::ostream &out_stream, bool binary) const {
    
    uint32 tmp_uint32;

    WriteToken(out_stream, binary, "<PHONETXCATACCS>");
    WriteToken(out_stream, binary, "<NUMSTATES>");
    tmp_uint32 = static_cast<uint32>(num_states_);
    WriteBasicType(out_stream, binary, tmp_uint32);
    WriteToken(out_stream, binary, "<NUMGaussians>");
    tmp_uint32 = static_cast<uint32>(num_gaussians_);
    WriteBasicType(out_stream, binary, tmp_uint32);
    WriteToken(out_stream, binary, "<FEATUREDIM>");
    tmp_uint32 = static_cast<uint32>(feature_dim_);
    WriteBasicType(out_stream, binary, tmp_uint32);
    WriteToken(out_stream, binary, "<NUMClusters>");
    tmp_uint32 = static_cast<uint32>(num_clusters_);
    WriteBasicType(out_stream, binary, tmp_uint32);
    WriteToken(out_stream, binary, "<NUMClusterWeightClasses>");
    tmp_uint32 = static_cast<uint32>(num_cluster_weight_classes_);
    WriteBasicType(out_stream, binary, tmp_uint32);
    if (!binary) out_stream << "\n";

    //(Deprecated)
    //if (Z_.size() != 0) {
    //  KALDI_ASSERT(gamma_.size() != 0);
    //  WriteToken(out_stream, binary, "<Z>");
    //  for (int32 j=0; j < num_states_; j++) {
    //    Z_[j].Write(out_stream, binary);
    //  }
    //}
    
    if (y_.size() != 0) {
      KALDI_ASSERT(gamma_.size() != 0);
      WriteToken(out_stream, binary, "<y>");
      for (int32 r = 0; r < num_cluster_weight_classes_; r++) {
        if (r > 0) {
          WriteToken(out_stream, binary, "<ClusterWeightClass>");
          WriteBasicType(out_stream, binary, r);
          if (!binary) out_stream << "\n";
        }
        for (int32 j = 0; j < num_states_; j++) {
          y_[r][j].Write(out_stream, binary);
        }
        if (!binary) out_stream << "\n";
      }
    }

    if (!binary) out_stream << "\n";
    
    if (G_.size() != 0) {
      KALDI_ASSERT(K_.size() != 0);
      WriteToken(out_stream, binary, "<G>");
      for (int32 i = 0; i < num_gaussians_; i++) {
        G_[i].Write(out_stream, binary);
      }
      WriteToken(out_stream, binary, "<K>");
      for (int32 i = 0; i < num_gaussians_; i++) {
        K_[i].Write(out_stream, binary);
      }
    }
    
    if (!binary) out_stream << "\n";

    if (L_.size() != 0) {
      KALDI_ASSERT(gamma_.size() != 0);
      WriteToken(out_stream, binary, "<L>");
      for (int32 i = 0; i < num_gaussians_; i++) {
        L_[i].Write(out_stream, binary);
      }
    }
    
    if (!binary) out_stream << "\n";

    if (gamma_.size() != 0) {
      WriteToken(out_stream, binary, "<gamma>");
      for (int32 j = 0; j < num_states_; j++) {
        gamma_[j].Write(out_stream, binary);
      }
    }
    
    if (!binary) out_stream << "\n";

    WriteToken(out_stream, binary, "<total_like>");
    WriteBasicType(out_stream, binary, total_like_);

    WriteToken(out_stream, binary, "<total_frames>");
    WriteBasicType(out_stream, binary, total_frames_);

    WriteToken(out_stream, binary, "</PHONETXCATACCS>");
  }

  void MleAmPhoneTxCATAccs::Read(std::istream &in_stream, bool binary,
      bool add) {
    uint32 tmp_uint32;
    string token;

    ExpectToken(in_stream, binary, "<PHONETXCATACCS>");

    ExpectToken(in_stream, binary, "<NUMSTATES>");
    ReadBasicType(in_stream, binary, &tmp_uint32);
    num_states_ = static_cast<int32>(tmp_uint32);
    ExpectToken(in_stream, binary, "<NUMGaussians>");
    ReadBasicType(in_stream, binary, &tmp_uint32);
    num_gaussians_ = static_cast<int32>(tmp_uint32);
    ExpectToken(in_stream, binary, "<FEATUREDIM>");
    ReadBasicType(in_stream, binary, &tmp_uint32);
    feature_dim_ = static_cast<int32>(tmp_uint32);
    ExpectToken(in_stream, binary, "<NUMClusters>");
    ReadBasicType(in_stream, binary, &tmp_uint32);
    num_clusters_ = static_cast<int32>(tmp_uint32);

    ReadToken(in_stream, binary, &token);

    num_cluster_weight_classes_ = 1;

    while (token != "</PHONETXCATACCS>") {
      // (Deprecated)
      // if (token == "<Z>") {
      //  Z_.resize(num_states_);
      //  for (size_t j = 0; j < Z_.size(); j++) {
      //    //Z_[j].Resize(num_gaussians_, feature_dim_);
      //    Z_[j].Read(in_stream, binary, add);
      //  }
      //  }
      if (token == "<NUMClusterWeightClasses>") {
        ReadBasicType(in_stream, binary, &tmp_uint32);
        num_cluster_weight_classes_ = static_cast<int32>(tmp_uint32);
      } else if (token == "<y>") {
        y_.resize(num_cluster_weight_classes_);
        for (int32 r = 0; r < num_cluster_weight_classes_; r++) {
          if (r > 0) {
            ExpectToken(in_stream, binary, "<ClusterWeightClass>");
            int32 tmp32;
            ReadBasicType(in_stream, binary, &tmp32);
            KALDI_ASSERT(r == tmp32);
          }
          y_[r].resize(num_states_);
          for (int32 j = 0; j < num_states_; j++) {
            y_[r][j].Read(in_stream, binary, add);
          }
        }
      } else if (token == "<G>") {
        G_.resize(num_gaussians_);
        for (size_t i = 0; i < G_.size(); i++) {
          //G_[i].Resize(num_clusters_);
          G_[i].Read(in_stream, binary, add);
        }
      } else if (token == "<K>") {
        K_.resize(num_gaussians_);
        for (size_t i = 0; i < K_.size(); i++) {
          //K_[i].Resize(num_clusters_, feature_dim_);
          K_[i].Read(in_stream, binary, add);
        }
      } else if (token == "<L>") {
        L_.resize(num_gaussians_);
        for (size_t i = 0; i < L_.size(); i++) {
          //L_[i].Resize(feature_dim_);
          L_[i].Read(in_stream, binary, add);
        }
      } else if (token == "<gamma>") {
        gamma_.resize(num_states_);
        for (int32 j = 0; j < num_states_; j++) {
          //gamma_[j].Resize(num_gaussians_);
          gamma_[j].Read(in_stream, binary, add);
        }
      } else if (token == "<total_like>") {
        double total_like;
        ReadBasicType(in_stream, binary, &total_like);
        if (add)
          total_like_ += total_like;
        else
          total_like_ = total_like;
      } else if (token == "<total_frames>") {
        double total_frames;
        ReadBasicType(in_stream, binary, &total_frames);
        if (add)
          total_frames_ += total_frames;
        else
          total_frames_ = total_frames;
      } else {
        KALDI_ERR << "Unexpected token '" << token 
          << "' in model file ";
      }
      ReadToken(in_stream, binary, &token);
    }
  }

  void MleAmPhoneTxCATAccs::Check(const AmPhoneTxCAT &model,
                                  bool show_properties) const {
    if (show_properties) {
      KALDI_LOG << "PhoneTxCATPdfModel: J = " << num_states_ << ", D = " <<
        feature_dim_ << ", P = " << num_clusters_ 
        << ", I = " << num_gaussians_;
    }
    
    KALDI_ASSERT(num_states_ == model.NumPdfs() && num_states_ > 0);
    KALDI_ASSERT(num_gaussians_ == model.NumGauss() && num_gaussians_ > 0);
    KALDI_ASSERT(feature_dim_ == model.FeatureDim() && feature_dim_ > 0);
    KALDI_ASSERT(num_clusters_ == model.NumClusters() && num_clusters_ > 0);

    std::ostringstream debug_str;

    //(Deprecated)
    //if (Z_.size() == 0) {
    //  debug_str << "Z: no.  ";
    //} else {
    //  KALDI_ASSERT(gamma_.size() != 0);
    //  KALDI_ASSERT(Z_.size() == static_cast<size_t>(num_states_));
    //  bool nz = false;
    //  for (int32 j = 0; j < num_states_; j++) {
    //    KALDI_ASSERT(Z_[j].NumRows() == num_gaussians_ &&
    //        Z_[j].NumCols() == feature_dim_);
    //    if(!nz && Z_[j](0,0) != 0) { nz = true; }
    //  }
    //  debug_str << "Z: yes, " << string(nz ? "nonzero. " : "zero. ");
    //}

    if (y_.size() == 0) {
      debug_str << "y: no.  ";
    } else {
      KALDI_ASSERT(gamma_.size() != 0);
      bool nz = false;
      for (int32 r = 0; r < num_cluster_weight_classes_; r++) {
        KALDI_ASSERT(y_[r].size() == static_cast<size_t>(num_states_));
        for (int32 j = 0; j < num_states_; j++) {
          KALDI_ASSERT(y_[r][j].Dim() == num_clusters_);
          if (!nz && y_[r][j](0) != 0) { nz = true; }
        }
      }
      debug_str << "y: yes, " << string(nz ? "nonzero. " : "zero. ");
    }

    if (G_.size() == 0) {
      KALDI_ASSERT(K_.size() == 0);
      debug_str << "G, K: no.   ";
    } else {
      KALDI_ASSERT(G_.size() == static_cast<size_t>(num_gaussians_));
      KALDI_ASSERT(K_.size() == static_cast<size_t>(num_gaussians_));
      bool G_nz = false, K_nz = false;
      for (int32 i = 0; i < num_gaussians_; i++) {
        KALDI_ASSERT(G_[i].NumRows() == num_clusters_);
        KALDI_ASSERT(K_[i].NumRows() == num_clusters_ &&
            K_[i].NumCols() == feature_dim_);
        if (!G_nz && G_[i](0,0) != 0) { G_nz = true; }
        if (!K_nz && K_[i](0,0) != 0) { K_nz = true; }
      }
      
      debug_str << "G: yes, " << string(G_nz ? "nonzero. " : "zero. ");
      debug_str << "K: yes, " << string(K_nz ? "nonzero. " : "zero. ");
    }

    if (L_.size() == 0) {
      debug_str << "L: no.  ";
    } else {
      KALDI_ASSERT(gamma_.size() != 0);
      KALDI_ASSERT(G_.size() != 0);
      KALDI_ASSERT(K_.size() != 0);
      bool L_nz = false;
      KALDI_ASSERT(L_.size() == static_cast<size_t>(num_gaussians_));
      for (int32 i = 0; i < num_gaussians_; i++) {
        KALDI_ASSERT(L_[i].NumRows() == feature_dim_);
        if (!L_nz && L_[i](0, 0) != 0) { L_nz = true; }
      }
      debug_str << "L: yes, " << string(L_nz ? "nonzero. " : "zero. ");
    }

    if (gamma_.size() == 0) {
      debug_str << "gamma: no.  ";
    } else {
      debug_str << "gamma: yes.  ";
      bool nz = false;
      KALDI_ASSERT(gamma_.size() == static_cast<size_t>(num_states_));
      for (int32 j = 0; j < num_states_; j++) {
        KALDI_ASSERT(gamma_[j].Dim() == num_gaussians_);
        if (!nz && gamma_[j](0) != 0) { nz = true; }
      }
      debug_str << "gamma: yes, " << string(nz ? "nonzero. " : "zero. ");
    }

    if (show_properties)
      KALDI_LOG << "Phone Transform CAT model properties: " 
        << debug_str.str() << '\n';
  }

  void MleAmPhoneTxCATAccs::ResizeAccumulators(const AmPhoneTxCAT &model,
      PhoneTxCATUpdateFlagsType flags) {
    num_states_ = model.NumPdfs();
    num_gaussians_ = model.NumGauss();
    feature_dim_ = model.FeatureDim();
    num_clusters_ = model.NumClusters();
    num_cluster_weight_classes_ = model.NumClusterWeightClasses();

    if (flags & (kPhoneTxCATStateVectors)) {
        y_.resize(num_cluster_weight_classes_);
        for (int32 r = 0; r < num_cluster_weight_classes_; r++) {
          y_[r].resize(num_states_);
          for (int32 j = 0; j < num_states_; j++) {
            y_[r][j].Resize(num_clusters_);
          }
        }
    } else {
      y_.clear();
    }

    if (flags & (kPhoneTxCATClusterTransforms | kPhoneTxCATCovarianceMatrix)) {
      G_.resize(num_gaussians_);
      K_.resize(num_gaussians_);
      for (int32 i = 0; i < num_gaussians_; i++) {
        G_[i].Resize(num_clusters_);
        K_[i].Resize(num_clusters_, feature_dim_);
      }
    } else {
      G_.clear();
      K_.clear();
    }

    if (flags & (kPhoneTxCATCovarianceMatrix)) {
      L_.resize(num_gaussians_);
      for (int32 i = 0; i < num_gaussians_; i++) {
        L_[i].Resize(feature_dim_);
      }
    } else {
      L_.clear();
    }

    if (flags & (kPhoneTxCATStateVectors |
          kPhoneTxCATCovarianceMatrix | kPhoneTxCATClusterTransforms |
          kPhoneTxCATGaussianWeights)) {
      gamma_.resize(num_states_);
      total_frames_ = total_like_ = 0;
      for (int32 j = 0; j < num_states_; j++) {
        gamma_[j].Resize(num_gaussians_);
      }
    } else {
      gamma_.clear();
      total_frames_ = total_like_ = 0;
    }
  }

  BaseFloat MleAmPhoneTxCATAccs::Accumulate(const AmPhoneTxCAT &model,
      const PhoneTxCATPerFrameDerivedVars &frame_vars, 
      int32 j, BaseFloat weight,
      PhoneTxCATUpdateFlagsType flags) {
    
    // Calculate Gaussian posteriors and collect statistics
    Vector<BaseFloat> posteriors;
    // Compute posteriors
    BaseFloat log_like = model.ComponentPosteriors(frame_vars, j, &posteriors);
    posteriors.Scale(weight);
    BaseFloat count = AccumulateFromPosteriors(model, frame_vars, posteriors,
                                             j, flags);
    // Note: total_frames_ is incremented in AccumulateFromPosteriors().
    total_like_ += count * log_like;
    return log_like;
  }

  BaseFloat MleAmPhoneTxCATAccs::AccumulateFromPosteriors(
      const AmPhoneTxCAT &model,
      const PhoneTxCATPerFrameDerivedVars &frame_vars,
      const Vector<BaseFloat> &posteriors,
      int32 j,
      PhoneTxCATUpdateFlagsType flags) {
    
    double tot_count = 0.0;
    const vector<int32> &gselect = frame_vars.gselect;
    
    for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
      int32 i = gselect[ki];
      int32 r = model.ClusterWeightClass(i);

      // gamma_{ji}(t) = p (j,i | t)
      BaseFloat gammat_ji = RandPrune(posteriors(ki), rand_prune_);

      // Accumulate statistics for non-zero gaussian posterior
      if (gammat_ji != 0.0) {
        tot_count += gammat_ji;
        if (flags & (kPhoneTxCATStateVectors |
              kPhoneTxCATCovarianceMatrix | kPhoneTxCATClusterTransforms |
              kPhoneTxCATGaussianWeights)) {
          // gamma_{ji} = \sum_t gamma_{ji}(t)
          gamma_[j](i) += gammat_ji;
        }

        if (flags & kPhoneTxCATStateVectors) {
          // Z_{j} = \sum_{jt} gamma_{ji}(t) * x(t)
          /// (Deprecated) Z_[j].Row(i).AddVec(gammat_ji, frame_vars.xt);
        
          // y_{j} = \sum_{t,i} gamma_{ji}(t) z_{i}(t)
          y_[r][j].AddVec(gammat_ji, frame_vars.zti.Row(ki));
        }

        if (flags & (kPhoneTxCATClusterTransforms |
              kPhoneTxCATCovarianceMatrix)) {
          Vector<BaseFloat> v_rj = model.StateVectors(r,j);
          // G_{i} = \sum_{jt} gamma_{ji} v_{r}{j} v_{r}{j}^T
          G_[i].AddVec2(gammat_ji, v_rj);
          // K_{i} = \sum_{jt} gamma_{ji} v_{r}{j} x(t)^T
          K_[i].AddVecVec(gammat_ji, v_rj, frame_vars.xt);
        }

        if (flags & kPhoneTxCATCovarianceMatrix) {
          // L_{i} = \sum_{jt} gamma_{ji} x(t) x(t)^T
          L_[i].AddVec2(gammat_ji, frame_vars.xt);
        }
      } // non-zero posteriors
    } // loop over selected Gaussians
    
    //if (flags & kPhoneTxCATStateVectors) {
    //  for (int32 r = 1; r < num_cluster_weight_classes_; r++) {
    //    y_[r][j] = y_[0][j];
    //  }
    //}

    total_frames_ += tot_count;
    return tot_count;
  }

  void MleAmPhoneTxCATAccs::GetStateOccupancies(Vector<BaseFloat> *occs) const {
    occs->Resize(gamma_.size());
    for (int32 j = 0, end = gamma_.size(); j < end; j++) {
      (*occs)(j) = gamma_[j].Sum();
    }
  }

  BaseFloat MleAmPhoneTxCATUpdater::Update(const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      PhoneTxCATUpdateFlagsType flags) {
    KALDI_ASSERT((flags & (kPhoneTxCATStateVectors |
            kPhoneTxCATClusterTransforms | kPhoneTxCATGaussianWeights |
            kPhoneTxCATCovarianceMatrix)) != 0);
    vector< SpMatrix<double> > H;
    
    BaseFloat tot_impr = 0.0;

    if (flags & kPhoneTxCATStateVectors) {
      model->ComputeH(&H);
      tot_impr += UpdateStateVectors(accs, model, H);
    }

    if (flags & kPhoneTxCATClusterTransforms) {
      if (update_options_.use_sequential_transform_update) {
        tot_impr += UpdateASequential(accs, model);
      } else {
        if (update_options_.use_sequential_multiple_txclass_update) {
          tot_impr += UpdateAParallel2(accs, model);
        } else {
          tot_impr += UpdateAParallel(accs, model);
        }
      }
    }

    if (flags & kPhoneTxCATGaussianWeights) {
      tot_impr += UpdateWeights(accs, model);
    }

    if (flags & kPhoneTxCATCanonicalMeans) {
      tot_impr += UpdateCanonicalMeans(accs, model);
    }

    if (flags & kPhoneTxCATCovarianceMatrix) {
      tot_impr += UpdateVars(accs, model);
    }

    KALDI_LOG << "*Overall auxf improvement, combining all parameters, is "
      << tot_impr;
  
    KALDI_LOG << "***Overall data likelihood is "
      << (accs.total_like_/accs.total_frames_)
      << " over " << (accs.total_frames_) << " frames.";
    
    model->ComputeNormalizers();
    return tot_impr;
  }

  class UpdateStateVectorsClass: public MultiThreadable {
    public:
      UpdateStateVectorsClass(const MleAmPhoneTxCATUpdater &updater,
                              const MleAmPhoneTxCATAccs &accs,
                              AmPhoneTxCAT *model,
                              const std::vector<SpMatrix<double> > &H,
                              double *auxf_impr,
                              double *like_impr):
        updater_(updater), accs_(accs), model_(model), H_(H), 
        auxf_impr_ptr_(auxf_impr), auxf_impr_(0.0),
        like_impr_ptr_(like_impr), like_impr_(0.0) {}

      ~UpdateStateVectorsClass() {
        *auxf_impr_ptr_ += auxf_impr_;
        *like_impr_ptr_ += like_impr_;
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to the total sums in the destructor.
        updater_.UpdateStateVectorsInternal(accs_, model_, H_,
            &auxf_impr_, &like_impr_,
            num_threads_, thread_id_);
      }
      
    private:
      const MleAmPhoneTxCATUpdater &updater_;
      const MleAmPhoneTxCATAccs &accs_;
      AmPhoneTxCAT *model_;
      const std::vector<SpMatrix<double> > &H_;
      double *auxf_impr_ptr_;
      double auxf_impr_;
      double *like_impr_ptr_;
      double like_impr_;
  };

  // Runs the state vectors update for states (called
  // multi-threaded).
  void MleAmPhoneTxCATUpdater::UpdateStateVectorsInternal(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      const std::vector< SpMatrix<double> > &H,
      double *auxf_impr,
      double *like_impr,
      int32 num_threads,
      int32 thread_id) const {
      
    int32 num_vectors = accs.num_cluster_weight_classes_ * accs.num_states_;
    int32 block_size = (num_vectors + (num_threads-1)) / num_threads;
    int32 n_start = block_size * thread_id;
    int32 n_end = std::min(num_vectors, n_start + block_size);

    for (int32 n = n_start; n < n_end; n++) {
      double state_count = 0.0, state_auxf_impr = 0.0, state_like_impr = 0.0;

      int32 r = n / accs.num_states_;
      int32 j = n % accs.num_states_;

      //TODO: gamma_[j] must be summed over the Gaussians in that cluster weight class
      double gamma_j = accs.gamma_[j].Sum();
      state_count += gamma_j;
      
      Vector<double> w_j;
      if(model->use_weight_projection_)
      {
        w_j.Resize(accs.num_gaussians_);
        //w_j.AddMatVec(1.0, Matrix<double>(model->w_), kNoTrans,
        //    Vector<double> (model->v_[j]), 0.0);
        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          int32 reg_class = model->cluster_weight_class_[i];
          w_j(i) = VecVec(Vector<double>(model->w_.Row(i)),
            Vector<double> (model->v_[reg_class][j]));
        }
        w_j.ApplySoftMax();
      }

      Vector<double> k_j(accs.y_[r][j]);
      SpMatrix<double> G_j(accs.num_clusters_);
      
      std::vector< Matrix<double> > M_SigmaInv_i;
      model->ComputeM_SigmaInv(&M_SigmaInv_i);

      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        if (model->cluster_weight_class_[i] != r) continue;

        double gamma_ji = accs.gamma_[j](i);
        
        if (gamma_ji != 0)
          G_j.AddSp(gamma_ji, H[i]);
        
        if (model->use_weight_projection_) {
          double quadratic_term = std::max(gamma_ji, gamma_j * w_j(i));
          double scalar = gamma_ji - gamma_j * w_j(i) + quadratic_term
            * VecVec(model->w_.Row(i), model->v_[r][j]);
          k_j.AddVec(scalar, model->w_.Row(i));
          
          if (quadratic_term > 1.0e-10) {
            G_j.AddVec2(static_cast<BaseFloat>(quadratic_term), model->w_.Row(i));
          }
        }
      }
 
      //(Deprecated)
      //{
      //  Vector<double> Z_ji(accs.Z_[j].Row(i));
      //  
      //  // k_{j} = Sum_i M_{i}^T SigmaInv_{i} Z_{ji}
      //  k_j.AddMatVec(1.0, M_SigmaInv_i[i], kNoTrans, Z_ji, 1.0);
      //}

      std::stringstream debug_info_term;
      debug_info_term << "vrj[" << r << "," << j << "]";
      
      Vector<double> vhat_j(model->v_[r][j]);
      double objf_impr_with_prior = 
        SolveQuadraticProblem(G_j, k_j, &vhat_j,
            static_cast<double>(update_options_.max_cond),
            static_cast<double>(update_options_.epsilon),
            debug_info_term.str().c_str(), true);

      SpMatrix<BaseFloat> G_j_flt(G_j);

      double objf_impr_noprior =
        (VecVec(vhat_j, k_j) 
         - 0.5 * VecSpVec(vhat_j, G_j, vhat_j))
        - (VecVec(model->v_[r][j],k_j)
            - 0.5 * VecSpVec(model->v_[r][j], G_j_flt, model->v_[r][j]));

      model->v_[r][j].CopyFromVec(vhat_j);

      if (r < 2 && j < 3 && thread_id == 0) {
        KALDI_LOG << "Objf impr for j = " << (j)
          << " r = " << r << " is "
          << (objf_impr_with_prior / (gamma_j + 1.0e-20))
          << " (with ad-hoc prior) "
          << (objf_impr_noprior / (gamma_j + 1.0e-20))
          << " (no prior) over " << (gamma_j) << " frames";
      }

      state_auxf_impr += objf_impr_with_prior;
      state_like_impr += objf_impr_noprior;

      *auxf_impr += state_auxf_impr;
      *like_impr += state_like_impr;
      if (j < 10 && thread_id == 0) {
        KALDI_LOG << "Objf impr for state j = " << (j) 
          << " r = " << r << "  is "
          << (state_auxf_impr / (state_count + 1.0e-20))
          << " (with ad-hoc prior) "
          << (state_like_impr / (state_count + 1.0e-20))
          << " (no prior) over " << (state_count) << " frames";
      }
    }
  }

  double MleAmPhoneTxCATUpdater::UpdateStateVectors(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      const vector< SpMatrix<double> > &H) {
    KALDI_LOG << "Updating state vectors";

    double count = 0.0, auxf_impr = 0.0, 
           like_impr = 0.0;  // sum over all states

    for (int32 j = 0; j < accs.num_states_; j++) 
      count += accs.gamma_[j].Sum();

    UpdateStateVectorsClass c(*this, accs, model, H, &auxf_impr, &like_impr);
    RunMultiThreaded(c);

    auxf_impr /= (count + 1.0e-20);
    like_impr /= (count + 1.0e-20);
    KALDI_LOG << "**Overall objf impr for v is " << auxf_impr
      << "(with ad-hoc prior) " << like_impr << " (no prior) over "
      << (count) << " frames";
    // Choosing to return actual likelihood impr here.
    return like_impr;
  }

  class UpdateASequentialClass: public MultiThreadable {
    public:
      UpdateASequentialClass(const MleAmPhoneTxCATUpdater &updater,
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          const int32 p,
          const std::vector< SpMatrix<double> > &xi2,
          const std::vector< Matrix<double> > &L,
          const Matrix<double> &variance, 
          const Vector<double> &class_count,
          std::vector<double> *init_step,
          double *auxf_impr,
          double *like_impr):
        updater_(updater), accs_(accs), model_(model),
        p_(p), xi2_(xi2), L_(L), variance_(variance), 
        class_count_(class_count), init_step_(init_step),
        auxf_impr_ptr_(auxf_impr), auxf_impr_(0.0),
        like_impr_ptr_(like_impr), like_impr_(0.0) { }

      ~UpdateASequentialClass() {
        *like_impr_ptr_ += like_impr_;
        *auxf_impr_ptr_ += auxf_impr_;
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to the total sums in the destructor.
        updater_.UpdateASequentialInternal(accs_, model_, p_,
            xi2_, L_, variance_,
            class_count_, init_step_, &auxf_impr_, &like_impr_,
            num_threads_, thread_id_);
      }

    private:
      const MleAmPhoneTxCATUpdater &updater_;
      const MleAmPhoneTxCATAccs &accs_;
      AmPhoneTxCAT *model_;
      const int32 p_;
      const std::vector< SpMatrix<double> > &xi2_;
      const std::vector< Matrix<double> > &L_;
      const Matrix<double> &variance_;
      const Vector<double> &class_count_;
      std::vector<double> *init_step_;
      double *auxf_impr_ptr_;
      double auxf_impr_;
      double *like_impr_ptr_;
      double like_impr_;
  };

  // Runs the transforms update for a subset of all q
  // where q is the transform class
  void MleAmPhoneTxCATUpdater::UpdateASequentialInternal(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      const int32 p,
      const std::vector< SpMatrix<double> > &xi2,
      const std::vector< Matrix<double> > &L,
      const Matrix<double> &variance,
      const Vector<double> &class_count,
      std::vector<double> *init_step,
      double *auxf_impr,
      double *like_impr,
      int32 num_threads,
      int32 thread_id) const {
    
    int32 num_vectors = model->NumTransformClasses();
    int32 block_size = (num_vectors + 
        (num_threads-1)) / num_threads,
    n_start = block_size * thread_id,
    n_end = std::min(num_vectors, n_start + block_size);
    
    for (int32 n = n_start; n < n_end; n++) {
      int32 tx_class = n;
      double class_like_impr = 0.0, class_auxf_impr = 0.0;
      
      double min_step = 0.0001;
      for (int32 k = 0; k < accs.feature_dim_; k++) {
        double step_size = (*init_step)[tx_class * accs.feature_dim_ + k];
        Vector<double> Apqk(model->A_[p][tx_class].Row(k));
        Vector<double> delta_Apqk(accs.feature_dim_+1);
        
        if (step_size < min_step) {
          if (p < 5) {
          KALDI_LOG << "Not updating Apqk[" << p << "," 
            << tx_class << "," << k << "]" 
            << " because initial step size " << step_size
            << " is lower than " << min_step;
          }
          continue;
        }

        SpMatrix<double> Gpqk(accs.feature_dim_+1);
        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          if (tx_class != model->transform_class_[i])
            continue;
          Gpqk.AddSp(accs.G_[i](p,p)/
              (variance(i,k)), 
              xi2[i]);
        }

        std::stringstream debug_info_term;
        debug_info_term << "A_pqk[" << p << "," << tx_class 
          << "," << k << "]";

        SolveQuadraticProblem(Gpqk, L[tx_class].Row(k),
            &delta_Apqk,
            static_cast<double>(update_options_.max_cond),
            static_cast<double>(update_options_.epsilon),
            debug_info_term.str().c_str(), true);

        try { // In case we have a problem in LogSub.
          for ( ; step_size >= min_step; step_size /= 2) {
            Vector<double> new_Apqk(Apqk);
            // copy it in case we do not commit this change.
            new_Apqk.AddVec(step_size, delta_Apqk);
            double predicted_impr = 
              step_size * VecVec(delta_Apqk, L[tx_class].Row(k)) -
              -0.5 * step_size * step_size * 
              VecSpVec(delta_Apqk, Gpqk, delta_Apqk);
            if (predicted_impr < -0.1) {
              KALDI_WARN << "Negative predicted auxf improvement " <<
                (predicted_impr) << 
                ", not updating transform for A" <<
                debug_info_term.str().c_str() <<
                " (either numerical problems or a code mistake.";
              break;
            }
            // Compute the observed objf change.
            double observed_impr = 0.0;

            Vector<double> step_delta_Apqk(delta_Apqk);
            step_delta_Apqk.Scale(step_size);

            //Matrix<double> new_Apq(model->A_[p][tx_class]);
            //new_Apq.Row(k).AddVec(1.0, step_delta_Apqk);
            //
            //Matrix<double> delta_Apq(accs.feature_dim_, accs.feature_dim_+1);
            //delta_Apq.Row(k).CopyFromVec(step_delta_Apqk);

            for (int32 i = 0; i < accs.num_gaussians_; i++) {
              if (tx_class != model->transform_class_[i])
                continue;
            
              double delta_mu_pk = VecVec(step_delta_Apqk, 
                  Vector<double> (model->u_.Row(i)));

              //Vector<double> delta_mu_p(accs.feature_dim_);
              //delta_mu_p.AddMatVec(1.0, delta_Apq, kNoTrans,  
              //    Vector<double> (model->u_.Row(i)),0.0);
              //
              //Vector<double> SInv_delta_mu(accs.feature_dim_);
              //SInv_delta_mu.AddSpVec(1.0, 
              //    SpMatrix<double> (model->SigmaInv_[i]),
              //    delta_mu_p, 0.0);

              observed_impr += delta_mu_pk * VecVec(
                  Vector<double>(accs.K_[i].Row(p)), 
                  Matrix<double>(model->SigmaInv_[i]).Row(k));

              //kaldi::ApproxEqual(delta_mu_pk * VecVec(Vector<double>(accs.K_[i].Row(p)), 
              //      Matrix<double>(model->SigmaInv_[i]).Row(k)), 
              //    VecVec(Vector<double>(accs.K_[i].Row(p)), SInv_delta_mu));
            
              Matrix<double> MiT(accs.num_clusters_, accs.feature_dim_);
              model->GetMTrans(i, &MiT);
            
              for (int32 q = 0; q < accs.num_clusters_; q++) {
                if(p==q) {
                  observed_impr -= accs.G_[i](p,p) * delta_mu_pk * 
                    VecVec(MiT.Row(p), Matrix<double>(model->SigmaInv_[i]).Row(k));

                  observed_impr -= 0.5 * accs.G_[i](p,p) *
                    delta_mu_pk * delta_mu_pk * static_cast<double>(model->SigmaInv_[i](k,k));

                  //Vector<double> new_mu_p(accs.feature_dim_);
                  //new_mu_p.AddMatVec(1.0, 
                  //    new_Apq, kNoTrans,
                  //    Vector<double> (model->u_.Row(i)), 0.0);
                  //kaldi::ApproxEqual(0.5 * accs.G_[i](p,p) * VecSpVec(new_mu_p, 
                  //      SpMatrix<double> (model->SigmaInv_[i]), new_mu_p) -
                  //    0.5 * accs.G_[i](p,p) * VecSpVec(MiT.Row(p), SpMatrix<double> (model->Sigma_Inv_[i]), MiT.Row(p)), 
                  //    accs.G_[i](p,p) * delta_mu_pk * VecVec(MiT.Row(p), Matrix<double>(model->SigmaInv_[i]).Row(k)) + 0.5 * accs.G_[i](p,p) *
                  //    delta_mu_pk * delta_mu_pk * static_cast<double>(model->SigmaInv_[i](k,k)));

                }
                else {
                  observed_impr -= accs.G_[i](p,q) * delta_mu_pk * 
                    VecVec(MiT.Row(q), Matrix<double>(model->SigmaInv_[i]).Row(k));
                  //kaldi::ApproxEqual(accs.G_[i](p,q)
                  //    * VecSpVec(delta_mu_p, SpMatrix<double> (model->SigmaInv_[i]), MiT.Row(q)),
                  //    accs.G_[i](p,q) * delta_mu_pk * 
                  //    VecVec(MiT.Row(q), Matrix<double>(model->SigmaInv_[i]).Row(k)));
                }
              }
            }

            if (observed_impr < 0.0) { // failed, so we reduce step size.
              // Does not print real log like impr. Need to divide appropriately.
              if (p < 5 && k < 10) {
                KALDI_LOG << "Updating transform row, for A_pqk[" << p << ","
                  << tx_class << "," << k << "]" << ", predicted auxf: " 
                  << (predicted_impr/(class_count(tx_class)/accs.feature_dim_ + 1.0e-20))
                  << ", observed "
                  << (observed_impr/(class_count(tx_class)/accs.feature_dim_ + 1.0e-20))
                  << " over " << class_count(tx_class)/accs.feature_dim_ << " frames. Reducing step size "
                  << "to " << (step_size/2);
              }
              if (predicted_impr / (class_count(tx_class)/accs.feature_dim_ + 1.0e-20) < 1.0e-07) {
                KALDI_WARN << "Not updating this transform as auxf decreased"
                  << " probably due to numerical issues (since small change).";
                break;
              }
            } else {
              class_auxf_impr += predicted_impr;
              class_like_impr += observed_impr;
              model->A_[p][tx_class].Row(k).CopyFromVec(new_Apqk);
              (*init_step)[tx_class * accs.feature_dim_ + k] = step_size; 
              break;
            }
          }
        } catch(...) {
          KALDI_WARN << "Warning: update for Apqk[" << p << "," 
            << tx_class << "," << k << "]"  
            << " failed, possible numerical problem.";
        }
      }

      // Does not print real log like impr. Need to divide appropriately.
      if (p < 5) {
        KALDI_LOG << "Updating Apq[" << p << "," << tx_class << "]" 
          << " gives predicted "
          << "per-frame like impr " << (class_auxf_impr / class_count(tx_class))
          << ", observed " << (class_like_impr / class_count(tx_class)) << ", over " 
          << class_count(tx_class) << " frames";
      }

      *auxf_impr += class_auxf_impr;
      *like_impr += class_like_impr;
    }
  }

  double MleAmPhoneTxCATUpdater::UpdateASequential(const MleAmPhoneTxCATAccs &accs, 
      AmPhoneTxCAT *model) {
    if (!model->use_full_covar_)
      return (this->UpdateA_DiagCov(accs, model));
   
    double count = 0.0;
    double auxf_impr = 0.0, like_impr = 0.0; // sum over all clusters
    for (int32 j = 0; j < accs.num_states_; j++) count += accs.gamma_[j].Sum();

    std::vector< Matrix<double> > K_SInv;
    K_SInv.resize(accs.num_gaussians_);
    
    std::vector< SpMatrix<double> > xi2;
    xi2.resize(accs.num_gaussians_);

    std::vector<std::vector< Matrix<double> > > SInv_KpT_uT;
    SInv_KpT_uT.resize(accs.num_gaussians_);

    Matrix<double> variance(accs.num_gaussians_, accs.feature_dim_);
    
    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      K_SInv[i].Resize(accs.num_clusters_, accs.feature_dim_);
      K_SInv[i].AddMatSp(1.0, Matrix<double> (accs.K_[i]), kNoTrans,  
          SpMatrix<double> (model->SigmaInv_[i]), 0.0); 
      
      xi2[i].Resize(accs.feature_dim_+1);
      xi2[i].AddVec2(1.0, Vector<double> (model->u_.Row(i)));
      
      int32 tx_class = model->transform_class_[i];

      SInv_KpT_uT[i].resize(accs.num_clusters_);
      for (int32 p = 0; p < accs.num_clusters_; p++) {
        Vector<double> mu_p(accs.feature_dim_);
        mu_p.AddMatVec(1.0, Matrix<double> (model->A_[p][tx_class]), kNoTrans, Vector<double> (model->u_.Row(i)), 0.0);
        SInv_KpT_uT[i][p].Resize(accs.feature_dim_, accs.feature_dim_+1);
        SInv_KpT_uT[i][p].AddVecVec(1.0, K_SInv[i].Row(p),
            Vector<double> (model->u_.Row(i)));
      }
      
      SpMatrix<double> Sigma_i(model->SigmaInv_[i]);
      Sigma_i.InvertDouble();
      for (int32 k = 0; k < accs.feature_dim_; k++) {
        variance(i,k) = Sigma_i(k,k);
      }
    }

    int32 num_tx_classes = model->NumTransformClasses();

    Vector<double> class_count(num_tx_classes);
    for (int32 j = 0; j < accs.num_states_; j++) {
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        class_count(model->transform_class_[i]) += accs.gamma_[j](i)/accs.feature_dim_;
      }
    }
    
    for (int32 p = 0; p < accs.num_clusters_; p++) {
      double cluster_auxf_impr = 0.0, cluster_like_impr = 0.0; // sum over all classes and feature dimensions
        
      std::vector<double> init_step(num_tx_classes*accs.feature_dim_);
      for (int32 n = 0; n < init_step.size(); n++) {
        init_step[n] = 1.0;
      }
      
      for (int32 iter = 0; 
          iter < update_options_.cluster_transforms_iters; iter++) {
        double this_like_impr = 0.0, this_auxf_impr = 0.0;
        
        std::vector< Matrix<double> > L;
        L.resize(num_tx_classes);
        
        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          int32 tx_class = model->transform_class_[i];
          if (L[tx_class].NumRows() == 0)
            L[tx_class].Resize(accs.feature_dim_, accs.feature_dim_+1);

          L[tx_class].AddMat(1.0, SInv_KpT_uT[i][p]);
          Matrix<double> A_xi2_sum(accs.feature_dim_, accs.feature_dim_+1);
          for (int32 l = 0; l < accs.num_clusters_; l++) {
            A_xi2_sum.AddMatSp(accs.G_[i](p,l), 
                Matrix<double> (model->A_[l][tx_class]), kNoTrans,
                xi2[i], 1.0);
          }
          L[tx_class].AddSpMat(-1.0, SpMatrix<double>(model->SigmaInv_[i]), 
              A_xi2_sum, kNoTrans , 1.0);
        }

        UpdateASequentialClass c(*this, accs, model, p,
            xi2, L, variance, class_count, &init_step,
            &this_auxf_impr, &this_like_impr);
        RunMultiThreaded(c);

        this_like_impr /= (count+1.0e-20);
        this_auxf_impr /= (count+1.0e-20);

        cluster_like_impr += this_like_impr;
        cluster_auxf_impr += this_like_impr;
      }
      auxf_impr += cluster_auxf_impr/accs.num_clusters_;
      like_impr += cluster_like_impr/accs.num_clusters_;
    }
    KALDI_LOG << "**Overall objf impr for A is " << auxf_impr
      << " (Predicted), " << like_impr << " (Actual) over "
      << (count) << " frames";
    
    // Choosing to return actual likelihood impr here.
    return like_impr;
  }

  class UpdateAParallelClass: public MultiThreadable {
    public:
      UpdateAParallelClass(const MleAmPhoneTxCATUpdater &updater,
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          std::vector<Matrix<double> > *delta_Ap,
          const int32 p,
          const std::vector< SpMatrix<double> > &xi2,
          const std::vector< Matrix<double> > &L,
          const Matrix<double> &variance):
        updater_(updater), accs_(accs), model_(model),
        delta_Ap_(delta_Ap), p_(p),
        xi2_(xi2), L_(L), variance_(variance) { }

      ~UpdateAParallelClass() {
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to the total sums in the destructor.
        updater_.UpdateAParallelInternal(accs_, model_, 
            delta_Ap_, p_, xi2_, L_, variance_,
            num_threads_, thread_id_);
      }

    private:
      const MleAmPhoneTxCATUpdater &updater_;
      const MleAmPhoneTxCATAccs &accs_;
      AmPhoneTxCAT *model_;
      std::vector< Matrix<double> > *delta_Ap_;
      const int32 p_;
      const std::vector< SpMatrix<double> > &xi2_;
      const std::vector< Matrix<double> > &L_;
      const Matrix<double> &variance_;
  };

  // Runs the transforms update for a subset of all q and k
  // where q is the transform class and k is a dimension in feature
  void MleAmPhoneTxCATUpdater::UpdateAParallelInternal(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      std::vector< Matrix<double> > *delta_Ap,
      const int32 p,
      const std::vector< SpMatrix<double> > &xi2,
      const std::vector< Matrix<double> > &L,
      const Matrix<double> &variance,
      int32 num_threads,
      int32 thread_id) const {
    
    int32 num_vectors = model->NumTransformClasses() * accs.feature_dim_;
    int32 block_size = (num_vectors + 
        (num_threads-1)) / num_threads,
    n_start = block_size * thread_id,
    n_end = std::min(num_vectors, n_start + block_size);
    
    for (int32 n = n_start; n < n_end; n++) {
      int32 tx_class = n / accs.feature_dim_;
      int32 k = n % accs.feature_dim_;

      Vector<double> Apqk(model->A_[p][tx_class].Row(k));
      Vector<double> delta_Apqk(accs.feature_dim_+1);

      SpMatrix<double> Gpqk(accs.feature_dim_+1);
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        if (tx_class != model->transform_class_[i])
          continue;
        Gpqk.AddSp(accs.G_[i](p,p)/
            (variance(i,k)), 
            xi2[i]);
      }

      std::stringstream debug_info_term;
      debug_info_term << "A_pqk[" << p << "," << tx_class 
        << "," << k << "]";

      SolveQuadraticProblem(Gpqk, L[tx_class].Row(k),
          &delta_Apqk,
          static_cast<double>(update_options_.max_cond),
          static_cast<double>(update_options_.epsilon),
          debug_info_term.str().c_str(), true);

      (*delta_Ap)[tx_class].Row(k).CopyFromVec(delta_Apqk);
      
      if (update_options_.use_diagonal_transform) {
        for (int32 d = 0; d < accs.feature_dim_; d++) {
          if ( d != k ) (*delta_Ap)[tx_class](k,d) = 0;
        }
      } else if (update_options_.use_block_diagonal_transform) {
        for (int32 d = 0; d < accs.feature_dim_; d++) {
          if ( k < accs.feature_dim_/3 ) {
            if ( d >= accs.feature_dim_/3 ) (*delta_Ap)[tx_class](k,d) = 0;
          } else if ( k < 2 * accs.feature_dim_/3 ) {
            if ( d < accs.feature_dim_/3 || 
                d >= 2*accs.feature_dim_/3 ) (*delta_Ap)[tx_class](k,d) = 0;
          } else {
            if ( d < 2*accs.feature_dim_/3 ) (*delta_Ap)[tx_class](k,d) = 0;
          }
        }
      }
      
    }
  }
  
  double MleAmPhoneTxCATUpdater::UpdateAParallel(const MleAmPhoneTxCATAccs &accs, 
      AmPhoneTxCAT *model) {

    if (!model->use_full_covar_)
      return (this->UpdateA_DiagCov(accs, model));
   
    double count = 0.0;
    double like_impr = 0.0; // sum over all clusters
    for (int32 j = 0; j < accs.num_states_; j++) count += accs.gamma_[j].Sum();

    std::vector< Matrix<double> > K_SInv;     // K_i * SigmaInv_i
    K_SInv.resize(accs.num_gaussians_);
    
    std::vector< Matrix<double> > SInv_mu_p;  // Sigma_inv_i * M_i^T
    SInv_mu_p.resize(accs.num_gaussians_);

    std::vector< SpMatrix<double> > xi2;      // zeta_i * zeta_i^T
    xi2.resize(accs.num_gaussians_);

    std::vector<std::vector< Matrix<double> > > SInv_KpT_uT;
    SInv_KpT_uT.resize(accs.num_gaussians_);

    Matrix<double> variance(accs.num_gaussians_, accs.feature_dim_);

    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      K_SInv[i].Resize(accs.num_clusters_, accs.feature_dim_);
      K_SInv[i].AddMatSp(1.0, Matrix<double> (accs.K_[i]), kNoTrans,  
          SpMatrix<double> (model->SigmaInv_[i]), 0.0); 
      
      xi2[i].Resize(accs.feature_dim_+1);
      xi2[i].AddVec2(1.0, Vector<double> (model->u_.Row(i)));
      
      int32 tx_class = model->transform_class_[i];

      SInv_mu_p[i].Resize(accs.num_clusters_, accs.feature_dim_);
      SInv_KpT_uT[i].resize(accs.num_clusters_);
      for (int32 p = 0; p < accs.num_clusters_; p++) {
        Vector<double> mu_p(accs.feature_dim_);
        mu_p.AddMatVec(1.0, Matrix<double> (model->A_[p][tx_class]), kNoTrans, Vector<double> (model->u_.Row(i)), 0.0);
        SInv_mu_p[i].Row(p).AddSpVec(1.0, 
            SpMatrix<double> (model->SigmaInv_[i]), mu_p, 0.0); 
        SInv_KpT_uT[i][p].Resize(accs.feature_dim_, accs.feature_dim_+1);
        SInv_KpT_uT[i][p].AddVecVec(1.0, K_SInv[i].Row(p),
            Vector<double> (model->u_.Row(i)));
      }

      SpMatrix<double> Sigma_i(model->SigmaInv_[i]);
      Sigma_i.InvertDouble();
      for (int32 k = 0; k < accs.feature_dim_; k++) {
        variance(i,k) = Sigma_i(k,k);
      }
    }

    int32 num_tx_classes = model->NumTransformClasses();
    
    for (int32 p = 0; p < accs.num_clusters_; p++) {
      std::vector< Matrix<double> > delta_Ap;
      delta_Ap.resize(num_tx_classes);
      
      double cluster_like_impr = 0.0;
      std::vector<double> step_size;
      
      if (update_options_.use_class_dep_steps) {
        for(int32 tx_class = 0; tx_class < num_tx_classes; tx_class++)
          step_size.push_back(1.0);
      } else
        step_size.push_back(1.0);
      
      double min_step = 0.0001;

      for (int32 iter = 0; 
          iter < update_options_.cluster_transforms_iters; iter++) {
        
        { // Check if any of the step_size is greater than min_step
          bool break_flag = true;
          std::vector<double>::iterator it = step_size.begin();
          for (; it < step_size.end(); it++) { 
            if (*it >= min_step) {
              break_flag = false;
              break;
            }
          }
          if (break_flag) {
            KALDI_LOG << "Not updating A_p[" << p << "]," 
              << " in iter " << iter 
              << " because initial step size " 
              << " is lower than " << min_step;
            break;  
            // break out of the iter loop. 
            // no more iterations can be done as step_size is less than min_step
          }
        }

        double this_like_impr = 0.0;

        std::vector< Matrix<double> > L;
        L.resize(num_tx_classes);
        
        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          int32 tx_class = model->transform_class_[i];
          if (L[tx_class].NumRows() == 0) {
            L[tx_class].Resize(accs.feature_dim_, accs.feature_dim_+1);
            delta_Ap[tx_class].Resize(accs.feature_dim_, accs.feature_dim_+1);
          }

          L[tx_class].AddMat(1.0, SInv_KpT_uT[i][p]);
          Matrix<double> A_xi2_sum(accs.feature_dim_, accs.feature_dim_+1);
          for (int32 l = 0; l < accs.num_clusters_; l++) {
            A_xi2_sum.AddMatSp(accs.G_[i](p,l), 
                Matrix<double> (model->A_[l][tx_class]), kNoTrans,
                xi2[i], 1.0);
          }
          L[tx_class].AddSpMat(-1.0, SpMatrix<double>(model->SigmaInv_[i]), 
              A_xi2_sum, kNoTrans , 1.0);
        }

        KALDI_LOG << "Stats accumulated for cluster " << p << "\n";

        UpdateAParallelClass c(*this, accs, model, &delta_Ap, p,
            xi2, L, variance);
        RunMultiThreaded(c);

        
        if (!update_options_.use_class_dep_steps) { // use same step size for all transform classes

          for (;step_size[0] >= min_step; step_size[0] /= 2) {

            // Compute the observed objf change.
            double observed_impr = 0.0;

            for (int32 i = 0; i < accs.num_gaussians_; i++) {
              int32 tx_class = model->transform_class_[i];

              Vector<double> delta_mu_p(accs.feature_dim_);
              delta_mu_p.AddMatVec(step_size[0], delta_Ap[tx_class], kNoTrans,
                  Vector<double> (model->u_.Row(i)), 0.0);

              observed_impr += VecVec(K_SInv[i].Row(p), delta_mu_p);

              Matrix<double> MiT(accs.num_clusters_, accs.feature_dim_);
              model->GetMTrans(i, &MiT);

              for  (int32 q = 0; q < accs.num_clusters_; q++) {
                if (p == q) {
                  observed_impr -= accs.G_[i](p,p) * VecSpVec(delta_mu_p,
                      SpMatrix<double> (model->SigmaInv_[i]),
                      MiT.Row(p));
                  observed_impr -= 0.5 * accs.G_[i](p,p) * 
                    VecSpVec(delta_mu_p, 
                        SpMatrix<double> (model->SigmaInv_[i]),
                        delta_mu_p);
                } else {
                  observed_impr -= accs.G_[i](p,q) * 
                    VecSpVec(delta_mu_p, 
                        SpMatrix<double> (model->SigmaInv_[i]),
                        MiT.Row(q));
                }
              }
            }

            if (observed_impr < 0.0) { // failed, so we reduce step size.
              // Does not print real log like impr. Need to divide appropriately.
              KALDI_LOG << "Updating A_p[" << p << "]"
                << ", observed "
                << (observed_impr/(count + 1.0e-20))
                << " over " << count << " frames. Reducing step size "
                << "to " << (step_size[0]/2);
            } else {
              for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {
                model->A_[p][tx_class].AddMat(step_size[0], 
                    Matrix<BaseFloat> (delta_Ap[tx_class]));
              }
              this_like_impr += observed_impr;
              break;
            }
          } // end step_size for loop
        } else {
          
          int32 t_class = 0;
          while (true) {
            // Compute the observed objf change.
            double observed_impr = 0.0;

            for (int32 i = 0; i < accs.num_gaussians_; i++) {
              int32 tx_class = model->transform_class_[i];

              Vector<double> delta_mu_p(accs.feature_dim_);
              delta_mu_p.AddMatVec(step_size[tx_class], delta_Ap[tx_class], kNoTrans,
                  Vector<double> (model->u_.Row(i)), 0.0);

              observed_impr += VecVec(K_SInv[i].Row(p), delta_mu_p);

              Matrix<double> MiT(accs.num_clusters_, accs.feature_dim_);
              model->GetMTrans(i, &MiT);

              for  (int32 q = 0; q < accs.num_clusters_; q++) {
                if (p == q) {
                  observed_impr -= accs.G_[i](p,p) * VecSpVec(delta_mu_p,
                      SpMatrix<double> (model->SigmaInv_[i]),
                      MiT.Row(p));
                  observed_impr -= 0.5 * accs.G_[i](p,p) * 
                    VecSpVec(delta_mu_p, 
                        SpMatrix<double> (model->SigmaInv_[i]),
                        delta_mu_p);
                } else {
                  observed_impr -= accs.G_[i](p,q) * 
                    VecSpVec(delta_mu_p, 
                        SpMatrix<double> (model->SigmaInv_[i]),
                        MiT.Row(q));
                }
              }
            }

            if (observed_impr < 0.0) { // failed, so we reduce step size.
              // Does not print real log like impr. Need to divide appropriately.
              KALDI_LOG << "Updating A_p[" << p << "]"
                << ", observed "
                << (observed_impr/(count + 1.0e-20))
                << " over " << count << " frames. Reducing step size for " << t_class << " class "
                << "to " << (step_size[t_class]/2);
            } else {
              for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {
                model->A_[p][tx_class].AddMat(step_size[tx_class], 
                    Matrix<BaseFloat> (delta_Ap[tx_class]));
              }
              this_like_impr += observed_impr;
              break; // out of the while (true) loop
            }
            
            step_size[t_class] /= 2;
            t_class = (t_class + 1) % num_tx_classes;

            { // Check if all step_sizes are less than min_step
              bool break_flag = true;

              // Break if step_size for all classes are less than min step
              std::vector<double>::iterator it = step_size.begin();
              for (; it < step_size.end(); it++) {
                if (*it >= min_step) {
                  break_flag = false;
                  break;
                }
              }

              if (break_flag) // break_flag is true if all t_classes have step_size less than min_step 
                break; // out of the while (true) loop
            }

            // If step_size for this t_class is less than min step, go to the next t_class
            // Repeat until you find a t_class whose step_size is still greater then min_step
            while (step_size[t_class] < min_step) {
              t_class = (t_class + 1) % num_tx_classes;
            }
          } // end while loop
        } // end if

        this_like_impr /= (count+1.0e-20);
        cluster_like_impr += this_like_impr;
        
        if (update_options_.use_diagonal_transform3) {
          for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {
            for (int32 k = 0; k < accs.feature_dim_; k++) {
              for (int32 d = 0; d < accs.feature_dim_; d++) {
                if ( d != k ) model->A_[p][tx_class](k,d) = 0;
              }
            }
          }
        } else if (update_options_.use_block_diagonal_transform3) {
          for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {
            for (int32 k = 0; k < accs.feature_dim_; k++) {
              for (int32 d = 0; d < accs.feature_dim_; d++) {
                if ( k < accs.feature_dim_/3 ) {
                  if ( d >= accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
                } else if ( k < 2 * accs.feature_dim_/3 ) {
                  if ( d < accs.feature_dim_/3 || 
                      d >= 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
                } else {
                  if ( d < 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
                }
              }
            }
          }
        }
      }

      if (update_options_.use_diagonal_transform2) {
        for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {
          for (int32 k = 0; k < accs.feature_dim_; k++) {
            for (int32 d = 0; d < accs.feature_dim_; d++) {
              if ( d != k ) model->A_[p][tx_class](k,d) = 0;
            }
          }
        }
      } else if (update_options_.use_block_diagonal_transform2) {
        for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {
          for (int32 k = 0; k < accs.feature_dim_; k++) {
            for (int32 d = 0; d < accs.feature_dim_; d++) {
              if ( k < accs.feature_dim_/3 ) {
                if ( d >= accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              } else if ( k < 2 * accs.feature_dim_/3 ) {
                if ( d < accs.feature_dim_/3 || 
                    d >= 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              } else {
                if ( d < 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              }
            }
          }
        }
      }

      like_impr += cluster_like_impr;
    }

    KALDI_LOG << "**Overall objf impr for A is " 
      << like_impr << " over "
      << (count) << " frames";

    return like_impr;
  }
  
  class UpdateAParallelClass2: public MultiThreadable {
    public:
      UpdateAParallelClass2(const MleAmPhoneTxCATUpdater &updater,
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          Matrix<double> *delta_Apq,
          const int32 p,
          const int32 tx_class,
          const std::vector< SpMatrix<double> > &xi2,
          const Matrix<double> &L,
          const Matrix<double> &variance):
        updater_(updater), accs_(accs), model_(model),
        delta_Apq_(delta_Apq), p_(p), tx_class_(tx_class),
        xi2_(xi2), L_(L), variance_(variance) { }

      ~UpdateAParallelClass2() {
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to the total sums in the destructor.
        updater_.UpdateAParallelInternal2(accs_, model_, 
            delta_Apq_, p_, tx_class_, xi2_, L_, variance_,
            num_threads_, thread_id_);
      }

    private:
      const MleAmPhoneTxCATUpdater &updater_;
      const MleAmPhoneTxCATAccs &accs_;
      AmPhoneTxCAT *model_;
      Matrix<double> *delta_Apq_;
      const int32 p_;
      const int32 tx_class_;
      const std::vector< SpMatrix<double> > &xi2_;
      const Matrix<double> &L_;
      const Matrix<double> &variance_;
  };

  // Runs the transforms update for a subset of all k
  // where k is a dimension in feature
  void MleAmPhoneTxCATUpdater::UpdateAParallelInternal2(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      Matrix<double> *delta_Apq,
      const int32 p,
      const int32 tx_class,
      const std::vector< SpMatrix<double> > &xi2,
      const Matrix<double> &L,
      const Matrix<double> &variance,
      int32 num_threads,
      int32 thread_id) const {
    
    int32 num_vectors = accs.feature_dim_;
    int32 block_size = (num_vectors + 
        (num_threads-1)) / num_threads,
    n_start = block_size * thread_id,
    n_end = std::min(num_vectors, n_start + block_size);
    
    for (int32 n = n_start; n < n_end; n++) {
      int32 k = n % accs.feature_dim_;

      Vector<double> Apqk(model->A_[p][tx_class].Row(k));
      Vector<double> delta_Apqk(accs.feature_dim_+1);

      SpMatrix<double> Gpqk(accs.feature_dim_+1);
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        if (tx_class != model->transform_class_[i])
          continue;
        Gpqk.AddSp(accs.G_[i](p,p)/
            (variance(i,k)), 
            xi2[i]);
      }

      std::stringstream debug_info_term;
      debug_info_term << "A_pqk[" << p << "," << tx_class 
        << "," << k << "]";

      SolveQuadraticProblem(Gpqk, L.Row(k),
          &delta_Apqk,
          static_cast<double>(update_options_.max_cond),
          static_cast<double>(update_options_.epsilon),
          debug_info_term.str().c_str(), true);

      (*delta_Apq).Row(k).CopyFromVec(delta_Apqk);
      
      if (update_options_.use_diagonal_transform) {
        for (int32 d = 0; d < accs.feature_dim_; d++) {
          if ( d != k ) (*delta_Apq)(k,d) = 0;
        }
      } else if (update_options_.use_block_diagonal_transform) {
        for (int32 d = 0; d < accs.feature_dim_; d++) {
          if ( k < accs.feature_dim_/3 ) {
            if ( d >= accs.feature_dim_/3 ) (*delta_Apq)(k,d) = 0;
          } else if ( k < 2 * accs.feature_dim_/3 ) {
            if ( d < accs.feature_dim_/3 || 
                d >= 2*accs.feature_dim_/3 ) (*delta_Apq)(k,d) = 0;
          } else {
            if ( d < 2*accs.feature_dim_/3 ) (*delta_Apq)(k,d) = 0;
          }
        }
      }
    }
  }
  
  double MleAmPhoneTxCATUpdater::UpdateAParallel2(const MleAmPhoneTxCATAccs &accs, 
      AmPhoneTxCAT *model) {

    if (!model->use_full_covar_)
      return (this->UpdateA_DiagCov(accs, model));
   
    double count = 0.0;
    double like_impr = 0.0; // sum over all clusters
    for (int32 j = 0; j < accs.num_states_; j++) count += accs.gamma_[j].Sum();

    std::vector< Matrix<double> > K_SInv;     // K_i * SigmaInv_i
    K_SInv.resize(accs.num_gaussians_);
    
    std::vector< Matrix<double> > SInv_mu_p;  // Sigma_inv_i * M_i^T
    SInv_mu_p.resize(accs.num_gaussians_);

    std::vector< SpMatrix<double> > xi2;      // zeta_i * zeta_i^T
    xi2.resize(accs.num_gaussians_);

    std::vector<std::vector< Matrix<double> > > SInv_KpT_uT;
    SInv_KpT_uT.resize(accs.num_gaussians_);

    Matrix<double> variance(accs.num_gaussians_, accs.feature_dim_);

    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      K_SInv[i].Resize(accs.num_clusters_, accs.feature_dim_);
      K_SInv[i].AddMatSp(1.0, Matrix<double> (accs.K_[i]), kNoTrans,  
          SpMatrix<double> (model->SigmaInv_[i]), 0.0); 
      
      xi2[i].Resize(accs.feature_dim_+1);
      xi2[i].AddVec2(1.0, Vector<double> (model->u_.Row(i)));
      
      int32 tx_class = model->transform_class_[i];

      SInv_mu_p[i].Resize(accs.num_clusters_, accs.feature_dim_);
      SInv_KpT_uT[i].resize(accs.num_clusters_);
      for (int32 p = 0; p < accs.num_clusters_; p++) {
        Vector<double> mu_p(accs.feature_dim_);
        mu_p.AddMatVec(1.0, Matrix<double> (model->A_[p][tx_class]), kNoTrans, Vector<double> (model->u_.Row(i)), 0.0);
        SInv_mu_p[i].Row(p).AddSpVec(1.0, 
            SpMatrix<double> (model->SigmaInv_[i]), mu_p, 0.0); 
        SInv_KpT_uT[i][p].Resize(accs.feature_dim_, accs.feature_dim_+1);
        SInv_KpT_uT[i][p].AddVecVec(1.0, K_SInv[i].Row(p),
            Vector<double> (model->u_.Row(i)));
      }

      SpMatrix<double> Sigma_i(model->SigmaInv_[i]);
      Sigma_i.InvertDouble();
      for (int32 k = 0; k < accs.feature_dim_; k++) {
        variance(i,k) = Sigma_i(k,k);
      }
    }

    int32 num_tx_classes = model->NumTransformClasses();
    
    for (int32 p = 0; p < accs.num_clusters_; p++) {
      for (int32 tx_class = 0; tx_class < num_tx_classes; tx_class++) {

        Matrix<double> delta_Apq;
      
        double transform_like_impr = 0.0;
        double step_size = 1.0;

        double min_step = 0.0001;

        for (int32 iter = 0; 
            iter < update_options_.cluster_transforms_iters; iter++) {
        
          if (step_size < min_step) {
            KALDI_LOG << "Not updating A_p[" << p << "]," 
              << " in iter " << iter 
              << " because initial step size " 
              << " is lower than " << min_step;
            break;  
          }

          double this_like_impr = 0.0;

          Matrix<double> L;
        
          for (int32 i = 0; i < accs.num_gaussians_; i++) {
            if (tx_class != model->transform_class_[i]) continue;
            if (L.NumRows() == 0) {
              L.Resize(accs.feature_dim_, accs.feature_dim_+1);
              delta_Apq.Resize(accs.feature_dim_, accs.feature_dim_+1);
            }

            L.AddMat(1.0, SInv_KpT_uT[i][p]);
            Matrix<double> A_xi2_sum(accs.feature_dim_, accs.feature_dim_+1);
            for (int32 l = 0; l < accs.num_clusters_; l++) {
              A_xi2_sum.AddMatSp(accs.G_[i](p,l), 
                  Matrix<double> (model->A_[l][tx_class]), kNoTrans,
                  xi2[i], 1.0);
            }
            L.AddSpMat(-1.0, SpMatrix<double>(model->SigmaInv_[i]), 
                A_xi2_sum, kNoTrans , 1.0);
          }

        KALDI_LOG << "Stats accumulated for cluster " << p << ", class " << tx_class << "\n";

        UpdateAParallelClass2 c(*this, accs, model, &delta_Apq, p, tx_class,
            xi2, L, variance);
        RunMultiThreaded(c);
        
        for (;step_size >= min_step; step_size /= 2) {

          // Compute the observed objf change.
          double observed_impr = 0.0;

          for (int32 i = 0; i < accs.num_gaussians_; i++) {
            if (tx_class != model->transform_class_[i]) continue;

            Vector<double> delta_mu_pq(accs.feature_dim_);
            delta_mu_pq.AddMatVec(step_size, delta_Apq, kNoTrans,
                Vector<double> (model->u_.Row(i)), 0.0);

            observed_impr += VecVec(K_SInv[i].Row(p), delta_mu_pq);

            Matrix<double> MiT(accs.num_clusters_, accs.feature_dim_);
            model->GetMTrans(i, &MiT);

            for  (int32 q = 0; q < accs.num_clusters_; q++) {
              if (p == q) {
                observed_impr -= accs.G_[i](p,p) * VecSpVec(delta_mu_pq,
                    SpMatrix<double> (model->SigmaInv_[i]),
                    MiT.Row(p));
                observed_impr -= 0.5 * accs.G_[i](p,p) * 
                  VecSpVec(delta_mu_pq, 
                      SpMatrix<double> (model->SigmaInv_[i]),
                      delta_mu_pq);
              } else {
                observed_impr -= accs.G_[i](p,q) * 
                  VecSpVec(delta_mu_pq, 
                      SpMatrix<double> (model->SigmaInv_[i]),
                      MiT.Row(q));
              }
            }
          }

          if (observed_impr < 0.0) { // failed, so we reduce step size.
            // Does not print real log like impr. Need to divide appropriately.
            KALDI_LOG << "Updating A_pq[" << p << "," << tx_class << "]"
              << ", observed "
              << (observed_impr/(count + 1.0e-20))
              << " over " << count << " frames. Reducing step size "
              << "to " << (step_size/2);
          } else {
            model->A_[p][tx_class].AddMat(step_size, 
                Matrix<BaseFloat> (delta_Apq));
            this_like_impr += observed_impr;
            break;
          }
        } // end step_size for loop

        this_like_impr /= (count+1.0e-20);
        transform_like_impr += this_like_impr;
        
        if (update_options_.use_diagonal_transform3) {
          for (int32 k = 0; k < accs.feature_dim_; k++) {
            for (int32 d = 0; d < accs.feature_dim_; d++) {
              if ( d != k ) model->A_[p][tx_class](k,d) = 0;
            }
          }
        } else if (update_options_.use_block_diagonal_transform3) {
          for (int32 k = 0; k < accs.feature_dim_; k++) {
            for (int32 d = 0; d < accs.feature_dim_; d++) {
              if ( k < accs.feature_dim_/3 ) {
                if ( d >= accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              } else if ( k < 2 * accs.feature_dim_/3 ) {
                if ( d < accs.feature_dim_/3 || 
                    d >= 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              } else {
                if ( d < 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              }
            }
          }
        }
        }

        if (update_options_.use_diagonal_transform2) {
          for (int32 k = 0; k < accs.feature_dim_; k++) {
            for (int32 d = 0; d < accs.feature_dim_; d++) {
              if ( d != k ) model->A_[p][tx_class](k,d) = 0;
            }
          }
        } else if (update_options_.use_block_diagonal_transform2) {
          for (int32 k = 0; k < accs.feature_dim_; k++) {
            for (int32 d = 0; d < accs.feature_dim_; d++) {
              if ( k < accs.feature_dim_/3 ) {
                if ( d >= accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              } else if ( k < 2 * accs.feature_dim_/3 ) {
                if ( d < accs.feature_dim_/3 || 
                    d >= 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              } else {
                if ( d < 2*accs.feature_dim_/3 ) model->A_[p][tx_class](k,d) = 0;
              }
            }
          }
        }

        like_impr += transform_like_impr;
      } // end tx_class loop
    } // end cluster loop

    KALDI_LOG << "**Overall objf impr for A is " 
      << like_impr << " over "
      << (count) << " frames";

    return like_impr;
  }

  class UpdateA_DiagCovParallelClass: public MultiThreadable {
    public: 
      UpdateA_DiagCovParallelClass(const MleAmPhoneTxCATUpdater &updater,
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model, 
          const int32 p,
          double *like_impr):
        updater_(updater), accs_(accs), model_(model),
        p_(p),
        like_impr_ptr_(like_impr), like_impr_(0.0) { }

      ~UpdateA_DiagCovParallelClass() {
        *like_impr_ptr_ += like_impr_;
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to the total sums in the destructor.
        updater_.UpdateA_DiagCovParallelInternal(accs_, model_, 
            p_, &like_impr_,
            num_threads_, thread_id_);
      }

    private:
      const MleAmPhoneTxCATUpdater &updater_;
      const MleAmPhoneTxCATAccs &accs_;
      AmPhoneTxCAT *model_;
      const int32 p_;
      double *like_impr_ptr_;
      double like_impr_;
  };

  // Runs the transforms update for a subset of all q and k
  // where q is the transform class and k is a dimension in feature
  void MleAmPhoneTxCATUpdater::UpdateA_DiagCovParallelInternal(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model,
      const int32 p,
      double *like_impr,
      int32 num_threads,
      int32 thread_id) const {
    
    int32 num_vectors = model->NumTransformClasses() * accs.feature_dim_;
    int32 block_size = (num_vectors + 
        (num_threads-1)) / num_threads,
    n_start = block_size * thread_id,
    n_end = std::min(num_vectors, n_start + block_size);
    
    for (int32 n = n_start; n < n_end; n++) {
      int32 tx_class = n / accs.feature_dim_;
      int32 k = n % accs.feature_dim_;
      
      double class_like_impr = 0.0;

      Vector<double> k_pqk;
      SpMatrix<double> G_pqk;
      k_pqk.Resize(model->A_[p][tx_class].NumCols());
      G_pqk.Resize(model->A_[p][tx_class].NumCols());
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        if (model->transform_class_[i] != tx_class)
          continue;
        Vector<double> ui(model->u_.Row(i));
        k_pqk.AddVec(accs.K_[i](p,k) * model->SigmaInv_[i](k,k), ui);
        for (int32 l = 0; l < accs.num_clusters_; l++) {
          if (l != p) {
            k_pqk.AddVec(-accs.G_[i](l,p) * 
                VecVec(model->A_[l][tx_class].Row(k),ui) *
                model->SigmaInv_[i](k,k), ui);
          }
        }
        G_pqk.AddVec2(accs.G_[i](p,p) * model->SigmaInv_[i](k,k), ui);
      }

      Vector<double> A_pqk(model->A_[p][tx_class].Row(k));

      std::stringstream debug_info_term;
      debug_info_term << "A_pqk[" << p << "," << tx_class << "," << k << "]";

      //impr += SolveQuadraticProblem(G_pk, k_pk, &A_pk,
      //    1.0e+05, 1.0e-40, debug_info_term.str().c_str(), true);
      class_like_impr += SolveQuadraticProblem(G_pqk, k_pqk, &A_pqk,
          static_cast<double>(update_options_.max_cond), 
          static_cast<double>(update_options_.epsilon), 
          debug_info_term.str().c_str(), true);

      model->A_[p][tx_class].Row(k).CopyFromVec(A_pqk);

      *like_impr += class_like_impr;
    }
  }


  double MleAmPhoneTxCATUpdater::UpdateA_DiagCov(const MleAmPhoneTxCATAccs &accs, 
      AmPhoneTxCAT *model) {
    double tot_count = 0.0, tot_like_impr = 0.0; // sum over all clusters
    //int32 num_transform_classes = model->NumTransformClasses();
    for (int32 p = 0; p < accs.num_clusters_; p++) {
      //for (int32 q = 0; q < num_transform_classes; q++) {
      double cluster_like_impr = 0;

      UpdateA_DiagCovParallelClass c(*this, accs, model, p,
          &cluster_like_impr);
      RunMultiThreaded(c);
        //for (int32 k = 0; k < accs.feature_dim_; k++) {
        //  Vector<double> k_pqk;
        //  SpMatrix<double> G_pqk;
        //  k_pqk.Resize(model->A_[p][q].NumCols());
        //  G_pqk.Resize(model->A_[p][q].NumCols());
        //  for (int32 i = 0; i < accs.num_gaussians_; i++) {
        //    if (model->transform_class_[i] != q)
        //      continue;
        //    Vector<double> ui(model->u_.Row(i));
        //    k_pqk.AddVec(accs.K_[i](p,k) * model->SigmaInv_[i](k,k), ui);
        //    for (int32 l = 0; l < accs.num_clusters_; l++) {
        //      if (l != p) {
        //        k_pqk.AddVec(-accs.G_[i](l,p) * 
        //            VecVec(model->A_[l][q].Row(k),ui) *
        //            model->SigmaInv_[i](k,k), ui);
        //      }
        //    }
        //    G_pqk.AddVec2(accs.G_[i](p,p) * model->SigmaInv_[i](k,k), ui);
        //  }
        //
        //  Vector<double> A_pqk(model->A_[p][q].Row(k));
        //
        //  std::stringstream debug_info_term;
        //  debug_info_term << "A_pqk[" << p << "," << q << "," << k << "]";
        //
        //  //impr += SolveQuadraticProblem(G_pk, k_pk, &A_pk,
        //  //    1.0e+05, 1.0e-40, debug_info_term.str().c_str(), true);
        //  impr += SolveQuadraticProblem(G_pqk, k_pqk, &A_pqk,
        //      static_cast<double>(update_options_.max_cond), 
        //      static_cast<double>(update_options_.epsilon), 
        //      debug_info_term.str().c_str(), true);
        //
        //  model->A_[p][q].Row(k).CopyFromVec(A_pqk);
        //}
      tot_like_impr += cluster_like_impr;
      //}
    }
    for (int32 j = 0; j < accs.num_states_; j++)
      tot_count += accs.gamma_[j].Sum();
    tot_like_impr /= (tot_count+1.0e-20);
    KALDI_LOG << "**Overall objf impr for A is " << tot_like_impr << " over "
      << tot_count << " frames";
    return tot_like_impr;
  }

  double MleAmPhoneTxCATUpdater::UpdateVars(const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model) {
    KALDI_LOG << "Update Covariance matrix";
    SpMatrix<double> Sigma_i(accs.feature_dim_), Sigma_i_ml(accs.feature_dim_);
    double tot_t = 0.0, tot_objf_impr = 0.0;
    SpMatrix<double> covfloor(accs.feature_dim_);
    Vector<double> gamma_vec(accs.num_gaussians_);
    Vector<double> objf_improv(accs.num_gaussians_);
  
    // First pass over all (shared) Gaussian components to calculate the
    // ML estimate of the covariances, and the total covariance for flooring.
    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      double gamma_i = 0.0;
      for (int32 j = 0; j < accs.num_states_; j++) {
        gamma_i += accs.gamma_[j](i);
      }

      Sigma_i_ml.SetZero();
      {
        Sigma_i_ml.AddSp(1.0, accs.L_[i]);

        std::vector< Vector<double> > mu_i;
        mu_i.resize(accs.num_clusters_);

        {
          for (int32 p = 0; p < accs.num_clusters_; p++) {
            if (mu_i[p].IsZero())
            {
              int32 tx_class = model->transform_class_[i];
              mu_i[p].Resize(accs.feature_dim_);
              mu_i[p].AddMatVec(1.0, Matrix<double>(model->A_[p][tx_class]), kNoTrans, 
                  Vector<double>(model->u_.Row(i)), 0.0);
            }
            Sigma_i_ml.AddVecVec(-1.0, accs.K_[i].Row(p), mu_i[p]);
          }
        }

        {
          Matrix<double> tmp(accs.feature_dim_, accs.feature_dim_);
          for (int32 p = 0; p < accs.num_clusters_; p++) {
            for (int32 q = 0; q < accs.num_clusters_; q++) {
              tmp.AddVecVec(accs.G_[i](p,q), mu_i[p], mu_i[q]);
            }
          }
          SpMatrix<double> mu_mu_T(accs.feature_dim_);
          mu_mu_T.CopyFromMat(tmp, kTakeMeanAndCheck);
          Sigma_i_ml.AddSp(1.0, mu_mu_T);
        }
      }
      
      gamma_vec(i) = gamma_i;
      // Accumulate gamma_i*Sigma_i_ml
      covfloor.AddSp(1.0, Sigma_i_ml);
      
      // So smoothing is done on updating Sigma_i_ml
      // to avoid inf in Sigma_i_ml
      if (gamma_i > 1.0e-20) {
        Sigma_i_ml.Scale(1 / (gamma_i + 1.0e-20));
      } else { 
        Sigma_i_ml.SetUnit();
      }
      KALDI_ASSERT(1.0 / Sigma_i_ml(0,0) != 0.0);
    
      // Compute the objective function with the old parameter values
      objf_improv(i) = model->SigmaInv_[i].LogPosDefDet() -
        TraceSpSp(SpMatrix<double>(model->SigmaInv_[i]), Sigma_i_ml);
      
      model->SigmaInv_[i].CopyFromSp(Sigma_i_ml);  // inverted in the next loop.
    }

    // Compute the covariance floor.
    if (gamma_vec.Sum() == 0) {  // If no count, use identity.
      KALDI_WARN << "Updating variances: zero counts. Setting floor to unit.";
      covfloor.SetUnit();
    } else {  // else, use the global average covariance.
      // scale the accumulated covfloor by f/ (sum_i gamma_i) 
      covfloor.Scale(update_options_.cov_floor / gamma_vec.Sum());
      int32 tmp;
      if ((tmp = covfloor.LimitCondDouble(update_options_.max_cond)) != 0) {
        KALDI_WARN << "Covariance flooring matrix is poorly conditioned. Fixed "
          << "up " << (tmp) << " eigenvalues.";
      }
    }
    
    if (update_options_.cov_diag_ratio > 1000 || !model->use_full_covar_) {
      KALDI_LOG << "Assuming you want to build a diagonal system since "
        << "cov_diag_ratio is large: making diagonal covFloor.";
      for (int32 i = 0; i < covfloor.NumRows(); i++)
        for (int32 j = 0; j < i; j++)
          covfloor(i, j) = 0.0;
    }
  
    // Second pass over all (shared) Gaussian components to calculate the
    // floored estimate of the covariances, and update the model.
    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      Sigma_i.CopyFromSp(model->SigmaInv_[i]);
      Sigma_i_ml.CopyFromSp(Sigma_i);
      // In case of insufficient counts, make the covariance matrix diagonal.
      // cov_diag_ratio is 2 by default, 
      // set to very large to always get diag-cov

      if (!model->use_full_covar_ || 
          gamma_vec(i) < update_options_.cov_diag_ratio * accs.feature_dim_) {
        if (model->use_full_covar_) {
          KALDI_WARN << "For Gaussian component " << i << ": Too low count "
            << gamma_vec(i) << " for covariance matrix estimation."
            << "Setting to diagonal";
        }

        for (int32 d = 0; d < accs.feature_dim_; d++)
          for (int32 e = 0; e < d; e++)
            Sigma_i(d, e) = 0.0;  // SpMatrix, can only set lower traingular part

        int floored = Sigma_i.ApplyFloor(covfloor);
        if (floored > 0) {
          KALDI_WARN << "For Gaussian component " << i 
            << ": Floored " << floored
            << " covariance eigenvalues.";
        }
        model->SigmaInv_[i].CopyFromSp(Sigma_i);
        model->SigmaInv_[i].InvertDouble();
        
        objf_improv(i) += Sigma_i.LogPosDefDet() +
          TraceSpSp(SpMatrix<double>(model->SigmaInv_[i]), Sigma_i_ml);
        objf_improv(i) *= (-0.5 * gamma_vec(i));  
        tot_objf_impr += objf_improv(i);
        tot_t += gamma_vec(i);
        if (i < 5) {
          KALDI_VLOG(2) << "objf impr from variance update =" 
            << objf_improv(i)
            / (gamma_vec(i) + 1.0e-20) << " over " << (gamma_vec(i))
            << " frames for i = " << (i);
        }
      } else { /// Updating the full covariance matrix.
        try {
          int floored = Sigma_i.ApplyFloor(covfloor);
          if (floored > 0) {
            KALDI_WARN << "For Gaussian component " << i << ": Floored "
              << floored << " covariance eigenvalues.";
          }

          model->SigmaInv_[i].CopyFromSp(Sigma_i);
          model->SigmaInv_[i].InvertDouble();
          
          objf_improv(i) += Sigma_i.LogPosDefDet() +
            TraceSpSp(SpMatrix<double>(model->SigmaInv_[i]), Sigma_i_ml);
          objf_improv(i) *= (-0.5 * gamma_vec(i));  
          tot_objf_impr += objf_improv(i);
          tot_t += gamma_vec(i);
          if (i < 5) {
            KALDI_VLOG(2) << "objf impr from variance update =" 
              << objf_improv(i)
              / (gamma_vec(i) + 1.0e-20) << " over " << (gamma_vec(i))
              << " frames for i = " << (i);
          }
        } catch(...) {
          KALDI_WARN << "Updating within-class covariance matrix i = " 
            << (i)
            << ", numerical problem";
          // This is a catch-all thing in case of unanticipated errors, but
          // flooring should prevent this occurring for the most part.
          model->SigmaInv_[i].SetUnit();  // Set to unit.
        }
      }
    }
    
    KALDI_LOG << "**Overall objf impr for variance update = "
      << (tot_objf_impr / (tot_t+ 1.0e-20))
      << " over " << (tot_t) << " frames";
    return tot_objf_impr / (tot_t + 1.0e-20);
  }
  
  // Deprecated. Was the UpdateVars() function in the older version
  // that dealt with only diagonal covariances
  double MleAmPhoneTxCATUpdater::UpdateDiagVars(const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model) {
    KALDI_LOG << "Update Covariance matrix";
    SpMatrix<double> Sigma_i(accs.feature_dim_), Sigma_i_ml(accs.feature_dim_);
    double tot_t = 0.0, tot_objf_impr = 0.0;
    SpMatrix<double> covfloor(accs.feature_dim_);
    Vector<double> gamma_vec(accs.num_gaussians_);
    Vector<double> objf_improv(accs.num_gaussians_);
  
    // First pass over all (shared) Gaussian components to calculate the
    // ML estimate of the covariances, and the total covariance for flooring.
    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      double gamma_i = 0.0;
      for (int32 j = 0; j < accs.num_states_; j++) {
        gamma_i += accs.gamma_[j](i);
      }

      Sigma_i_ml.SetZero();
      gamma_vec(i) = gamma_i;

      Sigma_i_ml.AddSp(1.0, accs.L_[i]);
      
      std::vector< Vector<double> > mu_i;
      mu_i.resize(accs.num_clusters_);
      
      {
        for (int32 p = 0; p < accs.num_clusters_; p++) {
          if (mu_i[p].IsZero())
          {
            Vector<BaseFloat> mu_p(accs.feature_dim_);
            int32 q = model->transform_class_[i];
            mu_p.AddMatVec(1.0, model->A_[p][q], kNoTrans, 
                model->u_.Row(i), 0.0);
            mu_i[p].Resize(accs.feature_dim_);
            mu_i[p].CopyFromVec(mu_p);
          }
          Sigma_i_ml.AddVecVec(-1.0, accs.K_[i].Row(p), mu_i[p]);
        }
      }

      {
        Matrix<double> tmp(accs.feature_dim_, accs.feature_dim_);
        for (int32 p = 0; p < accs.num_clusters_; p++) 
          for (int32 q = 0; q < accs.num_clusters_; q++) {
            tmp.AddVecVec(accs.G_[i](p,q), mu_i[p], mu_i[q]);
          }
        SpMatrix<double> mu_mu_T(accs.feature_dim_);
        mu_mu_T.CopyFromMat(tmp, kTakeMeanAndCheck);
        Sigma_i_ml.AddSp(1.0, mu_mu_T);
      }

      covfloor.AddSp(1.0, Sigma_i_ml);
      
      if (gamma_i > 1.0e-20)
        Sigma_i_ml.Scale(1 / (gamma_i + 1.0e-20));
      else 
        Sigma_i_ml.SetUnit();
      KALDI_ASSERT(1.0 / Sigma_i_ml(0,0) != 0.0);
    
      // Compute the objective function with the old parameter values
      objf_improv(i) = model->SigmaInv_[i].LogPosDefDet() -
        TraceSpSp(SpMatrix<double>(model->SigmaInv_[i]), Sigma_i_ml);
      
      model->SigmaInv_[i].CopyFromSp(Sigma_i_ml);  // inverted in the next loop.
    }

    // Compute the covariance floor.
    if (gamma_vec.Sum() == 0) {  // If no count, use identity.
      KALDI_WARN << "Updating variances: zero counts. Setting floor to unit.";
      covfloor.SetUnit();
    } else {  // else, use the global average covariance.
      // scale the accumulated covfloor by f/ (sum_i gamma_i) 
      covfloor.Scale(update_options_.cov_floor / gamma_vec.Sum());
      int32 tmp;
      if ((tmp = covfloor.LimitCondDouble(update_options_.max_cond)) != 0) {
        KALDI_WARN << "Covariance flooring matrix is poorly conditioned. Fixed "
          << "up " << (tmp) << " eigenvalues.";
      }
    }
    
    for (int32 i = 0; i < covfloor.NumRows(); i++)
      for (int32 j = 0; j < i; j++)
        covfloor(i, j) = 0.0;

    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      Sigma_i.CopyFromSp(model->SigmaInv_[i]);
      Sigma_i_ml.CopyFromSp(Sigma_i);

      for (int32 d = 0; d < accs.feature_dim_; d++)
        for (int32 e = 0; e < d; e++)
          Sigma_i(d, e) = 0.0;  // SpMatrix, can only set lower traingular part

      int floored = Sigma_i.ApplyFloor(covfloor);
      if (floored > 0) {
        KALDI_WARN << "For Gaussian component " << i 
          << ": Floored " << floored
          << " covariance eigenvalues.";
      }
      model->SigmaInv_[i].CopyFromSp(Sigma_i);
      model->SigmaInv_[i].InvertDouble();

      tot_t += gamma_vec(i);
    }
    
    KALDI_LOG << "**Overall objf impr for variance update = "
      << (tot_objf_impr / (tot_t+ 1.0e-20))
      << " over " << (tot_t) << " frames";
    return tot_objf_impr / (tot_t + 1.0e-20);
  }

  double MleAmPhoneTxCATUpdater::UpdateCanonicalMeans(
      const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model) {
    KALDI_LOG << "Updating canonical means";
    Vector<double> u_i(accs.feature_dim_), u_i_ml(accs.feature_dim_);
    double tot_t = 0.0, tot_objf_impr = 0.0;

    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      double gamma_i = 0.0;
      for (int32 j = 0; j < accs.num_states_; j++) {
        gamma_i += accs.gamma_[j](i);
        tot_t += gamma_i;
      }

      int32 tx_class = model->transform_class_[i];

      Vector<double> k_i(accs.feature_dim_);
      SpMatrix<double> G_i(accs.feature_dim_);

      std::vector<Matrix<double> > A;
      std::vector<Vector<double> > b;

      A.resize(accs.num_clusters_);
      b.resize(accs.num_clusters_);

      Matrix<double> G_tmp(accs.feature_dim_, accs.feature_dim_);
      std::vector<Matrix<double> > Ap_SigmaInv;
      Ap_SigmaInv.resize(accs.num_clusters_);

      for (int32 p = 0; p < accs.num_clusters_; p++) {
        for (int32 q = 0; q < accs.num_clusters_; q++) {
          if (p == 0) {
            A[q].Resize(accs.feature_dim_, accs.feature_dim_);
            A[q].CopyFromMat(model->A_[q][tx_class].Range(0, accs.feature_dim_,
                  0, accs.feature_dim_));
            Matrix<double> tmp(model->A_[q][tx_class]);
            tmp.Transpose();
            b[q].Resize(accs.feature_dim_);
            b[q].CopyFromVec(tmp.Row(accs.feature_dim_));
            Ap_SigmaInv[q].Resize(accs.feature_dim_, accs.feature_dim_);
            Ap_SigmaInv[q].AddMatSp(1.0, A[q], kTrans, 
                SpMatrix<double>(model->SigmaInv_[i]), 0.0);
          }
          G_tmp.AddMatMat(accs.G_[i](p,q), Ap_SigmaInv[p], 
              kNoTrans, A[q], kNoTrans, 1.0);
          k_i.AddMatVec(-accs.G_[i](p,q), Ap_SigmaInv[p], 
              kNoTrans, b[q], 1.0); 
        }
        k_i.AddMatVec(1.0, Ap_SigmaInv[p], 
            kNoTrans, accs.K_[i].Row(p), 1.0);
      }
      G_i.CopyFromMat(G_tmp, kTakeMeanAndCheck);

      SubVector<BaseFloat> zetahat_i(model->u_.Row(i));
      Vector<double> uhat_i(zetahat_i.Range(0, accs.feature_dim_));

      std::stringstream debug_info_term;
      debug_info_term << "u_i[" << i << "]";

      tot_objf_impr +=
        SolveQuadraticProblem(G_i, k_i, &uhat_i,
            static_cast<double>(update_options_.max_cond),
            static_cast<double>(update_options_.epsilon),
            debug_info_term.str().c_str(), true);

      SubVector<BaseFloat> ui(zetahat_i.Range(0, accs.feature_dim_));
      ui.CopyFromVec(uhat_i);
    }
    tot_objf_impr /= (tot_t+1.0e-20);
    KALDI_LOG << "**Overall objf impr for u is " << tot_objf_impr << " over "
      << tot_t << " frames";
    return tot_objf_impr;
  }

  double MleAmPhoneTxCATUpdater::UpdateWeights(
      const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model) {
    
    KALDI_LOG << "Updating mixture weights";

    if (model->use_weight_projection_) {
      if (update_options_.use_sequential_weight_update) {
        return (this->UpdateWSequential(accs, model));
      } else {
        return (this->UpdateWParallel(accs, model));
      }
    }
    
    if (model->NumWeightClasses() == model->NumPdfs()) {
      return (this->UpdateWeightsBasic(accs, model));
    }
    
    int32 num_weight_classes = model->NumWeightClasses();

    // Also set the vector gamma_j which is a cache of the state occupancies
    gamma_j_.Resize(accs.num_states_);
    // Also set the vector gamma_p_i which is an accumulation of the weight class occupancies
    Matrix<double> gamma_p_i;
    gamma_p_i.Resize(num_weight_classes, accs.num_gaussians_);

    double tot_gamma = 0.0, objf_impr = 0.0;
    
    for (int32 j = 0; j < accs.num_states_; j++) {
      gamma_j_(j) = accs.gamma_[j].Sum();
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        gamma_p_i(static_cast<size_t> (model->weights_map_[j]),i) += accs.gamma_[j](i);
      }
    }

    for (int32 p = 0; p < num_weight_classes; p++) {
      SubVector<double> gamma_p(gamma_p_i.Row(p));
      double occ_p = gamma_p.Sum();
      
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        double cur_weight = model->w_(p,i);
        if (cur_weight <= update_options_.min_gaussian_weight) {
          KALDI_WARN << "Zero or negative weight, flooring";
          cur_weight = update_options_.min_gaussian_weight;
        }
        double prob;

        if (occ_p > 0.0)
          prob = gamma_p_i(p,i)/occ_p;
        else
          prob = 1.0 / accs.num_gaussians_;
        
        if (prob > update_options_.min_gaussian_weight)
        {
          model->w_(p,i) = prob;
          objf_impr += log(model->w_(p,i) / cur_weight) * gamma_p_i(p,i);
        }
      }
      double sum_weights = model->w_.Row(p).Sum();
      KALDI_ASSERT(sum_weights > update_options_.min_gaussian_weight);
      if (sum_weights != 1) {
        KALDI_WARN << "Normalizing gaussian weights for weight class p = " << p; 
        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          model->w_(p,i) /= sum_weights;
        }
      }
      tot_gamma += occ_p;
    }

    objf_impr /= (tot_gamma + 1.0e-20);
    KALDI_LOG << "**Overall objf impr for w is " << objf_impr << ", over "
      << tot_gamma << " frames.";
    return objf_impr;
  }

  
  double MleAmPhoneTxCATUpdater::UpdateWeightsBasic(
      const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model) {
    // Also set the vector gamma_j which is a cache of the state occupancies
    gamma_j_.Resize(accs.num_states_);

    double tot_gamma = 0.0, objf_impr = 0.0;
    for (int32 j = 0; j < accs.num_states_; j++) {
      gamma_j_(j) = accs.gamma_[j].Sum();

      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        double cur_weight = model->w_(j,i);
        if (cur_weight <= update_options_.min_gaussian_weight) {
          KALDI_WARN << "Zero or negative weight, flooring";
          cur_weight = update_options_.min_gaussian_weight;
        }
        double prob;

        if (gamma_j_(j) > 0.0)
          prob = accs.gamma_[j](i)/gamma_j_(j);
        else
          prob = 1.0 / accs.num_gaussians_;

        if (prob > update_options_.min_gaussian_weight)
        {
          model->w_(j,i) = prob;
          objf_impr += log(model->w_(j,i) / cur_weight) * accs.gamma_[j](i);
        }
      }
      double sum_weights = model->w_.Row(j).Sum();
      KALDI_ASSERT(sum_weights > update_options_.min_gaussian_weight);
      if (sum_weights != 1) {
        KALDI_WARN << "Normalizing gaussian weights for state j = " << j; 
        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          model->w_(j,i) /= sum_weights;
        }
      }
      tot_gamma += gamma_j_(j);
    }

    objf_impr /= (tot_gamma + 1.0e-20);
    KALDI_LOG << "**Overall objf impr for w is " << objf_impr << ", over "
      << tot_gamma << " frames.";
    return objf_impr;
  }

  double MleAmPhoneTxCATUpdater::UpdateWSequential(
      const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model) {
    // Sequential approach as described in the SGMM paper

    KALDI_LOG << "Updating weight projections [original approach, checking each"
      << "Gaussian component].";

    SpMatrix<double> v_vT(accs.num_clusters_);

    // tot_like_{after, before} are totals over multiple iterations,
    // not valid likelihoods...
    // but difference is valid (when divided by tot_count).
    double tot_delta_predicted = 0.0, tot_delta_observed = 0.0,
           tot_count = 0.0;

    Vector<double> w_j(accs.num_gaussians_);
    Vector<double> g_i(accs.num_clusters_);
    SpMatrix<double> F_i(accs.num_clusters_);

    double k_count = 0.0;
    std::vector< double > gamma_j(accs.num_states_);

    for (int32 j = 0; j < accs.num_states_; j++) {   // Initialize gamma_j
      gamma_j[j] = accs.gamma_[j].Sum();
    }

    Matrix<double> w(model->w_);

    for (int32 iter = 0; iter < 
        update_options_.weight_projections_iters; iter++) {
      double k_delta_predicted = 0.0, k_delta_observed = 0.0;

      // log total of un-normalized weights for each j
      std::vector< double > weight_tots(accs.num_states_);

      // Initialize weight_tots
      for (int32 j = 0; j < accs.num_states_; j++) {
        for (int32 i = 0; i < accs.num_gaussians_; i ++) {
          int32 r = model->cluster_weight_class_[i];
          w_j(i) = VecVec(w.Row(i), Vector<double>(model->v_[r][j]));
        }
        weight_tots[j] = w_j.LogSumExp();
      }

      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        F_i.SetZero();
        g_i.SetZero();
        SubVector<double> w_i = w.Row(i);
        int32 r = model->cluster_weight_class_[i];

        for (int32 j = 0; j < accs.num_states_; j++) {
          double this_unnormalized_weight = VecVec(w_i, model->v_[r][j]);
          double normalizer = weight_tots[j];
          double this_log_w = this_unnormalized_weight - normalizer,
                 this_w = exp(this_log_w),
                 state_count = gamma_j[j],
                 this_count = accs.gamma_[j](i);
          double linear_term = this_count - state_count * this_w;
          double quadratic_term = std::max(this_count, state_count * this_w);

          g_i.AddVec(linear_term, model->v_[r][j]);

          if (quadratic_term != 0.0)
            F_i.AddVec2(static_cast<BaseFloat>(quadratic_term), model->v_[r][j]);
        }

        // auxf is formulated in terms of change in w.
        Vector<double> delta_w(accs.num_clusters_);
        // returns objf impr with step_size = 1,
        // but it may not be 1 so we recalculate it.

        SolveQuadraticProblem(F_i, g_i, &delta_w,
            static_cast<double>(update_options_.max_cond),
            static_cast<double>(update_options_.epsilon),
            "w", true);

        try {
          double step_size, min_step = 0.0001;
          for (step_size = 1.0; step_size >= min_step; step_size /= 2) {
            Vector<double> new_w_i(w_i);

            std::vector< double > new_weight_tots(weight_tots);
            new_w_i.AddVec(step_size, delta_w);
            double predicted_impr = step_size * VecVec(delta_w, g_i) -
              0.5 * step_size * step_size * VecSpVec(delta_w,  F_i, delta_w);
            if (predicted_impr < -0.1) {
              KALDI_WARN << "Negative predicted auxf improvement " <<
                (predicted_impr) << ", not updating this gaussian " <<
                "(either numerical problems or a code mistake.";
              break;
            }

            // Now compute observed objf change.
            double observed_impr = 0.0, this_tot_count = 0.0;

            for (int32 j = 0; j < accs.num_states_; j++) {
              double old_unnorm_weight = VecVec(w_i, model->v_[r][j]),
                     new_unnorm_weight = VecVec(new_w_i, model->v_[r][j]),
                     state_count = gamma_j[j],
                     this_count = accs.gamma_[j](i);

              this_tot_count += this_count;
              observed_impr += this_count *
                (new_unnorm_weight - old_unnorm_weight);
              double old_normalizer = new_weight_tots[j], delta;

              if (new_unnorm_weight > old_unnorm_weight) {
                delta = LogAdd(0, LogSub(new_unnorm_weight - old_normalizer,
                      old_unnorm_weight - old_normalizer));
              } else {
                delta = LogSub(0, LogSub(old_unnorm_weight - old_normalizer,
                      new_unnorm_weight - old_normalizer));
                // The if-statement above is equivalent to:
                // delta = LogAdd(LogSub(0,
                // old_unnorm_weight-old_normalizer),
                // new_unnorm_weight-old_normalizer)
                // but has better behaviour numerically.
              }
              observed_impr -= state_count * delta;
              new_weight_tots[j] += delta;
            }
            if (observed_impr < 0.0) {  // failed, so we reduce step size.
              KALDI_LOG << "Updating weights, for i = " << (i) << ", predicted "
                "auxf: " << (predicted_impr/(this_tot_count + 1.0e-20))
                << ", observed " << observed_impr/(this_tot_count + 1.0e-20)
                << " over " << this_tot_count << " frames. Reducing step size "
                << "to " << (step_size/2);
              if (predicted_impr / (this_tot_count + 1.0e-20) < 1.0e-07) {
                KALDI_WARN << "Not updating this weight vector as auxf decreased"
                  << " probably due to numerical issues (since small change).";
                break;
              }
            } else {
              if (i < 10)
                KALDI_LOG << "Updating weights, for i = " << (i)
                  << ", auxf change per frame is" << ": predicted " <<
                  (predicted_impr /(this_tot_count + 1.0e-20)) << ", observed "
                  << (observed_impr / (this_tot_count + 1.0e-20))
                  << " over " << (this_tot_count) << " frames.";

              k_delta_predicted += predicted_impr;
              k_delta_observed += observed_impr;
              w.Row(i).CopyFromVec(new_w_i);
              weight_tots = new_weight_tots;  // Copy over normalizers.
              break;
            }
          }
        } catch(...) {
          KALDI_LOG << "Warning: weight update for i = " << i
            << " failed, possible numerical problem.";
        }
      }
      KALDI_LOG << "For iteration " << iter << ", updating w gives predicted "
        << "per-frame like impr " << (k_delta_predicted / k_count) <<
        ", observed " << (k_delta_observed / k_count) << ", over " << (k_count)
        << " frames";
      if (iter == 0) tot_count += k_count;
      tot_delta_predicted += k_delta_predicted;
      tot_delta_observed += k_delta_observed;
    }

    model->w_.CopyFromMat(w);

    tot_delta_observed /= tot_count;
    tot_delta_predicted /= tot_count;
    KALDI_LOG << "**Overall objf impr for w is " << tot_delta_predicted
      << ", observed " << tot_delta_observed << ", over "
      << tot_count << " frames";
    return tot_delta_observed;
  }

  /// This function gets stats used inside UpdateWParallel, 
  /// where it accumulates
  /// the F_i and g_i quantities.  
  /// Note: F_i is viewed as a vector of SpMatrix
  /// (one for each i); each row of F_i is viewed as an 
  /// SpMatrix even though
  /// it's stored as a vector....
  /// Note: w is just a double-precision copy of the matrix model->w_

  void MleAmPhoneTxCATUpdater::UpdateWParallelGetStats(
      const MleAmPhoneTxCATAccs &accs,
      const AmPhoneTxCAT &model,
      const Matrix<double> &w,
      Matrix<double> *F_i,
      Matrix<double> *g_i,
      double *tot_like,
      int32 num_threads,
      int32 thread_id) {

    // Accumulate stats from a block of states 
    int32 block_size = (accs.num_states_ + (num_threads-1)) / num_threads,
    j_start = block_size * thread_id,
    j_end = std::min(accs.num_states_, j_start + block_size);

    SpMatrix<double> v_vT(accs.num_clusters_);
    
    for (int32 j = j_start; j < j_end; j++) {
      Vector<double> w_j(accs.num_gaussians_);
      // The linear term and quadratic term for each Gaussian-- 
      // two scalar for each Gaussian, 
      // they appear in the accumulation formulas.
      // w_j = softmax([w_{k1}^T ... w_{kD}^T] * v_{rjk})
      
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        int32 r = model.cluster_weight_class_[i];
        w_j(i) = VecVec(w.Row(i), Vector<double>(model.v_[r][j]));
      }

      double gamma_j = accs.gamma_[j].Sum();

      w_j.Add(-1.0 * w_j.LogSumExp());
      *tot_like += VecVec(w_j, accs.gamma_[j]);
      w_j.ApplyExp();
      
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        int32 r = model.cluster_weight_class_[i];

        v_vT.SetZero();
        // v_vT := v_{r}{j} v_{r}{j}^T (SpMatrix)
        v_vT.AddVec2(static_cast<BaseFloat>(1.0), Vector<double>(model.v_[r][j]));
      
        Vector<double> v_vT_vec(
            accs.num_clusters_*(accs.num_clusters_+1)/2);
        v_vT_vec.CopyFromPacked(v_vT);

        double linear_term = accs.gamma_[j](i) - gamma_j * w_j(i);
        double quadratic_term = std::max(accs.gamma_[j](i), gamma_j * w_j(i));
        
        g_i->Row(i).AddVec(linear_term, Vector<double>(model.v_[r][j]));
        F_i->Row(i).AddVec(quadratic_term, v_vT_vec);

      }
      //g_i->AddVecVec(1.0, linear_term, v_j_double);
      //F_i->AddVecVec(1.0, quadratic_term, v_vT_vec);
    } // loop over states
  }

  double MleAmPhoneTxCATUpdater::UpdateWParallel(
      const MleAmPhoneTxCATAccs &accs,
      AmPhoneTxCAT *model) {
    KALDI_LOG << "Update weight projections";

    // tot_like_{after, before} are totals over multiple iterations,
    // not valid likelihoods. but difference is valid (when divided by tot_count).
    double tot_predicted_like_impr = 0.0, tot_like_before = 0.0,
           tot_like_after = 0.0;

    Matrix<double> g_i(accs.num_gaussians_, accs.num_clusters_);
    // View F_i as a vector of SpMatrix.
    Matrix<double> F_i(accs.num_gaussians_,
        (accs.num_clusters_*(accs.num_clusters_+1))/2);

    Matrix<double> w(model->w_);
    double tot_count = 0.0;
    for (int32 j = 0; j < accs.num_states_; j++) 
      tot_count += accs.gamma_[j].Sum();

    for (int iter = 0; iter < 
        update_options_.weight_projections_iters; iter++) {
      F_i.SetZero();
      g_i.SetZero();
      double k_like_before = 0.0;

      UpdateWParallelClass c(accs, *model, w, &F_i, &g_i, &k_like_before);
      RunMultiThreaded(c);

      Matrix<double> w_orig(w);
      double k_predicted_like_impr = 0.0, k_like_after = 0.0;
      double min_step = 0.001, step_size;
      for (step_size = 1.0; step_size >= min_step; step_size /= 2) {
        k_predicted_like_impr = 0.0;
        k_like_after = 0.0;

        for (int32 i = 0; i < accs.num_gaussians_; i++) {
          // auxf is formulated in terms of change in w.
          Vector<double> delta_w(accs.num_clusters_);
          // returns objf impr with step_size = 1,
          // but it may not be 1 so we recalculate it.
          SpMatrix<double> this_F_i(accs.num_clusters_);
          this_F_i.CopyFromVec(F_i.Row(i));
          SolveQuadraticProblem(this_F_i, g_i.Row(i), &delta_w,
              static_cast<double>(update_options_.max_cond),
              static_cast<double>(update_options_.epsilon),
              "w",
              true);

          delta_w.Scale(step_size);
          double predicted_impr = VecVec(delta_w, g_i.Row(i)) -
            0.5 * VecSpVec(delta_w,  this_F_i, delta_w);

          // should never be negative because
          // we checked inside SolveQuadraticProblem.
          KALDI_ASSERT(predicted_impr >= -1.0e-05);

          if (i < 10) {
            KALDI_LOG 
              << "Predicted objf impr for w (not per frame), iter = " <<
              (iter) << ", i = " << (i) << " is " << (predicted_impr);
          }
          k_predicted_like_impr += predicted_impr;
          w.Row(i).AddVec(1.0, delta_w);
        }
        Vector<double> w_j_vec(accs.num_gaussians_);
        for (int32 j = 0; j < accs.num_states_; j++) {
          for (int32 i = 0; i < accs.num_gaussians_; i++) {
            int32 r = model->cluster_weight_class_[i];
            w_j_vec(i) = VecVec(w.Row(i), Vector<double>(model->v_[r][j]));
          }
          w_j_vec.Add((-1.0) * w_j_vec.LogSumExp());
          k_like_after += VecVec(w_j_vec, accs.gamma_[j]);
        }
        KALDI_VLOG(2) << "For iteration " << (iter) << ", updating w gives "
          << "predicted per-frame like impr "
          << (k_predicted_like_impr / tot_count) << ", actual "
          << ((k_like_after - k_like_before) / tot_count) << ", over "
          << (tot_count) << " frames";

        if (k_like_after < k_like_before) {
          w.CopyFromMat(w_orig);  // Undo what we computed.
          if (fabs(k_like_after - k_like_before) / tot_count < 1.0e-05) {
            k_like_after = k_like_before;
            KALDI_WARN << "Not updating weights as not increasing auxf and "
              << "probably due to numerical issues (since small change).";
            break;
          } else {
            KALDI_WARN << "Halving step size for weights as likelihood did "
              << "not increase";
          }
        } else {
          break;
        }
      }
      if (step_size < min_step) {
        // Undo any step as we have no confidence that this is right.
        w.CopyFromMat(w_orig);
      } else {
        tot_predicted_like_impr += k_predicted_like_impr;
        tot_like_after += k_like_after;
        tot_like_before += k_like_before;
      }
    }

    model->w_.CopyFromMat(w);

    tot_predicted_like_impr /= tot_count;
    tot_like_after = (tot_like_after - tot_like_before) / tot_count;
    KALDI_LOG << "**Overall objf impr for w is " << tot_predicted_like_impr
      << ", actual " << tot_like_after << ", over "
      << tot_count << " frames";
    return tot_like_after;
  }

} //end namespace kaldi

