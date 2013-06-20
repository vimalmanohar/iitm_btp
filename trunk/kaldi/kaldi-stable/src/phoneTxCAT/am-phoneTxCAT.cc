#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>
using std::vector;

#include "phoneTxCAT/am-phoneTxCAT.h"
#include "thread/kaldi-thread.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "tree/clusterable-classes.h"
#include "tree/cluster-utils.h"

namespace kaldi {

  PhoneTxCATUpdateFlagsType StringToPhoneTxCATUpdateFlags(std::string str) {
    PhoneTxCATUpdateFlagsType flags = 0;
    for (const char *c = str.c_str(); *c != '\0'; c++) {
      switch (*c) {
        case 'v': flags |= kPhoneTxCATClusterWeights; break;    //State vectors
        case 'A': flags |= kPhoneTxCATClusterTransforms; break;
        case 'w': flags |= kPhoneTxCATWeights; break;
        case 'S': flags |= kPhoneTxCATCovarianceMatrix; break;
        case 't': flags |= kPhoneTxCATTransitions; break;
        case 'u': flags |= kPhoneTxCATCanonicalMeans; break;
        case 'a': flags |= kPhoneTxCATAll; break;
        default: KALDI_ERR << "Invalid element " << CharToString(*c)
                 << " of PhoneTxCATUpdateFlagsType option string "
                   << str;
      }
    }
    return flags;
  }

  PhoneTxCATUpdateFlagsType StringToPhoneTxCATWriteFlags(std::string str) {
    PhoneTxCATWriteFlagsType flags = 0;
    for (const char *c = str.c_str(); *c != '\0'; c++) {
      switch (*c) {
        case 'g': flags |= kPhoneTxCATGlobalParams; break;
        case 's': flags |= kPhoneTxCATStateParams; break;
        case 'n': flags |= kPhoneTxCATNormalizers; break;
        case 'u': flags |= kPhoneTxCATBackgroundGmms; break;
        case 'a': flags |= kPhoneTxCATAll; break;
        default: KALDI_ERR << "Invalid element " << CharToString(*c)
                 << " of PhoneTxCATWriteFlagsType option string "
                   << str;
      }
    }
    return flags;
  }
    
  void AmPhoneTxCAT::Read(std::istream &in_stream, bool binary) {
    int32 num_states, feat_dim, num_gauss, num_clusters, 
          num_weight_classes = -1, num_transform_classes = 1,
          num_cluster_weight_classes = 1;
    std::string token;

    ExpectToken(in_stream, binary, "<PhoneTxCAT>");
    ExpectToken(in_stream, binary, "<NUMSTATES>");
    ReadBasicType(in_stream, binary, &num_states);
    ExpectToken(in_stream, binary, "<DIMENSION>");
    ReadBasicType(in_stream, binary, &feat_dim);
    KALDI_ASSERT(num_states > 0 && feat_dim > 0);

    ReadToken(in_stream, binary, &token);

    use_full_covar_ = false;

    while (token != "</PhoneTxCAT>") {
      if (token == "<NUMWeightClasses>") {
        ReadBasicType(in_stream, binary, &num_weight_classes);
      } else if (token == "<NUMClusterWeightClasses>") {
        ReadBasicType(in_stream, binary, &num_cluster_weight_classes);
      } else if (token == "<FULLC>") {
        use_full_covar_ = true;
      } else if (token == "<DIAGC>") {
        use_full_covar_ = false;
      } else if (token == "<FULL_UBM>") {
        full_ubm_.Read(in_stream, binary);
      } else if (token == "<DIAG_UBM>") {
        diag_ubm_.Read(in_stream, binary);
      } else if (token == "<SigmaInv>") {
        ExpectToken(in_stream, binary, "<NUMGaussians>");
        ReadBasicType(in_stream, binary, &num_gauss);
        SigmaInv_.resize(num_gauss);
        for (int32 i = 0; i < num_gauss; i++) {
          SigmaInv_[i].Read(in_stream, binary);
        }
      } else if (token == "<u>") {
        u_.Resize(num_gauss, feat_dim);
        u_.Read(in_stream, binary);
      } else if (token == "<A>") {
        ReadToken(in_stream, binary, &token);
        // For compatibility with older models,
        // NUMTransformClasses is written before NUMClusters
        // so that if NUMTransformClasses not seen, then it is 
        // implied that an older model is seen
        if (token == "<NUMTransformClasses>") {
          ReadBasicType(in_stream, binary, &num_transform_classes);
          ExpectToken(in_stream, binary, "<NUMClusters>");
          ReadBasicType(in_stream, binary, &num_clusters);
        }
        else if (token == "<NUMClusters>") {
          ReadBasicType(in_stream, binary, &num_clusters);
          num_transform_classes = 1;
        }
        else
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        A_.resize(num_clusters);
        for (int32 p = 0; p < num_clusters; p++) {
          A_[p].resize(num_transform_classes);
          for (int32 q = 0; q < num_transform_classes; q++) {
            A_[p][q].Read(in_stream, binary);
          }
        }
      } else if (token == "<w>") {
        Matrix<BaseFloat> tmp;
        tmp.Read(in_stream, binary);
        // Back compatipility. When w_ was a vector of Vectors.
        if (tmp.NumRows() == 1 && tmp.NumRows() != num_gauss) {
          use_weight_projection_ = false;
          if (num_weight_classes <= 0)
            num_weight_classes = num_states;
          w_.Resize(num_weight_classes, num_gauss);
          w_.CopyRowFromVec(tmp.Row(0), 0);
          for (int32 j = 1; j < num_weight_classes; j++) {
            Vector<BaseFloat> tmp_vec;
            tmp_vec.Read(in_stream, binary);
            w_.CopyRowFromVec(tmp_vec, j);
          }
        } else {
          KALDI_ASSERT(num_weight_classes == 0);
          use_weight_projection_ = true;
          w_.Resize(num_gauss, num_clusters);
          w_.CopyFromMat(tmp);
        }
      } else if (token == "<v>") {
        v_.resize(num_cluster_weight_classes);
        for (int32 r = 0; r < num_cluster_weight_classes; r++) {
          if (r>0) {
            ExpectToken(in_stream, binary, "<ClusterWeightClass>");
            int32 tmp32;
            ReadBasicType(in_stream, binary, &tmp32);
            KALDI_ASSERT(r == tmp32);
          }
          v_[r].resize(num_states);
          for (int32 j = 0; j < num_states; j++) {
            v_[r][j].Read(in_stream, binary);
          }
        }
      } else if (token == "<n>") {
        n_.resize(num_states);
        for (int32 j = 0; j < num_states; j++) {
          n_[j].Read(in_stream, binary);
        }
      } else if (token == "<PDFToClusterMap>") {
        Vector<BaseFloat> tmp;
        tmp.Read(in_stream, binary);
        pdf_id_to_cluster_.resize(tmp.Dim());
        for (int32 j = 0; j < tmp.Dim(); j++)
          pdf_id_to_cluster_[j] = static_cast<int32> (tmp(j));
      } else if (token == "<WeightsMap>") {
        Vector<BaseFloat> tmp;
        tmp.Read(in_stream, binary);
        weights_map_.resize(tmp.Dim());
        for (int32 j = 0; j < tmp.Dim(); j++)
          weights_map_[j] = static_cast<int32> (tmp(j));
        if (!use_weight_projection_) {
          //KALDI_ASSERT(static_cast<int32> (weights_map_.Max())
          //    == static_cast<int32> (num_weight_classes) - 1);
          KALDI_ASSERT(static_cast<int32> (*std::max_element(weights_map_.begin(), weights_map_.end()))
              == static_cast<int32> (num_weight_classes) - 1);
        }
        else {
          //KALDI_ASSERT(weights_map_.Dim() == 0);
          KALDI_ASSERT(weights_map_.size() == 0);
        }
      } else if (token == "<TransformClass>") {
        Vector<BaseFloat> tmp;
        tmp.Read(in_stream, binary);
        transform_class_.resize(tmp.Dim());
        for (int32 i = 0; i < tmp.Dim(); i++) {
          transform_class_[i] = static_cast<int32> (tmp(i));
        }
      } else if (token == "<ClusterWeightClass>") {
        Vector<BaseFloat> tmp;
        tmp.Read(in_stream, binary);
        cluster_weight_class_.resize(tmp.Dim());
        for (int32 i = 0; i < tmp.Dim(); i++) {
          cluster_weight_class_[i] = static_cast<int32> (tmp(i));
        }
      } else {
        KALDI_ERR << "Unexpected token '" << token << "' in model file ";
      }
      ReadToken(in_stream, binary, &token);
    }

    // TODO: Should Full UBM stored when not being used?
    //if (!use_full_covar_) {
    //  full_ubm_.CopyFromDiagGmm(diag_ubm_);
    //}

    if (transform_class_.empty()) {
      transform_class_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++)
        transform_class_[i] = 0;
    }

    if (cluster_weight_class_.empty()) {
      cluster_weight_class_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++)
        cluster_weight_class_[i] = 0;
    }

    if (n_.empty()) {
      ComputeNormalizers();
    }
  }

  void AmPhoneTxCAT::Write(std::ostream &out_stream, bool binary,
      PhoneTxCATWriteFlagsType write_params) const {
    int32 num_states = NumPdfs(),
    feat_dim = FeatureDim(),
    num_gauss = NumGauss(),
    num_clusters = NumClusters(),
    num_weight_classes = NumWeightClasses(),
    num_transform_classes = NumTransformClasses(),
    num_cluster_weight_classes = NumClusterWeightClasses();

    WriteToken(out_stream, binary, "<PhoneTxCAT>");
    if (!binary) out_stream << "\n";
    WriteToken(out_stream, binary, "<NUMSTATES>");
    WriteBasicType(out_stream, binary, num_states);
    WriteToken(out_stream, binary, "<DIMENSION>");
    WriteBasicType(out_stream, binary, feat_dim);
    if (use_full_covar_)
      WriteToken(out_stream, binary, "<FULLC>");
    else
      WriteToken(out_stream, binary, "<DIAGC>");
    WriteToken(out_stream, binary, "<NUMWeightClasses>");
    WriteBasicType(out_stream, binary, num_weight_classes);
    WriteToken(out_stream, binary, "<NUMClusterWeightClasses>");
    WriteBasicType(out_stream, binary, num_cluster_weight_classes);
    if (!binary) out_stream << "\n";

    if (write_params & kPhoneTxCATBackgroundGmms) {
      if(use_full_covar_) {
        WriteToken(out_stream, binary, "<FULL_UBM>");
        full_ubm_.Write(out_stream, binary);
      }
      WriteToken(out_stream, binary, "<DIAG_UBM>");
      diag_ubm_.Write(out_stream, binary);
    }

    if (write_params & kPhoneTxCATGlobalParams) {
      WriteToken(out_stream, binary, "<SigmaInv>");
      WriteToken(out_stream, binary, "<NUMGaussians>");
      WriteBasicType(out_stream, binary, num_gauss);
      if (!binary) out_stream << "\n";
      for (int32 i = 0; i < num_gauss; i++) {
        SigmaInv_[i].Write(out_stream, binary);
      }
      WriteToken(out_stream, binary, "<u>");
      u_.Write(out_stream, binary);
      WriteToken(out_stream, binary, "<A>");
      WriteToken(out_stream, binary, "<NUMTransformClasses>");
      WriteBasicType(out_stream, binary, num_transform_classes);
      WriteToken(out_stream, binary, "<NUMClusters>");
      WriteBasicType(out_stream, binary, num_clusters);
      if (!binary) out_stream << "\n";
      for (int32 p = 0; p < num_clusters; p++) {
        for (int32 q = 0; q < num_transform_classes; q++) {
          A_[p][q].Write(out_stream, binary);
        }
        if (!binary) out_stream << "\n";
      }
    }

    if (write_params & kPhoneTxCATStateParams) {
      WriteToken(out_stream, binary, "<v>");
      for (int32 r = 0; r < num_cluster_weight_classes; r++) {
        if (r > 0) {
          WriteToken(out_stream, binary, "<ClusterWeightClass>");
          WriteBasicType(out_stream, binary, r);
          if (!binary) out_stream << "\n";
        }
        for (int32 j = 0; j < num_states; j++) {
          v_[r][j].Write(out_stream, binary);
        }
        if (!binary) out_stream << "\n";
      }
      WriteToken(out_stream, binary, "<w>");
      if (use_weight_projection_)
        w_.Write(out_stream, binary);
      else {
        for (int32 j = 0; j < num_weight_classes; j++) {
          w_.Row(j).Write(out_stream, binary);
        }
      }
    }

    if (write_params & kPhoneTxCATNormalizers) {
      WriteToken(out_stream, binary, "<n>");
      if (n_.empty())
        KALDI_WARN << "Not writing normalizers since they are not present.";
      else
        for (int32 j = 0; j < num_states; j++)
          n_[j].Write(out_stream, binary);
    }

    {
      WriteToken(out_stream, binary, "<PDFToClusterMap>");
      Vector<BaseFloat> tmp(pdf_id_to_cluster_.size());
      for (int32 j = 0; j < pdf_id_to_cluster_.size(); j++)
        tmp(j) = static_cast<BaseFloat> (pdf_id_to_cluster_[j]);
      tmp.Write(out_stream, binary);
    }

    {
      WriteToken(out_stream, binary, "<WeightsMap>");
      Vector<BaseFloat> tmp(weights_map_.size());
      for (int32 j = 0; j < weights_map_.size(); j++)
        tmp(j) = static_cast<BaseFloat> (weights_map_[j]);
      tmp.Write(out_stream, binary);
    }

    {
      WriteToken(out_stream, binary, "<TransformClass>");
      Vector<BaseFloat> tmp(transform_class_.size());
      for (int32 i = 0; i < transform_class_.size(); i++)
        tmp(i) = static_cast<BaseFloat> (transform_class_[i]);
      tmp.Write(out_stream, binary);
    }
    
    {
      WriteToken(out_stream, binary, "<ClusterWeightClass>");
      Vector<BaseFloat> tmp(cluster_weight_class_.size());
      for (int32 i = 0; i < cluster_weight_class_.size(); i++)
        tmp(i) = static_cast<BaseFloat> (cluster_weight_class_[i]);
      tmp.Write(out_stream, binary);
    }

    WriteToken(out_stream, binary, "</PhoneTxCAT>");
  }

  void AmPhoneTxCAT::Check(bool show_properties) {
    int32 num_states = NumPdfs(),
          num_gauss = NumGauss(),
          feat_dim = FeatureDim(),
          num_clusters = NumClusters(),
          num_weight_classes = NumWeightClasses(),
          num_transform_classes = NumTransformClasses(),
          num_cluster_weight_classes = NumClusterWeightClasses();

    if (show_properties)
      KALDI_LOG << "AmPhoneTxCAT: #states = " << num_states << ", #Gaussians = "
        << num_gauss << ", feature dim = " << feat_dim
        << ", num-clusters =" << num_clusters;
    KALDI_ASSERT(num_states > 0 && num_gauss > 0 && feat_dim > 0 
        && num_clusters > 0);

    std::ostringstream debug_str;

    // Check the Mapping vectors
    //KALDI_ASSERT(pdf_id_to_cluster_.Dim() == num_states 
    //    && weights_map_.Dim() == num_states);
    KALDI_ASSERT(pdf_id_to_cluster_.size() == num_states 
        && weights_map_.size() == num_states);

    
    // If Full Covariance is used, check Full covariance UBM
    if(use_full_covar_) {
      KALDI_ASSERT(full_ubm_.NumGauss() == num_gauss);
      KALDI_ASSERT(full_ubm_.Dim() == feat_dim);
    }

    // Check the diagonal-covariance UBM.
    KALDI_ASSERT(diag_ubm_.NumGauss() == num_gauss);
    KALDI_ASSERT(diag_ubm_.Dim() == feat_dim);

    // Check the globally-shared covariance matrices
    // and the Canonical model
    KALDI_ASSERT(u_.NumRows() == static_cast<size_t>(num_gauss));
    KALDI_ASSERT(u_.NumCols() == static_cast<size_t>(feat_dim+1));
    KALDI_ASSERT(SigmaInv_.size() == static_cast<size_t>(num_gauss));
    for (int32 i = 0; i < num_gauss; i++) {
      KALDI_ASSERT(SigmaInv_[i].NumRows() == feat_dim &&
          SigmaInv_[i](0, 0) > 0.0);  // or it wouldn't be +ve definite.
    }

    KALDI_ASSERT(A_.size() == static_cast<int32>(num_clusters));
    for (int32 p = 0; p < num_clusters; p++) {
      KALDI_ASSERT(A_[p].size() == 
          static_cast<int32>(num_transform_classes));
      for (int32 q = 0; q < num_transform_classes; q++) {
        KALDI_ASSERT(A_[p][q].NumRows() == feat_dim && A_[p][q].NumCols() == feat_dim+1);
      }
    }

    {  // check v, w.
      KALDI_ASSERT(v_.size() == static_cast<size_t>(num_cluster_weight_classes));

      if (use_weight_projection_) {
        KALDI_ASSERT(w_.NumRows() == num_gauss && w_.NumCols() == num_clusters);
      } else {
        KALDI_ASSERT(w_.NumRows() == num_weight_classes && w_.NumCols() == num_gauss);
      }

      for (int32 r = 0; r < num_cluster_weight_classes; r++) {
        KALDI_ASSERT(v_[r].size() == static_cast<size_t>(num_states));
        for (int32 j = 0; j < num_states; j++) {
          KALDI_ASSERT(v_[r][j].Dim() == num_clusters);
          //KALDI_ASSERT(static_cast<int32> (pdf_id_to_cluster_(j)) < num_clusters);
          KALDI_ASSERT(static_cast<int32> (pdf_id_to_cluster_[j]) < num_clusters);
        }
      }
    }

    // check n.
    if (n_.size() == 0) {
      debug_str << "Normalizers: no.  ";
    } else {
      debug_str << "Normalizers: yes.  ";
      KALDI_ASSERT(n_.size() == static_cast<size_t>(num_states));
      for (int32 j = 0; j < num_states; j++) {
        KALDI_ASSERT(n_[j].Dim() == num_gauss);
      }
    }

    if (show_properties)
      KALDI_LOG << "Phone Tx CAT model properties: " << debug_str.str();
  }

  void AmPhoneTxCAT::InitializeFromFullGmm(const FullGmm &gmm,
      const std::vector<int32> &pdf_id_to_cluster,
      int32 num_transform_classes, bool use_full_covar) {
    this->InitializeFromFullGmm(gmm, pdf_id_to_cluster,
        std::vector<int32>(), true, num_transform_classes, use_full_covar);
  }

  void AmPhoneTxCAT::InitializeFromFullGmm(const FullGmm &gmm,
      const std::vector<int32> &pdf_id_to_cluster,
      bool use_weight_projection, bool use_full_covar) {
    this->InitializeFromFullGmm(gmm, pdf_id_to_cluster,
        std::vector<int32>(), use_weight_projection, 1, use_full_covar);
  }

  void AmPhoneTxCAT::InitializeFromFullGmm(const FullGmm &gmm,
      const std::vector<int32> &pdf_id_to_cluster,
      const std::vector<int32> &weights_map,
      int32 num_transform_classes, bool use_full_covar) {
    this->InitializeFromFullGmm(gmm, pdf_id_to_cluster,
        weights_map, true, num_transform_classes, use_full_covar);
  }

  void AmPhoneTxCAT::InitializeFromFullGmm(const FullGmm &gmm,
      const std::vector<int32> &pdf_id_to_cluster,
      const std::vector<int32> &weights_map,
      bool use_weight_projection,
      int32 num_transform_classes,
      bool use_full_covar,
      int32 num_cluster_weight_classes) {

    use_full_covar_ = use_full_covar;
    
    if (use_full_covar_) {
      full_ubm_.CopyFromFullGmm(gmm);
    }
    diag_ubm_.CopyFromFullGmm(gmm);

    v_.clear();
    w_.SetZero();
    SigmaInv_.clear();

    KALDI_LOG << "Initializing model";
    
    InitializeCanonicalMeans(); // Initialize from UBM
    InitializeCovars();         // Initialize from UBM

    KALDI_ASSERT(!pdf_id_to_cluster.empty());
    int32 num_states = static_cast<int32> (pdf_id_to_cluster.size());
    //pdf_id_to_cluster_.Resize(num_states);
    pdf_id_to_cluster_.resize(num_states);

    for (int32 j = 0; j < num_states; j++) {
      //pdf_id_to_cluster_(j) = static_cast<BaseFloat> (pdf_id_to_cluster[j]);
      pdf_id_to_cluster_[j] = static_cast<int32> (pdf_id_to_cluster[j]);
    }

    use_weight_projection_ = use_weight_projection;

    if (!use_weight_projection_) {
      if (weights_map.empty()) {
        //weights_map_.Resize(num_states);
        weights_map_.resize(num_states);
        for (int32 j = 0; j < num_states; j++) {
          //weights_map_(j) = static_cast<BaseFloat> (j);
          weights_map_[j] = static_cast<int32> (j);
        }
      } else {
        //weights_map_.Resize(num_states);
        weights_map_.resize(num_states);
        for (int32 j = 0; j < num_states; j++) {
          //weights_map_(j) = static_cast<BaseFloat> (weights_map[j]);
          weights_map_[j] = static_cast<int32> (weights_map[j]);
        }
      }
    } else {
      //weights_map_.Set(-1);
      weights_map_.clear();
    }
    
    ReClusterGaussians(num_transform_classes, &transform_class_);

    // Deprecated. This is now done by the function ReClusterGaussians
    /*
    if (num_transform_classes>1)
    { // generate the transform class map
      int32 feature_dim = FeatureDim();

      transform_class_.resize(num_gauss);
      
      //Vector<BaseFloat> tmp_weights(diag_ubm_.weights());
      
      std::vector<Clusterable*> stats;  
      for (int32 i = 0; i < num_gauss; i++) {
        Vector<BaseFloat> tmp_mean(feature_dim);
        Vector<BaseFloat> tmp_var(feature_dim);
        diag_ubm_.GetComponentMean(i, &tmp_mean);
        diag_ubm_.GetComponentVariance(i, &tmp_var);
        tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
        BaseFloat this_weight = 1.0;
        tmp_mean.Scale(this_weight);
        tmp_var.Scale(this_weight);
        stats.push_back(new GaussClusterable(tmp_mean, tmp_var,
              0.01, this_weight));  //var_floor = 0.01
      }
      ClusterBottomUp(stats, kBaseFloatMax, num_transform_classes, 
          NULL, &transform_class_);
      DeletePointers(&stats);
    }
    else {
      transform_class_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++)
        transform_class_[i] = 0;
    }
    */

    ReClusterGaussians(num_cluster_weight_classes, &cluster_weight_class_);
    // Deprecated. This is now done by the function ReClusterGaussians
    /*
    if (num_cluster_weight_classes>1)
    { // generate the transform class map
      int32 feature_dim = FeatureDim();

      cluster_weight_class_.resize(num_gauss);
      
      //Vector<BaseFloat> tmp_weights(diag_ubm_.weights());
      
      std::vector<Clusterable*> stats;  
      for (int32 i = 0; i < num_gauss; i++) {
        Vector<BaseFloat> tmp_mean(feature_dim);
        Vector<BaseFloat> tmp_var(feature_dim);
        diag_ubm_.GetComponentMean(i, &tmp_mean);
        diag_ubm_.GetComponentVariance(i, &tmp_var);
        tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
        BaseFloat this_weight = 1.0;
        tmp_mean.Scale(this_weight);
        tmp_var.Scale(this_weight);
        stats.push_back(new GaussClusterable(tmp_mean, tmp_var,
              0.01, this_weight));  //var_floor = 0.01
      }
      ClusterBottomUp(stats, kBaseFloatMax, num_cluster_weight_classes, 
          NULL, &cluster_weight_class_);
      DeletePointers(&stats);
    }
    else {
      cluster_weight_class_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++)
        cluster_weight_class_[i] = 0;
    }
    */

    InitializeA();
    InitializeVecs();
  }

  void AmPhoneTxCAT::CopyFromPhoneTxCAT(const AmPhoneTxCAT &other,
      bool copy_normalizers, bool recluster_gaussians, 
      bool reinitializeA) {

    KALDI_LOG << "Copying AmPhoneTxCAT";

    // Copy background GMMs
    diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
    use_full_covar_ = other.use_full_covar_;

    if (use_full_covar_)
      full_ubm_.CopyFromFullGmm(other.full_ubm_);
    
    // Copy global params
    SigmaInv_ = other.SigmaInv_;
    A_ = other.A_;
    u_ = other.u_;

    // Copy state-specific params, but only copy normalizers if requested.
    v_ = other.v_;
    w_ = other.w_;

    use_weight_projection_ = other.use_weight_projection_;

    weights_map_ = other.weights_map_;
    pdf_id_to_cluster_ = other.pdf_id_to_cluster_;
    cluster_weight_class_ = other.cluster_weight_class_;
    
    if (recluster_gaussians == false) {
      transform_class_ = other.transform_class_;
    } else {
      ReClusterGaussians(other.NumTransformClasses(), &transform_class_);
      if (reinitializeA) {
        InitializeA();
      }
    }
    
    if (copy_normalizers) n_ = other.n_;

    KALDI_LOG << "Done.";
  }

  // Copy global vectors from another model but initialize
  // the state vectors to zero
  void AmPhoneTxCAT::CopyGlobalsInitVecs(const AmPhoneTxCAT &other,
      const std::vector<int32> &pdf_id_to_cluster,
      const std::vector<int32> &weights_map) {
    KALDI_LOG << "Initializing model";

    // Copy background GMMs
    diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
    use_full_covar_ = other.use_full_covar_;

    if (use_full_covar_)
      full_ubm_.CopyFromFullGmm(other.full_ubm_);
    
    // Copy Canonical model
    int32 data_dim = diag_ubm_.Dim();
    u_.Resize(NumGauss(), data_dim+1);
    u_.CopyFromMat(other.u_,kNoTrans);

    // Copy covariance
    SigmaInv_ = other.SigmaInv_;

    KALDI_ASSERT(!pdf_id_to_cluster.empty());
    int32 num_states = static_cast<int32> (pdf_id_to_cluster.size());
    //pdf_id_to_cluster_.Resize(num_states);
    pdf_id_to_cluster_.resize(num_states);

    for (int32 j = 0; j < num_states; j++) {
      //pdf_id_to_cluster_(j) = static_cast<BaseFloat> (pdf_id_to_cluster[j]);
      pdf_id_to_cluster_[j] = static_cast<int32> (pdf_id_to_cluster[j]);
    }

    use_weight_projection_ = other.use_weight_projection_;

    if (!use_weight_projection_) {
      if (weights_map.empty()) {
        //weights_map_.Resize(num_states);
        weights_map_.resize(num_states);
        for (int32 j = 0; j < num_states; j++)
          //weights_map_(j) = j;
          weights_map_[j] = j;
      } else {
        for (int32 j = 0; j < num_states; j++) {
          //weights_map_(j) = weights_map[j];
          weights_map_[j] = weights_map[j];
        }
      }
    } else {
      //weights_map_.Set(-1);
      weights_map_.clear();
    }
    
    transform_class_ = other.transform_class_;
    cluster_weight_class_ = other.cluster_weight_class_;

    // Initialize Cluster transforms and state vectors
    InitializeA();
    InitializeVecs();

    if (use_weight_projection_)
      w_.CopyFromMat(other.w_);
  }

  void AmPhoneTxCAT::ComputePerFrameVars(const VectorBase<BaseFloat> &data,
      const std::vector<int32> &gselect,
      PhoneTxCATPerFrameDerivedVars *per_frame_vars) const {
    KALDI_ASSERT(!n_.empty() && "ComputeNormalizers() must be called.");

    if (per_frame_vars->NeedsResizing(gselect.size(),
          FeatureDim(),
          NumClusters()))
      per_frame_vars->Resize(gselect.size(), FeatureDim(), NumClusters());

    per_frame_vars->gselect = gselect;
    per_frame_vars->xt.CopyFromVec(data);

    for (int32 ki = 0, last = gselect.size(); ki < last; ki++) {
      Vector<BaseFloat> SigmaInv_xt(FeatureDim());
      int32 i = gselect[ki];
      SigmaInv_xt.AddSpVec(1.0, SigmaInv_[i], per_frame_vars->xt, 0.0);

      // z_{i}(t) = M_{i}^T \SigmaInv{i} x(t)
      Matrix<BaseFloat> MiTrans;
      MiTrans.Resize(NumClusters(),FeatureDim());

      for (int32 p = 0; p < NumClusters(); p++) {
        int32 q = transform_class_[i];
        MiTrans.Row(p).AddMatVec(1.0, A_[p][q], kNoTrans, u_.Row(i), 0.0);
      }

      per_frame_vars->zti.Row(ki).AddMatVec(1.0, MiTrans, kNoTrans, SigmaInv_xt, 0.0);

      // n_{i}(t) = -0.5 x(t) ^T \SigmaInv{i} x(t)
      per_frame_vars->nti(ki) = -0.5 * VecVec(per_frame_vars->xt, SigmaInv_xt);
    }
  }

  BaseFloat AmPhoneTxCAT::LogLikelihood(const PhoneTxCATPerFrameDerivedVars &per_frame_vars, int32 j, BaseFloat log_prune) const {
    KALDI_ASSERT(j < NumPdfs());
    const vector<int32> &gselect = per_frame_vars.gselect;

    // log p( x(t),i | j ) [indexed by j, ki]
    // Although the extra memory allocation of storing this as a
    // vector might seem unnecessary, we save time in the LogSumExp()
    // via more effective pruning.
    Vector<BaseFloat> logp_x(gselect.size());

    for (int32 ki = 0, last = gselect.size();  ki < last; ki++) {
      int32 i = gselect[ki];
      int32 r = cluster_weight_class_[i];
      // Compute z_{i}^T v_{r}{j}
      logp_x(ki) = VecVec(per_frame_vars.zti.Row(ki), v_[r][j]);
      logp_x(ki) += n_[j](i);
      logp_x(ki) += per_frame_vars.nti(ki);
    }

    // log p(x(t)/j) = log \sum_{i} p( x(t),i | j )
    return logp_x.LogSumExp(log_prune);
  }

  BaseFloat AmPhoneTxCAT::ComponentPosteriors(
      const PhoneTxCATPerFrameDerivedVars &per_frame_vars,
      int32 j,
      Vector<BaseFloat> *post) const {
    KALDI_ASSERT(j < NumPdfs());
    if (post == NULL) KALDI_ERR << "NULL pointer passed as return argument.";

    const vector<int32> &gselect = per_frame_vars.gselect;
    int32 num_gselect = gselect.size();
    post->Resize(num_gselect);

    // log p( x(t),i | j ) = z_{i}^T v_{r}{j} // for the gselect-ed gaussians
    // post->AddMatVec(1.0, per_frame_vars.zti, kNoTrans, v_[j], 0.0);

    for (int32 ki = 0; ki < num_gselect; ki++) {
      int32 i = gselect[ki];
      int32 r = cluster_weight_class_[i];
      // Compute z_{i}^T v_{r}{j}
      (*post)(ki) = VecVec(per_frame_vars.zti.Row(ki), v_[r][j]);
      // log p( x(t),i | j ) += n_{ji} + n_{i}(t)
      (*post)(ki) += n_[j](i);
      (*post)(ki) += per_frame_vars.nti(ki);
    }

    // log p(x(t)|j) = log \sum_{i} p(x(t), i|j)
    return post->ApplySoftMax();
  }

  void AmPhoneTxCAT::ComputeDerivedVars() {
    if (n_.empty()) {
      ComputeNormalizers();
    }
  }

  class ComputeNormalizersClass: public MultiThreadable { 
    // For multi-threaded.
    public:
      ComputeNormalizersClass(AmPhoneTxCAT *am_phoneTxCAT,
          int32 *entropy_count_ptr,
          double *entropy_sum_ptr):
        am_phoneTxCAT_(am_phoneTxCAT), entropy_count_ptr_(entropy_count_ptr),
        entropy_sum_ptr_(entropy_sum_ptr), entropy_count_(0),
        entropy_sum_(0.0) { }

      ~ComputeNormalizersClass() {
        *entropy_count_ptr_ += entropy_count_;
        *entropy_sum_ptr_ += entropy_sum_;
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to original pointer in the destructor.
        am_phoneTxCAT_->ComputeNormalizersInternal(num_threads_, thread_id_,
            &entropy_count_,
            &entropy_sum_);
      }
    private:
      ComputeNormalizersClass() { } // Disallow empty constructor.
      AmPhoneTxCAT *am_phoneTxCAT_;
      int32 *entropy_count_ptr_;
      double *entropy_sum_ptr_;
      int32 entropy_count_;
      double entropy_sum_;

  };

  void AmPhoneTxCAT::ComputeNormalizers() {
    KALDI_LOG << "Computing normalizers";
    n_.resize(NumPdfs());         // NumPdfs == Num_TiedStates
    int32 entropy_count = 0;
    double entropy_sum = 0.0;

    ComputeNormalizersClass c(this, &entropy_count, &entropy_sum);
    RunMultiThreaded(c);
    KALDI_LOG << "Entropy of weights in states is "
      << (entropy_sum / entropy_count) << " over " << entropy_count
      << " states, equivalent to perplexity of "
      << (exp(entropy_sum /entropy_count));
    KALDI_LOG << "Done computing normalizers";
  }

  void AmPhoneTxCAT::ComputeNormalizersInternal(int32 num_threads,
      int32 thread, int32 *entropy_count,
      double *entropy_sum) {
    BaseFloat DLog2pi = FeatureDim() * log(2 * M_PI);
    Vector<BaseFloat> log_det_Sigma(NumGauss());
    
    for (int32 i = 0; i < NumGauss(); i++) {
      try {
        log_det_Sigma(i) = - SigmaInv_[i].LogPosDefDet();
      } catch(...) {
        if(thread == 0) // just for one thread,
                        // print errors
                        // else print duplicates
          KALDI_WARN << "Covariance is not positive definite, setting to unit";
        SigmaInv_[i].SetUnit();
        log_det_Sigma(i) = 0.0;
      }
    }

    int block_size = (NumPdfs() + num_threads-1) / num_threads;
    int j_start = thread * block_size, j_end = std::min(NumPdfs(), j_start + block_size);

    for (int32 j = j_start; j < j_end; j++) {
      Vector<BaseFloat> log_w_j(NumGauss());
      n_[j].Resize(NumGauss());

      if (use_weight_projection_) { 
        //log_w_j.AddMatVec(1.0, w_, kNoTrans, v_[j], 0.0);
        for (int32 i = 0; i < NumGauss(); i++) {
          int32 r = cluster_weight_class_[i];
          log_w_j(i) = VecVec(w_.Row(i), v_[r][j]);
        }
        log_w_j.Add(-1.0 * log_w_j.LogSumExp());

        { // DIAGNOSTIC CODE
          (*entropy_count)++;
          for (int32 i = 0; i < NumGauss(); i++) {
            (*entropy_sum) -= log_w_j(i) * exp(log_w_j(i));
          }
        }
      } else { // DIAGNOSTIC CODE
        (*entropy_count)++;
        for (int32 i = 0; i < NumGauss(); i++) {
          //if (w_(static_cast<size_t> (weights_map_(j)),i) == 0)
          if (w_(static_cast<size_t> (weights_map_[j]),i) == 0)
          {
            log_w_j(i) = -1e+40;
            continue;
          }
          //log_w_j(i) = log(w_(static_cast<size_t> (weights_map_(j)),i));
          log_w_j(i) = log(w_(static_cast<size_t> (weights_map_[j]),i));
          (*entropy_sum) -= log_w_j(i) * 
            //w_(static_cast<size_t> (weights_map_(j)),i);
            w_(static_cast<size_t> (weights_map_[j]),i);
        }
      }

      for (int32 i = 0; i < NumGauss(); i++) {    
        Vector<BaseFloat> mu_ji(FeatureDim());
        Vector<BaseFloat> SigmaInv_mu(FeatureDim());

        // mu_ji = M_{i} * v_{r}{j}
        GetStateMean(j, i, &mu_ji);
        SigmaInv_mu.AddSpVec(1.0, SigmaInv_[i], mu_ji, 0.0);

        // mu_{ji} * \Sigma_{i}^{-1} * mu_{ji}
        BaseFloat mu_SigmaInv_mu = VecVec(mu_ji, SigmaInv_mu);

        n_[j](i) = log_w_j(i) - 0.5 * (log_det_Sigma(i) + DLog2pi
            + mu_SigmaInv_mu);

        {  // Mainly diagnostic code.  Not necessary.
          BaseFloat tmp = n_[j](i);
          if (!KALDI_ISFINITE(tmp)) {  // NaN or inf
            KALDI_LOG << "Warning: normalizer for j = " << j 
              << ", i = " << i << " is infinite or NaN " << tmp << "= "
              << "+" << (log_w_j(i)) << "+" << (-0.5 *
                  log_det_Sigma(i)) << "+" << (-0.5 * DLog2pi)
              << "+" << (mu_SigmaInv_mu) << ", setting to finite.";
            n_[j](i) = -1.0e+40;  // future work(arnab): get rid of magic number
          }
        }
      }
    }
  }

  void AmPhoneTxCAT::ReClusterGaussians(int num_clusters, std::vector<int32> *cluster_map) {
    
    int32 num_gauss = NumGauss();

    if (num_clusters>1)
    { // generate the transform class map
      int32 feature_dim = FeatureDim();
      (*cluster_map).resize(num_gauss);
      
      //Vector<BaseFloat> tmp_weights(diag_ubm_.weights());
      
      std::vector<Clusterable*> stats;  
      for (int32 i = 0; i < num_gauss; i++) {
        Vector<BaseFloat> tmp_mean(
            SubVector<BaseFloat>(u_.Row(i), 0, feature_dim));
        Vector<BaseFloat> tmp_var(feature_dim);
        for (int32 d = 0; d < feature_dim; d++) {
          tmp_var(d) = static_cast<BaseFloat>(SigmaInv_[i](d,d));
        }
        tmp_var.AddVec2(1.0, tmp_mean);  // make it x^2 stats.
        BaseFloat this_weight = 1.0;
        tmp_mean.Scale(this_weight);
        tmp_var.Scale(this_weight);
        stats.push_back(new GaussClusterable(tmp_mean, tmp_var,
              0.01, this_weight));  //var_floor = 0.01
      }
      ClusterBottomUp(stats, kBaseFloatMax, num_clusters, 
          NULL, cluster_map);
      DeletePointers(&stats);
    }
    else {
      (*cluster_map).resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++)
        (*cluster_map)[i] = 0;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////

  template<class Real>
    void AmPhoneTxCAT::ComputeH(std::vector< SpMatrix<Real> > *H) const {
      KALDI_ASSERT(NumGauss() != 0);
      (*H).resize(NumGauss());
      for (int32 i = 0; i < NumGauss(); i++) {
        Matrix<BaseFloat> MiTrans;
        MiTrans.Resize(NumClusters(), FeatureDim());
        SpMatrix<BaseFloat> H_tmp;

        // The pth column of Mi is A{p}*u_{i}
        for (int32 p = 0; p < NumClusters(); p++) {
          int32 q = transform_class_[i];
          MiTrans.Row(p).AddMatVec(1.0, A_[p][q], kNoTrans, u_.Row(i), 0.0);
        }

        H_tmp.Resize(NumClusters());
        // H_{i} = M_{i}^T SigmaInv_{i} M_{i}
        H_tmp.AddMat2Sp(1.0, MiTrans, kNoTrans, SigmaInv_[i], 0.0);
        (*H)[i].Resize(NumClusters());
        (*H)[i].CopyFromSp(H_tmp);
      }
    }

  // Instantiate the template.
  template
    void AmPhoneTxCAT::ComputeH(std::vector< SpMatrix<float> > *H) const;
  template
    void AmPhoneTxCAT::ComputeH(std::vector< SpMatrix<double> > *H) const;
  
  template<class Real>
    void AmPhoneTxCAT::ComputeM_SigmaInv(std::vector< Matrix<Real> > *M_SigmaInv_i) const {
      KALDI_ASSERT(NumGauss() != 0);
      (*M_SigmaInv_i).resize(NumGauss());
      
      for (int32 i = 0; i < NumGauss(); i++) {
        Matrix<BaseFloat> Mi(FeatureDim(), NumClusters());
        GetModelSpaceProjection(i, &Mi);

        Matrix<BaseFloat> tmp(NumClusters(), FeatureDim());
        // Compute M_{i}^T SigmaInv_{i}
        tmp.AddMatSp(1.0, Mi, kTrans, SigmaInv_[i], 0.0);
        (*M_SigmaInv_i)[i].Resize(NumClusters(), FeatureDim());
        (*M_SigmaInv_i)[i].CopyFromMat(tmp);
      }
    }

  // Instantiate the template.
  template
    void AmPhoneTxCAT::ComputeM_SigmaInv(
        std::vector< Matrix<float> > *M_SigmaInv_i) const;
  template
    void AmPhoneTxCAT::ComputeM_SigmaInv(
        std::vector< Matrix<double> > *M_SigmaInv_i) const;
  
  // Initialize cannonical model means u_{i}
  void AmPhoneTxCAT::InitializeCanonicalMeans() {
    int32 ddim = diag_ubm_.Dim();
    int32 num_gauss = diag_ubm_.NumGauss();

    u_.Resize(num_gauss, ddim+1);
    
    Matrix<BaseFloat> ubm_means(num_gauss, ddim);
    diag_ubm_.GetMeans(&ubm_means);
    
    SubMatrix<BaseFloat> mu(u_, 0, num_gauss, 0, ddim);
    mu.CopyFromMat(ubm_means);

    for (int32 i = 0; i < num_gauss; i++) {
      u_(i,ddim) = 1;
    }
  }

  // Initializes the matrices A_{p}{q}
  void AmPhoneTxCAT::InitializeA() {
    //KALDI_ASSERT(pdf_id_to_cluster_.Dim()>0);
    KALDI_ASSERT(pdf_id_to_cluster_.size()>0);
    //int32 num_clusters = static_cast<int32> (pdf_id_to_cluster_.Max()) + 1;
    int32 num_clusters = static_cast<int32> (*std::max_element(pdf_id_to_cluster_.begin(), pdf_id_to_cluster_.end())) + 1;
    
    int32 num_transform_classes = NumTransformClasses();
    KALDI_ASSERT(num_transform_classes > 0);

    int32 ddim = diag_ubm_.Dim();
    A_.resize(num_clusters);
    for (int32 p = 0; p < num_clusters; p++) {
      A_[p].resize(num_transform_classes);
      for (int32 q = 0; q < num_transform_classes; q++) {
        A_[p][q].Resize(ddim, ddim+1);
        for (int32 i = 0; i < ddim; i++) {
          A_[p][q](i,i) = 1.0;
        }
      }
    } 
  }

  // Initialize vectors v_{j} and w_{j}
  void AmPhoneTxCAT::InitializeVecs() {
    
    int32 num_clusters = NumClusters();
    //int32 num_states = static_cast<int32> (pdf_id_to_cluster_.Dim());
    int32 num_states = static_cast<int32> (pdf_id_to_cluster_.size());
    int32 num_gauss = diag_ubm_.NumGauss();
    int32 num_weight_classes = 0;
    int32 num_cluster_weight_classes = NumClusterWeightClasses();

    KALDI_ASSERT(num_states > 0);
    KALDI_ASSERT(num_clusters > 0 && "Initialize A first!\n");

    v_.resize(num_cluster_weight_classes);

    if (!use_weight_projection_) {
      num_weight_classes = NumWeightClasses();
      w_.Resize(num_weight_classes, num_gauss);
    } else {
      w_.Resize(num_gauss, num_clusters);
    }

    for (int32 r = 0; r < num_cluster_weight_classes; r++) {
      v_[r].resize(num_states);
      for (int32 j = 0; j < num_states; j++) {
        //int32 p = static_cast<int32> (pdf_id_to_cluster_(j));
        int32 p = static_cast<int32> (pdf_id_to_cluster_[j]);
        KALDI_ASSERT(p < num_clusters);

        v_[r][j].Resize(num_clusters);
        v_[r][j](p) = 1.0;

        if (!use_weight_projection_) {
          if (j<num_weight_classes) {
            w_.CopyRowFromVec(diag_ubm_.weights(),j);
          }
        } else {
          w_.SetZero();
        }
      }
    }
  }

  // Initializes the within-class vars Sigma_{i}
  void AmPhoneTxCAT::InitializeCovars() {
    if (use_full_covar_) {
      std::vector< SpMatrix<BaseFloat> > &inv_covars(full_ubm_.inv_covars());
      int32 num_gauss = full_ubm_.NumGauss();
      int32 dim = full_ubm_.Dim();
      SigmaInv_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        SigmaInv_[i].Resize(dim);
        SigmaInv_[i].CopyFromSp(inv_covars[i]);
      }
      return;
    }

    const Matrix<BaseFloat> &inv_covars(diag_ubm_.inv_vars());
    int32 num_gauss = diag_ubm_.NumGauss();
    int32 dim = diag_ubm_.Dim();
    SigmaInv_.resize(num_gauss);
    for (int32 i = 0; i < num_gauss; i++) {
      SigmaInv_[i].Resize(dim);
      for (int32 d = 0; d < dim; d++) {
        SigmaInv_[i](d,d) = inv_covars(i,d);
      }
    }
  }

  BaseFloat AmPhoneTxCAT::GaussianSelection(const PhoneTxCATGselectConfig &config,
      const VectorBase<BaseFloat> &data,
      std::vector<int32> *gselect) const {

    if(!use_full_covar_) {
      return (this->GaussianSelectionDiag(config, data, gselect));
    }

    KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
        diag_ubm_.NumGauss() == full_ubm_.NumGauss() &&
        diag_ubm_.Dim() == data.Dim());
    KALDI_ASSERT(config.diag_gmm_nbest > 0 && config.full_gmm_nbest > 0 &&
        config.full_gmm_nbest < config.diag_gmm_nbest);
    int32 num_gauss = diag_ubm_.NumGauss();

    std::vector< std::pair<BaseFloat, int32> > pruned_pairs;
    if (config.diag_gmm_nbest < num_gauss) {
      Vector<BaseFloat> loglikes(num_gauss);
      diag_ubm_.LogLikelihoods(data, &loglikes);
      Vector<BaseFloat> loglikes_copy(loglikes);
      BaseFloat *ptr = loglikes_copy.Data();
      std::nth_element(ptr, ptr+num_gauss-config.diag_gmm_nbest, ptr+num_gauss);
      BaseFloat thresh = ptr[num_gauss-config.diag_gmm_nbest];
      for (int32 g = 0; g < num_gauss; g++)
        if (loglikes(g) >= thresh)  // met threshold for diagonal phase.
          pruned_pairs.push_back(
              std::make_pair(full_ubm_.ComponentLogLikelihood(data, g), g));
    } else {
      Vector<BaseFloat> loglikes(num_gauss);
      full_ubm_.LogLikelihoods(data, &loglikes);
      for (int32 g = 0; g < num_gauss; g++)
        pruned_pairs.push_back(std::make_pair(loglikes(g), g));
    }
    KALDI_ASSERT(!pruned_pairs.empty());
    if (pruned_pairs.size() > static_cast<size_t>(config.full_gmm_nbest)) {
      std::nth_element(pruned_pairs.begin(),
          pruned_pairs.end() - config.full_gmm_nbest,
          pruned_pairs.end());
      pruned_pairs.erase(pruned_pairs.begin(),
          pruned_pairs.end() - config.full_gmm_nbest);
    }
    Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
    KALDI_ASSERT(gselect != NULL);
    gselect->resize(pruned_pairs.size());
    // Make sure pruned Gaussians appear from best to worst.
    std::sort(pruned_pairs.begin(), pruned_pairs.end(),
        std::greater< std::pair<BaseFloat, int32> >());
    for (size_t i = 0; i < pruned_pairs.size(); i++) {
      loglikes_tmp(i) = pruned_pairs[i].first;
      (*gselect)[i] = pruned_pairs[i].second;
    }
    return loglikes_tmp.LogSumExp();
  }

  BaseFloat AmPhoneTxCAT::GaussianSelectionDiag(
      const PhoneTxCATGselectConfig &config,
      const VectorBase<BaseFloat> &data,
      std::vector<int32> *gselect) const {
    KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
        diag_ubm_.Dim() == data.Dim());

    KALDI_ASSERT(config.diag_gmm_nbest > 0);
    int32 num_gauss = diag_ubm_.NumGauss();

    std::vector< std::pair<BaseFloat, int32> > pruned_pairs;

    Vector<BaseFloat> loglikes(num_gauss);
    diag_ubm_.LogLikelihoods(data, &loglikes);  // Per-component likelihoods
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();

    if (config.diag_gmm_nbest < num_gauss) {
      // Make the element at the (ptr+num_gauss-nbest)^th place to be 
      // the element it would be if the list was sorted
      // The elements to the left of this nth element would be lower than
      // the nth element and vice versa for the right elements
      std::nth_element(ptr, ptr+num_gauss-config.diag_gmm_nbest, ptr+num_gauss);
      BaseFloat thresh = ptr[num_gauss-config.diag_gmm_nbest];

      for (int32 g=0; g < num_gauss; g++)
        if (loglikes(g) >= thresh)
          pruned_pairs.push_back(
              std::make_pair(loglikes(g),g));
    } else {
      for (int32 g = 0; g < num_gauss; g++)
        pruned_pairs.push_back(std::make_pair(loglikes(g), g));
    }

    KALDI_ASSERT(!pruned_pairs.empty());

    Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
    KALDI_ASSERT(gselect != NULL);
    gselect->resize(pruned_pairs.size());
  // Make sure pruned Gaussians appear from best to worst.
    std::sort(pruned_pairs.begin(), pruned_pairs.end(),
        std::greater< std::pair<BaseFloat, int32> >());

    for (size_t i = 0; i < pruned_pairs.size(); i++) {
      loglikes_tmp(i) = pruned_pairs[i].first;
      (*gselect)[i] = pruned_pairs[i].second;
    }
    return loglikes_tmp.LogSumExp();
  }

  BaseFloat AmPhoneTxCAT::GaussianSelectionPreselect(
      const PhoneTxCATGselectConfig &config,
      const VectorBase<BaseFloat> &data,
      const std::vector<int32> &preselect,
      std::vector<int32> *gselect) const {

    if (!use_full_covar_) {
      return (this->GaussianSelectionPreselectDiag(config,
            data, preselect, gselect));
    }

    KALDI_ASSERT(IsSortedAndUniq(preselect) && !preselect.empty());
    KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
        diag_ubm_.NumGauss() == full_ubm_.NumGauss() &&
        diag_ubm_.Dim() == data.Dim());

    int32 num_preselect = preselect.size();

    KALDI_ASSERT(config.diag_gmm_nbest > 0 && config.full_gmm_nbest > 0 &&
        config.full_gmm_nbest < num_preselect);

    std::vector<std::pair<BaseFloat, int32> > pruned_pairs;
    if (config.diag_gmm_nbest < num_preselect) {
      Vector<BaseFloat> loglikes(num_preselect);
      diag_ubm_.LogLikelihoodsPreselect(data, preselect, &loglikes);
      Vector<BaseFloat> loglikes_copy(loglikes);
      BaseFloat *ptr = loglikes_copy.Data();
      std::nth_element(ptr, ptr+num_preselect-config.diag_gmm_nbest,
          ptr+num_preselect);
      BaseFloat thresh = ptr[num_preselect-config.diag_gmm_nbest];
      for (int32 p = 0; p < num_preselect; p++) {
        if (loglikes(p) >= thresh) {  // met threshold for diagonal phase.
          int32 g = preselect[p];
          pruned_pairs.push_back(
              std::make_pair(full_ubm_.ComponentLogLikelihood(data, g), g));
        }
      }
    } else {
      for (int32 p = 0; p < num_preselect; p++) {
        int32 g = preselect[p];
        pruned_pairs.push_back(
            std::make_pair(full_ubm_.ComponentLogLikelihood(data, g), g));
      }
    }
    KALDI_ASSERT(!pruned_pairs.empty());
    if (pruned_pairs.size() > static_cast<size_t>(config.full_gmm_nbest)) {
      std::nth_element(pruned_pairs.begin(),
          pruned_pairs.end() - config.full_gmm_nbest,
          pruned_pairs.end());
      pruned_pairs.erase(pruned_pairs.begin(),
          pruned_pairs.end() - config.full_gmm_nbest);
    }
    // Make sure pruned Gaussians appear from best to worst.
    std::sort(pruned_pairs.begin(), pruned_pairs.end(),
        std::greater<std::pair<BaseFloat, int32> >());
    Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
    KALDI_ASSERT(gselect != NULL);
    gselect->resize(pruned_pairs.size());
    for (size_t i = 0; i < pruned_pairs.size(); i++) {
      loglikes_tmp(i) = pruned_pairs[i].first;
      (*gselect)[i] = pruned_pairs[i].second;
    }
    return loglikes_tmp.LogSumExp();
  }

  BaseFloat AmPhoneTxCAT::GaussianSelectionPreselectDiag(
      const PhoneTxCATGselectConfig &config, 
      const VectorBase<BaseFloat> &data,
      const std::vector<int32> &preselect,
      std::vector<int32> *gselect) const {

    KALDI_ASSERT(IsSortedAndUniq(preselect) && !preselect.empty());
    KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
        diag_ubm_.Dim() == data.Dim());

    int32 num_preselect = preselect.size();

    KALDI_ASSERT(config.diag_gmm_nbest > 0 &&
        config.diag_gmm_nbest < num_preselect);

    std::vector<std::pair<BaseFloat, int32> > pruned_pairs;

    if (config.diag_gmm_nbest < num_preselect) {

      Vector<BaseFloat> loglikes(num_preselect);
      diag_ubm_.LogLikelihoodsPreselect(data, preselect, &loglikes);
      Vector<BaseFloat> loglikes_copy(loglikes);
      BaseFloat *ptr = loglikes_copy.Data();

      std::nth_element(ptr, ptr+num_preselect-config.diag_gmm_nbest,
          ptr+num_preselect);
      BaseFloat thresh = ptr[num_preselect-config.diag_gmm_nbest];

      for (int32 p=0; p < num_preselect; p++)
        if (loglikes(p) >= thresh) {  // met threshold for diagonal phase.
          int32 g = preselect[p];
          pruned_pairs.push_back(
              std::make_pair(diag_ubm_.ComponentLogLikelihood(data, g), g));
        } 
    } else {
      for (int32 p = 0; p < num_preselect; p++) {
        int32 g = preselect[p];
        pruned_pairs.push_back(
            std::make_pair(diag_ubm_.ComponentLogLikelihood(data, g), g));
      }
    }
    KALDI_ASSERT(!pruned_pairs.empty());

    // Make sure pruned Gaussians appear from best to worst.
    std::sort(pruned_pairs.begin(), pruned_pairs.end(),
        std::greater<std::pair<BaseFloat, int32> >());
    Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
    KALDI_ASSERT(gselect != NULL);
    gselect->resize(pruned_pairs.size());
    for (size_t i = 0; i < pruned_pairs.size(); i++) {
      loglikes_tmp(i) = pruned_pairs[i].first;
      (*gselect)[i] = pruned_pairs[i].second;
    }
    return loglikes_tmp.LogSumExp();
  }

  void PhoneTxCATGauPost::Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<PhoneTxCATGauPost>");
    int32 T = this->size();
    WriteBasicType(os, binary, T);
    for (int32 t = 0; t < T; t++) {
      WriteToken(os, binary, "<gselect>");
      WriteIntegerVector(os, binary, (*this)[t].gselect);
      WriteToken(os, binary, "<tids>");
      WriteIntegerVector(os, binary, (*this)[t].tids);
      KALDI_ASSERT((*this)[t].tids.size() == (*this)[t].posteriors.size());
      for (size_t i = 0; i < (*this)[t].posteriors.size(); i++) {
        (*this)[t].posteriors[i].Write(os, binary);
      }
    }
    WriteToken(os, binary, "</PhoneTxCATGauPost>");
  }

  void PhoneTxCATGauPost::Read(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<PhoneTxCATGauPost>");
    int32 T;
    ReadBasicType(is, binary, &T);
    KALDI_ASSERT(T >= 0);
    this->resize(T);
    for (int32 t = 0; t < T; t++) {
      ExpectToken(is, binary, "<gselect>");
      ReadIntegerVector(is, binary, &((*this)[t].gselect));
      ExpectToken(is, binary, "<tids>");
      ReadIntegerVector(is, binary, &((*this)[t].tids));
      size_t sz = (*this)[t].tids.size();
      (*this)[t].posteriors.resize(sz);
      for (size_t i = 0; i < sz; i++)
        (*this)[t].posteriors[i].Read(is, binary);
    }
    ExpectToken(is, binary, "</PhoneTxCATGauPost>");
  }

} // namespace kaldi
