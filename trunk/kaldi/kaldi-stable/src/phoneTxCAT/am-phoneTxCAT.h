#ifndef KALDI_PHONETXCAT_AM_PHONETXCAT_H_
#define KALDI_PHONETXCAT_AM_PHONETXCAT_H_

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

  enum PhoneTxCATUpdateFlags {  /// The letters correspond to the variable names.
    kPhoneTxCATClusterWeights              = 0x001,  /// v
    kPhoneTxCATStateVectors                = 0x001,  /// v (Compatibility)
    kPhoneTxCATClusterTransforms           = 0x002,  /// A
    kPhoneTxCATWeights                     = 0x004,  /// w (Compatibility)
    kPhoneTxCATGaussianWeights             = 0x004,  /// w
    kPhoneTxCATCovarianceMatrix            = 0x008,  /// S
    kPhoneTxCATTransitions                 = 0x010,  /// t .. not really part of SGMM.
    kPhoneTxCATCanonicalMeans              = 0x020,  /// u
    kPhoneTxCATAll                         = 0x0FF   /// a (won't normally use this).  
  };

  typedef uint16 PhoneTxCATUpdateFlagsType;  ///< Bitwise OR of the above flags.
  PhoneTxCATUpdateFlagsType StringToPhoneTxCATUpdateFlags(std::string str);

  enum PhoneTxCATWriteFlags {
    kPhoneTxCATGlobalParams    = 0x001,  /// g
    kPhoneTxCATStateParams     = 0x002,  /// s
    kPhoneTxCATNormalizers     = 0x004,  /// n
    kPhoneTxCATBackgroundGmms  = 0x008,  /// u
    kPhoneTxCATWriteAll        = 0x00F  /// a
  };

  typedef uint16 PhoneTxCATWriteFlagsType;  ///< Bitwise OR of the above flags.

  PhoneTxCATWriteFlagsType StringToPhoneTxCATWriteFlags(std::string str);

  struct PhoneTxCATGselectConfig {
    /// Number of highest-scoring diagonal-covariance Gaussians per frame.
    int32 diag_gmm_nbest;
    int32 full_gmm_nbest;

    PhoneTxCATGselectConfig() {
      full_gmm_nbest = 15;
      diag_gmm_nbest = 50;
    }

    void Register(ParseOptions *po) {
      po->Register("diag-gmm-nbest", &diag_gmm_nbest, "Number of highest-scoring"
          " diagonal-covariance Gaussians selected per frame.");
      po->Register("full-gmm-nbest", &full_gmm_nbest, "Number of highest-scoring"
          " full-covariance Gaussians selected per frame.");
    }
  };

  /** \struct PhoneTxCATPerFrameDerivedVars
   *  Holds the per-frame precomputed quantities x(t), z_{i}(t), and
   *  n_{i}(t) for the PhoneTxCAT, as well as the cached Gaussian
   *  selection records.
   */
  struct PhoneTxCATPerFrameDerivedVars {
    std::vector<int32> gselect;
    Vector<BaseFloat> xt;   ///< x(t), dim = [D]
    /// Just the observation vector.
    /// In future, FMLLR can be added
    Matrix<BaseFloat> zti;  ///< z_{i}(t), dim = [I][S]
    Vector<BaseFloat> nti;  ///< n_{i}(t), dim = [I]

    PhoneTxCATPerFrameDerivedVars() : xt(0), zti(0,0), nti(0) {}

    void Resize(int32 ngauss, int32 feat_dim, int32 num_clusters) {
      xt.Resize(feat_dim);
      zti.Resize(ngauss, num_clusters);
      nti.Resize(ngauss);
    }

    bool IsEmpty() const {
      return (xt.Dim() == 0 || zti.NumRows() == 0 || nti.Dim() == 0);
    }

    bool NeedsResizing(int32 ngauss, int32 feat_dim, int32 num_clusters) const {
      return (xt.Dim() != feat_dim 
          || zti.NumRows() != ngauss || zti.NumCols() != num_clusters
          || nti.Dim() != ngauss);
    }


  };

  /** \class AmPhoneTxCAT
   *  Class for definition of the Phone Transform CAT acoustic model
   */
  class AmPhoneTxCAT {
    public:
      AmPhoneTxCAT() {}
      void Read(std::istream &rIn, bool binary);
      void Write(std::ostream &out, bool binary,
          PhoneTxCATWriteFlagsType write_params) const;

      /// Checks the various components for correct sizes. With wrong sizes,
      /// assertion failure occurs. When the argument is set to true, dimensions of
      /// the various components are printed.
      void Check(bool show_properties = true);

      /// Initializes the PhoneTxCAT parameters from a diag-covariance UBM.
      void InitializeFromFullGmm(const FullGmm &gmm,
          const std::vector<int32> &pdf_id_to_cluster,
          const std::vector<int32> &weights_map,
          bool use_weight_projection = true,
          int32 num_transform_classes = 1,
          bool use_full_covar = false,
          int32 num_cluster_weight_classes = 1); 
      
      void InitializeFromFullGmm(const FullGmm &gmm,
          const std::vector<int32> &pdf_id_to_cluster,
          const std::vector<int32> &weights_map,
          int32 num_transform_classes, 
          bool use_full_covar = false); 
      
      void InitializeFromFullGmm(const FullGmm &gmm,
          const std::vector<int32> &pdf_id_to_cluster,
          bool use_weight_projection = true,
          bool use_full_covar = false); 
      
      void InitializeFromFullGmm(const FullGmm &gmm,
          const std::vector<int32> &pdf_id_to_cluster,
          int32 num_transform_classes, 
          bool use_full_covar = false); 

      /// Used to copy models (useful in update)
      void CopyFromPhoneTxCAT(const AmPhoneTxCAT &other, 
          bool copy_normalizers, bool recluster_gaussians = false, 
          bool reinitializeA = false);

      /// Copies the global parameters from the supplied model,
      /// but initializes the Cluster transforms and State vectors
      void CopyGlobalsInitVecs(const AmPhoneTxCAT &other,
          const std::vector<int32> &pdf_id_to_cluster,
          const std::vector<int32> &weights_map);

      /// Computes the top-scoring Gaussian indices (used for pruning of later
      /// stages of computation). Returns frame log-likelihood given selected
      /// Gaussians from diag UBM.
      BaseFloat GaussianSelectionDiag(const PhoneTxCATGselectConfig &config,
          const VectorBase<BaseFloat> &data,
          std::vector<int32> *gselect) const;
      
      /// Computes the top-scoring Gaussian indices (used for pruning of later
      /// stages of computation). Returns frame log-likelihood given selected
      /// Gaussians from UBM.
      BaseFloat GaussianSelection(const PhoneTxCATGselectConfig &config,
          const VectorBase<BaseFloat> &data,
          std::vector<int32> *gselect) const;

      /// As GaussianSelection, but limiting it to a provided list of
      /// preselected Gaussians (e.g. for gender dependency).
      /// The list "preselect" must be sorted and uniq.
      BaseFloat GaussianSelectionPreselectDiag(const PhoneTxCATGselectConfig &config,
          const VectorBase<BaseFloat> &data,
          const std::vector<int32> &preselect,
          std::vector<int32> *gselect) const;
      
      BaseFloat GaussianSelectionPreselect(const PhoneTxCATGselectConfig &config,
          const VectorBase<BaseFloat> &data,
          const std::vector<int32> &preselect,
          std::vector<int32> *gselect) const;

      /// This needs to be called with each new frame of data, prior to accumulation
      /// or likelihood evaluation: it computes various pre-computed quantities. 
      /// We are not using FMLLR currently
      void ComputePerFrameVars(const VectorBase<BaseFloat> &data,
          const std::vector<int32> &gselect,
          PhoneTxCATPerFrameDerivedVars *per_frame_vars) const;

      /// This does a likelihood computation for a given state using the
      /// top-scoring Gaussian components (in per_frame_vars).  If the
      /// log_prune parameter is nonzero (e.g. 5.0), the LogSumExp() stage is
      /// pruned, which is a significant speedup... smaller values are faster.
      BaseFloat LogLikelihood(const PhoneTxCATPerFrameDerivedVars &per_frame_vars,
          int32 state_index, BaseFloat log_prune = 0.0) const;

      /// Similar to LogLikelihood() function above, but also computes the posterior
      /// probabilities for the top-scoring Gaussian components 
      BaseFloat ComponentPosteriors(
          const PhoneTxCATPerFrameDerivedVars &per_frame_vars,
          int32 state, Vector<BaseFloat> *post) const;

      /// Computes (and initializes if necessary) derived vars...
      /// for now this is just the normalizers "n" and the diagonal UBM.
      void ComputeDerivedVars();

      /// Computes the data-independent terms in the log-likelihood computation
      /// for each Gaussian component and all substates. Eq. (31)
      void ComputeNormalizers();

      void ReClusterGaussians(int32 num_clusters, std::vector<int32> *cluster_map);

      /// Various model dimensions.
      int32 NumPdfs() const { return v_[0].size(); }
      int32 NumClusters() const { return A_.size(); }
      int32 NumGauss() const { return SigmaInv_.size(); }
      int32 FeatureDim() const { return diag_ubm_.Dim(); }
      int32 NumWeightClasses() const { 
        //if (weights_map_.Dim() > 0)
        //  return (static_cast<int32> (weights_map_.Max()) + 1); 
        //else
        //  return 0;
        if (weights_map_.size() > 0)
          return (static_cast<int32> (*std::max_element(weights_map_.begin(), 
                  weights_map_.end())) + 1);
        else
          return 0;
      }
      int32 NumTransformClasses() const {
        if (transform_class_.size() > 0)
          return (static_cast<int32> (*std::max_element(
                  transform_class_.begin(), transform_class_.end())) + 1);
        else
          return 1;
      }
      int32 NumClusterWeightClasses() const {
        if (cluster_weight_class_.size() > 0)
          return (static_cast<int32> (*std::max_element(
                  cluster_weight_class_.begin(), cluster_weight_class_.end())) + 1);
        else
          return 1;
      }

      /// Accessors
      const DiagGmm& diag_ubm() const { return diag_ubm_; }
      const FullGmm& full_ubm() const { return full_ubm_; }

      const Vector<BaseFloat>& StateVectors(int32 cluster_weight_class, int32 state_index) const {
        return v_[cluster_weight_class][state_index];
      }

      const Vector<BaseFloat>& ClusterWeights(int32 cluster_weight_class, int32 state_index) const {
        return StateVectors(cluster_weight_class, state_index);
      }

      const SpMatrix<BaseFloat>& InvCovars(int32 gauss_index) const {
        return SigmaInv_[gauss_index];
      }

      const Matrix<BaseFloat>& ClusterTransforms(int32 cluster_index,
          int32 transform_class = 0) const {
        return A_[cluster_index][transform_class];
      }
      
      const Matrix<BaseFloat>& CanonicalMeans() const {
        return u_;
      }

      int32 ClusterWeightClass(int32 i) const {
        return cluster_weight_class_[i];
      }
      
      int32 TransformClass(int32 i) const {
        return transform_class_[i];
      }

      /// Templated accessors (used to accumulate in different precision)
      template<typename Real>
        void GetInvCovars(int32 gauss_index, SpMatrix<Real> *out) const;

      template<typename Real>
        void GetStateMean(int32 j, int32 i,
            VectorBase<Real> *mean_out) const;

      template<typename Real>
        void GetModelSpaceProjection(int32 i, MatrixBase<Real> *Mi) const;
      
      template<typename Real>
        void GetMTrans(int32 i, MatrixBase<Real> *MiT) const;

      /// Computes quantities H = M_i Sigma_i^{-1} M_i^T.
      template<class Real>
        void ComputeH(std::vector< SpMatrix<Real> > *H_i) const;

      /// Computes quantities M_i^T Sigma_i^{-1}.
      template<class Real>
        void ComputeM_SigmaInv(std::vector< Matrix<Real> > *M_SigmaInv) const;

    protected:
      friend class ComputeNormalizersClass;

    private:
      /// Compute a subset of normalizers; used in multi-threaded implementation.
      void ComputeNormalizersInternal(int32 num_threads, int32 thread,
          int32 *entropy_count, double *entropy_sum);

      /// Initialize the within-class covariances
      void InitializeCovars();

      /// Initialize the cannonical means
      void InitializeCanonicalMeans();

      /// Initializes the matrices A
      void InitializeA();

      /// Initialize the state-vectors v_j and w_j
      void InitializeVecs();

    private:
      /// These contain the "background" model associated with the subspace GMM.
      DiagGmm diag_ubm_;
      FullGmm full_ubm_;

      /// Globally shared parameters of the Phone Transform CAT.
      /// The various quantities are: I = number of Gaussians, D = data dimension,
      /// P = number of clusters, 
      /// J = number of states

      /// Inverse within-class (full/diag) covariances; dim is [I][D][D].
      /// Stored and used as as full covariances
      std::vector< SpMatrix<BaseFloat> > SigmaInv_;
      /// Cluster Transforms. Dimension is [P][Q][D][D+1]
      std::vector< std::vector< Matrix<BaseFloat> > > A_;
      /// Canonical model. Dimension is [I][D+1]
      Matrix<BaseFloat> u_;

      /// The parameters in a particular state.

      /// v_{r}{j}, per-state cluster weight vectors. Dimension is [R][J][P].
      /// where r is the regression class
      std::vector< std::vector < Vector<BaseFloat> > > v_;
 
      ///(Deprecated) std::vector< Vector<BaseFloat> > w_;

      /// w_{ji}, mixture weights. Dimension is [W][I] or
      /// w_{i}, Weight Projection. Dimension is [I][P]
      Matrix<BaseFloat> w_;

      /// n_{ji}, normalizers. Dimension is [J][I]
      std::vector< Vector<BaseFloat> > n_;

      /// Mapping from tied state to cluster
      //Vector<BaseFloat> pdf_id_to_cluster_;
      std::vector<int32> pdf_id_to_cluster_;

      /// Mapping for the weights of each tied state
      /// In base version 1, each weights_map_[j] = j
      /// In modified version 1.1, weight_map_[j] = pdf_id_to_cluster_[j]  
      //Vector<BaseFloat> weights_map_;
      std::vector<int32> weights_map_;

      //Vector<BaseFloat> transform_class_;
      std::vector<int32> transform_class_;
      std::vector<int32> cluster_weight_class_;
      
      bool use_weight_projection_;
      bool use_full_covar_;

      KALDI_DISALLOW_COPY_AND_ASSIGN(AmPhoneTxCAT);
      friend class MleAmPhoneTxCATUpdater;
      friend class AmPhoneTxCATFunctions;  // misc functions that need access.
  };

  template<typename Real>
    inline void AmPhoneTxCAT::GetInvCovars(int32 gauss_index,
        SpMatrix<Real> *out) const {
      out->Resize(SigmaInv_[gauss_index].NumRows(), kUndefined);
      out->CopyFromSp(SigmaInv_[gauss_index]);
    }

  template<typename Real>
    inline void AmPhoneTxCAT::GetStateMean(int32 j, int32 i,
        VectorBase<Real> *mean_out) const {
      KALDI_ASSERT(mean_out != NULL);
      KALDI_ASSERT(j < NumPdfs() && i < NumGauss());
      KALDI_ASSERT(mean_out->Dim() == FeatureDim());
      Vector<BaseFloat> mean_tmp(FeatureDim());
      Matrix<BaseFloat> Mi(FeatureDim(), NumClusters());
      GetModelSpaceProjection(i, &Mi);
      // mean = M_{i} v_{j}
      int32 r = cluster_weight_class_[i];
      mean_tmp.AddMatVec(1.0, Mi, kNoTrans, v_[r][j], 0.0);
      mean_out->CopyFromVec(mean_tmp);
    }

  template<typename Real>
    inline void AmPhoneTxCAT::GetModelSpaceProjection(int32 i, 
        MatrixBase<Real> *Mi) const {
      KALDI_ASSERT(Mi != NULL);
      KALDI_ASSERT(i < NumGauss());
      int32 num_clusters = NumClusters();
      KALDI_ASSERT(Mi->NumCols() == num_clusters &&
          Mi->NumRows() == FeatureDim());

      for (int32 p = 0; p < num_clusters; p++) {
        Vector<BaseFloat> mu_p(FeatureDim());
        // The pth column of M_{i} is A_{p}*u_{i}
        int32 q = transform_class_[i];
        mu_p.AddMatVec(1.0, A_[p][q], kNoTrans, u_.Row(i), 0.0);
        Mi->CopyColFromVec(mu_p, p); 
      }
    }
  
  template<typename Real>
    inline void AmPhoneTxCAT::GetMTrans(int32 i, 
        MatrixBase<Real> *MiT) const {
      KALDI_ASSERT(MiT != NULL);
      KALDI_ASSERT(i < NumGauss());
      int32 num_clusters = NumClusters();
      KALDI_ASSERT(MiT->NumRows() == num_clusters &&
          MiT->NumCols() == FeatureDim());

      for (int32 p = 0; p < num_clusters; p++) {
        Vector<BaseFloat> mu_p(FeatureDim());
        // The pth row of M^T{i} is A_{p}*u_{i}
        int32 q = transform_class_[i];
        mu_p.AddMatVec(1.0, A_[p][q], kNoTrans, u_.Row(i), 0.0);
        MiT->Row(p).CopyFromVec(mu_p); 
      }
    }

  /// This is the entry for a single time.
  struct PhoneTxCATGauPostElement {
    // Need gselect info here, since "posteriors" is  relative to this set of
    // selected Gaussians.
    std::vector<int32> gselect;
    std::vector<int32> tids;  // transition-ids for each entry in "posteriors"
    std::vector<Matrix<BaseFloat> > posteriors;
  };

  class PhoneTxCATGauPost: public std::vector<PhoneTxCATGauPostElement> {
    public:
      // Add the standard Kaldi Read and Write routines so
      // we can use KaldiObjectHolder with this type.
      explicit PhoneTxCATGauPost(size_t i) : 
        std::vector<PhoneTxCATGauPostElement>(i) {}
      PhoneTxCATGauPost() {}
      void Write(std::ostream &os, bool binary) const;
      void Read(std::istream &is, bool binary);
  };

  typedef KaldiObjectHolder<PhoneTxCATGauPost> PhoneTxCATGauPostHolder;
  typedef RandomAccessTableReader<PhoneTxCATGauPostHolder> RandomAccessPhoneTxCATGauPostReader;
  typedef SequentialTableReader<PhoneTxCATGauPostHolder> SequentialPhoneTxCATGauPostReader;
  typedef TableWriter<PhoneTxCATGauPostHolder> PhoneTxCATGauPostWriter;

} // namespace kaldi

#endif
