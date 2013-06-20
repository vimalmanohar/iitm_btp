#ifndef KALDI_PHONETXCAT_ESTIMATE_AM_PHONETXCAT_H_
#define KALDI_PHONETXCAT_ESTIMATE_AM_PHONETXCAT_H_ 1

#include <string>
#include <vector>

#include "phoneTxCAT/am-phoneTxCAT.h"
#include "gmm/model-common.h"
#include "util/parse-options.h"
//#include "phoneTxCAT/phoneTxCAT-clusterable.h"
#include "thread/kaldi-thread.h"  // for MultiThreadable

namespace kaldi {

  /** \struct MleAmPhoneTxCATOptions
   *  Configuration variables needed in the Phone Transform CAT
   *  estimation process.
   */
  struct MleAmPhoneTxCATOptions {
    /// Configuration Parameters.  
    /// See initialization code for more comments.

    /// Floor covariance matrices Sigma_i to this times average cov.
    BaseFloat cov_floor;
    /// ratio to dim below which we use diagonal. 
    /// default 2, set to >1000 for diag.
    BaseFloat cov_diag_ratio;

    /// Max on condition of matrices in update beyond which 
    /// we do not update.
    /// Should probably be related to numerical properties of machine
    /// or BaseFloat type.
    BaseFloat max_cond;

    BaseFloat epsilon;  ///< very small value used to prevent SVD crashing.
    //BaseFloat min_gaussian_occupancy;
    BaseFloat min_gaussian_weight;

    /// The "sequential" weight update that checks each i in turn.
    /// (if false, uses the "parallel" one).
    bool use_sequential_weight_update;
    bool use_sequential_transform_update;

    int32 weight_projections_iters;
    int32 cluster_transforms_iters;
  
    bool use_block_diagonal_transform;
    bool use_diagonal_transform;
    
    bool use_block_diagonal_transform2;
    bool use_diagonal_transform2;
    
    bool use_block_diagonal_transform3;
    bool use_diagonal_transform3;

    bool use_class_dep_steps;
    bool use_sequential_multiple_txclass_update;

    MleAmPhoneTxCATOptions() {
      cov_floor = 0.025;
      cov_diag_ratio = 2.0;  // set to very large to get diagonal-cov models.
      max_cond = 1.0e+05;
      epsilon = 1.0e-40;
      min_gaussian_weight     = 1.0e-10;
      //min_gaussian_occupancy  = 10.0;
      use_sequential_weight_update = false;
      use_sequential_transform_update = true;
      weight_projections_iters = 10;
      cluster_transforms_iters = 10;

      use_block_diagonal_transform = false;
      use_diagonal_transform = false;
      use_block_diagonal_transform2 = false;
      use_diagonal_transform2 = false;
      use_block_diagonal_transform3 = false;
      use_diagonal_transform3 = false;

      use_class_dep_steps = false;
      use_sequential_multiple_txclass_update = false;
    }

    void Register(ParseOptions *po) {
      std::string module = "MleAmPhoneTxCATOptions: ";
      po->Register("cov-floor", &cov_floor, module+
          "Covariance floor (fraction of average covariance).");
      po->Register("cov-diag-ratio", &cov_diag_ratio, module+
          "Minumum occ/dim ratio below which use diagonal covariances.");
      po->Register("max-cond", &max_cond, module+
          "Maximum condition number beyond\
          which matrices are not updated.");
      po->Register("min-gaussian-weight", &min_gaussian_weight,
          module+"Min Gaussian weight before we floor it");
      po->Register("weight-projections-iters", &weight_projections_iters, module+
          "Iterations for Weight Projection estimation");
      po->Register("cluster-transforms-iters", &cluster_transforms_iters, module+
          "Iterations for Cluster Transform estimation");
      po->Register("use-sequential-weight-update", 
          &use_sequential_weight_update, 
          module+"Update Weight projection sequentially");
      po->Register("use-sequential-transform-update", 
          &use_sequential_transform_update, 
          module+"Update Transforms sequentially");
      po->Register("use-block-diagonal-transform",
          &use_block_diagonal_transform,
          module+"Use Block Diagonal MLLR Transform");
      po->Register("use-diagonal-transform",
          &use_diagonal_transform,
          module+"Use Diagonal MLLR Transform");
      po->Register("use-block-diagonal-transform2",
          &use_block_diagonal_transform2,
          module+"Use Block Diagonal MLLR Transform 2");
      po->Register("use-diagonal-transform2",
          &use_diagonal_transform2,
          module+"Use Diagonal MLLR Transform 2");
      po->Register("use-block-diagonal-transform3",
          &use_block_diagonal_transform3,
          module+"Use Block Diagonal MLLR Transform 3");
      po->Register("use-diagonal-transform3",
          &use_diagonal_transform3,
          module+"Use Diagonal MLLR Transform 3");
      po->Register("use-class-dep-steps", 
          &use_class_dep_steps,
          module+"Use different step sizes for different transform classes");
      po->Register("use-sequential-multiple-txclass-update", 
          &use_sequential_multiple_txclass_update, 
          module+"Update Transforms of multiple classes sequentially");

      if (use_block_diagonal_transform2 || use_diagonal_transform2) {
        use_diagonal_transform = false;
        use_block_diagonal_transform = false;
      }

      if (use_block_diagonal_transform3 || use_diagonal_transform3) {
        use_diagonal_transform2 = false;
        use_block_diagonal_transform2 = false;
      }
      //po->Register("min-gaussian-occupancy", &min_gaussian_occupancy,
      //    module+"Minimum occupancy to update a Gaussian.");
    }
  };

  /** \class MleAmPhoneTxCATAccs
   *  Class for the accumulators associated with the Phone Transform CAT
   *  parameters 
   */
  class MleAmPhoneTxCATAccs {
    public:
      explicit MleAmPhoneTxCATAccs(BaseFloat rand_prune = 1.0e-05)
        : total_frames_(0.0), total_like_(0.0), feature_dim_(0),
        num_clusters_(0), num_gaussians_(0),
        num_states_(0), rand_prune_(rand_prune) {}

      MleAmPhoneTxCATAccs(const AmPhoneTxCAT &model, 
          PhoneTxCATUpdateFlagsType flags,
          BaseFloat rand_prune = 1.0e-05)
        : total_frames_(0.0), total_like_(0.0), 
        rand_prune_(rand_prune) {
          ResizeAccumulators(model, flags);
        }

      ~MleAmPhoneTxCATAccs() {};

      void Read(std::istream &in_stream, bool binary, bool add);
      void Write(std::ostream &out_stream, bool binary) const;

      /// Checks the various accumulators for correct sizes given a model.
      /// With wrong sizes, assertion failure occurs. 
      /// When the show_properties argument
      /// is set to true, dimensions and presence/absence of the various
      /// accumulators are printed. 
      /// For use when accumulators are read from file.
      void Check(const AmPhoneTxCAT &model, bool show_properties = true) const;

      /// Resizes the accumulators to the correct sizes given the model. 
      /// The flags argument control which accumulators to resize.
      void ResizeAccumulators(const AmPhoneTxCAT &model, 
          PhoneTxCATUpdateFlagsType flags);

      /// Returns likelihood.
      BaseFloat Accumulate(const AmPhoneTxCAT &model,
          const PhoneTxCATPerFrameDerivedVars &frame_vars, 
          int32 j, BaseFloat weight,
          PhoneTxCATUpdateFlagsType flags); 

      /// Returns count accumulated (may differ from posteriors.Sum()
      /// due to weight pruning).
      BaseFloat AccumulateFromPosteriors(
          const AmPhoneTxCAT &model,
          const PhoneTxCATPerFrameDerivedVars &frame_vars,
          const Vector<BaseFloat> &posteriors,
          int32 j,
          PhoneTxCATUpdateFlagsType flags);

      /// Accessors
      void GetStateOccupancies(Vector<BaseFloat> *occs) const;
      const std::vector< Vector<double> >& GetOccs() const {
        return gamma_;
      }
      int32 FeatureDim() const { return feature_dim_; }
      int32 NumClusters() const { return num_clusters_; }
      int32 NumStates() const { return num_states_; }
      int32 NumGauss() const { return num_gaussians_; }
      int32 NumClusterWeightClasses() const { return num_cluster_weight_classes_; }
      double TotalFrames() const { return total_frames_; }
      double TotalLike() const { return total_like_; }

    private:
      /// The stats which are not tied to any state.
      /// Stats G_{i} for cluster transforms A. Dim is [I][P][P]
      std::vector< SpMatrix<double> > G_;
      /// Stats K_{i} for cluster transforms A. Dim is [I][P][D]
      std::vector< Matrix<double> > K_;
      /// Stats L_{i} for shared covariance S. Dim is [i][D][D]
      std::vector< SpMatrix<double> > L_;

      /// The state specific stats
      /// Stats Z_{j} for cluster weights v_{j}. Dim is [J][I][D]
      /// (Deprecated) std::vector< Matrix<double> > Z_;
      
      /// Stats y_{r}{j} for cluster weights v_{r}{j}. Dim is [R][J][P]
      std::vector< std::vector< Vector<double> > > y_;

      /// Gaussian occupancies gamma_{ji} for each state. Dim is [J][I]
      std::vector< Vector<double> > gamma_;

      double total_frames_, total_like_;

      /// Dimensionality
      int32 feature_dim_, num_clusters_;
      int32 num_gaussians_, num_states_, num_cluster_weight_classes_;

      BaseFloat rand_prune_;

      KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmPhoneTxCATAccs);
      friend class MleAmPhoneTxCATUpdater;
      friend class MleAmPhoneTxCATGlobalAccs;
  };

  /** \class MleAmPhoneTxCATUpdater
   *  Contains the functions needed to update the 
   *  Phone Transform CAT parameters.
   */
  class MleAmPhoneTxCATUpdater {
    public:
      explicit MleAmPhoneTxCATUpdater(const MleAmPhoneTxCATOptions &options)
        : update_options_(options) {}
      void Reconfigure(const MleAmPhoneTxCATOptions &options) {
        update_options_ = options;
      }

      /// Main update function: Computes some overall stats, does parameter updates
      /// and returns the total improvement of the different auxiliary functions.
      BaseFloat Update(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          PhoneTxCATUpdateFlagsType flags);
 
    protected:
      friend class UpdateWParallelClass;
      friend class UpdateStateVectorsClass;
      friend class UpdateASequentialClass;
      friend class UpdateAParallelClass;
      friend class UpdateAParallelClass2;
      friend class UpdateA_DiagCovParallelClass;
    private:
      MleAmPhoneTxCATOptions update_options_;

      Vector<double> gamma_j_;  ///< State occupancies

      // UpdateStateVectors function
      double UpdateStateVectors(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          const std::vector< SpMatrix<double> > &H);

      void UpdateStateVectorsInternal(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          const std::vector< SpMatrix<double> > &H,
          double *auxf_impr,
          double *like_impr,
          int32 num_threads,
          int32 thread_id) const;

      double UpdateASequential(const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model);
      
      void UpdateASequentialInternal(
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
          int32 thread_id) const;
      
      double UpdateAParallel(const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model);
      void UpdateAParallelInternal(
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          std::vector< Matrix<double> > *delta_A,
          const int32 p,
          const std::vector< SpMatrix<double> > &xi2,
          const std::vector< Matrix<double> > &L,
          const Matrix<double> &variance,
          int32 num_threads,
          int32 thread_id) const;
      
      double UpdateAParallel2(const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model);
      void UpdateAParallelInternal2(
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          Matrix<double> *delta_Apq,
          const int32 p,
          const int32 tx_class,
          const std::vector< SpMatrix<double> > &xi2,
          const Matrix<double> &L,
          const Matrix<double> &variance,
          int32 num_threads,
          int32 thread_id) const;

      double UpdateA_DiagCov(const MleAmPhoneTxCATAccs &accs, AmPhoneTxCAT *model);
      void UpdateA_DiagCovParallelInternal(
          const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model,
          const int32 p,
          double *like_impr,
          int32 num_threads,
          int32 thread_id) const;
      
      
      double UpdateWeights(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);
      
      double UpdateWeightsBasic(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);

      double UpdateWParallel(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);
      
      /// Called, multithreaded, inside UpdateWParallel
      static
        void UpdateWParallelGetStats(const MleAmPhoneTxCATAccs &accs,
            const AmPhoneTxCAT &model,
            const Matrix<double> &w,
            Matrix<double> *F_i,
            Matrix<double> *g_i,
            double *tot_like,
            int32 num_threads, 
            int32 thread_id);
      
      double UpdateWSequential(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);
      
      double UpdateVars(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);
      double UpdateDiagVars(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);
    
      double UpdateCanonicalMeans(const MleAmPhoneTxCATAccs &accs,
          AmPhoneTxCAT *model);

      KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmPhoneTxCATUpdater);
      MleAmPhoneTxCATUpdater() {}
  };

  // This class, used in multi-core implementation 
  // of the updates of the "w_i" quantities
  // It is responsible for
  // computing, in parallel, the F_i and g_i quantities 
  // used in the updates of w_i.
  
  class UpdateWParallelClass: public MultiThreadable {
    public:
      UpdateWParallelClass(const MleAmPhoneTxCATAccs &accs,
          const AmPhoneTxCAT &model,
          const Matrix<double> &w,
          Matrix<double> *F_i,
          Matrix<double> *g_i,
          double *tot_like):
        accs_(accs), model_(model), w_(w),
        F_i_ptr_(F_i), g_i_ptr_(g_i), tot_like_ptr_(tot_like) {
          tot_like_ = 0.0;
          F_i_.Resize(F_i->NumRows(), F_i->NumCols());
          g_i_.Resize(g_i->NumRows(), g_i->NumCols());
        }

      ~UpdateWParallelClass() {
        F_i_ptr_->AddMat(1.0, F_i_, kNoTrans);
        g_i_ptr_->AddMat(1.0, g_i_, kNoTrans);
        *tot_like_ptr_ += tot_like_;
      }

      inline void operator() () {
        // Note: give them local copy of the sums we're computing,
        // which will be propagated to the total sums in the destructor.
        MleAmPhoneTxCATUpdater::UpdateWParallelGetStats(accs_, model_, w_,
            &F_i_, &g_i_, &tot_like_,
            num_threads_, thread_id_);
      }
    private:
      const MleAmPhoneTxCATAccs &accs_;
      const AmPhoneTxCAT &model_;
      const Matrix<double> &w_;
      Matrix<double> *F_i_ptr_;
      Matrix<double> *g_i_ptr_;
      Matrix<double> F_i_;
      Matrix<double> g_i_;
      double *tot_like_ptr_;
      double tot_like_;
  };





} // end namespace kaldi

#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_H_

