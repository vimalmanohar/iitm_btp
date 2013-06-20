#include "gmm/model-test-common.h"
#include "phoneTxCAT/am-phoneTxCAT.h"
#include "util/kaldi-io.h"

using kaldi::AmPhoneTxCAT;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the initialization routines: InitializeFromDiagGmm(), CopyFromPhoneTxCAT()
// and CopyGlobalsInitVecs().
void TestPhoneTxCATInit(const AmPhoneTxCAT &phoneTxCAT) {
  using namespace kaldi;
  int32 dim = phoneTxCAT.FeatureDim();
  kaldi::PhoneTxCATGselectConfig config;
  config.diag_gmm_nbest = std::min(config.diag_gmm_nbest, phoneTxCAT.NumGauss());
  // Ensure that the number of selected gaussians does not exceed the 
  // number of gaussians in the model
  
  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++){
    feat(d) = kaldi::RandGauss();
  } // Initialize a random feature vector

  kaldi::PhoneTxCATPerFrameDerivedVars frame_vars;
  frame_vars.Resize(phoneTxCAT.NumGauss(), phoneTxCAT.FeatureDim(),
      phoneTxCAT.NumClusters());

  // Do gaussian selection for this frame
  std::vector<int32> gselect;
  phoneTxCAT.GaussianSelection(config, feat, &gselect);
  
  // Compute per frame vars
  PhoneTxCATPerFrameDerivedVars per_frame; 
  phoneTxCAT.ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike = phoneTxCAT.LogLikelihood(per_frame, 0);

  // First, test the CopyFromPhoneTxCAT() method:
  AmPhoneTxCAT *phoneTxCAT1 = new AmPhoneTxCAT();
  phoneTxCAT1->CopyFromPhoneTxCAT(phoneTxCAT, true);
  phoneTxCAT1->GaussianSelection(config, feat, &gselect);
  phoneTxCAT1->ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike1 = phoneTxCAT1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike1, 1e-4);
  delete phoneTxCAT1;
  
  AmPhoneTxCAT *phoneTxCAT2 = new AmPhoneTxCAT();
  phoneTxCAT2->CopyFromPhoneTxCAT(phoneTxCAT, false);
  phoneTxCAT2->ComputeNormalizers();
  phoneTxCAT2->GaussianSelection(config, feat, &gselect);
  phoneTxCAT2->ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike2 = phoneTxCAT2->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete phoneTxCAT2;

  // Next, initialize using the UBM from the current model
  AmPhoneTxCAT *phoneTxCAT3 = new AmPhoneTxCAT();
  phoneTxCAT3->InitializeFromDiagGmm(phoneTxCAT.diag_ubm(), 
      phoneTxCAT.NumPdfs(), phoneTxCAT.NumClusters());
  phoneTxCAT3->ComputeNormalizers();
  phoneTxCAT3->GaussianSelection(config, feat, &gselect);
  phoneTxCAT3->ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike3 = phoneTxCAT3->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike3, 1e-4);
  delete phoneTxCAT3;

  // Finally, copy the global parameters from the current model
  AmPhoneTxCAT *phoneTxCAT4 = new AmPhoneTxCAT();
  phoneTxCAT4->CopyGlobalsInitVecs(phoneTxCAT,
      phoneTxCAT.NumClusters(), phoneTxCAT.NumPdfs());
  phoneTxCAT4->ComputeNormalizers();
  phoneTxCAT4->GaussianSelection(config, feat, &gselect);
  phoneTxCAT4->ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike4 = phoneTxCAT4->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike4, 1e-4);
  delete phoneTxCAT4;
}

// Tests the Read() and Write() methods, in both binary and ASCII mode, as well
// as Check(), and methods in likelihood computations.

void TestPhoneTxCATIO(const AmPhoneTxCAT &phoneTxCAT) {
  using namespace kaldi;
  int32 dim = phoneTxCAT.FeatureDim();
  kaldi::PhoneTxCATGselectConfig config;
  config.diag_gmm_nbest = std::min(config.diag_gmm_nbest, 
      phoneTxCAT.NumGauss());

  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++){
    feat(d) = kaldi::RandGauss();
  } // Initialize a random feature vector
  
  kaldi::PhoneTxCATPerFrameDerivedVars frame_vars;
  frame_vars.Resize(phoneTxCAT.NumGauss(), phoneTxCAT.FeatureDim(),
      phoneTxCAT.NumClusters());

  // Do gaussian selection for this frame
  std::vector<int32> gselect;
  phoneTxCAT.GaussianSelection(config, feat, &gselect);
  
  // Compute per frame vars
  PhoneTxCATPerFrameDerivedVars per_frame; 
  phoneTxCAT.ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike = phoneTxCAT.LogLikelihood(per_frame, 0);

  // First, non-binary write
  phoneTxCAT.Write(kaldi::Output("tmpf", false).Stream(), false, kaldi::kPhoneTxCATWriteAll);

  bool binary_in;
  AmPhoneTxCAT *phoneTxCAT1 = new AmPhoneTxCAT();

  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  phoneTxCAT1->Read(ki1.Stream(), binary_in);
  phoneTxCAT1->Check(true);
  phoneTxCAT1->GaussianSelection(config, feat, &gselect);
  phoneTxCAT1->ComputePerFrameVars(feat, gselect, &per_frame);

  BaseFloat loglike1 = phoneTxCAT1->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike1, 1e-4);
  
  // Next, binary write
  phoneTxCAT1->Write(kaldi::Output("tmpfb", true).Stream(), true,
      kaldi::kPhoneTxCATWriteAll);
  delete phoneTxCAT1;

  AmPhoneTxCAT *phoneTxCAT2 = new AmPhoneTxCAT();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  phoneTxCAT2->Read(ki2.Stream(), binary_in);
  phoneTxCAT2->Check(true);
  phoneTxCAT2->GaussianSelection(config, feat, &gselect);
  phoneTxCAT2->ComputePerFrameVars(feat, gselect, &per_frame);
  BaseFloat loglike2 = phoneTxCAT2->LogLikelihood(per_frame, 0);
  kaldi::AssertEqual(loglike, loglike2, 1e-4);
  delete phoneTxCAT2;
}

void UnitTestPhoneTxCAT() {
  size_t dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  size_t num_comp = 3 + kaldi::RandInt(0, 9);  // random number of mixtures;
  // make sure it's more than one or we get errors initializing the phoneTxCAT.
  kaldi::DiagGmm diag_gmm;
  ut::InitRandDiagGmm(dim, num_comp, &diag_gmm);

  size_t num_states = 1;
  AmPhoneTxCAT phoneTxCAT;
  kaldi::PhoneTxCATGselectConfig config;
  phoneTxCAT.InitializeFromDiagGmm(diag_gmm, num_states, dim+1);
  phoneTxCAT.ComputeNormalizers();
  TestPhoneTxCATInit(phoneTxCAT);
  TestPhoneTxCATIO(phoneTxCAT);
}

int main() {
  for (int i = 0; i < 10; i++)
    UnitTestPhoneTxCAT();
  std::cout << "Test OK.\n";
  return 0;
}
