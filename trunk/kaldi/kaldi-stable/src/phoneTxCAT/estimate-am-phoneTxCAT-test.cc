// phoneTxCAT/estimate-am-phoneTxCAT-test.cc

// Author: Vimal Manohar

#include "gmm/model-test-common.h"
#include "phoneTxCAT/am-phoneTxCAT.h"
#include "phoneTxCAT/estimate-am-phoneTxCAT.h"
#include "util/kaldi-io.h"

using kaldi::AmPhoneTxCAT;
using kaldi::MleAmPhoneTxCATAccs;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

// Tests the Read() and Write() methods for the accumulators, in both binary
// and ASCII mode, as well as Check().
void TestPhoneTxCATAccsIO(const AmPhoneTxCAT &phoneTxCAT,
                    const kaldi::Matrix<BaseFloat> &feats) {
  using namespace kaldi;
  kaldi::PhoneTxCATUpdateFlagsType flags = kaldi::kPhoneTxCATAll;
  kaldi::PhoneTxCATPerFrameDerivedVars frame_vars;
  frame_vars.Resize(phoneTxCAT.NumGauss(), phoneTxCAT.FeatureDim(),
                    phoneTxCAT.NumClusters());
  kaldi::PhoneTxCATGselectConfig phoneTxCAT_config;
  phoneTxCAT_config.diag_gmm_nbest = std::min(phoneTxCAT_config.diag_gmm_nbest,
      phoneTxCAT.NumGauss());
  MleAmPhoneTxCATAccs accs(phoneTxCAT, flags);
  BaseFloat loglike = 0.0;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    std::vector<int32> gselect;
    phoneTxCAT.GaussianSelection(phoneTxCAT_config, feats.Row(i), &gselect);
    phoneTxCAT.ComputePerFrameVars(feats.Row(i), gselect, &frame_vars);
    loglike += accs.Accumulate(phoneTxCAT, frame_vars, 0, 1.0, flags);
  }

  kaldi::MleAmPhoneTxCATOptions update_opts;

  AmPhoneTxCAT *phoneTxCAT1 = new AmPhoneTxCAT();
  phoneTxCAT1->CopyFromPhoneTxCAT(phoneTxCAT, false);
  kaldi::MleAmPhoneTxCATUpdater updater(update_opts);
  updater.Update(accs, phoneTxCAT1, flags);
  std::vector<int32> gselect;

  phoneTxCAT1->GaussianSelection(phoneTxCAT_config, feats.Row(0), &gselect);
  phoneTxCAT1->ComputePerFrameVars(feats.Row(0), gselect, &frame_vars);
  BaseFloat loglike1 = phoneTxCAT1->LogLikelihood(frame_vars, 0);
  delete phoneTxCAT1;

  // First, non-binary write
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);
  bool binary_in;
  MleAmPhoneTxCATAccs *accs1 = new MleAmPhoneTxCATAccs();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  accs1->Check(phoneTxCAT, true);
  AmPhoneTxCAT *phoneTxCAT2 = new AmPhoneTxCAT();
  phoneTxCAT2->CopyFromPhoneTxCAT(phoneTxCAT, false);
  updater.Update(*accs1, phoneTxCAT2, flags);

  phoneTxCAT2->GaussianSelection(phoneTxCAT_config, feats.Row(0), &gselect);
  phoneTxCAT2->ComputePerFrameVars(feats.Row(0), gselect, &frame_vars);
  BaseFloat loglike2 = phoneTxCAT2->LogLikelihood(frame_vars, 0);
  kaldi::AssertEqual(loglike1, loglike2, 1e-4);
  delete accs1;

  // Next, binary write
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  MleAmPhoneTxCATAccs *accs2 = new MleAmPhoneTxCATAccs();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  accs2->Check(phoneTxCAT, true);
  AmPhoneTxCAT *phoneTxCAT3 = new AmPhoneTxCAT();
  phoneTxCAT3->CopyFromPhoneTxCAT(phoneTxCAT, false);
  updater.Update(*accs2, phoneTxCAT3, flags);
  phoneTxCAT3->GaussianSelection(phoneTxCAT_config, feats.Row(0), &gselect);
  phoneTxCAT3->ComputePerFrameVars(feats.Row(0), gselect, &frame_vars);
  BaseFloat loglike3 = phoneTxCAT3->LogLikelihood(frame_vars, 0);
  kaldi::AssertEqual(loglike1, loglike3, 1e-6);
  delete accs2;
  delete phoneTxCAT2;
  delete phoneTxCAT3;
}

void UnitTestEstimatePhoneTxCAT() {
  int32 dim = 1 + kaldi::RandInt(0, 9);  // random dimension of the gmm
  int32 num_comp = 2 + kaldi::RandInt(0, 9);  // random mixture size
  kaldi::DiagGmm diag_gmm;
  ut::InitRandDiagGmm(dim, num_comp, &diag_gmm);

  int32 num_states = 1;
  AmPhoneTxCAT phoneTxCAT;
  kaldi::PhoneTxCATGselectConfig config;
  phoneTxCAT.InitializeFromDiagGmm(diag_gmm, num_states, dim);
  phoneTxCAT.ComputeNormalizers();

  kaldi::Matrix<BaseFloat> feats;

  {  // First, generate random means and variances
    int32 num_feat_comp = num_comp + kaldi::RandInt(-num_comp/2, num_comp/2);
    kaldi::Matrix<BaseFloat> means(num_feat_comp, dim),
        vars(num_feat_comp, dim);
    for (int32 m = 0; m < num_feat_comp; m++) {
      for (int32 d= 0; d < dim; d++) {
        means(m, d) = kaldi::RandGauss();
        vars(m, d) = exp(kaldi::RandGauss()) + 1e-2;
      }
    }
    // Now generate random features with those means and variances.
    feats.Resize(num_feat_comp * 200, dim);
    for (int32 m = 0; m < num_feat_comp; m++) {
      kaldi::SubMatrix<BaseFloat> tmp(feats, m*200, 200, 0, dim);
      ut::RandDiagGaussFeatures(200, means.Row(m), vars.Row(m), &tmp);
    }
  }
  TestPhoneTxCATAccsIO(phoneTxCAT, feats);
}

int main() {
  KALDI_ERR << "Code not in shape. Need to be edited!";
  for (int i = 0; i < 10; i++)
    UnitTestEstimatePhoneTxCAT();
  std::cout << "Test OK.\n";
  return 0;
}

