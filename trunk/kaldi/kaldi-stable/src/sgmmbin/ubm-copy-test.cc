#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/model-test-common.h"
#include "gmm/full-gmm.h"
#include "util/kaldi-io.h"

using kaldi::FullGmm;
using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

void TestFullGmmHTKIO(const FullGmm &full_gmm) {
  int32 dim = full_gmm.Dim();

  kaldi::Vector<BaseFloat> feat(dim);
  for (int32 d = 0; d < dim; d++) {
    feat(d) = kaldi::RandGauss();
  }
  
  BaseFloat loglike = full_gmm.LogLikelihood(feat);

  // First, non-binary write
  full_gmm.Write(kaldi::Output("tmpf", false).Stream(), false);

  bool binary_in;
  FullGmm *full_gmm1 = new FullGmm();
  //Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  full_gmm1->Read(ki1.Stream(), binary_in);
  BaseFloat loglike1 = full_gmm1->LogLikelihood(feat);

  kaldi::AssertEqual(loglike, loglike1, 1e-4);
  
  delete full_gmm1;
}

void UnitTestFullGmm() {
  int32 dim = 1 + kaldi::RandInt(0, 9),  // random dimension of the gmm
      num_pdfs = 5 + kaldi::RandInt(0, 9);  // random number of states

  int32 num_comp = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  kaldi::FullGmm gmm;
  ut::InitRandFullGmm(dim, num_comp, &gmm);

  TestFullGmmHTKIO(gmm);
}

int main() {
  for (int i = 0; i < 10; i++)
    UnitTestFullGmm();
  std::cout << "Test OK.\n";
  return 0;
}




