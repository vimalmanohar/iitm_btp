// phoneTxCATbin/phoneTxCAT-sum-accs.cc

#include "util/common-utils.h"
#include "phoneTxCAT/estimate-am-phoneTxCAT.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for Phone Transform CAT training.\n"
        "Usage: phoneTxCAT-sum-accs [options] stats-out stats-in1 stats-in2 ...\n";
    
    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);
    
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_out_filename = po.GetArg(1);
    kaldi::Vector<double> transition_accs;
    kaldi::MleAmPhoneTxCATAccs phoneTxCAT_accs;

    for (int i = 2, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read, true /* add values */);
      phoneTxCAT_accs.Read(ki.Stream(), binary_read, true /* add values */);
    }

    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
      transition_accs.Write(ko.Stream(), binary);
      phoneTxCAT_accs.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written stats to " << stats_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



