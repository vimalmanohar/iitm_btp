#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "thread/kaldi-thread.h"
#include "phoneTxCAT/am-phoneTxCAT.h"
#include "hmm/transition-model.h"
#include "phoneTxCAT/estimate-am-phoneTxCAT.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Estimate Phone Transform CAT model parameters from accumulated stats.\n"
        "Usage: phoneTxCAT-est [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = true;
    std::string update_flags_str = "vAwuSt";
    std::string write_flags_str = "gsnu";
    kaldi::MleTransitionUpdateConfig tcfg;
    kaldi::MleAmPhoneTxCATOptions phoneTxCAT_opts;

    std::string occs_out_filename;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vAwuSt.");
    po.Register("write-flags", &write_flags_str, "Which SGMM parameters to "
                "write: subset of gsnu");
    po.Register("num-threads", &g_num_threads, "Number of threads to use in "
                "weight update and normalizer computation");
    tcfg.Register(&po);
    phoneTxCAT_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);
    
    kaldi::PhoneTxCATUpdateFlagsType update_flags =
        StringToPhoneTxCATUpdateFlags(update_flags_str);
    kaldi::SgmmWriteFlagsType write_flags =
        StringToPhoneTxCATWriteFlags(write_flags_str);
    
    AmPhoneTxCAT am_phoneTxCAT;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_phoneTxCAT.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    MleAmPhoneTxCATAccs phoneTxCAT_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      phoneTxCAT_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }
    
    if (update_flags & kPhoneTxCATTransitions) {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: Overall " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }
    
    // Check the dimension and the attributes of the statistics 
    // in the accs and ensure it matches with the model's
    phoneTxCAT_accs.Check(am_phoneTxCAT, true); // Will check consistency and print some diagnostics.

    { // Do the update.
      kaldi::MleAmPhoneTxCATUpdater updater(phoneTxCAT_opts);
      updater.Update(phoneTxCAT_accs, &am_phoneTxCAT, update_flags);
    }

    if (!occs_out_filename.empty()) {  // get state occs
      Vector<BaseFloat> state_occs;
      phoneTxCAT_accs.GetStateOccupancies(&state_occs);
      
      if (!occs_out_filename.empty()) {
        kaldi::Output ko(occs_out_filename, false /* no binary write */);
        state_occs.Write(ko.Stream(), false  /* no binary write */);
      }
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_phoneTxCAT.Write(ko.Stream(), binary_write, write_flags);
    }
    
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



