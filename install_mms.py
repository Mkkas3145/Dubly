"""Prepare the official multilingual aligner weights, without any ASR model."""
from core import ROOT

def main():
    from torchaudio.pipelines import MMS_FA
    folder=ROOT/'models/mms-forced-aligner';folder.mkdir(parents=True,exist_ok=True)
    MMS_FA.get_model(with_star=False,dl_kwargs={'model_dir':str(folder)})
    print('MMS-FA alignment weights are ready.')

if __name__=='__main__':main()
