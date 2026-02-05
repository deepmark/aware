import numpy as np
import librosa
from pathlib import Path
from aware.utils import logger
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from attacks import NoAttack, PCMBitDepthConversion, MP3Compression, DeleteSamples, PitchShift, TimeStretch, Resample, RandomBandstop, SampleSupression, LowPassFilter, HighPassFilter
from aware.metrics.audio import PESQ, BER, STOI

import logging
logger.setLevel(logging.DEBUG)

def main():
    attack_list = [ NoAttack(), PCMBitDepthConversion(8),  
                    MP3Compression(9), MP3Compression(0), 
                    DeleteSamples(0.5), TimeStretch(0.7),TimeStretch(1.5), PitchShift(cents=10),
                    Resample(), RandomBandstop(), LowPassFilter() , HighPassFilter()] 

    print("Watermark Test Pipeline")
    print("=" * 50)
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    audio_folder_path = project_root / "libri"
    

    # Check if audio file exists
    if not audio_folder_path.exists():
        logger.error(f"Audio file path not found: {audio_folder_path}")
        logger.error("Please place correct folder path")
        return
    
    
    embedder, detector = load()

    watermark_length = 20

    pesq_metric = PESQ()
    stoi_metric = STOI()
    ber_metric = BER()    

    input_dir = Path(audio_folder_path)
    rec={}
    rec["pesq"] = []
    rec["stoi"] = []

    for audio_file_path in input_dir.glob('*.*'):
        audio, sr = librosa.load(str(audio_file_path), sr=None, mono=True)
        
        logger.info("Processing " + audio_file_path.name)

        watermark_bits = np.random.randint(0, 2, size=watermark_length, dtype=np.int32)
        
        
        if sr != 16000:
            from scipy.signal import resample_poly
            up, down = 16000, sr
            audio = resample_poly(audio, up, down)

        sr=16000

        
        try:
            watermarked_audio = embed_watermark(audio, sample_rate=sr, watermark_bits = watermark_bits, model = embedder)
        except ValueError as e:
            # handle a specific exception
            print(f"Bad_input {audio_file_path.name}: ", e)
            continue


        try:
            pesq_ = pesq_metric(watermarked_audio, audio, sr)
            rec["pesq"].append( pesq_ )
            logger.debug(f"PESQ : {pesq_}")
        except Exception as e:
            logger.debug("Not enough speach contnent for calculating PESQ")


        stoi_ = stoi_metric(watermarked_audio, audio, sr)

        #if stoi calculation was successfull
        if stoi_ > 0.1:
            rec["stoi"].append( stoi_ )
            logger.debug(f"STOI : {stoi_}")
        else:
            logger.debug("Not enough speach contnent for calculating STOI")
        


        for attack in attack_list:
            name = attack.name
            
            wm_attacked = attack.apply(watermarked_audio, sr)
            
            detected_pattern, confidence = detect_watermark(wm_attacked, sr, detector)
            
            ber = ber_metric(watermark_bits, detected_pattern)
            
            if name not in rec:
                rec[name] = []
            rec[name].append(ber)

            logger.debug(name + ": " + f"{ber:.2f}" + f" (confidence: {confidence:.4f})")
            
    
    for att in rec.keys():
        items = np.array(rec[att])

        mean = items.mean()
        
        logger.info(att + ": " + f"mean: {mean:.4f}") 

if __name__ == "__main__":
    main()