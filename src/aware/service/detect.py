import numpy as np
from aware.detection.multibit_detector import AWAREDetector
from aware.utils.watermark import PatternDecoder
from aware.utils.logger import logger
from aware.utils.utils import BRH_activation_to_probability


def detect_watermark(audio: np.ndarray, sample_rate: int, detector: AWAREDetector):
    """
    Detects the presence of a watermark in audio data and returns the detection decision, extracted watermark, and confidence statistics.

    Args:
        audio (np.ndarray): The audio data.
        sample_rate (int): Sampling rate of the audio.
    Returns:
        watermark bits
    """
    pattern_postprocess_pipeline = [PatternDecoder(encoder_mode=detector.pattern_mode, threshold=detector.threshold)]
    
    if sample_rate != 16000:
        logger.error(f"Invalid sample rate. Expected 16000Hz, got {sample_rate}Hz.")
        raise ValueError("Invalid sample rate. Expected 16000Hz.")

    if audio.ndim != 1:
        logger.error(f"Invalid audio shape. Expected mono (1D) signal, got shape {audio.shape}.")
        raise ValueError(f"Invalid audio shape. Expected mono (1D) signal, got shape {audio.shape}.")
    

    if detector.mode_name == "full_length":
        detected_values = detector.detect(audio, sample_rate)

        watermark_bits = detected_values
        for processor in pattern_postprocess_pipeline:
            watermark_bits = processor(watermark_bits)

        return watermark_bits, BRH_activation_to_probability(np.sum(np.abs(detected_values))/detector.output_length, 0.0, detector.mode_name)

    elif detector.mode_name == "segments":
    
        chunk_duration = sample_rate+128
        step = chunk_duration//24
        start = 0
        max_act = 0
        all_watermarks = []
        all_acts = []
        while start + chunk_duration < len(audio):
            audio_ = audio[start:start+chunk_duration]
            start += step

            detected_values = detector.detect(audio_, sample_rate)
            
            act = np.sum(np.abs(detected_values))
            
            watermark_bits = detected_values
            for processor in pattern_postprocess_pipeline:
                watermark_bits = processor(watermark_bits)

            all_watermarks.append(watermark_bits)  
            all_acts.append(act)

        wms_final_round = []
        acts_final_round = []
        start = 0
        tau = 0.4
        while start < len(all_acts):
            part = all_acts[start: start+22]
            idx = np.argmax(part)

            if part[idx] > tau:
                wms_final_round.append(all_watermarks[start+idx])
                acts_final_round.append(part[idx])
                start += (idx+9)
            else:
                start += 22

        if len(all_acts) == 0:
            return np.array([]), 0.0, 0

        if len(wms_final_round) == 0:
            best_idx = int(np.argmax(all_acts))
            final_watermark = all_watermarks[best_idx]
        else:
            stacked = np.stack(wms_final_round)          
            votes = np.sum(stacked, axis=0)          
            n = len(wms_final_round)
            majority = (votes > n / 2).astype(stacked.dtype)
            
            tie_mask = votes == n / 2
            if np.any(tie_mask):
                best = stacked[int(np.argmax(acts_final_round))]
                majority[tie_mask] = best[tie_mask]
            final_watermark = majority

        if len(wms_final_round) > 3:
            def hamming_dist(a, b):
                return int(np.sum(a != b))

            best_count = 0
            for candidate in wms_final_round:
                count = sum(hamming_dist(candidate, wm) <= 2 for wm in wms_final_round)
                if count > best_count:
                    best_count = count
            
            bit_agreement = float(best_count / len(wms_final_round))
        else:
            bit_agreement = 0.0


        return final_watermark, BRH_activation_to_probability(np.max(all_acts)/detector.output_length, bit_agreement, detector.mode_name)


    else:
        logger.error("Invalid audio shape. Expected 1D or 2D numpy array.")
        raise ValueError("Invalid audio shape. Expected 1D or 2D numpy array.")
