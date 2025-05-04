# CMFF: A Cross-Modal Multi-Layer Feature Fusion Network for Multimodal Sentiment Analysis



## Download Data
The three datasets (CMU-MOSI and CMU-MOSEI) are available from this link: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

## Directory
To run our preprocessing and training codes directly, please put the necessary files from downloaded data in separate folders as described below.

```
/datasets/
    mosi/
        Raw/
        label.csv
    mosei/
        Raw/
        label.csv
    data_convert.py
    extract_audio_feature.py
    extract_text_embedding.py
    extract_wav_multithread.py
    
/cmff/
    main.py
    config.py
    m1/
    analysis/
    ...
```

## Audio Extraction
use extract_wav_multithread.py to extract audio wav file.
