# DSP Morse Transceiver

A Streamlit-based acoustic Morse code transceiver that can synthesize Morse audio from English text and decode Morse signals from microphone recordings or uploaded WAV files.

## Features

- English-to-Morse transcription
- Synthetic Morse audio generation with adjustable carrier and unit timing
- Live microphone recording and audio playback
- Loopback testing with simulated noise
- FFT-based carrier detection
- Bandpass filtering and envelope decoding for Morse recovery
- DSP diagnostic plots for signal analysis

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- This project uses the laptop microphone/speaker and is intended for local experimentation.
- You may need to allow microphone access in your browser or OS when using the recording input.
- For best results, use the app with a phone or speaker generating the Morse tones while recording with the laptop microphone.
