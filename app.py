import os
# Prevent OpenBLAS multithread memory exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import io

# ---------------- MORSE CODE DICTIONARY ----------------
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '0': '-----', ' ': '/'
}
REVERSE_MORSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}

# ---------------- TRANSMITTER (ENGLISH -> AUDIO) ----------------
def text_to_morse(text):
    return ' '.join(MORSE_CODE_DICT.get(ch, '') for ch in text.upper() if ch in MORSE_CODE_DICT)

def synthesize_morse_audio(morse_str, f_c=800, fs=44100, unit_ms=100):
    unit_samples = int(fs * (unit_ms / 1000.0))
    audio_segments = []

    def make_tone(duration_units):
        N = duration_units * unit_samples
        t = np.arange(N) / fs
        tone = np.cos(2 * np.pi * f_c * t)
        # Apply Hanning window taper at edges (10% of pulse) to prevent clicks
        taper_len = max(int(0.1 * unit_samples), 20)
        if 2 * taper_len < N:
            win = np.hanning(2 * taper_len)
            tone[:taper_len] *= win[:taper_len]
            tone[-taper_len:] *= win[taper_len:]
        return tone

    def make_silence(duration_units):
        return np.zeros(duration_units * unit_samples)

    # Add 0.3s padding silence at start and end
    audio_segments.append(np.zeros(int(0.3 * fs)))

    words = morse_str.split(' / ')
    for w_idx, word in enumerate(words):
        letters = word.split(' ')
        for l_idx, letter in enumerate(letters):
            for s_idx, symbol in enumerate(letter):
                if symbol == '.':
                    audio_segments.append(make_tone(1))
                elif symbol == '-':
                    audio_segments.append(make_tone(3))
                if s_idx < len(letter) - 1:
                    audio_segments.append(make_silence(1)) # Intra-character gap
            if l_idx < len(letters) - 1:
                audio_segments.append(make_silence(3))     # Inter-character gap
        if w_idx < len(words) - 1:
            audio_segments.append(make_silence(7))         # Word gap

    audio_segments.append(np.zeros(int(0.3 * fs)))
    return np.concatenate(audio_segments).astype(np.float32)

# ---------------- RECEIVER DSP PIPELINE (AUDIO -> ENGLISH) ----------------
def design_bandpass_filter(filter_family, f_low, f_high, fs, order, rp=1.0):
    nyq = 0.5 * fs
    low = max(f_low / nyq, 0.01)
    high = min(f_high / nyq, 0.99)

    if filter_family == "FIR - Hamming Window":
        b = signal.firwin(order + 1, [low, high], pass_zero=False, window='hamming')
        a = [1.0]
    elif filter_family == "FIR - Hanning Window":
        b = signal.firwin(order + 1, [low, high], pass_zero=False, window='hann')
        a = [1.0]
    elif filter_family == "FIR - Rectangular Window":
        b = signal.firwin(order + 1, [low, high], pass_zero=False, window='boxcar')
        a = [1.0]
    elif filter_family == "IIR - Butterworth (BLT)":
        b, a = signal.butter(order, [low, high], btype='bandpass')
    elif filter_family == "IIR - Chebyshev Type I (BLT)":
        b, a = signal.cheby1(order, rp, [low, high], btype='bandpass')
    return b, a

def process_and_decode_audio(raw_audio, fs_orig, decimation_factor, filter_family, order, bw, thresh_ratio, unit_ms_est):
    # Multirate decimation (downsampling with anti-aliasing filter)
    if decimation_factor > 1:
        sig_dec = signal.decimate(raw_audio, decimation_factor, ftype='fir')
        fs = fs_orig // decimation_factor
    else:
        sig_dec = raw_audio.copy()
        fs = fs_orig

    # Normalize decimated signal
    sig_dec = sig_dec / (np.max(np.abs(sig_dec)) + 1e-8)

    # Radix-2 FFT carrier frequency detection
    N_fft = 1 << (len(sig_dec) - 1).bit_length() # Next power of 2 for Radix-2 FFT
    fft_spectrum = np.abs(np.fft.rfft(sig_dec, n=N_fft))
    fft_freqs = np.fft.rfftfreq(N_fft, d=1.0 / fs)

    # Search for dominant Morse carrier between 300 Hz and 2500 Hz
    valid_idx = np.where((fft_freqs >= 300) & (fft_freqs <= 2500))[0]
    peak_idx = valid_idx[np.argmax(fft_spectrum[valid_idx])]
    f_carrier = fft_freqs[peak_idx]

    # Digital bandpass filtering around the detected carrier
    f_low = max(100.0, f_carrier - bw / 2.0)
    f_high = min(fs / 2.0 - 100.0, f_carrier + bw / 2.0)
    b_bp, a_bp = design_bandpass_filter(filter_family, f_low, f_high, fs, order)
    filtered_sig = signal.lfilter(b_bp, a_bp, sig_dec)

    # Envelope detection: full-wave rectification + digital low-pass filter (25 Hz cutoff)
    rectified = np.abs(filtered_sig)
    b_lp, a_lp = signal.butter(4, 25.0 / (0.5 * fs), btype='low')
    envelope = signal.lfilter(b_lp, a_lp, rectified)
    envelope = envelope / (np.max(envelope) + 1e-8)

    # Binary thresholding and run-length decoding
    binary_env = (envelope > thresh_ratio).astype(int)

    # Compute SNR Improvement Estimate (In-band tone power vs Out-of-band noise power)
    on_mask = binary_env == 1
    off_mask = binary_env == 0
    if np.any(on_mask) and np.any(off_mask):
        snr_in = 10 * np.log10((np.mean(sig_dec[on_mask]**2) + 1e-10) / (np.mean(sig_dec[off_mask]**2) + 1e-10))
        snr_out = 10 * np.log10((np.mean(filtered_sig[on_mask]**2) + 1e-10) / (np.mean(filtered_sig[off_mask]**2) + 1e-10))
    else:
        snr_in, snr_out = 0.0, 0.0

    # Run-length encoding of binary states
    diffs = np.diff(binary_env)
    change_indices = np.where(diffs != 0)[0] + 1
    segments = np.split(binary_env, change_indices)

    states = [seg[0] for seg in segments if len(seg) > 0]
    durations_ms = [(len(seg) / fs) * 1000.0 for seg in segments if len(seg) > 0]

    # Filter out tiny glitches (< 25 ms)
    clean_states, clean_durations = [], []
    for st_val, dur in zip(states, durations_ms):
        if dur < 25.0 and len(clean_durations) > 0:
            clean_durations[-1] += dur
        else:
            clean_states.append(st_val)
            clean_durations.append(dur)

    # Automatic or Manual Unit Timing Estimation
    on_durations = [d for s, d in zip(clean_states, clean_durations) if s == 1]
    if len(on_durations) > 0 and unit_ms_est == 0:
        # Auto-estimate dot duration from the 25th percentile of ON pulses
        T_unit = np.percentile(on_durations, 25)
    else:
        T_unit = float(unit_ms_est) if unit_ms_est > 0 else 100.0

    # Decode runs into Morse symbols
    morse_out = []
    for st_val, dur in zip(clean_states, clean_durations):
        if st_val == 1:
            if dur < 2.0 * T_unit:
                morse_out.append('.')
            else:
                morse_out.append('-')
        else:
            # Ignore leading/trailing silence
            if len(morse_out) == 0:
                continue
            if dur >= 5.0 * T_unit:
                morse_out.append(' / ')
            elif dur >= 2.0 * T_unit:
                morse_out.append(' ')

    morse_string = ''.join(morse_out).strip()
    decoded_words = []
    for word in morse_string.split('/'):
        chars = [REVERSE_MORSE_DICT.get(code, '?') for code in word.strip().split(' ') if code]
        decoded_words.append(''.join(chars))
    decoded_text = ' '.join(decoded_words)

    return {
        "fs": fs, "sig_dec": sig_dec, "fft_freqs": fft_freqs, "fft_spectrum": fft_spectrum,
        "f_carrier": f_carrier, "filtered_sig": filtered_sig, "envelope": envelope,
        "binary_env": binary_env, "b_bp": b_bp, "a_bp": a_bp,
        "snr_in": snr_in, "snr_out": snr_out, "morse_string": morse_string,
        "decoded_text": decoded_text, "T_unit": T_unit
    }

# ---------------- STREAMLIT UI DASHBOARD ----------------
st.set_page_config(page_title="DSP Acoustic Morse Transceiver", layout="wide")
st.title("Bidirectional Acoustic Morse Code Transceiver (DSP Mini-Project)")

tab_tx, tab_rx = st.tabs(["Transmitter (English to Morse Audio)", "Receiver & DSP Analyzer (Mic/Audio to English)"])

with tab_tx:
    st.subheader("Discrete-Time Sinusoidal Morse Synthesizer")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_text = st.text_input("Enter English Text:", "SOS DSP")
    with col2:
        tx_fc = st.slider("Carrier Frequency f_c (Hz):", 400, 1500, 800, 50)
    with col3:
        tx_unit_ms = st.slider("Dot Unit Duration (ms):", 50, 250, 100, 10)

    morse_code = text_to_morse(input_text)
    st.markdown(f"**Generated Morse Code:** `{morse_code}`")

    tx_audio = synthesize_morse_audio(morse_code, f_c=tx_fc, fs=44100, unit_ms=tx_unit_ms)

    # Save to memory buffer for browser playback or download to phone
    wav_buf = io.BytesIO()
    wav.write(wav_buf, 44100, (tx_audio * 32767).astype(np.int16))
    st.audio(wav_buf, format="audio/wav")

    if st.button("Play Directly Through Laptop Speaker"):
        sd.play(tx_audio, samplerate=44100)
        sd.wait()
        st.success("Playback complete!")

with tab_rx:
    st.subheader("Multirate FFT & Digital Filter Receiver")

    # Sidebar / Top DSP Parameters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        filter_choice = st.selectbox("Select Digital Bandpass Filter:", [
            "FIR - Hamming Window",
            "FIR - Hanning Window",
            "FIR - Rectangular Window",
            "IIR - Butterworth (BLT)",
            "IIR - Chebyshev Type I (BLT)"
        ])
    with c2:
        default_order = 64 if "FIR" in filter_choice else 4
        max_order = 128 if "FIR" in filter_choice else 10
        filt_order = st.slider("Filter Order (N):", 2, max_order, default_order, 2)
    with c3:
        decim_factor = st.selectbox("Decimation Factor (M):", [1, 2, 4, 5], index=3)
    with c4:
        thresh_val = st.slider("Envelope Threshold:", 0.15, 0.70, 0.35, 0.05)

    c5, c6, c7 = st.columns(3)
    with c5:
        bw_val = st.slider("Bandpass Bandwidth (Hz):", 50, 400, 150, 25)
    with c6:
        unit_est = st.number_input("Dot Duration (0 = Auto-Detect ms):", min_value=0, max_value=300, value=0)
    with c7:
        rec_duration = st.slider("Mic Recording Duration (sec):", 2, 15, 6)

    audio_source = st.radio("Select Input Audio Source:", [
        "Record Live from Laptop Microphone (Play Morse from Phone)",
        "Loopback Test (Use Transmitter Audio + Synthetic Room Noise)",
        "Upload Recorded .wav File"
    ], horizontal=True)

    raw_rx_audio = None
    fs_rx = 44100

    if "Record Live" in audio_source:
        if st.button("Start Recording from Microphone"):
            with st.spinner(f"Listening for {rec_duration} seconds... Play your phone beeps now!"):
                rec = sd.rec(int(rec_duration * fs_rx), samplerate=fs_rx, channels=1, dtype='float32')
                sd.wait()
                raw_rx_audio = rec.flatten()
                st.session_state['last_recorded_audio'] = raw_rx_audio
        if 'last_recorded_audio' in st.session_state:
            raw_rx_audio = st.session_state['last_recorded_audio']

    elif "Loopback Test" in audio_source:
        noise_sigma = st.slider("Simulated Ambient Room Noise Level (σ):", 0.0, 1.5, 0.4, 0.05)
        np.random.seed(42)
        raw_rx_audio = tx_audio + noise_sigma * np.random.randn(len(tx_audio)).astype(np.float32)

    elif "Upload" in audio_source:
        uploaded = st.file_uploader("Upload .wav file", type=["wav"])
        if uploaded is not None:
            fs_rx, data = wav.read(uploaded)
            if data.ndim > 1:
                data = data[:, 0]
            raw_rx_audio = data.astype(np.float32)

    if raw_rx_audio is not None:
        res = process_and_decode_audio(
            raw_rx_audio, fs_rx, decim_factor, filter_choice,
            filt_order, bw_val, thresh_val, unit_est
        )

        # Display Metrics & Decoded Output
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected Carrier (FFT)", f"{res['f_carrier']:.1f} Hz")
        m2.metric("Input SNR", f"{res['snr_in']:.2f} dB")
        m3.metric("Filtered Output SNR", f"{res['snr_out']:.2f} dB", f"{res['snr_out'] - res['snr_in']:+.2f} dB Gain")
        m4.metric("Decimated Sample Rate", f"{res['fs']} Hz (M={decim_factor})")

        st.success(f"### Translated English Output: `{res['decoded_text']}`")
        st.info(f"**Detected Morse Sequence:** `{res['morse_string']}`  |  **Estimated Dot Unit:** `{res['T_unit']:.1f} ms`")

        # Plot 4 DSP Diagnostic Graphs
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))
        t_axis = np.arange(len(res['sig_dec'])) / res['fs']

        # Plot 1: Raw Decimated Audio
        axes[0, 0].plot(t_axis, res['sig_dec'], color='gray', alpha=0.8)
        axes[0, 0].set_title("Noisy Input Signal (Decimated Time-Domain)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Normalized Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Radix-2 FFT Magnitude Spectrum
        axes[0, 1].plot(res['fft_freqs'], res['fft_spectrum'], color='purple')
        axes[0, 1].axvline(res['f_carrier'], color='red', linestyle='--', label=f"Peak = {res['f_carrier']:.1f} Hz")
        axes[0, 1].set_xlim(0, 2500)
        axes[0, 1].set_title("Radix-2 FFT Spectrum & Automatic Peak Detection")
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("|X(f)|")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Bandpass Filtered Signal + LPF Envelope + Binary Threshold
        axes[1, 0].plot(t_axis, res['filtered_sig'], color='lightblue', alpha=0.6, label="Bandpass Filtered")
        axes[1, 0].plot(t_axis, res['envelope'], color='darkblue', linewidth=1.5, label="LPF Envelope (25 Hz)")
        axes[1, 0].plot(t_axis, res['binary_env'], color='red', linestyle='--', linewidth=1.2, label="Decoded Binary Pulses")
        axes[1, 0].set_title(f"{filter_choice} Output & Envelope Detection")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].legend(loc="upper right")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Digital Filter Frequency Response |H(e^jw)|
        w, h = signal.freqz(res['b_bp'], res['a_bp'], worN=2048, fs=res['fs'])
        axes[1, 1].plot(w, 20 * np.log10(np.abs(h) + 1e-10), color='green')
        axes[1, 1].set_xlim(0, 2500)
        axes[1, 1].set_ylim(-80, 5)
        axes[1, 1].set_title(f"Digital Filter Frequency Response (Order N={filt_order})")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Magnitude (dB)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)