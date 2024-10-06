import numpy as np
from scipy.io.wavfile import write
from scipy.signal import convolve
import os

# Constants
SAMPLE_RATE = 44100  # Sample rate in Hz
DURATION = 50        # Duration in seconds

def normalize_parameters(parameters_list):
    """Normalize parameters across the entire set of images."""
    parameters_array = np.array(parameters_list)
    min_vals = parameters_array.min(axis=0)
    max_vals = parameters_array.max(axis=0)
    normalized_parameters = (parameters_array - min_vals) / (max_vals - min_vals)
    return normalized_parameters.tolist()

def get_chord_frequencies(chord_name):
    """Return the frequencies of the notes in the chord."""
    # Define frequencies for notes in one octave
    note_freqs = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
    }
    # Chord formulas
    chords = {
        'C': ['C', 'E', 'G'],
        'Cm': ['C', 'D#', 'G'],
        'D': ['D', 'F#', 'A'],
        'Dm': ['D', 'F', 'A'],
        'E': ['E', 'G#', 'B'],
        'Em': ['E', 'G', 'B'],
        'F': ['F', 'A', 'C'],
        'Fm': ['F', 'G#', 'C'],
        'G': ['G', 'B', 'D'],
        'Gm': ['G', 'A#', 'D'],
        'A': ['A', 'C#', 'E'],
        'Am': ['A', 'C', 'E'],
        'B': ['B', 'D#', 'F#'],
        'Bm': ['B', 'D', 'F#'],
    }
    # Get the notes in the chord
    chord_notes = chords.get(chord_name, ['C', 'E', 'G'])
    # Get frequencies
    frequencies = [note_freqs[note] for note in chord_notes]
    return frequencies

def generate_waveform(parameters, duration):
    """Generate audio waveform from chord progressions."""
    num_samples = int(SAMPLE_RATE * duration)
    audio = np.zeros(num_samples)

    # Map parameters to chords
    mean_brightness, saturation, blue_pct, green_pct, red_pct = parameters
    # Determine the key
    if blue_pct > red_pct and blue_pct > green_pct:
        key = 'A minor'
    elif red_pct > blue_pct and red_pct > green_pct:
        key = 'C major'
    elif green_pct > blue_pct and green_pct > red_pct:
        key = 'G major'
    else:
        key = 'E minor'

    # Choose chord progression based on saturation
    chord_progressions = {
        'C major': [['C', 'F', 'G', 'Am'], ['C', 'Dm', 'G', 'C'], ['C', 'Em', 'F', 'G']],
        'A minor': [['Am', 'F', 'G', 'Em'], ['Am', 'C', 'G', 'Dm'], ['Am', 'Dm', 'Em', 'Am']],
        'G major': [['G', 'C', 'D', 'Em'], ['G', 'Bm', 'C', 'D'], ['G', 'Em', 'C', 'D']],
        'E minor': [['Em', 'C', 'D', 'Bm'], ['Em', 'G', 'D', 'C'], ['Em', 'Am', 'D', 'G']],
    }
    progression_choices = chord_progressions[key]
    progression_index = int(saturation * len(progression_choices)) % len(progression_choices)
    chords_list = progression_choices[progression_index]
    # Duration per chord
    chord_duration = duration / len(chords_list)
    t = np.linspace(0, chord_duration, int(SAMPLE_RATE * chord_duration), endpoint=False)

    # Generate chords
    for i, chord_name in enumerate(chords_list):
        frequencies = get_chord_frequencies(chord_name)
        chord_wave = np.zeros_like(t)
        for freq in frequencies:
            chord_wave += np.sin(2 * np.pi * freq * t)
        # Normalize chord_wave
        chord_wave /= len(frequencies)
        # Apply gentle low-pass filter to soften the sound
        chord_wave = lowpass_filter(chord_wave, 2000)
        # Apply envelope to smooth the chord transitions
        envelope = np.ones_like(t)
        attack = int(0.1 * SAMPLE_RATE)  # 0.1 second attack
        release = int(0.1 * SAMPLE_RATE)  # 0.1 second release
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        chord_wave *= envelope
        # Add to audio
        start_idx = int(i * chord_duration * SAMPLE_RATE)
        end_idx = start_idx + len(chord_wave)
        audio[start_idx:end_idx] += chord_wave

    # Normalize the audio
    audio /= np.max(np.abs(audio)) * 1.2  # Reduce overall volume to prevent clipping
    return audio

def generate_melody(parameters, duration):
    """Generate a melody based on parameters and different musical scales."""
    num_samples = int(SAMPLE_RATE * duration)
    melody = np.zeros(num_samples)

    # Select musical scale based on color distribution
    _, _, blue_pct, green_pct, red_pct = parameters
    scales = {
        'Harmonic Minor': [0, 2, 3, 5, 7, 8, 11, 12],
        'Major': [0, 2, 4, 5, 7, 9, 11, 12],
        'Dorian': [0, 2, 3, 5, 7, 9, 10, 12],
        'Pentatonic': [0, 2, 4, 7, 9, 12],
        'Chromatic': list(range(13)),
        'Phrygian': [0, 1, 3, 5, 7, 8, 10, 12],
    }

    if blue_pct > red_pct and blue_pct > green_pct:
        scale_name = 'Harmonic Minor'
    elif red_pct > blue_pct and red_pct > green_pct:
        scale_name = 'Major'
    elif green_pct > blue_pct and green_pct > red_pct:
        scale_name = 'Dorian'
    else:
        scale_name = 'Pentatonic'

    scale_intervals = scales[scale_name]

    # Use parameters to determine the root note
    root_note_freq = 261.63  # Middle C (C4)
    # Shift the root note based on mean_brightness
    mean_brightness, saturation, _, _, _ = parameters
    root_note_freq *= 2 ** ((mean_brightness - 0.5))  # Shift by up to an octave

    # Select waveform based on saturation
    waveforms = ['sine', 'triangle', 'sawtooth']
    waveform_idx = int(saturation * len(waveforms)) % len(waveforms)
    waveform_type = waveforms[waveform_idx]

    # Determine tempo based on brightness
    tempo = 60 + int(mean_brightness * 80)  # Tempo between 60 and 140 BPM
    beats_per_second = tempo / 60
    note_duration = 1 / beats_per_second  # Duration of one beat
    num_notes = int(duration / note_duration)

    t_note = np.linspace(0, note_duration, int(SAMPLE_RATE * note_duration), endpoint=False)

    for i in range(num_notes):
        # Choose a note from the scale
        note_idx = (int((saturation + np.random.rand()) * len(scale_intervals)) + i) % len(scale_intervals)
        interval = scale_intervals[note_idx]
        freq = root_note_freq * (2 ** (interval / 12.0))
        # Generate the note waveform
        if waveform_type == 'sine':
            note_wave = np.sin(2 * np.pi * freq * t_note)
        elif waveform_type == 'triangle':
            note_wave = 2 * np.abs(2 * (t_note * freq - np.floor(0.5 + t_note * freq))) - 1
        elif waveform_type == 'sawtooth':
            note_wave = 2 * (t_note * freq - np.floor(0.5 + t_note * freq))
        # Apply envelope
        envelope = np.exp(-3 * t_note)
        note_wave *= envelope
        # Soften the note by applying a low-pass filter
        note_wave = lowpass_filter(note_wave, 3000)
        # Add to the melody
        start_idx = int(i * note_duration * SAMPLE_RATE)
        end_idx = start_idx + len(note_wave)
        if end_idx > num_samples:
            end_idx = num_samples
            note_wave = note_wave[:end_idx - start_idx]
        melody[start_idx:end_idx] += note_wave

    # Normalize the melody
    melody /= np.max(np.abs(melody)) * 1.2  # Reduce overall volume
    return melody

def apply_fade(audio):
    """Apply fade-in and fade-out to the audio."""
    fade_in_duration = int(SAMPLE_RATE * 10)   # 10 seconds fade-in
    fade_out_duration = int(SAMPLE_RATE * 10)  # 10 seconds fade-out

    # Apply fade-in
    fade_in = np.linspace(0, 1, fade_in_duration)
    audio[:fade_in_duration] *= fade_in

    # Apply fade-out
    fade_out = np.linspace(1, 0, fade_out_duration)
    audio[-fade_out_duration:] *= fade_out

    return audio

def lowpass_filter(data, cutoff):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * SAMPLE_RATE
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * SAMPLE_RATE
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(2, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def highpass_filter(data, cutoff):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * SAMPLE_RATE
    high = cutoff / nyq
    b, a = butter(2, high, btype='highpass')
    y = filtfilt(b, a, data)
    return y

def generate_drum_sound():
    """Generate a simple drum kit with kick, snare, and hi-hat."""
    duration = 0.2  # 200 ms
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Kick drum - low frequency sine wave with fast decay
    kick_freq = 60  # 60 Hz
    kick_envelope = np.exp(-40 * t)
    kick = np.sin(2 * np.pi * kick_freq * t) * kick_envelope

    # Snare drum - noise with bandpass filter
    snare_noise = np.random.randn(len(t))
    snare_envelope = np.exp(-20 * t)
    snare = snare_noise * snare_envelope
    # Apply bandpass filter to simulate snare frequencies
    snare = bandpass_filter(snare, 1500, 6000)

    # Hi-hat - high frequency noise with fast decay
    hihat_noise = np.random.randn(len(t))
    hihat_envelope = np.exp(-50 * t)
    hihat = hihat_noise * hihat_envelope
    # Apply highpass filter
    hihat = highpass_filter(hihat, 8000)

    # Soften the drums
    kick = lowpass_filter(kick, 2000)
    snare = lowpass_filter(snare, 4000)
    hihat = lowpass_filter(hihat, 10000)

    # Combine drum sounds
    drum_kit = {
        'kick': kick,
        'snare': snare,
        'hihat': hihat
    }
    return drum_kit

def add_drum_beats(audio, parameters):
    """Add dynamic drum beats based on brightness and enhance drum volume."""
    drum_sounds = generate_drum_sound()
    mean_brightness, _, _, _, _ = parameters
    # Determine tempo based on mean_brightness
    tempo = 60 + int(mean_brightness * 80)  # Tempo between 60 and 140 BPM
    beat_interval = int(SAMPLE_RATE * 60 / tempo)  # Samples between beats

    num_beats = len(audio) // beat_interval
    for i in range(num_beats):
        # Create a basic drum pattern
        if i % 4 == 0:
            # Kick on every downbeat
            drum = drum_sounds['kick'] * (0.5 + 0.5 * mean_brightness)
        elif i % 4 == 2:
            # Snare on the offbeat
            drum = drum_sounds['snare'] * (0.5 + 0.5 * mean_brightness)
        else:
            # Hi-hat on other beats
            drum = drum_sounds['hihat'] * (0.3 + 0.7 * mean_brightness)

        start_idx = i * beat_interval
        end_idx = start_idx + len(drum)
        if end_idx > len(audio):
            end_idx = len(audio)
            drum = drum[:end_idx - start_idx]
        audio[start_idx:end_idx] += drum

    # Soften the drum mix
    audio = lowpass_filter(audio, 5000)
    return audio

def add_ambient_sounds(audio, parameters):
    """Add ambient background sounds based on saturation and color percentages."""
    _, saturation, blue_pct, green_pct, red_pct = parameters
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    # Add smooth transitioning frequencies for ambient effect
    min_ambient_freq = 110.0  # A2
    max_ambient_freq = 880.0  # A5
    freq_transition = min_ambient_freq + (max_ambient_freq - min_ambient_freq) * saturation
    ambient_wave = np.sin(2 * np.pi * freq_transition * t)

    # Modulate ambient sound with low-frequency oscillator
    lfo_freq = 0.2 + saturation  # LFO frequency varies with saturation
    lfo = np.sin(2 * np.pi * lfo_freq * t)
    ambient_wave *= lfo

    # Adjust volume based on color intensity
    ambient_strength = (blue_pct + green_pct + red_pct) / 3
    ambient_wave *= ambient_strength * 0.2  # Lower volume to avoid overpowering

    # Apply low-pass filter to soften the ambient sound
    ambient_wave = lowpass_filter(ambient_wave, 2000)

    # Add to audio
    audio += ambient_wave

    # Normalize
    audio /= np.max(np.abs(audio)) * 1.2  # Reduce overall volume
    return audio

def add_reverb(audio):
    """Add reverb effect to the audio."""
    # Create a simple reverb kernel
    kernel_size = int(SAMPLE_RATE * 0.3)  # 300ms kernel
    kernel = np.zeros(kernel_size)
    kernel[0] = 1
    kernel[int(kernel_size / 4)] = 0.6
    kernel[int(kernel_size / 2)] = 0.3
    kernel[int(3 * kernel_size / 4)] = 0.1
    # Apply convolution
    audio_reverb = convolve(audio, kernel, mode='full')[:len(audio)]
    # Normalize
    audio_reverb /= np.max(np.abs(audio_reverb)) * 1.2  # Reduce overall volume
    return audio_reverb

def save_audio(audio, filename):
    """Save the audio array to a WAV file."""
    # Ensure the audio doesn't exceed [-1, 1]
    audio = np.clip(audio, -1, 1)
    audio_int16 = np.int16(audio * 32767)
    write(filename, SAMPLE_RATE, audio_int16)

def main(parameters_list, output_filename):
    normalized_parameters_list = normalize_parameters(parameters_list)
    for i, parameters in enumerate(normalized_parameters_list):
        chords = generate_waveform(parameters, DURATION)
        melody = generate_melody(parameters, DURATION)
        audio = chords + melody
        audio = apply_fade(audio)
        audio = add_drum_beats(audio, parameters)
        audio = add_ambient_sounds(audio, parameters)
        audio = add_reverb(audio)
        # Final normalization and volume adjustment
        audio /= np.max(np.abs(audio)) * 1.1  # Slightly reduce volume to prevent clipping
        save_audio(audio, f"{output_filename}_{i+1}.wav")
        print(f"Audio saved to {output_filename}_{i+1}.wav")

if __name__ == "__main__":
    parameters_list = [
        (95.67081018257157, 104.00398915556075, 37.560184283573896, 33.56203174576044, 28.877783970665654),
        (98.31761255411256, 127.87768441558441, 34.800069674506624, 30.694017564405705, 34.50591276108767),
        (63.37436556122449, 121.8607193877551, 32.389142380986655, 26.833149316074735, 40.777708302938606),
        (60.76245325571856, 225.891983915133, 15.56681367212644, 21.37698437046624, 63.05620195740732),
        (33.84658200027061, 145.97826327955602, 42.865173291223144, 28.677000792450798, 28.45782591632606),
        (27.36372715318869, 24.27243063773833, 32.272895001461606, 32.08753480929896, 35.639570189239436),
        (41.6946805, 50.32173575, 32.521102519856434, 34.06214900297807, 33.4167484771655),
        (57.77738977517241, 131.47161864212364, 31.89717622418156, 24.239975437112367, 43.86284833870607),
        (49.28377592076757, 86.47343502674978, 32.24922911307847, 29.99983750792406, 37.75093337899747),
        (57.55572479954181, 58.157160080183274, 35.97727329628657, 30.136077768719076, 33.88664893499435)
    ]
    output_filename = "atomic_a1_cosmic_music_best"
    main(parameters_list, output_filename)
