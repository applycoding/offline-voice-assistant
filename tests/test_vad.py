from offline_voice.audio import concat, silence, tone
from offline_voice.vad import EnergyVAD


def _detector() -> EnergyVAD:
    return EnergyVAD(threshold=0.02, silence_ms=300, min_speech_ms=200, pad_ms=60, frame_ms=30)


def test_energy_vad_trims_silence_around_a_tone() -> None:
    clip = concat(
        [
            silence(0.3, 16000),
            tone(0.5, 16000, 440.0, amplitude=0.5),
            silence(0.4, 16000),
        ]
    )
    speech = _detector().trim(clip)
    assert speech is not None
    assert 0.4 < speech.duration_s < clip.duration_s


def test_energy_vad_ignores_silence_and_short_blips() -> None:
    assert _detector().trim(silence(0.8, 16000)) is None
    blip = concat([silence(0.2, 16000), tone(0.05, 16000, 440.0, amplitude=0.5), silence(0.2, 16000)])
    assert _detector().trim(blip) is None


def test_quiet_noise_is_not_speech() -> None:
    quiet = tone(0.6, 16000, 200.0, amplitude=0.001)
    assert _detector().trim(quiet) is None
