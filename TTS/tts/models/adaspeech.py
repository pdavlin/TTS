from TTS.tts.configs.adaspeech_config import AdaSpeechConfig
from TTS.tts.layers.align_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.io import load_fsspec


class AdaSpeech(BaseTTS):
    """AdaSpeech model.

    Paper::
        https://arxiv.org/abs/2103.00993

    Paper abstract::
        Custom voice, a specific text to speech (TTS) service in commercial speech platforms, aims to adapt a source
        TTS model to synthesize personal voice for a target speaker using few speech data. Custom voice presents two
        unique challenges for TTS adaptation: 1) to support diverse customers, the adaptation model needs to handle
        diverse acoustic conditions that could be very different from source speech data, and 2) to support a large
        number of customers, the adaptation parameters need to be small enough for each target speaker to reduce memory
        usage while maintaining high voice quality. In this work, we propose AdaSpeech, an adaptive TTS system for
        high-quality and efficient customization of new voices. We design several techniques in AdaSpeech to address
        the two challenges in custom voice: 1) To handle different acoustic conditions, we use two acoustic encoders
        to extract an utterance-level vector and a sequence of phoneme-level vectors from the target speech during
        training; in inference, we extract the utterance-level vector from a reference speech and use an acoustic
        predictor to predict the phoneme-level vectors. 2) To better trade off the adaptation parameters and voice
        quality, we introduce conditional layer normalization in the mel-spectrogram decoder of AdaSpeech, and
        fine-tune this part in addition to speaker embedding for adaptation. We pre-train the source TTS model on
        LibriTTS datasets and fine-tune it on VCTK and LJSpeech datasets (with different acoustic conditions from
        LibriTTS) with few adaptation data, e.g., 20 sentences, about 1 minute speech. Experiment results show that
        AdaSpeech achieves much better adaptation quality than baseline methods, with only about 5K specific parameters
        for each speaker, which demonstrates its effectiveness for custom voice.

    """

    def __init__(
        self,
        config: AdaSpeechConfig,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        self.encoder = Encoder()
        self.duration_predictor = DurationPredictor(
            self.args.hidden_channels + self.embedded_speaker_dim,
            self.args.duration_predictor_hidden_channels,
            self.args.duration_predictor_kernel_size,
            self.args.duration_predictor_dropout_p,
        )
        self.energy_predictor = None # TODO: add energy predictor
        self.energy_embed = torch.nn.Linear() # TODO: add energy embedding
        self.pitch_predictor = DurationPredictor(
            self.args.hidden_channels + self.embedded_speaker_dim,
            self.args.pitch_predictor_hidden_channels,
            self.args.pitch_predictor_kernel_size,
            self.args.pitch_predictor_dropout_p,
        )
        self.pitch_emb = torch.nn.Linear() # TODO: add pitch embedding
        self.length_regulator = LengthRegulator()

        ###### AdaSpeech-specific layers ###### TODO: add these lol

        self.utterance_encoder = UtteranceEncoder(idim=hp.audio.n_mels)


        self.phoneme_level_encoder = PhonemeLevelEncoder(idim=hp.audio.n_mels)

        self.phoneme_level_predictor = PhonemeLevelPredictor(idim=hp.model.adim)

        self.phone_level_embed = torch.nn.Linear(hp.model.phn_latent_dim, hp.model.adim)

        self.acoustic_criterion = AcousticPredictorLoss()

        self.decoder = Decoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.decoder_type,
            self.args.decoder_params,
        )