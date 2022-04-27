from dataclasses import dataclass, field
import os
import random
from typing import Dict, Tuple

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast


from TTS.tts.layers.feed_forward.acoustic_encoder import PhonemeLevelEncoder, UtteranceEncoder
from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.helpers import (
    average_mel_over_duration,
    sequence_mask,
)
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram


@dataclass
class AdaSpeechArgs(Coqpit):
    """AdaSpeech Model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels. Defaults to 80.

        hidden_channels (int):
            Number of base hidden channels of the model. Defaults to 512.

        use_aligner (bool):
            Whether to use aligner network to learn the text to speech alignment or use pre-computed durations.
            If set False, durations should be computed by `TTS/bin/compute_attention_masks.py` and path to the
            pre-computed durations must be provided to `config.datasets[0].meta_file_attn_mask`. Defaults to True.

        use_pitch (bool):
            Use pitch predictor to learn the pitch. Defaults to True.

        duration_predictor_hidden_channels (int):
            Number of hidden channels in the duration predictor. Defaults to 256.

        duration_predictor_dropout_p (float):
            Dropout rate for the duration predictor. Defaults to 0.1.

        duration_predictor_kernel_size (int):
            Kernel size of conv layers in the duration predictor. Defaults to 3.

        pitch_predictor_hidden_channels (int):
            Number of hidden channels in the pitch predictor. Defaults to 256.

        pitch_predictor_dropout_p (float):
            Dropout rate for the pitch predictor. Defaults to 0.1.

        pitch_predictor_kernel_size (int):
            Kernel size of conv layers in the pitch predictor. Defaults to 3.

        pitch_embedding_kernel_size (int):
            Kernel size of the projection layer in the pitch predictor. Defaults to 3.

        positional_encoding (bool):
            Whether to use positional encoding. Defaults to True.

        positional_encoding_use_scale (bool):
            Whether to use a learnable scale coeff in the positional encoding. Defaults to True.

        length_scale (int):
            Length scale that multiplies the predicted durations. Larger values result slower speech. Defaults to 1.0.

        encoder_type (str):
            Type of the encoder module. One of the encoders available in :class:`TTS.tts.layers.feed_forward.encoder`.
            Defaults to `fftransformer` as in the paper.

        encoder_params (dict):
            Parameters of the encoder module. Defaults to ```{"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}```

        decoder_type (str):
            Type of the decoder module. One of the decoders available in :class:`TTS.tts.layers.feed_forward.decoder`.
            Defaults to `fftransformer` as in the paper.

        decoder_params (str):
            Parameters of the decoder module. Defaults to ```{"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}```

        detach_duration_predictor (bool):
            Detach the input to the duration predictor from the earlier computation graph so that the duraiton loss
            does not pass to the earlier layers. Defaults to True.

        max_duration (int):
            Maximum duration accepted by the model. Defaults to 75.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

    """

    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 384
    use_aligner: bool = True
    use_pitch: bool = False
    use_adaspeech: bool = False
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    duration_predictor_hidden_channels: int = 256
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout_p: float = 0.1
    positional_encoding: bool = True
    poisitonal_encoding_use_scale: bool = True
    length_scale: int = 1
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024,
                                 "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024,
                                 "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )
    detach_duration_predictor: bool = False
    max_duration: int = 75
    num_speakers: int = 1
    use_speaker_embedding: bool = False
    speakers_file: str = None
    use_d_vector_file: bool = False
    d_vector_dim: int = None
    d_vector_file: str = None
    predictor_train_epoch: int = 400

    stored_average_mel = None
    stored_phoneme_pred = None
    use_adaspeech = True


class AdaSpeech(ForwardTTS):
    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager = None):
        super().__init__(config)

        self.speaker_manager = speaker_manager
        self.init_multispeaker(config)

        self.max_duration = self.args.max_duration
        self.use_aligner = self.args.use_aligner
        self.use_pitch = self.args.use_pitch
        self.use_binary_alignment_loss = False
        self.use_adaspeech = self.args.use_adaspeech
        self.predictor_train_epoch = self.args.predictor_train_epoch

        self.length_scale = (
            float(self.args.length_scale) if isinstance(
                self.args.length_scale, int) else self.args.length_scale
        )

        self.emb = nn.Embedding(self.args.num_chars, self.args.hidden_channels)

        self.encoder = Encoder(
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.encoder_type,
            self.args.encoder_params,
            self.embedded_speaker_dim,
        )

        if self.args.positional_encoding:
            self.pos_encoder = PositionalEncoding(self.args.hidden_channels)

        self.decoder = Decoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.decoder_type,
            self.args.decoder_params,
        )

        self.duration_predictor = DurationPredictor(
            self.args.hidden_channels + self.embedded_speaker_dim,
            self.args.duration_predictor_hidden_channels,
            self.args.duration_predictor_kernel_size,
            self.args.duration_predictor_dropout_p,
        )

        if self.args.use_aligner:
            self.aligner = AlignmentNetwork(
                in_query_channels=self.args.out_channels, in_key_channels=self.args.hidden_channels
            )
        self.utterance_encoder = UtteranceEncoder(
            idim=config.audio.num_mels, n_chans=self.args.hidden_channels)
        self.phoneme_level_encoder = PhonemeLevelEncoder(
            idim=config.audio.num_mels, n_chans=self.args.hidden_channels)
        self.phoneme_level_predictor = PhonemeLevelEncoder(
            idim=self.args.hidden_channels, n_chans=self.args.hidden_channels)
        self.train_predictor = False

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        y_lengths: torch.LongTensor,
        y: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
        pitch: torch.FloatTensor = None,
        aux_input: Dict = {"d_vectors": None,
                           "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> Dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            y (torch.LongTensor): Ground truth mel for sequence.
            x_lengths (torch.LongTensor): Input sequence lengths.
            y_lengths (torch.LongTensor): Output sequnce lengths. Defaults to None.
            y (torch.FloatTensor): Spectrogram frames. Only used when the alignment network is on. Defaults to None.
            dr (torch.IntTensor): Character durations over the spectrogram frames. Only used when the alignment network is off. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Only used when the pitch predictor is on. Defaults to None.
            aux_input (Dict): Auxiliary model inputs for multi-speaker training. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - y_lengths: :math:`[B]`
            - y: :math:`[B, T_max2]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T]`
        """
        g = self._set_speaker_input(aux_input)
        # compute sequence masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).float()
        x_mask = torch.unsqueeze(sequence_mask(
            x_lengths, x.shape[1]), 1).float()
        # encoder pass
        o_en, x_mask, g, x_emb = self._forward_encoder(x, x_mask, g)

        # aligner
        o_alignment_dur = None
        alignment_soft = None
        alignment_logprob = None
        alignment_mas = None
        if self.use_aligner:
            o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_aligner(
                x_emb, y, x_mask, y_mask
            )
            alignment_soft = alignment_soft.transpose(1, 2)
            alignment_mas = alignment_mas.transpose(1, 2)
            dr = o_alignment_dur
        # pitch predictor pass
        o_pitch = None
        avg_pitch = None

        # AdaSpeech

        avg_mel = average_mel_over_duration(
            y.transpose(1, 2), dr).transpose(1, 2)
        self.stored_average_mel = avg_mel

        o_phn = self.phoneme_level_encoder(
            avg_mel.transpose(1, 2)).transpose(1, 2)
        o_en = o_en + o_phn
        o_phn_pred = self.phoneme_level_predictor(o_en.detach()).transpose(1,2)

        # Utterance encoder pass
        # print('---adaspeech---')
        utterance_vec = self.utterance_encoder(y.transpose(1, 2))
        # print('utterance_vec:        {}'.format(utterance_vec.size()))
        o_en = o_en + utterance_vec

        # duration predictor pass
        if self.args.detach_duration_predictor:
            o_dr_log = self.duration_predictor(o_en.detach(), x_mask)
        else:
            o_dr_log = self.duration_predictor(o_en, x_mask)
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)
        # generate attn mask from predicted durations
        o_attn = self.generate_attn(o_dr.squeeze(1), x_mask)
        # # aligner
        # o_alignment_dur = None
        # alignment_soft = None
        # alignment_logprob = None
        # alignment_mas = None
        # if self.use_aligner:
        #     o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_aligner(
        #         x_emb, y, x_mask, y_mask
        #     )
        #     alignment_soft = alignment_soft.transpose(1, 2)
        #     alignment_mas = alignment_mas.transpose(1, 2)
        #     dr = o_alignment_dur
        # # pitch predictor pass
        # o_pitch = None
        # avg_pitch = None

        # print('step count:           {}'.format(self.));



        # decoder pass
        o_de, attn = self._forward_decoder(
            o_en, dr, x_mask, y_lengths, g=None
        )  # TODO: maybe pass speaker embedding (g) too
        outputs = {
            "model_outputs": o_de,  # [B, T, C]
            "durations_log": o_dr_log.squeeze(1),  # [B, T]
            "durations": o_dr.squeeze(1),  # [B, T]
            "attn_durations": o_attn,  # for visualization [B, T_en, T_de']
            "pitch_avg": o_pitch,
            "pitch_avg_gt": avg_pitch,
            "alignments": attn,  # [B, T_de, T_en]
            "alignment_soft": alignment_soft,
            "alignment_mas": alignment_mas,
            "o_alignment_dur": o_alignment_dur,
            "alignment_logprob": alignment_logprob,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "avg_mel": avg_mel,
            "o_phn": o_phn,
            "o_phn_pred": o_phn_pred
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None, "utterance_vec": None}):  # pylint: disable=unused-argument
        """Model's inference pass.

        Args:
            x (torch.LongTensor): Input character sequence.
            aux_input (Dict): Auxiliary model inputs. Defaults to `{"d_vectors": None, "speaker_ids": None}`.

        Shapes:
            - x: [B, T_max]
            - x_lengths: [B]
            - g: [B, C]
        """
        g = self._set_speaker_input(aux_input)
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        x_mask = torch.unsqueeze(sequence_mask(
            x_lengths, x.shape[1]), 1).to(x.dtype).float()
        # encoder pass
        o_en, x_mask, g, _ = self._forward_encoder(x, x_mask, g)

        # AdaSpeech
        
        y = self._set_utterance_input(aux_input)

        # phoneme predictor pass
        o_phn_pred = self.phoneme_level_predictor(o_en)
        print('o_phn_pred size:      {}'.format(o_phn_pred.size()))
        print('o_en size:            {}'.format(o_en.size()))
        o_en = o_en + o_phn_pred.transpose(1,2)

        # utterance encoder pass
        utterance_vec = self.utterance_encoder(y)
        o_en = o_en + utterance_vec

        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en, x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)
        # pitch predictor pass
        o_pitch = None  # TODO: How do I safely delete this?



        # decoder pass
        o_de, attn = self._forward_decoder(
            o_en, o_dr, x_mask, y_lengths, g=None)
        outputs = {
            "model_outputs": o_de,
            "alignments": attn,
            "pitch": o_pitch, # TODO: How do I safely delete this?
            "durations_log": o_dr_log,
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        pitch = batch["pitch"] if self.args.use_pitch else None
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        durations = batch["durations"]
        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}

        # forward pass
        outputs = self.forward(
            text_input, text_lengths, mel_lengths, y=mel_input, dr=durations, pitch=pitch, aux_input=aux_input
        )
        # use aligner's output as the duration target
        if self.use_aligner:
            durations = outputs["o_alignment_dur"]
        # use float32 in AMP
        with autocast(enabled=False):
            # compute loss
            loss_dict = criterion(
                decoder_output=outputs["model_outputs"],
                decoder_target=mel_input,
                decoder_output_lens=mel_lengths,
                dur_output=outputs["durations_log"],
                dur_target=durations,
                pitch_output=outputs["pitch_avg"] if self.use_pitch else None,
                pitch_target=outputs["pitch_avg_gt"] if self.use_pitch else None,
                input_lens=text_lengths,
                alignment_logprob=outputs["alignment_logprob"] if self.use_aligner else None,
                alignment_soft=outputs["alignment_soft"] if self.use_binary_alignment_loss else None,
                alignment_hard=outputs["alignment_mas"] if self.use_binary_alignment_loss else None,
                phoneme_encoder_output=outputs["o_phn"],
                phoneme_predictor_output=outputs["o_phn_pred"]
            )
            # compute duration error
            durations_pred = outputs["durations"]
            duration_error = torch.abs(
                durations - durations_pred).sum() / text_lengths.sum()
            loss_dict["duration_error"] = duration_error

        return outputs, loss_dict

    def eval_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        pitch = batch["pitch"] if self.args.use_pitch else None
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        durations = batch["durations"]
        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}

        # forward pass
        outputs = self.forward(
            text_input, text_lengths, mel_lengths, y=mel_input, dr=durations, pitch=pitch, aux_input=aux_input
        )
        # use aligner's output as the duration target
        if self.use_aligner:
            durations = outputs["o_alignment_dur"]
        # use float32 in AMP
        with autocast(enabled=False):
            # compute loss
            loss_dict = criterion(
                decoder_output=outputs["model_outputs"],
                decoder_target=mel_input,
                decoder_output_lens=mel_lengths,
                dur_output=outputs["durations_log"],
                dur_target=durations,
                pitch_output=outputs["pitch_avg"] if self.use_pitch else None,
                pitch_target=outputs["pitch_avg_gt"] if self.use_pitch else None,
                input_lens=text_lengths,
                alignment_logprob=outputs["alignment_logprob"] if self.use_aligner else None,
                alignment_soft=outputs["alignment_soft"] if self.use_binary_alignment_loss else None,
                alignment_hard=outputs["alignment_mas"] if self.use_binary_alignment_loss else None,
                phoneme_encoder_output=outputs["o_phn"],
                phoneme_predictor_output=outputs["o_phn_pred"]
            )
            # compute duration error
            durations_pred = outputs["durations"]
            duration_error = torch.abs(
                durations - durations_pred).sum() / text_lengths.sum()
            loss_dict["duration_error"] = duration_error

        return outputs, loss_dict

    def test_run(self, assets: Dict) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For `tts` models, it must include `{'audio_processor': ap}`.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        ap = assets["audio_processor"]
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        print(test_sentences)
        testvar = self._get_random_speakerfile()
        print(testvar)
        # TODO: This isn't a sustainable solution (if it works at all)
        style_wav = os.path.abspath(testvar)
        style_mel = torch.FloatTensor(ap.melspectrogram(
            ap.load_wav(style_wav, sr=ap.sample_rate))).unsqueeze(0)
        style_mel_in = style_mel.cuda()
        aux_inputs = self._get_test_aux_input()
        for idx, sen in enumerate(test_sentences):
            outputs_dict = synthesis(
                self,
                sen,
                self.config,
                "cuda" in str(next(self.parameters()).device),
                ap,
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=style_wav,
                enable_eos_bos_chars=self.config.enable_eos_bos_chars,
                use_griffin_lim=True,
                do_trim_silence=False,
            )
            test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                outputs_dict["outputs"]["model_outputs"], ap, output_fig=False
            )
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                outputs_dict["outputs"]["alignments"], output_fig=False
            )
        return test_figures, test_audios

    def _set_utterance_input(self, aux_input: Dict):
        print(" | > Setting utterance input.")
        style_mel = aux_input.get("style_mel", None)
        return style_mel

    def _get_test_aux_input(
        self,
    ) -> Dict:

        d_vector = None
        if self.config.use_d_vector_file:
            d_vector = [self.speaker_manager.d_vectors[name]["embedding"]
                        for name in self.speaker_manager.d_vectors]
            d_vector = (random.sample(sorted(d_vector), 1),)

        aux_inputs = {
            "speaker_id": None
            if not self.config.use_speaker_embedding
            else random.sample(sorted(self.speaker_manager.speaker_ids.values()), 1),
            "d_vector": d_vector,
            "style_wav": None,  # TODO: handle GST style input
        }
        return aux_inputs

    def _get_random_speakerfile(traindir: str):
        print(' | > Getting random speaker file.')
        files = os.listdir(
            '/home/pdavlin/school/TTS/recipes/ljspeech/LJSpeech-1.1/wavs')
        # print(files)
        return '/home/pdavlin/school/TTS/recipes/ljspeech/LJSpeech-1.1/wavs/' + random.choice(files)

    def get_criterion(self):
        from TTS.tts.layers.losses import ForwardTTSLoss  # pylint: disable=import-outside-toplevel

        return ForwardTTSLoss(self.config)

    #############################
    # CALLBACKS
    #############################

    def on_epoch_start(self, trainer):
        if (trainer.epochs_done == self.predictor_train_epoch):
            print('Starting predictor training'.upper())
            self.train_predictor = True
