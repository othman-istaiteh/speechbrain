"""This lobe enables the integration of HuggingFace pretrained SeamlessM4T models
for speech-to-text tasks (speech recognition and speech translation).

Reference: https://arxiv.org/abs/2308.11596 (SeamlessM4T)
           https://arxiv.org/abs/2312.05187 (SeamlessM4T v2)
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Othman Istaiteh 2026
"""

import torch

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.integrations.huggingface.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

# Groups of sub-modules (prefixes of ``model.named_modules()``) that can be
# frozen with a single flag. The input embeddings are tied with the lm_head.
FREEZABLE_GROUPS = {
    "encoder": ["speech_encoder"],
    "feature_projection": ["speech_encoder.feature_projection"],
    "adapter": ["speech_encoder.intermediate_ffn", "speech_encoder.adapter"],
    "decoder": ["text_decoder", "lm_head", "shared"],
    "embeddings": ["shared", "text_decoder.embed_tokens", "lm_head"],
}


class SeamlessM4T(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained SeamlessM4T
    (v1 and v2) models for speech-to-text tasks.

    Only the speech encoder and the text decoder are loaded (the
    ``SeamlessM4T[v2]ForSpeechToText`` sub-model of transformers), so the text
    encoder and the speech generation modules never occupy memory.

    The same model performs speech recognition and speech translation: the
    task is selected by the target language. For ASR, set ``target_lang`` to
    the language spoken in the audio; for ST, set it to the language of the
    translation. Language codes follow the Seamless convention (3 letters,
    e.g. "eng", "fra", "arb").

    The decoder is fed with ``[decoder_start, target_lang, tokens...]``, as in
    the original model. To stay compatible with the SpeechBrain searchers,
    ``decoder_start`` is prepended internally: recipes and searchers only deal
    with ``[target_lang, tokens...]`` (see ``bos`` and ``encode_target``).

    Arguments
    ---------
    source : str
        HuggingFace hub name or local path, e.g. "facebook/seamless-m4t-v2-large".
    save_path : str
        Path (dir) of the downloaded model.
    target_lang : str (default: "eng")
        Seamless code of the language the decoder must produce.
    sampling_rate : int (default: 16000)
        Sampling rate of the input audio.
    encoder_only : bool (default: False)
        If True, the text decoder is deleted and the forward function returns
        the speech encoder outputs only.
    freeze : bool (default: False)
        If True, the whole model is frozen and run without gradients.
    freeze_encoder : bool (default: False)
        If True, the speech encoder (including its adapter) is frozen.
    freeze_feature_projection : bool (default: False)
        If True, the feature projection at the input of the speech encoder is
        frozen.
    freeze_adapter : bool (default: False)
        If True, the length adapter between the encoder and the decoder is
        frozen.
    freeze_decoder : bool (default: False)
        If True, the text decoder (including the embeddings and lm_head) is
        frozen.
    freeze_embeddings : bool (default: False)
        If True, the token embeddings and the tied lm_head are frozen. They
        hold a large part of the parameters (256k entries), so freezing them
        saves a lot of memory with little impact on fine-tuning.
    frozen_modules : list of str (default: None)
        Extra sub-modules to freeze, given as prefixes of the HuggingFace
        module names, e.g. ["speech_encoder.encoder.layers.0"].
    output_attentions : bool (default: False)
        If True, the decoder returns the last cross-attention weights. This
        disables the fused attention kernels, so it is slower.
    gradient_checkpointing : bool (default: False)
        If True, enables (non-reentrant) gradient checkpointing to reduce the
        activation memory during fine-tuning.

    Example
    -------
    >>> model_hub = "facebook/hf-seamless-m4t-medium"
    >>> save_path = "savedir"
    >>> model = SeamlessM4T(model_hub, save_path, target_lang="fra")
    >>> wav = torch.randn([2, 16000])
    >>> wav_lens = torch.tensor([1.0, 0.75])
    >>> tokens_bos = torch.tensor([[model.bos]] * 2)
    >>> enc_out, enc_lens, logits = model(wav, tokens_bos, wav_lens)
    >>> logits.shape[:2]
    torch.Size([2, 1])
    """

    def __init__(
        self,
        source,
        save_path,
        target_lang="eng",
        sampling_rate=16000,
        encoder_only=False,
        freeze=False,
        freeze_encoder=False,
        freeze_feature_projection=False,
        freeze_adapter=False,
        freeze_decoder=False,
        freeze_embeddings=False,
        frozen_modules=None,
        output_attentions=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(
            source=source, save_path=save_path, freeze=freeze, **kwargs
        )
        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.output_attentions = output_attentions

        self.load_feature_extractor(
            source, save_path, sampling_rate=sampling_rate
        )

        if encoder_only:
            self.tokenizer = None
            del self.model.text_decoder
            del self.model.lm_head
            del self.model.shared
            self.model.text_decoder = None
            self.model.lm_head = None
        else:
            self.load_tokenizer(source, cache_dir=save_path)
            self.set_target_lang(target_lang)

        self.frozen_prefixes = []
        if not self.freeze:
            flags = {
                "encoder": freeze_encoder,
                "feature_projection": freeze_feature_projection,
                "adapter": freeze_adapter,
                "decoder": freeze_decoder and not encoder_only,
                "embeddings": freeze_embeddings and not encoder_only,
            }
            for group, enabled in flags.items():
                if enabled:
                    self.frozen_prefixes += FREEZABLE_GROUPS[group]
            self.frozen_prefixes += list(frozen_modules or [])
            self._freeze_submodules(self.frozen_prefixes)

        if gradient_checkpointing and not self.freeze:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _from_pretrained(self, source, save_path, cache_dir, **kwargs):
        """Loads the speech-to-text sub-model matching the checkpoint version.

        Arguments
        ---------
        source : str
            HuggingFace hub name or local path.
        save_path : str
            Path (dir) of the downloaded model.
        cache_dir : str
            Path (dir) in which a downloaded pretrained model configuration
            should be cached.
        **kwargs
            Extra keyword arguments passed to the `from_pretrained` function.
        """
        from transformers import (
            SeamlessM4TForSpeechToText,
            SeamlessM4Tv2ForSpeechToText,
        )

        speech_to_text_classes = {
            "seamless_m4t": SeamlessM4TForSpeechToText,
            "seamless_m4t_v2": SeamlessM4Tv2ForSpeechToText,
        }
        if self.config.model_type not in speech_to_text_classes:
            raise ValueError(
                f"{source} is a {self.config.model_type} model, "
                f"expected one of {list(speech_to_text_classes)}."
            )
        self.auto_class = speech_to_text_classes[self.config.model_type]
        super()._from_pretrained(source, save_path, cache_dir, **kwargs)

    def _freeze_submodules(self, prefixes):
        """Freezes the parameters of the sub-modules matching the prefixes.

        Arguments
        ---------
        prefixes : list of str
            Prefixes of the HuggingFace module names to freeze.
        """
        for name, module in self._frozen_submodules(prefixes):
            logger.warning(
                f"speechbrain.integrations.huggingface.seamless_m4t - {name} is frozen."
            )
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def _frozen_submodules(self, prefixes):
        """Returns the (name, module) pairs matching the prefixes.

        Arguments
        ---------
        prefixes : list of str
            Prefixes of the HuggingFace module names.

        Returns
        -------
        list of (str, torch.nn.Module)
            The matching sub-modules (parents only, not their children).

        Raises
        ------
        ValueError
            If a prefix does not match any sub-module.
        """
        modules = dict(self.model.named_modules())
        matched = []
        for prefix in dict.fromkeys(prefixes):
            if prefix not in modules:
                raise ValueError(
                    f"Cannot freeze '{prefix}': no such sub-module in "
                    f"{type(self.model).__name__}."
                )
            matched.append((prefix, modules[prefix]))
        return matched

    def train(self, mode=True):
        """Sets the training mode while keeping the frozen parts in eval mode,
        so that dropout and layerdrop are not applied to them.

        Arguments
        ---------
        mode : bool
            Whether to set training mode (True) or evaluation mode (False).

        Returns
        -------
        self
        """
        super().train(mode)
        if getattr(self, "freeze", False):
            self.model.eval()
        else:
            for _, module in self._frozen_submodules(
                getattr(self, "frozen_prefixes", [])
            ):
                module.eval()
        return self

    def set_target_lang(self, target_lang):
        """Sets the language produced by the decoder.

        Arguments
        ---------
        target_lang : str
            Seamless language code, e.g. "fra".

        Raises
        ------
        KeyError
            If the language is not in the tokenizer vocabulary.
        """
        token = f"__{target_lang}__"
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise KeyError(f"Language {target_lang} not found in tokenizer.")
        self.target_lang = target_lang
        self.tokenizer.tgt_lang = target_lang

    @property
    def bos(self):
        """Token id of the target language, used as first decoder input."""
        return self.tokenizer.convert_tokens_to_ids(f"__{self.target_lang}__")

    @property
    def eos(self):
        """Token id of the end of sequence."""
        return self.config.eos_token_id

    @property
    def pad(self):
        """Token id of the padding."""
        return self.config.pad_token_id

    @property
    def decoder_start(self):
        """Token id prepended internally to every decoder input."""
        return self.config.decoder_start_token_id

    @property
    def vocab_size(self):
        """Size of the output vocabulary."""
        return self.config.vocab_size

    def encode_target(self, text):
        """Tokenizes a target sentence as ``[target_lang] + tokens + [eos]``.

        ``[:-1]`` gives the decoder input and ``[1:]`` gives the targets.

        Arguments
        ---------
        text : str
            The transcription or translation.

        Returns
        -------
        list of int
            The token ids.
        """
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        return [self.bos] + tokens + [self.eos]

    def forward(self, wav, decoder_input_ids=None, wav_lens=None):
        """Computes the speech features, encodes them and runs the decoder.

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals, shape (batch, time).
        decoder_input_ids : torch.Tensor
            Decoder inputs starting with the target language token,
            shape (batch, tokens). Ignored if ``encoder_only``.
        wav_lens : torch.Tensor
            Relative lengths of the signals in SpeechBrain format.

        Returns
        -------
        enc_out : torch.Tensor
            The speech encoder outputs, shape (batch, frames, hidden).
        enc_lens : torch.Tensor
            Relative lengths of the encoder outputs. They are the lengths to
            give to the SpeechBrain searchers.
        logits : torch.Tensor
            The decoder logits, shape (batch, tokens, vocab). Only returned if
            the model is not ``encoder_only``.
        """

        def _forward():
            """Forward pass of the model"""
            input_features, attention_mask = self.compute_features(
                wav, wav_lens
            )
            enc_out, enc_lens = self.forward_encoder(
                input_features, attention_mask
            )
            if self.encoder_only:
                return enc_out, enc_lens
            abs_enc_lens = torch.round(enc_lens * enc_out.shape[1])
            logits, _ = self.forward_decoder(
                enc_out, decoder_input_ids, abs_enc_lens
            )
            return enc_out, enc_lens, logits

        if self.freeze:
            with torch.no_grad():
                return _forward()
        return _forward()

    def compute_features(self, wav, wav_lens=None):
        """Computes the fbank features expected by the speech encoder with the
        HuggingFace feature extractor (80 mel bins stacked by 2, normalised
        per utterance).

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals, shape (batch, time).
        wav_lens : torch.Tensor
            Relative lengths of the signals in SpeechBrain format.

        Returns
        -------
        input_features : torch.Tensor
            Features, shape (batch, frames, 160).
        attention_mask : torch.Tensor
            Mask of the valid frames, shape (batch, frames).
        """
        wav_cpu = wav.detach().float().cpu()
        if wav_lens is None:
            wav_list = list(wav_cpu)
        else:
            wav_list = undo_padding(wav_cpu, wav_lens.detach().cpu())
        inputs = self.feature_extractor(
            [torch.as_tensor(w).numpy() for w in wav_list],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return (
            inputs["input_features"].to(wav.device),
            inputs["attention_mask"].to(wav.device),
        )

    def forward_encoder(self, input_features, attention_mask=None):
        """Encodes the speech features.

        Arguments
        ---------
        input_features : torch.Tensor
            Features from ``compute_features``, shape (batch, frames, 160).
        attention_mask : torch.Tensor
            Mask of the valid frames, shape (batch, frames).

        Returns
        -------
        enc_out : torch.Tensor
            The speech encoder outputs, shape (batch, enc_frames, hidden).
        enc_lens : torch.Tensor
            Relative lengths of the encoder outputs.
        """
        enc_out = self.model.speech_encoder(
            input_features=input_features, attention_mask=attention_mask
        ).last_hidden_state

        if attention_mask is None:
            return enc_out, enc_out.new_ones(enc_out.shape[0])

        if self.config.add_adapter:
            # Same lengths as the ones HF computes after the length adapter.
            subsample = (
                self.model._compute_sub_sample_lengths_from_attention_mask
            )
            frame_lens = subsample(attention_mask)
        else:
            frame_lens = attention_mask.sum(dim=1)
        enc_lens = (frame_lens.float() / enc_out.shape[1]).clamp(max=1.0)
        return enc_out, enc_lens.to(enc_out.device)

    def forward_decoder(self, encoder_states, decoder_input_ids, enc_lens=None):
        """Runs the decoder on a sequence starting with the target language token.

        Arguments
        ---------
        encoder_states : torch.Tensor
            The speech encoder outputs, shape (batch, enc_frames, hidden).
        decoder_input_ids : torch.Tensor
            Decoder inputs starting with the target language token.
        enc_lens : torch.Tensor
            Absolute lengths of the encoder outputs, used to mask padding in
            the cross-attention. If None, nothing is masked.

        Returns
        -------
        logits : torch.Tensor
            The logits for each input position, shape (batch, tokens, vocab).
        attn : torch.Tensor | None
            The last cross-attention weights if ``output_attentions``.
        """
        decoder_input_ids = decoder_input_ids.long()
        start = decoder_input_ids.new_full(
            (decoder_input_ids.shape[0], 1), self.decoder_start
        )
        decoder_input_ids = torch.cat([start, decoder_input_ids], dim=1)

        encoder_attention_mask = None
        if enc_lens is not None:
            encoder_attention_mask = length_to_mask(
                enc_lens.long(),
                max_len=encoder_states.shape[1],
                dtype=encoder_states.dtype,
                device=encoder_states.device,
            )

        output = self.model.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=self.output_attentions,
            use_cache=False,
        )
        logits = self.model.lm_head(output.last_hidden_state).float()

        attn = None
        if self.output_attentions:
            attn = output.cross_attentions[-1][:, :, 1:]

        # The first position predicts the language token, which is given.
        return logits[:, 1:], attn

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """Decoding step used by the SpeechBrain searchers.

        Arguments
        ---------
        tgt : torch.Tensor
            The tokens decoded so far, starting with the target language token.
        encoder_out : torch.Tensor
            The speech encoder outputs.
        enc_len : torch.Tensor
            Absolute lengths of the encoder outputs.

        Returns
        -------
        logits : torch.Tensor
            The logits for each input position.
        attn : torch.Tensor | None
            The last cross-attention weights if ``output_attentions``.
        """
        return self.forward_decoder(encoder_out, tgt, enc_len)
