"""
+----------------------------------------------------------------------------------------------------------------------+
neuro_ntp.py
NeuroNTP model: next-token predictor backbones, auxiliary and downstream heads

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Defines the NeuroNTP model, which combines the multimodal tokenizer (MMT), a causal next-token predictor (NTP)
    and the prediction heads:
        next-token predictors    xLSTM (NX-AI and HuggingFace implementations), Transformer, Linear Transformer,
                                 Mamba (requires the optional mamba-ssm package); selected with ntp_type
        multitask head           auxiliary heads of pretraining: token type / marker presence, time past and time
                                 to seizure, segment type, marker reconstruction
        downstream heads         task-specific heads for fine-tuning, defined in the downstream_tasks section of the
                                 model configuration
    Stateful forward passes for streaming inference with recurrent backbones are included (experimental).
+----------------------------------------------------------------------------------------------------------------------+
"""
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from neuro_ntp.model.mmt.mm_tokenizer import MultiModalTokenizer
from neuro_ntp.model.mmt.core.mmt_structs import MultiModalTokenSequence
from neuro_ntp.model.mmt.core.token_type import TokenType
import neuro_ntp.misc.nntp_utils as nnutils

from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig        # NX-AI xLSTM implementation (paper/research)
try:
    from transformers import xLSTMConfig, xLSTMModel            # HuggingFace implementation with cache params
    XLSTM_HF_AVAILABLE = True
except ImportError:
    XLSTM_HF_AVAILABLE = False
try:
    from mamba_ssm import Mamba as _MambaCUDA
    from mamba_ssm.utils.generation import InferenceParams
    _MAMBA_CUDA = True
except Exception:
    _MAMBA_CUDA = False
    InferenceParams = None


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        tr_emb_dim,
        depth,
        heads,
        causal=True,
        ff_mult=4,
        attn_dropout=0.2,
        layer_dropout=0.1,
        activation="gelu",
    ):
        super(DecoderOnlyTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.tr_emb_dim = tr_emb_dim
        self.depth = depth
        self.heads = heads
        self.causal = causal

        self.projected = self.embedding_dim != self.tr_emb_dim
        if self.projected:
            self.in_proj = nn.Linear(self.embedding_dim, self.tr_emb_dim)
            self.out_proj = nn.Linear(self.tr_emb_dim, self.embedding_dim)

        ff_dim = self.tr_emb_dim * ff_mult

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.tr_emb_dim,
            nhead=self.heads,
            dim_feedforward=ff_dim,
            dropout=attn_dropout,          # general dropout inside layer
            activation=activation,
            batch_first=True,
            norm_first=True,               # often a good modern default
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.depth,
            norm=nn.LayerNorm(self.tr_emb_dim),
        )

        self.layer_dropout = nn.Dropout(layer_dropout)
        self.frozen = False

    def set_frozen(self, freeze):
        self.frozen = freeze
        for param in self.parameters():
            param.requires_grad = not freeze

    def _generate_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Tensor of shape [B, T, D]
            padding_mask: optional BoolTensor of shape [B, T]
                True indicates padding positions to ignore.

        Returns:
            Tensor of shape [B, T, D]
        """
        if self.projected:
            x = self.in_proj(x)

        attn_mask = None
        if self.causal:
            attn_mask = self._generate_causal_mask(x.size(1), x.device)

        x = self.layer_dropout(x)
        x = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )

        if self.projected:
            x = self.out_proj(x)

        return x


class DecoderOnlyLinTrafo(nn.Module):
    def __init__(self, embedding_dim, tr_emb_dim, depth, heads, causal=True):

        super(DecoderOnlyLinTrafo, self).__init__()
        self.embedding_dim = embedding_dim
        self.tr_emb_dim = tr_emb_dim
        self.projected = self.embedding_dim != tr_emb_dim
        if self.projected:
            self.in_proj = nn.Linear(embedding_dim, tr_emb_dim)
            self.out_proj = nn.Linear(tr_emb_dim, embedding_dim)
        self.linear_transformer = LinearAttentionTransformer(
            dim=tr_emb_dim,
            depth=depth,
            max_seq_len=3000,
            heads=heads,
            causal=causal,
            attn_layer_dropout=0.1,
            attn_dropout=0.2,
        )
        self.frozen = False

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def forward(self, x):
        if self.projected:
            x = self.in_proj(x)
            x = self.linear_transformer(x)
            x = self.out_proj(x)
        else:
            x = self.linear_transformer(x)
        return x


class xLSTMNtp(nn.Module):

    def __init__(self, xlstm_cfg):
        super().__init__()
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))

        self.stack = xLSTMBlockStack(cfg)

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def forward(self, x):            # x: (B, T, D)
        return self.stack(x)         # (B, T, D)

    @torch.no_grad()
    def forward_stateful(self, x: torch.Tensor, state=None):
        return self.stack.step(x, state)

    @torch.no_grad()
    def forward_stateful_sequence(self, x: torch.Tensor, state=None):
        if state is None:
            state = {}

        new_state = {}

        for block_idx, block in enumerate(self.stack.blocks):
            block_state = state.get(f"block_{block_idx}", {})

            # only safe for sLSTM blocks
            if not hasattr(block.xlstm, "forward"):
                raise RuntimeError(f"Block {block_idx} has no xlstm.forward")

            # normalized input
            x_norm = block.xlstm_norm(x)

            # sequence forward with carried state
            x_xlstm, xlstm_state = block.xlstm(
                x_norm,
                conv_state=block_state.get("conv_state", None),
                slstm_state=block_state.get("slstm_state", None),
                return_last_state=True,
            )

            x = x + x_xlstm

            if block.ffn is not None:
                x = x + block.ffn(block.ffn_norm(x))

            new_state[f"block_{block_idx}"] = xlstm_state

        x = self.stack.post_blocks_norm(x)
        return x, new_state


class xLSTMNtpHF(nn.Module):
    """
    HuggingFace xLSTM backend - for stateful inference on a continuous stream of embeddings.
    """

    def __init__(self, xlstm_cfg: dict):
        super().__init__()

        d = int(xlstm_cfg.get("embedding_dim", 1024))
        num_blocks = int(xlstm_cfg.get("num_blocks", xlstm_cfg.get("num_hidden_layers", 6)))
        num_heads = int(xlstm_cfg.get("num_heads", 16))
        chunk_size = int(xlstm_cfg.get("context_length", xlstm_cfg.get("chunk_size", 64)))

        cfg = xLSTMConfig(
            hidden_size=d,
            embedding_dim=d,
            num_blocks=num_blocks,
            num_hidden_layers=num_blocks,
            num_heads=num_heads,
            chunk_size=chunk_size,
            use_cache=True,
            return_last_states=True,
            ffn_proj_factor=1.3,
        )

        self.model = xLSTMModel(cfg)

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) float
        returns: (B, T, D)
        """
        out = self.model(inputs_embeds=x, use_cache=False)  # stateless (training-style)
        return out.last_hidden_state

    @torch.no_grad()
    def forward_stateful(self, x: torch.Tensor, cache_params=None):
        """
        Stateful streaming call:
        x: (B, T, D)
        cache_params: HF xLSTMCache or None
        returns:
          y: (B, T, D)
          new_cache: xLSTMCache
        """
        out = self.model(inputs_embeds=x, use_cache=True, cache_params=cache_params)
        return out.last_hidden_state, out.cache_params


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block (CUDA mixer from mamba-ssm)."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1,
                 force_pt=False, layer_idx=None):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mixer = _MambaCUDA(d_model=dim, d_state=d_state, d_conv=d_conv,
                                expand=expand, layer_idx=layer_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        return x + self.dropout(
            self.mixer(self.norm(x), inference_params=inference_params)
        )


class DecoderOnlyMamba(nn.Module):
    def __init__(self, embedding_dim, tr_emb_dim, depth,
                 d_state=16, d_conv=4, expand=2, dropout=0.1, force_pt=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tr_emb_dim = tr_emb_dim
        self.projected = embedding_dim != tr_emb_dim
        if self.projected:
            self.in_proj = nn.Linear(embedding_dim, tr_emb_dim)
            self.out_proj = nn.Linear(tr_emb_dim, embedding_dim)
        self.layers = nn.ModuleList([
            MambaBlock(tr_emb_dim, d_state, d_conv, expand, dropout, force_pt,
                       layer_idx=i)
            for i in range(depth)
        ])
        self.norm_f = RMSNorm(tr_emb_dim)
        self.frozen = False

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def _apply_stack(self, x, inference_params=None):
        if self.projected:
            x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        x = self.norm_f(x)
        if self.projected:
            x = self.out_proj(x)
        return x

    def forward(self, x):
        return self._apply_stack(x, inference_params=None)

    def forward_stateful_sequence(self, x, state=None):
        B, L, _ = x.shape
        if state is None:
            # max_seqlen is nominal for pure Mamba: conv/ssm state is sized from
            # module dims (batch, d_inner, d_conv/d_state), not sequence length.
            state = InferenceParams(max_seqlen=L, max_batch_size=B)

        if state.seqlen_offset == 0:
            y = self._apply_stack(x, inference_params=state)  # fused prefill, writes state
            state.seqlen_offset += L
        else:
            outs = []
            for t in range(L):
                outs.append(self._apply_stack(x[:, t:t + 1], inference_params=state))
                state.seqlen_offset += 1
            y = torch.cat(outs, dim=1)
        return y, state


class TokenTypeClf(nn.Module):
    def __init__(self, embedding_dim, num_ttypes):

        super(TokenTypeClf, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_ttypes)

    def forward(self, x):
        return self.fc(x)


class TpsTtsRegressor(nn.Module):
    def __init__(self, input_dim):
        super(TpsTtsRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.regressor(x).squeeze(-1)


class SingleLayerBinClf(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None, dropout_rate=None):
        super(SingleLayerBinClf, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.fc(x)


class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.3):

        super(BinaryClassifier, self).__init__()
        self.frozen = False
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, 1)
        )

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def forward(self, x):
        return self.classifier(x)


class MarkerDecoder(nn.Module):
    def __init__(self, embedding_dim, marker_emb_dim):

        super(MarkerDecoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim // 2)
        self.emb_fc = nn.Linear(embedding_dim // 2, marker_emb_dim)
        self.act_fn = nn.ELU()

    def forward(self, x):
        x = self.act_fn(self.fc(x))
        return self.emb_fc(x)


class RandomTokenDropout(nn.Module):

    def __init__(self, dropout_prob=0.1):
        super(RandomTokenDropout, self).__init__()
        assert 0.0 <= dropout_prob <= 1.0, "dropout probability must be between 0 and 1."
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if not self.training or self.dropout_prob == 0.0:
            return x

        # mask for tokens to keep/drop
        batch_size, seq_len, data_dim = x.shape
        # random mask for entire tokens (batch, seq, 1)
        token_mask = torch.rand(batch_size, seq_len, 1, device=x.device) > self.dropout_prob
        token_mask = token_mask.expand(-1, -1, data_dim)

        # Apply mask (drop tokens)
        return x * token_mask


class MultitaskHead(nn.Module):

    def __init__(self, config, causal=False):
        super(MultitaskHead, self).__init__()

        self.causal = causal
        self.config = config
        # AMT - Auxiliary multi-task heads
        if self.config['fusion_level'] == "timestep":
            # -- marker-presence classifier - only in "timestep" fusion mode
            self.mp_clf = SingleLayerBinClf(config['embedding_dim'])
        else:
            # -- token-type classifier - only in "channel" fusion mode
            self.tt_clf = TokenTypeClf(config['embedding_dim'], TokenType.max_id() + 1)

        # -- TPS - time-past-seizure regressor
        if self.config['predict_tps']:
            self.tps_reg = TpsTtsRegressor(config['embedding_dim'])

        # -- TTS - time-to-seizure regressor
        if self.config['predict_tts']:
            self.tts_reg = TpsTtsRegressor(config['embedding_dim'])

        # -- ST - segment type classifier (binary)
        if self.config['predict_segtype']:
            self.st_clf = SingleLayerBinClf(config['embedding_dim'])

        # -- MKR - marker decoder lin. mapping (to retrieve SBERT embeddings)
        if self.config['decode_markers']:
            self.mkr_decoder = MarkerDecoder(config['embedding_dim'], config['marker_emb_dim'])

    def forward(self, x):
        token_types, tokens = x

        if self.config['fusion_level'] == "timestep":
            pred_ttypes = self.mp_clf(tokens)  # marker presence detection
        else:
            pred_ttypes = self.tt_clf(tokens) if self.config['predict_token_type'] else None  # token-type classifier
        pred_tps = self.tps_reg(tokens) if self.config['predict_tps'] else None  # time-past-seizure regressor
        pred_tts = self.tts_reg(tokens) if self.config['predict_tts'] else None  # time-to-seizure regressor
        pred_st = self.st_clf(tokens) if self.config['predict_segtype'] else None  # segment type classifier

        # marker reconstruction - only at expected and predicted marker position
        pred_mkr, marker_pos = None, None
        if self.config.get("decode_markers", False) and (pred_ttypes is not None):
            tt_in = token_types.to(pred_ttypes.device)  # (B,T)
            expected_tt = (tt_in == int(TokenType.Marker))[:, int(self.causal):]  # predicts positions 1..T-1 for ntp
            mask = expected_tt
            if mask.any():
                marker_pos = mask.nonzero(as_tuple=False)
                pred_mkr = self.mkr_decoder(tokens[mask])

        return pred_ttypes, pred_tps, pred_tts, pred_st, pred_mkr, marker_pos

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)


class NeuroNTP(nn.Module):

    def __init__(self, config: dict = None):
        super(NeuroNTP, self).__init__()

        self.config = config
        self.signal_channels = config['signal_channels']

        # MMT - multimodal tokenizer
        self.mm_tokenizer = MultiModalTokenizer(config=config)

        # NTP - next-token predictor (causal sequence model)
        if self.config['ntp_type'] == 'xLSTM':
            self.ntp = xLSTMNtp(config['xlstm'])
        elif self.config['ntp_type'] == 'xLSTM_HF':
            self.ntp = xLSTMNtpHF(config['xlstm'])
        elif self.config['ntp_type'] == 'transformer':
            self.ntp = DecoderOnlyTransformer(embedding_dim=config['embedding_dim'],
                                           tr_emb_dim=config['trf_emb_dim'],
                                           depth=config['trf_depth'], heads=config['trf_heads'])
        elif self.config['ntp_type'] == 'lintrafo':
            self.ntp = DecoderOnlyLinTrafo(embedding_dim=config['embedding_dim'],
                                           tr_emb_dim=config['trf_emb_dim'],
                                           depth=config['trf_depth'], heads=config['trf_heads'])
        elif self.config['ntp_type'] == 'mamba':
            mamba_config = config['mamba']
            self.ntp = DecoderOnlyMamba(embedding_dim=config['embedding_dim'],
                                        tr_emb_dim=mamba_config['emb_dim'],
                                        depth=mamba_config['depth'],
                                        d_state=mamba_config.get('d_state', 16),
                                        d_conv=mamba_config.get('d_conv', 4),
                                        expand=mamba_config.get('expand', 2))

        # AUX - Multitask head
        if not self.config.get('linprobe', False):
            self.multitask_head = MultitaskHead(config, causal=True)

        #------------------------------------------------------------
        # DOWNSTREAM task heads / relevant in finetuning
        # -----------------------------------------------------------
        self.dws_heads = nn.ModuleDict()
        self.dws_specs = {}  # non-module metadata per task

        dws_cfg = self.config.get("downstream_tasks", {}) or {}
        if bool(dws_cfg.get("enabled", False)):
            tasks_cfg = dws_cfg.get("tasks", {}) or {}
            if not tasks_cfg:
                raise ValueError("downstream.enabled=true but downstream.tasks is empty")

            for task_name, spec in tasks_cfg.items():
                attach_on = spec.get("attach_on", "ntp")
                cls_path = spec["class"]
                weight = float(spec.get("weight", 1.0))
                kwargs = dict(spec.get("kwargs", {}) or {})

                if attach_on not in ("mmt", "ntp"):
                    raise ValueError(f"{task_name}: attach_on must be 'mmt' or 'ntp', got {attach_on}")

                hidden_dim = int(kwargs.pop("hidden_dim", self.config['embedding_dim'] // 4))
                dropout = float(kwargs.pop("dropout", kwargs.pop("dropout_rate", 0.25)))

                head = nnutils.instantiate(cls_path, self.config['embedding_dim'], hidden_dim, dropout_rate=dropout, **kwargs)

                self.dws_heads[task_name] = head
                self.dws_specs[task_name] = {
                    "attach_on": attach_on,
                    "weight": weight,
                }

        self.dropout = nn.Dropout(p=config['emb_dropout'])
        self.token_dropout_prob = config['token_dropout_prob']
        self.token_dropout = RandomTokenDropout(self.token_dropout_prob)
        self.frozen = False

    def registries(self) -> dict:
        regs = {}
        if hasattr(self, "mm_tokenizer"):
            # expose as "mmt.encoders" to make it explicit at the call site
            regs["mmt.encoders"] = self.mm_tokenizer.encoders
        return regs

    def export_meta(self) -> dict:
        """small dict to reconstruct model at load time (no tensors)."""
        return {
            "version": 1.1,
            "config": self.config,  # contains the configuration of the model
            "ntp_type": self.config.get("ntp_type", "transformer"), # type of the causal model
            "token_types": getattr(self.mm_tokenizer, "token_types", None), # token types
            "token_codes": getattr(self.mm_tokenizer, "token_codes", None), # token codes
        }

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def set_downstream_heads_frozen(self, frozen: bool):
        for head in self.dws_heads.values():
            if hasattr(head, "set_frozen"):
                head.set_frozen(frozen)
            else:
                for p in head.parameters():
                    p.requires_grad = not frozen

    def forward(self, x):
        mmt_seq = self.mm_tokenizer(x)               # multi-modal tokenizer
        mmt_tokens = mmt_seq.tokens                  # encoded tokens

        if not self.config.get('linprobe', False):
            mmt_tokens = self.dropout(mmt_tokens)           # dropout in emb.dim
            mmt_tokens = self.token_dropout(mmt_tokens)     # dropout of token before ntp

        # main task - next-token prediction
        pred_tokens = self.ntp(mmt_tokens[:, :-1, :])         # causal sequence model

        # auxiliary tasks - only if not linear probing
        pred_ttypes = pred_tps = pred_tts = pred_st = pred_mkr = marker_pos = None
        if not self.config.get('linprobe', False):
            pred_ttypes, pred_tps, pred_tts, pred_st, pred_mkr, marker_pos = (
                self.multitask_head((mmt_seq.token_types, pred_tokens)))

        # downstream heads
        dws_pred = {}
        if len(self.dws_heads) > 0:
            for task_name, head in self.dws_heads.items():
                attach_on = self.dws_specs[task_name]["attach_on"]
                x = mmt_seq.target_tokens[:, 1:, :] if attach_on == "mmt" else pred_tokens
                dws_pred[task_name] = head(x)

        # next-token predicted sequence
        ntp_seq = MultiModalTokenSequence(pred_tokens, target_tokens=None, token_types=pred_ttypes, tps=pred_tps,
                                          tts=pred_tts, seg_types=pred_st, markers=pred_mkr, marker_pos=marker_pos)
        return ntp_seq, mmt_seq, dws_pred


    @torch.no_grad()
    def forward_stateful_channel(self, x, state=None):
        # tokenize full window (channel fusion always re-tokenizes)
        mmt_seq = self.mm_tokenizer(x)
        tokens = mmt_seq.tokens  # (B,S,D)
        token_types = mmt_seq.token_types  # (B,S)

        S = tokens.size(1)
        if S < 2:
            raise RuntimeError(f"Need at least 2 tokens, got S={S}")

        eos = (token_types == int(TokenType.EOS))  # (B, S)
        idx = torch.arange(S - 1, device=eos.device)
        prev_eos = torch.where(eos[:, :-1], idx, -1).amax(dim=1)  # (B,)
        start = int(prev_eos[0]) + 1  # batch size 1 in streaming mode

        new_state = state
        for i in range(start, S):
            x_step = tokens[:, i:i+1, :]  # (B,1,D)
            pred_tokens, new_state = self.ntp.forward_stateful(x_step, new_state)  # (B,1,D)

        # last token is EOS target; last input token is S-2
        inp_pos = S - 2

        # multitask head needs (B,2) token_types so tf[:,1:] aligns to (B,1)
        tf_token_types = token_types[:, inp_pos:inp_pos + 2]  # (B,2)

        pred_ttypes, pred_tps, pred_tts, pred_st, pred_mkr, marker_pos = self.multitask_head((tf_token_types, pred_tokens))

        # downstream heads (attach_on="ntp" case)
        dws_pred = {name: head(pred_tokens) for name, head in self.dws_heads.items()}

        ntp_seq = MultiModalTokenSequence(
            pred_tokens, target_tokens=None,
            token_types=pred_ttypes, tps=pred_tps, tts=pred_tts,
            seg_types=pred_st, markers=pred_mkr, marker_pos=marker_pos
        )
        return ntp_seq, mmt_seq, dws_pred, new_state

