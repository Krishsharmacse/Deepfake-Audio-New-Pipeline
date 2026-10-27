import gc
import math
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm as tqdm_auto
from transformers import WavLMModel

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
ASVSPOOF_ROOT = Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA")
TRACK = "LA"
SR = 16000
MAX_LEN = int(6 * SR)
N_FFT = 512
HOP = 160
FEAT_DIM = 128

N_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_PER_GPU = 16
BATCH_SIZE = BATCH_PER_GPU * max(N_GPUS, 1)
ACCUM_STEPS = 4

LR = 3e-4 * max(N_GPUS, 1)
WAVLM_LR = 5e-6
EPOCHS = 50
PATIENCE = 10
MODEL_PATH = "/kaggle/working/best_qsr_v9.pt"
RESUME_PATH = None

# Loss regularizer weights
LAMBDA_AUX = 0.1
LAMBDA_FID = 0.1
LAMBDA_ROUTE = 0.05
LAMBDA_ENTROPY = 0.03
LAMBDA_PURITY = 0.05
LAMBDA_ORTHO = 0.03
LAMBDA_DIVERSITY = 0.02
LAMBDA_NOISE = 0.01
LAMBDA_INTERACTION = 0.02
USE_QUANTUM = True

N_WAVLM_LAYERS = 12
N_ROUTING_STATES = N_WAVLM_LAYERS + 2  # 12 WavLM layers + Raw + STFT
STATE_NAMES = [f"L{i+1}" for i in range(12)] + ["Raw", "STFT"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUDIO AUGMENTATION (ASVspoof 2021 DF Simulation)
# ══════════════════════════════════════════════════════════════════════════════
class ASVspoof2021Augmentor(nn.Module):
    def __init__(self, p_codec=0.6, p_noise=0.5, p_reverb=0.3):
        super().__init__()
        self.p_codec, self.p_noise, self.p_reverb = p_codec, p_noise, p_reverb
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)

    def forward(self, wav):
        B, T = wav.shape
        dev = wav.device
        wav_aug = wav.clone()

        # 1. Codec Degradation Simulation
        if torch.rand(1).item() < self.p_codec:
            for i in range(B):
                if torch.rand(1).item() < 0.5:
                    w = wav_aug[i : i + 1]
                    w_8k = torchaudio.functional.resample(w, 16000, 8000)
                    wav_aug[i : i + 1] = torchaudio.functional.resample(w_8k, 8000, 16000)
                else:
                    bits = torch.randint(6, 10, (1,)).item()
                    q_level = 2**bits
                    wav_aug[i] = torch.round(wav_aug[i] * q_level) / q_level

        # 2. Additive Pink Noise
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(wav_aug)
            freqs = torch.fft.rfftfreq(T, d=1 / 16000, device=dev)
            pink_filter = 1.0 / (freqs + 1e-6) ** 0.5
            pink_filter = pink_filter / pink_filter.max()
            noise_fft = torch.fft.rfft(noise, dim=-1)
            noise = torch.fft.irfft(noise_fft * pink_filter, n=T)
            snr_db = torch.empty(B, 1, device=dev).uniform_(5, 20)
            sig_pow = wav_aug.pow(2).mean(dim=1, keepdim=True)
            noise_pow = noise.pow(2).mean(dim=1, keepdim=True)
            noise_scaled = noise * torch.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10) + 1e-8))
            wav_aug = wav_aug + noise_scaled

        # 3. Simple Reverberation
        if torch.rand(1).item() < self.p_reverb:
            delay_samples = torch.randint(160, 1600, (1,), device=dev).item()
            decay = torch.empty(1, device=dev).uniform_(0.2, 0.5).item()
            wav_aug = wav_aug + decay * F.pad(wav_aug[:, :-delay_samples], (delay_samples, 0))

        # 4. Light SpecAugment
        if torch.rand(1).item() < 0.5:
            spec = torch.stft(wav_aug, n_fft=512, hop_length=160, return_complex=True)
            mag, phase = spec.abs(), spec.angle()
            mag = self.time_mask(mag)
            mag = self.freq_mask(mag)
            wav_aug = torch.istft(mag * torch.exp(1j * phase), n_fft=512, hop_length=160, length=T)

        return wav_aug.clamp(-1.0, 1.0)


augment_2021 = ASVspoof2021Augmentor().to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET & PROTOCOLS
# ══════════════════════════════════════════════════════════════════════════════
class ASVspoofDataset(Dataset):
    def __init__(self, df, wav_dir, training=True, max_samples=None):
        self.df = df.reset_index(drop=True)
        if max_samples:
            self.df = self.df.head(max_samples)
        self.wav_dir = Path(wav_dir)
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = self.wav_dir / f"{row['file_id']}.flac"
        wav, _ = torchaudio.load(str(wav_path))
        wav = wav.mean(0)

        if wav.numel() >= MAX_LEN:
            if self.training:
                start = torch.randint(0, wav.numel() - MAX_LEN + 1, (1,)).item()
                wav = wav[start : start + MAX_LEN]
            else:
                wav = wav[:MAX_LEN]
        else:
            wav = F.pad(wav, (0, MAX_LEN - wav.numel()))

        return wav, float(row["y"]), row.get("attack", "unknown")


def get_paths(root, track):
    t = root / track
    return {
        "train": t / f"ASVspoof2019_{track}_train/flac",
        "dev": t / f"ASVspoof2019_{track}_dev/flac",
        "eval": t / f"ASVspoof2019_{track}_eval/flac",
    }, t / f"ASVspoof2019_{track}_cm_protocols"


def read_protocol(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            rows.append({"file_id": p[1], "attack": p[3], "y": 1 if p[4] == "bonafide" else 0})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM CIRCUIT PRIMITIVES (PYTORCH COMPLEX TENSORS)
# ══════════════════════════════════════════════════════════════════════════════
_CNOT_CACHE = {}
_CNOT_LOCK = threading.Lock()


def _get_cnot(ctrl, tgt, n, device):
    key = (ctrl, tgt, n, device.type, device.index or 0)
    if key not in _CNOT_CACHE:
        with _CNOT_LOCK:
            if key not in _CNOT_CACHE:
                dim = 2**n
                U = torch.eye(dim, dtype=torch.complex64, device=device)
                idx = torch.arange(dim, device=device)
                ctrl_mask = ((idx >> (n - 1 - ctrl)) & 1).bool()
                flipped = idx ^ (1 << (n - 1 - tgt))
                for i in range(dim):
                    if ctrl_mask[i] and i < flipped[i].item():
                        j = flipped[i].item()
                        U[[i, j]] = U[[j, i]].clone()
                _CNOT_CACHE[key] = U
    return _CNOT_CACHE[key]


def _ry(theta):
    c = torch.cos(theta / 2).to(torch.complex64)
    s = torch.sin(theta / 2).to(torch.complex64)
    m = torch.zeros(theta.shape[0], 2, 2, dtype=torch.complex64, device=theta.device)
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = c, -s, s, c
    return m


def _rz(theta):
    c = torch.cos(theta / 2).to(torch.complex64)
    s = torch.sin(theta / 2).to(torch.complex64)
    m = torch.zeros(theta.shape[0], 2, 2, dtype=torch.complex64, device=theta.device)
    m[:, 0, 0] = c - 1j * s
    m[:, 1, 1] = c + 1j * s
    return m


def _apply_gate(state, gate, qubit, n):
    B = state.shape[0]
    state = state.view(B, 2**qubit, 2, 2 ** (n - qubit - 1))
    return torch.einsum("bij,bxjy->bxiy", gate, state).reshape(B, 2**n)


# ══════════════════════════════════════════════════════════════════════════════
# 5. BRANCH ENCODERS
# ══════════════════════════════════════════════════════════════════════════════
class WavLMMultiLayerEncoder(nn.Module):
    def __init__(self, D=FEAT_DIM, model_name="microsoft/wavlm-base-plus", unfreeze_last_n=3):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name).to(DEVICE)
        self.unfreeze_last_n = unfreeze_last_n

        for param in self.wavlm.parameters():
            param.requires_grad = False

        total_layers = 12
        for i in range(total_layers - unfreeze_last_n, total_layers):
            layer = self.wavlm.encoder.layers[i]
            for param in layer.parameters():
                param.requires_grad = True

        self.hidden_size = self.wavlm.config.hidden_size
        self.layer_poolers = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten()) for _ in range(N_WAVLM_LAYERS)
        ])
        self.layer_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, D) for _ in range(N_WAVLM_LAYERS)
        ])

    def forward(self, wav):
        out = self.wavlm(wav.float(), output_hidden_states=True)
        all_hidden = out.hidden_states[1:]

        pooled_layers = []
        for i, h in enumerate(all_hidden):
            h_t = h.transpose(1, 2)
            pooled = self.layer_poolers[i](h_t)
            projected = self.layer_projs[i](pooled)
            pooled_layers.append(projected)

        return torch.stack(pooled_layers, dim=1)


class RawEncoder(nn.Module):
    def __init__(self, D=FEAT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 128, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, D, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(D),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, wav):
        feat = self.encoder(wav.unsqueeze(1))
        return self.pool(feat).squeeze(-1)


class STFTEncoder(nn.Module):
    def __init__(self, n_fft=N_FFT, hop=HOP, D=FEAT_DIM):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, D, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _frontend(self, wav):
        Z = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop, window=self.window, return_complex=True
        )
        mag = torch.log(Z.abs().clamp(min=1e-8))
        phase = Z.angle()
        gd = torch.diff(phase, dim=1, prepend=phase[:, :1, :])
        inst = torch.diff(phase, dim=2, prepend=phase[:, :, :1])
        return torch.stack([mag, phase, gd, inst], dim=1)

    def forward(self, wav):
        x = self.net(self._frontend(wav))
        return self.pool(x).view(x.size(0), -1)


class DriftEncoder(nn.Module):
    def __init__(self, n_fft=N_FFT, hop=HOP, D=FEAT_DIM):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.frame_enc = nn.Sequential(
            nn.Linear(n_fft // 2 + 1, 128), nn.GELU(), nn.Linear(128, 64)
        )
        self.proj = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, D))
        self.score_head = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )
        self.register_buffer("window", torch.hann_window(n_fft))
        self.score_head[-1].bias.data.fill_(0.0)

    def _feats(self, wav):
        Z = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop, window=self.window, return_complex=True
        )
        logmag = torch.log(Z.abs().clamp(min=1e-8))
        emb = F.normalize(self.frame_enc(logmag.permute(0, 2, 1)), dim=-1)
        e1, e2 = emb[:, :-1, :], emb[:, 1:, :]
        step_cos = (e1 * e2).sum(-1)
        vel = (e2 - e1).norm(dim=-1)
        T_half = emb.shape[1] // 2
        el_cos = (
            F.normalize(emb[:, :T_half].mean(1), dim=-1)
            * F.normalize(emb[:, T_half:].mean(1), dim=-1)
        ).sum(-1)
        K = 8
        edges = torch.linspace(0, emb.shape[1], K + 1, device=emb.device).long()
        chunks = F.normalize(
            torch.stack(
                [emb[:, int(edges[i]) : int(edges[i + 1])].mean(1) for i in range(K)], dim=1
            ),
            dim=-1,
        )
        sim = torch.bmm(chunks, chunks.transpose(1, 2))
        mask = torch.triu(torch.ones(K, K, device=emb.device), diagonal=1).bool()
        return torch.stack(
            [
                step_cos.mean(1),
                step_cos.std(1),
                step_cos.min(1).values,
                vel.mean(1),
                torch.quantile(vel, 0.95, dim=1),
                el_cos,
                sim[:, mask].min(1).values,
            ],
            dim=1,
        )

    def forward(self, wav):
        feat = self._feats(wav)
        desc = self.proj(feat)
        return desc, torch.sigmoid(self.score_head(desc)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUANTUM SEMANTIC ROUTER WITH CANONICAL POVM
# ══════════════════════════════════════════════════════════════════════════════
class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        d = input.size(self.dim)
        input_sorted, _ = torch.sort(input, descending=True, dim=self.dim)
        cumsum = torch.cumsum(input_sorted, dim=self.dim)
        k = torch.arange(1, d + 1, device=input.device, dtype=input.dtype).view(
            [1] * (input.dim() - 1) + [-1]
        )
        support = (input_sorted - (cumsum - 1) / k) > 0
        k_max = support.sum(dim=self.dim, keepdim=True).clamp(min=1)
        tau = (cumsum.gather(self.dim, k_max - 1) - 1) / k_max.float()
        return torch.clamp(input - tau, min=0)


class QuantumSemanticRouter(nn.Module):
    def __init__(self, D=FEAT_DIM, n_qubits=4, n_layers=3, n_reupload=2, kraus_rank=4):
        super().__init__()
        self.D = D
        self.n = n_qubits
        self.n_l = n_layers
        self.dim = 2**n_qubits
        self.n_states = 14
        self.n_reupload = n_reupload
        self.kraus_rank = kraus_rank

        self.shared_encoder = nn.Sequential(
            nn.Linear(D, 128), nn.GELU(), nn.Linear(128, n_qubits)
        )
        self.ry_params = nn.Parameter(torch.randn(n_reupload, n_layers, n_qubits) * 0.01)
        self.rz_params = nn.Parameter(torch.randn(n_reupload, n_layers, n_qubits) * 0.01)

        self.mixing_network = nn.Sequential(
            nn.Linear(D * self.n_states, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.n_states),
        )
        self.kraus_U = nn.Parameter(torch.randn(self.n_states, self.dim, kraus_rank) * 0.1)
        self.kraus_V = nn.Parameter(torch.randn(self.n_states, kraus_rank, self.dim) * 0.1)

        self.noise_network = nn.Sequential(
            nn.Linear(D * self.n_states, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.sparsemax = Sparsemax(dim=-1)

    def _encode_with_reuploading(self, all_states):
        B = all_states.shape[0]
        dev = all_states.device
        flat = all_states.reshape(B * self.n_states, self.D)
        theta = (torch.tanh(self.shared_encoder(flat)) + 1.0) * (math.pi / 2.0)
        theta = theta.reshape(B, self.n_states, self.n)

        state = torch.zeros(B * self.n_states, self.dim, dtype=torch.complex64, device=dev)
        state[:, 0] = 1.0 + 0j
        state = state.reshape(B, self.n_states, self.dim)

        for r in range(self.n_reupload):
            for s in range(self.n_states):
                for q in range(self.n):
                    state_s = state[:, s, :].clone()
                    state_s = _apply_gate(state_s, _ry(theta[:, s, q]), q, self.n)
                    state = state.clone()
                    state[:, s, :] = state_s

            for l in range(self.n_l):
                for s in range(self.n_states):
                    state_s = state[:, s, :].clone()
                    for q in range(self.n):
                        ry_gate = _ry(self.ry_params[r, l, q].expand(B).contiguous())
                        rz_gate = _rz(self.rz_params[r, l, q].expand(B).contiguous())
                        state_s = _apply_gate(state_s, ry_gate, q, self.n)
                        state_s = _apply_gate(state_s, rz_gate, q, self.n)

                    for ctrl, tgt in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                        cnot = _get_cnot(ctrl, tgt, self.n, dev).T
                        state_s = torch.matmul(state_s, cnot)

                    state = state.clone()
                    state[:, s, :] = state_s

        return state

    def _compute_fidelity_matrix(self, quantum_states):
        inner = torch.einsum("bni,bmi->bnm", quantum_states.conj(), quantum_states)
        return inner.abs().pow(2)

    def _compute_orthogonality_loss(self, fidelity_matrix):
        N = fidelity_matrix.shape[1]
        mask = torch.triu(torch.ones(N, N, device=fidelity_matrix.device), diagonal=1)
        ortho = (fidelity_matrix * mask.unsqueeze(0)).sum(dim=(1, 2))
        return ortho / (N * (N - 1) / 2)

    def _build_superposition_density_matrix(self, quantum_states, all_states):
        B, N, _ = quantum_states.shape
        flat_states = all_states.reshape(B, -1)
        weights = torch.softmax(self.mixing_network(flat_states), dim=-1)
        sqrt_weights = torch.sqrt(weights)
        superposition = torch.einsum(
            "bn,bnd->bd", sqrt_weights.to(quantum_states.dtype), quantum_states
        )
        norm = torch.norm(superposition, dim=-1, keepdim=True).clamp(min=1e-8)
        superposition = superposition / norm
        rho = torch.einsum("bi,bj->bij", superposition, superposition.conj())
        return rho, weights

    def _learnable_depolarizing_channel(self, rho, all_states):
        B = rho.shape[0]
        flat_states = all_states.reshape(B, -1)
        p = 0.5 * self.noise_network(flat_states)
        identity = (
            torch.eye(self.dim, dtype=torch.complex64, device=rho.device)
            .unsqueeze(0)
            .expand_as(rho)
        )
        rho_noisy = (1 - p.unsqueeze(-1)) * rho + p.unsqueeze(-1) * identity / self.dim
        return rho_noisy, p.squeeze(-1)

    def _build_canonical_povm(self):
        A = torch.matmul(self.kraus_U.to(torch.complex64), self.kraus_V.to(torch.complex64))
        E_raw = torch.matmul(A.conj().transpose(-1, -2), A)
        G = E_raw.sum(dim=0)
        _, S, Vh = torch.linalg.svd(G, full_matrices=True)
        S = S.clamp(min=1e-8)
        S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S)).to(torch.complex64)
        G_inv_sqrt = torch.matmul(Vh.conj().T, torch.matmul(S_inv_sqrt, Vh))
        return torch.matmul(G_inv_sqrt, torch.matmul(E_raw, G_inv_sqrt))

    def _compute_povm_diversity_loss(self, povm_elements):
        N = povm_elements.shape[0]
        diff = povm_elements.unsqueeze(0) - povm_elements.unsqueeze(1)
        dist_sq = (diff.abs() ** 2).sum(dim=(-2, -1))
        mask = torch.triu(torch.ones(N, N, device=povm_elements.device), diagonal=1)
        return (dist_sq * mask).sum() / (N * (N - 1) / 2)

    def _compute_purity(self, rho):
        rho_sq = torch.matmul(rho, rho)
        return torch.einsum("bii->b", rho_sq).real

    def _von_neumann_entropy(self, rho):
        eigvals = torch.linalg.eigvalsh(rho).real.clamp(min=1e-10)
        return -(eigvals * torch.log(eigvals)).sum(dim=-1)

    def _adaptive_measurement_softness(self, probabilities, entropy):
        max_entropy = math.log(self.dim)
        temperature = 1.0 + 2.0 * (entropy / max_entropy)
        scaled_probs = probabilities / temperature.unsqueeze(-1)
        routing_weights = self.sparsemax(scaled_probs)
        return routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def forward(self, all_states):
        B, N, _ = all_states.shape
        if not USE_QUANTUM:
            routing_weights = torch.ones(B, N, device=all_states.device) / N
            dummy = torch.zeros(B, self.dim, self.dim, device=all_states.device)
            zeros_b = torch.zeros(B, device=all_states.device)
            return (
                routing_weights,
                torch.zeros(B, N, device=all_states.device),
                torch.zeros(B, N, device=all_states.device),
                dummy,
                zeros_b,
                zeros_b,
                zeros_b,
                zeros_b,
                zeros_b,
                zeros_b,
                zeros_b,
                zeros_b,
                zeros_b,
            )

        quantum_states = self._encode_with_reuploading(all_states)
        fidelity_matrix = self._compute_fidelity_matrix(quantum_states)
        ortho_loss = self._compute_orthogonality_loss(fidelity_matrix)
        rho_pure, mixing_weights = self._build_superposition_density_matrix(
            quantum_states, all_states
        )
        rho_noisy, noise_level = self._learnable_depolarizing_channel(rho_pure, all_states)
        povm_elements = self._build_canonical_povm()
        povm_diversity = self._compute_povm_diversity_loss(povm_elements)

        raw_probabilities = torch.einsum("bij,qji->bq", rho_noisy, povm_elements).real.clamp(
            min=0.0
        )
        entropy = self._von_neumann_entropy(rho_noisy)
        purity = self._compute_purity(rho_noisy)
        routing_weights = self._adaptive_measurement_softness(raw_probabilities, entropy)

        mask_off_diag = 1.0 - torch.eye(N, device=all_states.device)
        coherence_strength = (fidelity_matrix * mask_off_diag.unsqueeze(0)).mean(dim=(1, 2))

        return (
            routing_weights,
            routing_weights,
            1.0 - routing_weights,
            rho_noisy,
            entropy,
            ortho_loss,
            coherence_strength,
            purity,
            mixing_weights,
            fidelity_matrix,
            noise_level,
            povm_diversity,
            raw_probabilities,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. INTEGRATED PIPELINE MODEL
# ══════════════════════════════════════════════════════════════════════════════
class QuantumSemanticRouterModel(nn.Module):
    def __init__(self, D=FEAT_DIM, n_qubits=4, n_reupload=2, kraus_rank=4):
        super().__init__()
        self.D = D
        self.wavlm_enc = WavLMMultiLayerEncoder(D=D)
        self.raw_enc = RawEncoder(D=D)
        self.stft_enc = STFTEncoder(D=D)
        self.drift_enc = DriftEncoder(D=D)
        self.quantum_router = QuantumSemanticRouter(
            D=D, n_qubits=n_qubits, n_layers=3, n_reupload=n_reupload, kraus_rank=kraus_rank
        )
        self.drift_film = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, D * 2))
        self.classifier = nn.Sequential(
            nn.Linear(D, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.classifier[-1].bias.data.fill_(0.0)

    def forward(self, wav):
        wavlm_layers = self.wavlm_enc(wav)
        raw_pooled = self.raw_enc(wav)
        stft_pooled = self.stft_enc(wav)
        drift_desc, s_drift = self.drift_enc(wav)

        all_states = torch.cat(
            [wavlm_layers, raw_pooled.unsqueeze(1), stft_pooled.unsqueeze(1)], dim=1
        )

        (
            routing_weights,
            fid_real,
            fid_fake,
            rho_noisy,
            entropy,
            ortho_loss,
            coherence_strength,
            purity,
            mixing_weights,
            fidelity_matrix,
            noise_level,
            povm_diversity,
            raw_probabilities,
        ) = self.quantum_router(all_states)

        semantic_emb = torch.sum(all_states * routing_weights.unsqueeze(-1), dim=1)
        gamma, beta = torch.chunk(self.drift_film(drift_desc), 2, dim=-1)
        semantic_emb = gamma * semantic_emb + beta + semantic_emb

        score_main = torch.sigmoid(self.classifier(semantic_emb)).squeeze(-1)
        fid_all = fid_real.mean(dim=-1)

        return (
            score_main,
            s_drift,
            fid_all,
            routing_weights,
            fid_real,
            fid_fake,
            entropy,
            ortho_loss,
            coherence_strength,
            purity,
            mixing_weights,
            fidelity_matrix,
            noise_level,
            povm_diversity,
            raw_probabilities,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8. LOSS OBJECTIVES & DATA AUGMENTATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target >= 0.5, pred, 1.0 - pred)
        return (self.alpha * (1.0 - pt) ** self.gamma * bce).mean()


def mixup_data(wav, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(wav.size(0), device=wav.device)
    return lam * wav + (1.0 - lam) * wav[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════════════════════
# 9. EVALUATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_set(model, loader, set_name, apply_aug=True):
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    all_prob, all_y, all_attacks = [], [], []
    all_routing, all_noise_level, all_purity, all_entropy = [], [], [], []

    aug_tag = "2021-Aug" if apply_aug else "Clean"

    with torch.no_grad():
        for wav, y, attack in tqdm_auto(loader, desc=f"Eval {set_name} [{aug_tag}]", leave=False):
            wav = wav.to(DEVICE)
            if apply_aug:
                wav = augment_2021(wav)

            (
                score_main,
                _,
                _,
                routing_w,
                _,
                _,
                entropy,
                _,
                _,
                purity,
                _,
                _,
                noise_level,
                _,
                _,
            ) = model(wav)

            all_prob.extend(score_main.cpu().tolist())
            all_y.extend(y.tolist())
            all_attacks.extend(attack)
            all_routing.append(routing_w.cpu())
            all_noise_level.append(noise_level.cpu())
            all_purity.append(purity.cpu())
            all_entropy.append(entropy.cpu())

    y_true = np.array(all_y)
    scores = np.array(all_prob)
    routing_all = torch.cat(all_routing, dim=0).numpy()
    noise_all = torch.cat(all_noise_level, dim=0).numpy()
    purity_all = torch.cat(all_purity, dim=0).numpy()
    entropy_all = torch.cat(all_entropy, dim=0).numpy()

    if len(np.unique(y_true)) < 2:
        eer, auc, tdcf = 0.5, 0.5, 1.0
    else:
        fpr, tpr, _ = roc_curve(y_true, scores)
        fnr = 1.0 - tpr
        min_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
        auc = roc_auc_score(y_true, scores)
        tdcf = float((1 * 0.05 * fnr + 10 * 0.95 * fpr).min())

    bonafide_mask = y_true == 1
    spoof_mask = y_true == 0
    print(f"\n   🌌 Quantum Routing Analysis [{set_name} - {aug_tag}]:")
    print(f"   {'State':<8} | {'Bonafide':>10} | {'Spoof':>10} | {'Delta':>10}")
    print(f"   {'-'*45}")
    for i, name in enumerate(STATE_NAMES):
        w_b = routing_all[bonafide_mask, i].mean() if bonafide_mask.any() else 0
        w_s = routing_all[spoof_mask, i].mean() if spoof_mask.any() else 0
        print(f"   {name:<8} | {w_b:>10.4f} | {w_s:>10.4f} | {w_b-w_s:>+10.4f}")

    return {
        "eer": eer,
        "auc": auc,
        "tdcf": tdcf,
        "noise": (noise_all[bonafide_mask].mean(), noise_all[spoof_mask].mean()),
        "purity": (purity_all[bonafide_mask].mean(), purity_all[spoof_mask].mean()),
        "entropy": (entropy_all[bonafide_mask].mean(), entropy_all[spoof_mask].mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN TRAINING WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Load dataset protocols
    wav_dirs, proto_dir = get_paths(ASVSPOOF_ROOT, TRACK)
    df_train = read_protocol(proto_dir / f"ASVspoof2019.{TRACK}.cm.train.trn.txt")
    df_dev = read_protocol(proto_dir / f"ASVspoof2019.{TRACK}.cm.dev.trl.txt")

    train_dataset = ASVspoofDataset(df_train, wav_dirs["train"], training=True)
    lbl_np = df_train["y"].values.astype(int)
    counts = np.bincount(lbl_np)
    weights = 1.0 / counts[lbl_np]
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(train_dataset), replacement=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    dev_dataset = ASVspoofDataset(df_dev, wav_dirs["dev"], training=False)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # Initialize model
    base_model = QuantumSemanticRouterModel(
        D=FEAT_DIM, n_qubits=4, n_reupload=2, kraus_rank=4
    ).to(DEVICE)
    model = nn.DataParallel(base_model) if N_GPUS > 1 else base_model

    # Differential learning rate setup
    wavlm_params = [p for p in base_model.wavlm_enc.wavlm.parameters() if p.requires_grad]
    other_params = [
        p
        for n, p in base_model.named_parameters()
        if not n.startswith("wavlm_enc.wavlm") and p.requires_grad
    ]

    optimizer = AdamW([
        {"params": wavlm_params, "lr": WAVLM_LR, "weight_decay": 1e-3},
        {"params": other_params, "lr": LR, "weight_decay": 5e-4},
    ])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[WAVLM_LR, LR],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.15,
        anneal_strategy="cos",
    )

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_bce = nn.BCELoss()

    best_eer = float("inf")
    wait = 0

    print("\n" + "=" * 65)
    print("Training QSR v17.0 (Quantum Semantic Router)")
    print("=" * 65 + "\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_n = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (wav, y, _) in enumerate(
            tqdm_auto(train_loader, desc=f"Ep {epoch:>2d} train", leave=False)
        ):
            wav, y = wav.to(DEVICE), y.float().to(DEVICE)
            wav = augment_2021(wav)

            use_mixup = torch.rand(1).item() < 0.5
            if use_mixup:
                wav_mix, y_a, y_b, lam = mixup_data(wav, y)
            else:
                wav_mix, y_a, y_b, lam = wav, y, y, 1.0

            (
                score_main,
                s_drift,
                fid_all,
                routing_w,
                fid_real,
                fid_fake,
                entropy,
                ortho_loss,
                coherence_strength,
                purity,
                mixing_weights,
                _,
                noise_level,
                povm_diversity,
                _,
            ) = model(wav_mix)

            loss_main = (
                mixup_criterion(criterion, score_main, y_a, y_b, lam)
                if use_mixup
                else criterion(score_main, y_a)
            )
            loss_aux = criterion_bce(s_drift, y)
            loss_fid_ = criterion_bce(fid_all, y)

            # Quantum Regularization Losses
            y_expanded = y.unsqueeze(1).expand_as(fid_real)
            loss_routing = F.mse_loss(fid_real, y_expanded) + F.mse_loss(
                fid_fake, 1 - y_expanded
            )
            loss_entropy = entropy.mean()
            loss_purity = F.mse_loss(purity, y)
            loss_ortho = ortho_loss.mean()
            loss_diversity = povm_diversity.mean()
            loss_noise = F.mse_loss(noise_level, 1 - y)
            loss_interaction = -coherence_strength.mean()
            mixing_entropy = -(mixing_weights * torch.log(mixing_weights + 1e-8)).sum(-1).mean()

            total_loss = (
                loss_main
                + LAMBDA_AUX * loss_aux
                + LAMBDA_FID * loss_fid_
                + LAMBDA_ROUTE * loss_routing
                + LAMBDA_ENTROPY * loss_entropy
                + LAMBDA_PURITY * loss_purity
                + LAMBDA_ORTHO * loss_ortho
                + LAMBDA_DIVERSITY * loss_diversity
                + LAMBDA_NOISE * loss_noise
                + LAMBDA_INTERACTION * loss_interaction
                + 0.01 * mixing_entropy
            )

            (total_loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            tr_loss += loss_main.item() * wav.size(0)
            tr_n += wav.size(0)

        tr_loss /= tr_n

        # Validation every 2 epochs
        if epoch % 2 == 0 or epoch == 1:
            dev_results = evaluate_set(model, dev_loader, "ASVspoof2019-Dev", apply_aug=True)
            dev_eer = dev_results["eer"]
            dev_auc = dev_results["auc"]
            print(
                f"\nEp {epoch:>2d} | loss={tr_loss:.4f} | Dev_EER(2021-Aug)={dev_eer*100:.2f}% | AUC={dev_auc:.4f} | {time.time()-t0:.0f}s"
            )

            if dev_eer < best_eer - 1e-5:
                best_eer = dev_eer
                wait = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": base_model.state_dict(),
                        "best_eer": best_eer,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    MODEL_PATH,
                )
                print(f"  💾 Saved Checkpoint — Best Dev EER: {best_eer*100:.2f}%")
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"⏹️ Early stopping triggered at epoch {epoch}")
                    break


if __name__ == "__main__":
    main()