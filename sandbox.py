import argparse
import csv
import datetime
import math
import random
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== BIGGER VOCAB (512 unique words) ====================
_VOCAB_BLOB = """
the quick brown fox jumps over lazy dog and then what happens in system of mind
reason cause effect stable flow attractor signal pattern past present future problem solution question answer because
therefore however although if but so understands creates builds dissolves reaches appears exists clear quickly time
change cat bird tree house car computer science logic truth knowledge wisdom idea thought concept process
result outcome a an as at be by do go he is it me my no
on or to up us we act add age air all any arm art ask bad
bag bar bat bed bet big bit box boy bus can cap cop cup cut dad
dam day did die dig dry due ear eat egg end eye fan far fat fed
few fig fit fix fly fog for fun gap gas get got gum gun guy had
has hat her hid him hip his hit hot how hub hug hum ice ill ink
its jam jar jaw jet job jog joy jug key kid kin lab lad lag law
lay led leg let lid lie lip lit log lot low mad man map mat men
met mix mob mom mud mug nap net new nod nor not now nut oak off
oil old one out owe owl own pad pal pan pat paw pay pen pet pie
pig pin pit pot pub put rag ram ran rap rat raw ray red rib rid
rig rim rob rod rot row rub rug run rut sad sap sat saw say sea
set sew she shy sin sip sir sit six ski sky sly sob sod son sow
soy spy sub sum sun sup tab tag tan tap tar tax tea ten tie tin
tip toe ton top tot tow toy try tub tug two use van vet vie vow
wag war was wax way web wed wet who why wig win wit wok won wow
yak yen yes yet you zap zip zone zoom able acid acre aged also area army
atom aunt auto away baby back bake bald ball band bank bare barn base bath beam
bean bear beat been beer bell belt bend bent best bike bill bind bite blow blue
boat body boil bomb bone book boom boot bore born both bowl bulk burn bush busy
cafe cage cake calf call calm camp card care cart case cash cast cave cell chap
chat chef chin chip chop cite city clan clay clip club coal coat code coin cold
come cook cool cope copy cord core corn cost crew crop crow cube cuff cult curb
cure curl cute dale damp dark data dawn days dead deal dear debt deck deep deer
deny desk dial dice diet dime dine dirt dish disk dive dock does done door dose
down drag draw drew drop drum duck dull dumb dump dust duty each earn ease east
easy edge edit else emit ends epic even ever evil exam exit face fact fade fail
fair fake fall fame farm fast fate fear feed feel feet fell felt file fill film
find fine fire firm fish five flag flat flaw flea flex flip float flock floor flour
fluid flush focus force forge forth found frame fresh front frost fruit fully funny gains games
gauge ghost giant given glass glide globe glove going goods grace grade grain grand grant grass
grave great green greet grief grill grind groan group grown guard guess guest guide habit happy
harsh harvest haste hasty hatch haven hazard heady heart heavy hedge hello helps hence herbs hitch
hobby hoist holly honey honor horse hotel hover human humor humph hurry ideal image imply inner
input issue ivory jelly joint judge juice jumpy jolly jumbo kneel knife knock label labor laden
lager large laser later laugh layer learn lease least leave legal lemon level lever light limit
linen liner liquid listen litter little liver lobby local loose lorry lover lower loyal lucky lunar
lunch lunge lyric magic major maker march marry match maybe mayor medal media melon mercy merge
merit merry metal meter micro might minor minus model moist money month moral motor mount mouse
mouth movie music naive naked nappy nasty naval needy nerve never newly night ninja noble noise
noisy north noted novel nurse nylon oasis occur ocean offer often olive onion opera order organ
other ought ounce outer owner paint panel paper party paste patch pause peace peach pearl pedal
penny perch peril petal phase phone photo piano piece pilot pinch pitch place plain plane plant
plate plaza plead pluck point polar porch pound power press price pride prime print prior prism
privy prize probe proof proud prove proxy pulse puppy purge quack quake qualm quart queen query
quest queue quiet quilt quirk quota quote radar radio raise rally ranch range rapid ratio raven
razor reach react ready realm rebel refer refit relax relay relic remit renew repel reply reset
resin retro retry reuse revel rhyme rigid riled risk river roast robot rocky rogue roomy roots
roost rough round route royal rugby ruler rumba rural rusty sadly safer saint salad salon salty
sandy satin sauce sauna saved saver scale scalp scant scare scarf scene scent scoop scope score
scour scout scrap scrub scuba sedan sense serve setup seven shade shady shaft shake shall shame
shape share sharp shave shear sheet shelf shell shift shine shiny shirt shock shoot shore short
shout shown shred shrug sight sigma silky silly since singe sinus siren sixth skate sketch skill
skull slack slain slang slash slate slave sleek sleep slice slide sling sloop slope slosh sloth
slug small smart smash smell smelt smile smirk smoke snack snail snake snare sneak snide sniff
snore snort snowy sober solar solid solve sonic sorry sound south space spade spare spark speak
speed spell spend spice spicy spike spill spine spiral spite split spoil spoke spoof spook spoon
sport spray spree sprig squad squat stack staff stage stain stair stake stale stalk stall stamp
stand stare stark start state stave steak steal steam steel steep steer stem stern stick stiff
still stilt sting stink stock stoic stomp stone stool stoop store storm story stout strap straw
stray streak stream street stress stretch strut stuck study stuff stump stung stunt style suave sugar
suite sulky sunny super surer surge sushi swami swamp swarm swear sweat sweep sweet swell swift
swing swirl sword syrup table tacky taffy taken taker tales tally tamer tangy taper tardy taste
tasty teach tease teeth tempo tenet tenor tense tenth tepee tepid terms terra terse thank theft
their theme there these thick thief thigh thing think third thirsty thorn those three threw thrive
throw thumb thump tiara tidal tiger tight timer timid titan title toast today token tonal tonic
tooth topaz topic torch total touch tough towel tower toxic trace track tract trade trail train
trait tramp trans trash treat trend trial tribe trick tried tripe trite troll troop trout truce
truck truer truly trunk trust tubal tulip tumor tuner tunic turbo tutor twang tweak tweed twice
twine twist tying udder ultra uncle under undid unify union unite unity until upper upset urban
usage usher usual utter vague valet valid valor value valve vapid vault vegan venom venue verge
verse video vigor villa vinyl viola viper viral virus visit vista vital vivid vocal vodka voice
vomit voter vouch vowel wacky waist waive waken waltz warty waste watch water waver weary weave
wedge weedy weigh weird welch welsh wench whack whale wharf wheat wheel whelp where which whiff
while whine whiny whirr whisk white whole whoop whose widen wider widow width wield wight wimpy
wince winch windy wiser wispy witch witty woken woman women world worry worse worst worth would
wound woven wrack wrap wrath wreak wreck wrest wring wrist write wrong wrote xerox yacht yearn
yeast yield young youth zebra zesty zonal stays flows into
"""

_CORPUS_LINES = [
    "the problem appears but the solution exists because the reason is clear therefore the system stays stable",
    "mind understands the cause and the effect creates a stable system",
    "the quick brown fox jumps over the lazy dog and then the pattern flows into the future",
]

_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = _REPO_ROOT / "data" / "corpus.txt"


def _unique_words_from_corpus_file(path: Path) -> set[str]:
    """Words from a line-oriented corpus file (same rules as load_corpus). Missing file → empty."""
    if not path.is_file():
        return set()
    words: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.update(line.lower().split())
    return words


_corpus_words: set[str] = set()
for line in _CORPUS_LINES:
    _corpus_words.update(line.lower().split())
_corpus_words.update(_unique_words_from_corpus_file(DEFAULT_CORPUS_PATH))

_seen: set[str] = set()
BASE_VOCAB: list[str] = []
sorted_corpus = sorted(_corpus_words)
if len(sorted_corpus) > 512:
    print(
        f"Warning: {len(sorted_corpus)} unique words in legacy + default corpus files; "
        "keeping the first 512 alphabetically (raise vocab cap or shrink corpus to include more).",
        flush=True,
    )
    sorted_corpus = sorted_corpus[:512]
for w in sorted_corpus:
    BASE_VOCAB.append(w)
    _seen.add(w)
for w in _VOCAB_BLOB.split():
    wl = w.lower()
    if wl in _seen:
        continue
    _seen.add(wl)
    BASE_VOCAB.append(wl)
    if len(BASE_VOCAB) >= 512:
        break

assert len(BASE_VOCAB) == 512, len(BASE_VOCAB)

FULL_VOCAB = sorted(set(BASE_VOCAB))

# Anti-collapse: trajectory drift in readout; entropy floor for sampling / training logits.
DRIFT_MIN = 0.008
# Min entropy (nats) before extra logit noise; ~log(V) is max. Too low (e.g. 0.02) fires on every confident step.
ENTROPY_FLOOR = 2.0
# Training: lighter exploratory noise when floor triggers (large σ destroys the CE signal).
ENTROPY_FLOOR_NOISE = 0.12
TRAIN_LOGIT_NOISE = 0.005
# Generation: top-k caps tail mass; repeat penalties reduce "effect effect" / same-token loops.
GEN_TOP_K = 28
GEN_REPEAT_LOGIT_PENALTY = 1.35
# Extra penalty on the single most recent token (blocks immediate repeats harder).
GEN_NO_REPEAT_LAST_EXTRA = 5.0
# Training: bigram bias scale (too high pulls logits toward embedding self-similarity loops).
BIGRAM_TRAIN_WEIGHT = 0.025
LABEL_SMOOTHING = 0.06

# Trajectory training (default): match evolved state of context window to evolved state of shifted
# teacher window [x2..xW, next_token]. Auxiliary CE on readout(pred_state) keeps decoding trainable.
TOKEN_AUX_CE_WEIGHT_DEFAULT = 0.2

# --- GOAT-TS-style tension (adaptive dynamics + symplectic readout) ---
# T ≈ |ΔE_state| + λ(1 - cos(fast,slow)) + μ·H(logits); used to adapt inner steps and modulate noise.
TENSION_LAMBDA = 0.35
TENSION_MU = 0.08
TENSION_TOL = 0.85
MAX_CONVERGENCE_STEPS = 12
TENSION_BREAK_THRESH = 2.5
TENSION_NOISE_GAIN = 0.15
GEN_TENSION_TEMP_SCALE = 0.035

# Training: full-window state (W × D), tension-adaptive interaction + step_state (no attention).
NUM_WINDOW_DYNAMICS_STEPS = 8  # legacy default for max_window_steps if not overridden
MAX_WINDOW_STEPS = 16
# Window tension: two regimes — with readout entropy (scale ~0.7–0.9) vs geometry-only (much smaller).
WINDOW_TENSION_USE_ENTROPY = False
WINDOW_TENSION_TOL_GEOMETRY = 0.05
WINDOW_TENSION_HIGH_GEOMETRY = 0.22
WINDOW_TENSION_TOL_ENTROPY = 0.75
WINDOW_TENSION_HIGH_ENTROPY = 0.92
TRAJECTORY_BATCH_SIZE_DEFAULT = 16
# Larger outer step → more movement per iteration (target mean_cos(step) ~0.98–0.995 vs ~0.997+).
WINDOW_INTERACTION_DT_INIT = 0.09
# Sharper distance decay → less mixing of far positions (reduces over-smoothing vs nonlinearity).
WINDOW_POSITION_GAMMA_INIT = 0.52
# Lower than earlier defaults: coupling was washing token rows together (low mean_var in logs).
WINDOW_INTERACTION_SCALE_INIT = 0.07
# Extra gain on tanh(c) in step_state_batch for the window path only (differentiation vs collapse).
WINDOW_NONLINEAR_GAIN = 4.0
# Scales (1 + strength * tanh(asym) * sign(j−i)); was 0.5, too weak for left/right contrast.
POSITION_ASYM_STRENGTH = 1.25


def sample_next_token_id(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    recent_token_ids: list,
    repeat_penalty: float,
    no_repeat_last_extra: float,
) -> int:
    """Apply repetition penalty, optional top-k, temperature, multinomial sample."""
    lo = logits.clone()
    for tid in recent_token_ids[-4:]:
        lo[tid] -= repeat_penalty
    if recent_token_ids:
        lo[recent_token_ids[-1]] -= no_repeat_last_extra
    if top_k > 0 and top_k < lo.numel():
        tk_logits, tk_idx = torch.topk(lo, top_k)
        scaled = (tk_logits - tk_logits.max()) / temperature
        probs = F.softmax(scaled, dim=-1)
        j = torch.multinomial(probs, 1).item()
        return int(tk_idx[j].item())
    scaled = (lo - lo.max()) / temperature
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# ==================== FIXED TORCH MODEL (shape bugs corrected) ====================
class TorchAttractorLanguageModel(nn.Module):
    def __init__(
        self,
        vocab,
        state_dim=512,
        convergence_steps=4,
        slow_decay=0.05,
        slow_lr=0.05,
        w_fast=1.0,
        w_slow=0.3,
        gamma_init=0.2,
        generation_temperature=1.02,
        max_convergence_steps=MAX_CONVERGENCE_STEPS,
        train_window_size: int = 6,
        max_window_steps: int = MAX_WINDOW_STEPS,
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # O(1) word → index (list.index is O(V) and dominated training/data prep).
        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(vocab)}
        self.state_dim = state_dim
        self.train_window_size = train_window_size
        self.max_window_steps = max_window_steps
        _wtol = (
            WINDOW_TENSION_TOL_ENTROPY
            if WINDOW_TENSION_USE_ENTROPY
            else WINDOW_TENSION_TOL_GEOMETRY
        )
        _whigh = (
            WINDOW_TENSION_HIGH_ENTROPY
            if WINDOW_TENSION_USE_ENTROPY
            else WINDOW_TENSION_HIGH_GEOMETRY
        )
        self.register_buffer("window_tension_tol", torch.tensor(float(_wtol)))
        self.register_buffer("window_tension_high", torch.tensor(float(_whigh)))
        # Partial updates per token (path-dependent evolution; not full relaxation).
        self.convergence_steps = convergence_steps
        self.max_convergence_steps = max_convergence_steps
        # Slow memory: slow = (1 - slow_decay) * slow + slow_lr * fast (decay prevents unbounded growth).
        self.register_buffer("slow_decay", torch.tensor(float(slow_decay)))
        self.slow_lr = nn.Parameter(torch.tensor(float(slow_lr)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Decode / context mix: symplectic half-step uses w_fast/w_slow on midpoint fast + slow.
        self.register_buffer("w_fast", torch.tensor(float(w_fast)))
        self.register_buffer("w_slow", torch.tensor(float(w_slow)))
        # Extra temperature at generation time (escapes shallow attractors in sampling).
        self.register_buffer("generation_temperature", torch.tensor(float(generation_temperature)))
        # Context-dependent signal injection strength (trajectory sensitivity).
        self.register_buffer("signal_eps", torch.tensor(1e-6))
        self.dynamics = SimpleAttractorDynamics(state_dim)
        self.embedder = nn.Embedding(self.vocab_size, state_dim)
        self.norm = nn.LayerNorm(state_dim, elementwise_affine=False)
        self.readout = nn.Linear(self.state_dim, self.vocab_size, bias=False)
        # Training: readout from full converged window tensor flattened (context interaction path).
        self.readout_window = nn.Linear(
            train_window_size * state_dim, self.vocab_size, bias=False
        )
        # Positional interaction strength (softplus > 0); left/right differ via |i−j|.
        self.position_gamma_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(WINDOW_POSITION_GAMMA_INIT) - 1.0))
        )
        self.interaction_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(WINDOW_INTERACTION_SCALE_INIT) - 1.0))
        )
        self.interaction_dt_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(WINDOW_INTERACTION_DT_INIT) - 1.0))
        )
        # Left vs right neighbor strength (tanh-bounded inside coupling).
        self.position_asym = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_vocab_ids", torch.arange(self.vocab_size, dtype=torch.long))
        # Unconstrained raw; effective temperature = softplus(raw) > 0 (learnable temp can hit 0 otherwise -> inf logits).
        t0 = 0.12
        self.temperature_raw = nn.Parameter(torch.tensor(math.log(math.exp(t0) - 1.0)))
        # Tension coefficients (buffers; can tune without breaking checkpoints if names stable).
        self.register_buffer("tension_lambda", torch.tensor(float(TENSION_LAMBDA)))
        self.register_buffer("tension_mu", torch.tensor(float(TENSION_MU)))
        self.register_buffer("tension_tol", torch.tensor(float(TENSION_TOL)))
        self.register_buffer("tension_break_thresh", torch.tensor(float(TENSION_BREAK_THRESH)))
        self.tension_noise_gain = nn.Parameter(torch.tensor(float(TENSION_NOISE_GAIN)))
        self.agent_blend_weight = nn.Parameter(torch.tensor(-0.4))
        # Last inner-step tension (float) for generation temperature adaptation.
        self._last_tension_val = 0.0
        # Symplectic readout: fast at start of token vs end (midpoint).
        self._fast_start_snapshot: torch.Tensor | None = None
        # Multi-agent light: recent token signals (normalized embedding directions).
        self._context_ring: list[torch.Tensor] = []
        # Debug: attractor keys and last-step metrics (set by evolve_token when track_attractors=True).
        self.track_attractors = False
        self._attractor_counts: Counter = Counter()
        self._last_state_norm = 0.0
        self._last_state_delta = 0.0
        self._last_combined_norm = 0.0
        self._last_slow_norm = 0.0
        # Trajectory drift pressure in readout (reset at each new sequence via reset_readout_trajectory).
        self._prev_combined = None
        # Filled when collect_dynamics_metrics=True in forward_training_window / run_window_dynamics.
        self._last_dynamics_logs: list[dict] | None = None
        self._last_window_tension_mean: torch.Tensor | None = None
        self._last_adaptive_window_steps: int = 0
        # Mean tension after each outer step (last run_window_dynamics with record_tension_log=True).
        self._last_window_tension_curve: list[float] = []

    def reset_readout_trajectory(self):
        """Clear stored combined state for drift pressure (call once per training window / at generate start)."""
        self._prev_combined = None
        self._context_ring = []
        self._fast_start_snapshot = None
        self._last_tension_val = 0.0

    def _state_energy(self, fast: torch.Tensor) -> torch.Tensor:
        return torch.sum(fast * fast)

    def _normalized_token_embedding(self, token_id: int) -> torch.Tensor:
        """Single-token path: LayerNorm row + unit direction (matches batched embed + norm)."""
        row = self.embedder.weight[token_id].unsqueeze(0)
        emb = self.norm(row).squeeze(0)
        n0 = torch.linalg.vector_norm(emb).clamp(min=1e-12)
        return emb / n0

    def compute_tension(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        logits: torch.Tensor,
        prev_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scalar tension T and components; logits are vocab logits for entropy term."""
        e = self._state_energy(fast)
        de = torch.abs(e - prev_energy)
        fnf = torch.linalg.vector_norm(fast)
        fns = torch.linalg.vector_norm(slow)
        cos_fs = ((fast * slow).sum() / (fnf * fns + 1e-12)).clamp(-1.0, 1.0)
        div = 1.0 - cos_fs
        probs = F.softmax(logits, dim=-1)
        H = -(probs * (probs.clamp(min=1e-9)).log()).sum(dim=-1)
        T = de + self.tension_lambda * div + self.tension_mu * H
        return T, de, div, H

    def _symplectic_combined(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        """Half-step (Störmer-style) blend: midpoint in fast, static slow for this sub-step."""
        fast, slow = self._init_dual_state(fast, slow)
        fs = self._fast_start_snapshot
        if fs is None:
            fs = fast
        fast_mid = 0.5 * (fast + fs)
        return self.w_fast * fast_mid + self.w_slow * slow

    def _logits_for_tension(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        combined = self._symplectic_combined(fast, slow)
        state = combined / (torch.linalg.vector_norm(combined) + 1e-8)
        logits = self.readout(state) / self.effective_temperature()
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)

    def effective_temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw).clamp(min=1e-6)

    def _context_vector(self, fast_state, slow_state):
        """Unit direction from fast (or weighted combined) for context injection; expects inited dual state."""
        combined = self._symplectic_combined(fast_state, slow_state)
        fast_norm = torch.linalg.vector_norm(fast_state)
        eps = self.signal_eps
        device = fast_state.device
        dtype = fast_state.dtype
        if float(fast_norm.detach()) > 1e-8:
            return fast_state / (fast_norm + eps)
        cn = torch.linalg.vector_norm(combined)
        if float(cn.detach()) > 1e-8:
            return combined / (cn + eps)
        return torch.zeros(self.state_dim, device=device, dtype=dtype)

    def get_signal(self, token_id: int, fast_state=None, slow_state=None) -> torch.Tensor:
        """Context-sensitive input: base embedding + gamma * normalized context; then unit-scale signal."""
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        self._fast_start_snapshot = fast_state.detach().clone()
        base_signal = self._normalized_token_embedding(token_id)
        if len(self._context_ring) >= 2:
            w = torch.sigmoid(self.agent_blend_weight)
            ring_mean = torch.stack(self._context_ring).mean(0)
            base_signal = (1.0 - w) * base_signal + w * ring_mean
        self._context_ring.append(base_signal.detach().clone())
        if len(self._context_ring) > 4:
            self._context_ring.pop(0)
        context_vector = self._context_vector(fast_state, slow_state)
        signal = base_signal + self.gamma * context_vector
        sn = torch.linalg.vector_norm(signal)
        signal = signal / (sn + self.signal_eps)
        return signal

    def all_signals(self, fast_state, slow_state):
        """All vocab signals in one batched pass (avoids 512× Python loop and duplicate graphs)."""
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        ids = self._vocab_ids.to(device=fast_state.device)
        emb = self.norm(self.embedder(ids))
        n0 = torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        base_signals = emb / n0
        ctx = self._context_vector(fast_state, slow_state)
        signals = base_signals + self.gamma * ctx
        sn = torch.linalg.vector_norm(signals, dim=-1, keepdim=True).clamp(min=1e-12)
        return signals / (sn + self.signal_eps)

    def _init_dual_state(self, fast_state, slow_state):
        if fast_state is None:
            fast_state = torch.zeros(self.state_dim, device=self.embedder.weight.device, dtype=self.embedder.weight.dtype)
        if slow_state is None:
            slow_state = torch.zeros(self.state_dim, device=self.embedder.weight.device, dtype=self.embedder.weight.dtype)
        return fast_state, slow_state

    def evolve_token(self, fast_state, slow_state, signal, num_steps=None):
        """Tension-adaptive inner steps on fast_state, then slow memory; symplectic readout uses token start/end fast."""
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        if self._fast_start_snapshot is None:
            self._fast_start_snapshot = fast_state.detach().clone()
        base = int(num_steps) if num_steps is not None else self.convergence_steps
        max_steps = self.max_convergence_steps
        prev_energy = self._state_energy(fast_state)
        brk = float(self.tension_break_thresh)
        tol = float(self.tension_tol)
        i = 0
        while i < max_steps:
            prev_fast = fast_state.detach()
            t_prev = self._last_tension_val
            noise_mul = (1.0 + F.softplus(self.tension_noise_gain) * min(t_prev, 3.0)).detach()
            fast_state = self.dynamics(fast_state, signal, noise_scale_mul=noise_mul)
            logits_t = self._logits_for_tension(fast_state, slow_state)
            T, _de, _div, _H = self.compute_tension(
                fast_state, slow_state, logits_t, prev_energy
            )
            prev_energy = self._state_energy(fast_state)
            t_item = T.detach().item()
            self._last_tension_val = t_item
            if t_item > brk:
                fast_state = fast_state + 0.02 * torch.randn_like(fast_state)
                nrm = torch.linalg.vector_norm(fast_state)
                fast_state = fast_state / (nrm + 1e-8)
            self._last_state_norm = float(torch.linalg.vector_norm(fast_state.detach()))
            self._last_state_delta = float(
                torch.linalg.vector_norm((fast_state - prev_fast).detach())
            )
            if self.track_attractors:
                print(
                    f"  [dyn] ||fast||={self._last_state_norm:.4f}  "
                    f"||Δfast||={self._last_state_delta:.4f}  T={t_item:.4f}"
                )
            i += 1
            if i >= base and t_item < tol:
                break
        slow_state = (1.0 - self.slow_decay) * slow_state + self.slow_lr * fast_state
        sn_slow = torch.linalg.vector_norm(slow_state)
        if float(sn_slow.detach()) > 0.5:
            slow_state = slow_state * (0.5 / (sn_slow + 1e-12))
        combined = self._symplectic_combined(fast_state, slow_state)
        self._last_slow_norm = float(torch.linalg.vector_norm(slow_state.detach()))
        self._last_combined_norm = float(torch.linalg.vector_norm(combined.detach()))
        if self.track_attractors:
            aid = torch.round(combined, decimals=2)
            key = aid.detach().cpu().numpy().tobytes()
            self._attractor_counts[key] += 1
            print(
                f"  [token] ||fast||={self._last_state_norm:.4f}  ||slow||={self._last_slow_norm:.4f}  "
                f"||combined||={self._last_combined_norm:.4f}  attractor_id[:4]={aid[:4].tolist()}"
            )
        return fast_state, slow_state

    def step_token(self, fast_state, slow_state, signal):
        """Single dynamics update per token (num_steps=1); use for maximal path dependence."""
        return self.evolve_token(fast_state, slow_state, signal, num_steps=1)

    def combined_state(self, fast_state, slow_state):
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        return self._symplectic_combined(fast_state, slow_state)

    def next_token_logits(self, fast_state, slow_state):
        combined = self.combined_state(fast_state, slow_state)
        if self._prev_combined is not None:
            prev = self._prev_combined.to(device=combined.device, dtype=combined.dtype)
            drift = torch.linalg.vector_norm(combined - prev)
            if float(drift.detach()) < DRIFT_MIN:
                combined = combined + torch.randn_like(combined) * 0.05
        if not self.training:
            combined = combined + torch.randn_like(combined) * 0.01
        state = combined / (torch.linalg.vector_norm(combined) + 1e-8)
        logits = self.readout(state)
        logits = logits / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        self._prev_combined = combined.detach().clone()
        return logits

    def next_token_logits_distance(self, fast_state, slow_state):
        """Distance-to-embedding decoding (baseline / comparison experiments)."""
        state = self.combined_state(fast_state, slow_state)
        all_signals = self.all_signals(fast_state, slow_state)
        dists = torch.linalg.vector_norm(all_signals - state.unsqueeze(0), dim=-1)
        logits = -dists / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        return logits

    def compute_window_tension(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, W, D) normalized states

        Returns:
            scalar tension per batch (B,)
        """
        assert state.dim() == 3 and state.size(1) >= 2
        lam = self.tension_lambda.to(device=state.device, dtype=state.dtype)
        delta = state[:, 1:] - state[:, :-1]
        energy = delta.pow(2).mean(dim=(1, 2))
        cos = F.cosine_similarity(state[:, 1:], state[:, :-1], dim=-1)
        misalign = (1 - cos).mean(dim=1)
        T = energy + lam * misalign
        if WINDOW_TENSION_USE_ENTROPY:
            mu = self.tension_mu.to(device=state.device, dtype=state.dtype)
            flat = state.reshape(state.size(0), -1)
            logits = self.readout_window(flat)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            T = T + mu * entropy
        return T

    def _single_window_step(self, S: torch.Tensor) -> torch.Tensor:
        """One coupling + relaxation step; S is (B, W, D)."""
        B, W, D = S.shape
        assert W == self.train_window_size
        dyn = self.dynamics
        zero_sig = torch.zeros(B, W, D, device=S.device, dtype=S.dtype)
        pos_g = F.softplus(self.position_gamma_raw) + 1e-6
        isc = F.softplus(self.interaction_scale_raw)
        idt = F.softplus(self.interaction_dt_raw)
        ns = (
            dyn.noise_scale.to(device=S.device, dtype=S.dtype)
            if self.training
            else torch.tensor(0.0, device=S.device, dtype=S.dtype)
        )
        delta = positional_coupling_delta(S, pos_g, self.position_asym)
        S = S + idt * isc * delta
        S = step_state_batch(
            S,
            dyn.diffusion,
            zero_sig,
            dyn.dt,
            dyn.cubic_scale,
            beta=dyn.beta,
            noise_scale=ns,
            lambda_decay=dyn.lambda_decay,
            signal_scale=dyn.signal_scale,
            state_norm_eps=dyn.state_norm_eps,
            nonlinear_gain=WINDOW_NONLINEAR_GAIN,
        )
        return S

    def run_window_dynamics(
        self,
        S: torch.Tensor,
        collect_metrics: bool = False,
        record_tension_log: bool = True,
    ) -> tuple[torch.Tensor, list[dict] | None]:
        """
        Tension-adaptive evolution: (W, D) or (B, W, D). Gradients flow through all steps.
        If record_tension_log, fills _last_window_tension_curve with mean(T) after each step
        (use False on teacher-only passes so the student curve is preserved).
        """
        single = S.dim() == 2
        if single:
            S = S.unsqueeze(0)
        B, W, D = S.shape
        assert W == self.train_window_size
        step_logs: list[dict] | None = [] if collect_metrics else None
        tension_curve: list[float] = []
        tol = self.window_tension_tol.to(device=S.device, dtype=S.dtype)
        thigh = self.window_tension_high.to(device=S.device, dtype=S.dtype)
        for step in range(self.max_window_steps):
            S0 = S.detach().clone() if collect_metrics else None
            S = self._single_window_step(S)
            T = self.compute_window_tension(S)
            self._last_window_tension_mean = T.mean().detach()
            self._last_adaptive_window_steps = step + 1
            if record_tension_log:
                tension_curve.append(float(T.mean().detach()))
            if collect_metrics and S0 is not None and step_logs is not None:
                with torch.no_grad():
                    diff = S - S0
                    nd = float(torch.linalg.vector_norm(diff).item())
                    tok_var = float(
                        S.var(dim=(0, 1), unbiased=False).mean().item()
                    )
                    cos = float(
                        F.cosine_similarity(S.flatten(), S0.flatten(), dim=0).item()
                    )
                    mn = float(torch.linalg.vector_norm(S, dim=-1).mean().item())
                    step_logs.append(
                        {
                            "norm_delta": nd,
                            "token_var_mean": tok_var,
                            "cosine_to_prev": cos,
                            "mean_row_norm": mn,
                        }
                    )
            if (T < tol).all():
                break
            if (T > thigh).any():
                noise = 0.01 * torch.randn_like(S)
                S = S + noise
                S = S / (torch.linalg.vector_norm(S, dim=-1, keepdim=True) + 1e-8)
        if record_tension_log:
            self._last_window_tension_curve = tension_curve
        if single:
            S = S.squeeze(0)
        return S, step_logs

    def trajectory_contrastive_loss(
        self, state_a: torch.Tensor, state_b: torch.Tensor
    ) -> torch.Tensor:
        """
        state_a: evolved(context); state_b: evolved(shifted_context + target).
        Same shape (B, W, D).
        """
        a = state_a.reshape(state_a.size(0), -1)
        b = state_b.reshape(state_b.size(0), -1)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        pos = (a * b).sum(dim=-1)
        b_neg = b[torch.randperm(b.size(0), device=b.device)]
        neg = (a * b_neg).sum(dim=-1)
        return F.relu(0.2 - pos + neg).mean()

    def window_ids_from_sequence(self, seq_ids: list[int]) -> list[int]:
        """Last W token ids; left-pad with seq_ids[0] if shorter than window (full window for dynamics)."""
        W = self.train_window_size
        if not seq_ids:
            seq_ids = [0]
        if len(seq_ids) >= W:
            return seq_ids[-W:]
        pad = seq_ids[0]
        return [pad] * (W - len(seq_ids)) + list(seq_ids)

    def embed_window(self, context_ids: list[int]) -> torch.Tensor:
        assert len(context_ids) == self.train_window_size
        device = self.embedder.weight.device
        dtype = self.embedder.weight.dtype
        ids = torch.tensor(context_ids, device=device, dtype=torch.long)
        emb = self.norm(self.embedder(ids))
        n0 = torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        return emb / n0

    def forward_training_window(
        self, context_ids: list[int], collect_dynamics_metrics: bool = False
    ) -> torch.Tensor:
        """
        Embed window tokens to (W, D), run multi-step interacting dynamics, read out vocab logits.
        Training, validation, and generation all use this path.
        """
        assert len(context_ids) == self.train_window_size
        S = self.embed_window(context_ids)
        S, dyn_logs = self.run_window_dynamics(S, collect_metrics=collect_dynamics_metrics)
        self._last_dynamics_logs = dyn_logs
        logits = self.readout_window(S.reshape(-1))
        logits = logits / self.effective_temperature()
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)

    def shifted_next_window(self, context_ids: list[int], target_id: int) -> list[int]:
        """One-step shift: [x2, …, xW, next_token] — teacher window for trajectory consistency."""
        assert len(context_ids) == self.train_window_size
        return context_ids[1:] + [target_id]

    def trajectory_contrastive_loss_and_logits(
        self, contexts: list[list[int]], targets: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched trajectory contrastive loss + readout logits from pred state.
        Teacher states are detached (no grad through shifted window).
        """
        B = len(contexts)
        assert B == len(targets) and B >= 1
        S_pred = torch.stack([self.embed_window(c) for c in contexts], dim=0)
        S_pred, _ = self.run_window_dynamics(
            S_pred, collect_metrics=False, record_tension_log=True
        )
        with torch.no_grad():
            S_tgt = torch.stack(
                [
                    self.embed_window(self.shifted_next_window(c, t))
                    for c, t in zip(contexts, targets, strict=True)
                ],
                dim=0,
            )
            S_tgt, _ = self.run_window_dynamics(
                S_tgt, collect_metrics=False, record_tension_log=False
            )
        loss_traj = self.trajectory_contrastive_loss(S_pred, S_tgt)
        logits = self.readout_window(S_pred.reshape(B, -1)) / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        return loss_traj, logits

    @staticmethod
    def summarize_dynamics_logs(logs: list[dict] | None) -> str:
        if not logs:
            return ""
        nds = [x["norm_delta"] for x in logs]
        tvs = [x["token_var_mean"] for x in logs]
        coss = [x["cosine_to_prev"] for x in logs]
        mns = [x["mean_row_norm"] for x in logs]
        return (
            f"steps={len(logs)}  "
            f"mean|Δ|={statistics.mean(nds):.4f}  "
            f"mean_var={statistics.mean(tvs):.6f}  "
            f"mean_cos(step)={statistics.mean(coss):.4f}  "
            f"mean||row||={statistics.mean(mns):.4f}"
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Run window dynamics on the trailing context; return converged state (W, D)."""
        self.reset_readout_trajectory()
        w2i = self._word_to_idx
        tokens = [w for w in prompt.lower().split() if w in w2i] or ["the"]
        input_ids = [w2i[w] for w in tokens]
        wid = self.window_ids_from_sequence(input_ids)
        with torch.inference_mode():
            S = self.embed_window(wid)
            S, _ = self.run_window_dynamics(S, collect_metrics=False)
        return S

    def _print_attractor_diversity(self, top_k: int = 5):
        ctr = self._attractor_counts
        total = sum(ctr.values())
        n_unique = len(ctr)
        if total == 0:
            print("[diversity] no attractor samples")
            return
        probs = [c / total for _, c in ctr.most_common()]
        entropy = -sum(p * math.log(p + 1e-30) for p in probs if p > 0)
        top = ctr.most_common(top_k)
        most_common_count = top[0][1] if top else 0
        print(
            f"[diversity] unique={n_unique}  total_tokens={total}  "
            f"most_common_count={most_common_count}  entropy={entropy:.4f}"
        )
        print(f"[diversity] top-{top_k} raw counts: {[c for _, c in top]}")

    def generate(
        self,
        prompt: str,
        max_tokens=40,
        debug_track=False,
        log_dynamics: bool = False,
    ):
        """Autoregressive generation: each step uses last-W context → dynamics → readout (same as training)."""
        w2i = self._word_to_idx
        tokens = [w for w in prompt.lower().split() if w in w2i] or ["the"]
        input_ids = [w2i[w] for w in tokens]

        was_training = self.training
        self.eval()
        self.reset_readout_trajectory()
        generated = tokens[:]
        generated_ids = list(input_ids)
        with torch.inference_mode():
            base_gen_temp = self.generation_temperature
            if torch.is_tensor(base_gen_temp):
                base_gen_temp = float(base_gen_temp.detach())
            for ti in range(max_tokens):
                wid = self.window_ids_from_sequence(generated_ids)
                want_metrics = bool(log_dynamics or debug_track)
                logits = self.forward_training_window(wid, collect_dynamics_metrics=want_metrics)
                if want_metrics and self._last_dynamics_logs:
                    show = log_dynamics or (debug_track and ti < 4)
                    if show:
                        summ = self.summarize_dynamics_logs(self._last_dynamics_logs)
                        curve = self._last_window_tension_curve
                        curve_s = (
                            "[" + ", ".join(f"{x:.4f}" for x in curve) + "]"
                            if curve
                            else "[]"
                        )
                        print(
                            f"  [dyn t={ti}] tension_curve={curve_s}  "
                            f"steps={self._last_adaptive_window_steps}  {summ}"
                        )
                next_id = sample_next_token_id(
                    logits,
                    base_gen_temp,
                    GEN_TOP_K,
                    generated_ids,
                    GEN_REPEAT_LOGIT_PENALTY,
                    GEN_NO_REPEAT_LAST_EXTRA,
                )
                next_word = self.vocab[next_id]
                generated.append(next_word)
                generated_ids.append(next_id)

        if debug_track:
            print("[generate] window path: last step dynamics summary (if logged above).")

        if was_training:
            self.train()
        return " ".join(generated)


class SimpleAttractorDynamics(nn.Module):
    def __init__(
        self,
        dim=512,
        dt=0.04,
        cubic_scale=0.008,
        beta_init=0.75,
        noise_scale=1e-3,
        lambda_decay=0.1,
        signal_scale=0.5,
        state_norm_eps=1e-8,
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cubic_scale = cubic_scale
        self.diffusion = nn.Parameter(make_diffusion_matrix(dim))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale)))
        self.register_buffer("lambda_decay", torch.tensor(float(lambda_decay)))
        self.register_buffer("signal_scale", torch.tensor(float(signal_scale)))
        self.register_buffer("state_norm_eps", torch.tensor(float(state_norm_eps)))

    def forward(self, state, signal, noise_scale_mul=1.0):
        ns = self.noise_scale * noise_scale_mul
        return step_state(
            state,
            self.diffusion,
            signal,
            self.dt,
            self.cubic_scale,
            beta=self.beta,
            noise_scale=ns,
            lambda_decay=self.lambda_decay,
            signal_scale=self.signal_scale,
            state_norm_eps=self.state_norm_eps,
        )


def make_diffusion_matrix(dim):
    torch.manual_seed(42)
    q = torch.linalg.qr(torch.randn(dim, dim))[0]
    u = torch.rand(dim)
    eigenvalues = -0.2 - (0.05 + 0.3 * u)
    return (q * eigenvalues) @ q.T


def compare_prompts(model: "TorchAttractorLanguageModel", prompt1: str, prompt2: str):
    """Encode two prompts via window dynamics and compare converged window states (flattened)."""
    model.eval()
    with torch.inference_mode():
        S1 = model.encode_prompt(prompt1)
        S2 = model.encode_prompt(prompt2)
    v1 = S1.reshape(-1)
    v2 = S2.reshape(-1)
    dist = torch.linalg.vector_norm(v1 - v2).item()
    cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()
    print(
        f"[compare_prompts] L2(window)={dist:.6f}  cosine={cos:.6f}  "
        f"||S1||={torch.linalg.vector_norm(v1).item():.4f}  ||S2||={torch.linalg.vector_norm(v2).item():.4f}"
    )


def run_quick_window_tests(model: TorchAttractorLanguageModel) -> None:
    """Sanity checks: divergent states for different orderings; dynamics summary (no training)."""
    print("--- quick window / context test ---", flush=True)
    model.eval()
    w2i = model._word_to_idx
    W = model.train_window_size

    def to_ids(text: str) -> list[int]:
        return [w2i[w] for w in text.lower().split() if w in w2i]

    long_a = to_ids("the cat sat on the mat and then there was a reason")
    long_b = to_ids("there was a reason and then the cat sat on the mat")
    if len(long_a) < W + 1 or len(long_b) < W + 1:
        long_a = to_ids("the quick brown fox jumps over lazy dog and then".split())
        long_b = to_ids("and then lazy dog jumps over the quick brown fox".split())
    wid_a = model.window_ids_from_sequence(long_a)
    wid_b = model.window_ids_from_sequence(long_b)
    with torch.inference_mode():
        Sa = model.embed_window(wid_a)
        Sa, _ = model.run_window_dynamics(Sa)
        Sb = model.embed_window(wid_b)
        Sb, _ = model.run_window_dynamics(Sb)
    va = Sa.reshape(-1)
    vb = Sb.reshape(-1)
    dist = torch.linalg.vector_norm(va - vb).item()
    cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0), dim=1).item()
    print(
        f"  different order (trailing window): L2={dist:.6f}  cosine={cos:.6f}",
        flush=True,
    )
    with torch.inference_mode():
        _, logs = model.run_window_dynamics(model.embed_window(wid_a), collect_metrics=True)
    if logs:
        print(f"  single-window dynamics: {model.summarize_dynamics_logs(logs)}", flush=True)
    print("--- end quick test ---", flush=True)


def step_state(
    state,
    diffusion,
    applied_signal,
    dt,
    cubic_scale,
    beta=1.0,
    noise_scale=0.0,
    lambda_decay=0.1,
    signal_scale=0.5,
    state_norm_eps=1e-8,
):
    c = state - state.mean()
    # Bounded nonlinearity (avoids cubic blow-up at large |c|).
    nonlinear = cubic_scale * torch.tanh(c)
    scaled_signal = signal_scale * applied_signal
    drift = state @ diffusion.T + nonlinear + beta * scaled_signal - lambda_decay * state
    s = state + dt * drift
    if noise_scale is not None and float(noise_scale) > 0:
        s = s + noise_scale * torch.randn_like(s)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = torch.linalg.vector_norm(s)
    eps = state_norm_eps
    if torch.is_tensor(eps):
        eps = eps.to(device=s.device, dtype=s.dtype)
    s = s / (nrm + eps)
    return torch.clamp(s, -10.0, 10.0)


def step_state_batch(
    state_batch: torch.Tensor,
    diffusion: torch.Tensor,
    applied_signal_batch: torch.Tensor,
    dt: float,
    cubic_scale: float,
    beta: torch.Tensor | float = 1.0,
    noise_scale: torch.Tensor | float = 0.0,
    lambda_decay: torch.Tensor | float = 0.1,
    signal_scale: torch.Tensor | float = 0.5,
    state_norm_eps: torch.Tensor | float = 1e-8,
    nonlinear_gain: float = 1.0,
) -> torch.Tensor:
    """
    Same physics as step_state applied row-wise: each s_i is (D,) like a single state vector.
    c = s_i - mean(s_i) over D (matches step_state on shape (D,)).
    nonlinear_gain scales tanh(c) (window training uses >1 to resist row collapse).
    """
    if state_batch.dim() == 3:
        B = state_batch.size(0)
        outs = [
            step_state_batch(
                state_batch[b],
                diffusion,
                applied_signal_batch[b],
                dt,
                cubic_scale,
                beta=beta,
                noise_scale=noise_scale,
                lambda_decay=lambda_decay,
                signal_scale=signal_scale,
                state_norm_eps=state_norm_eps,
                nonlinear_gain=nonlinear_gain,
            )
            for b in range(B)
        ]
        return torch.stack(outs, dim=0)
    c = state_batch - state_batch.mean(dim=-1, keepdim=True)
    nonlinear = cubic_scale * float(nonlinear_gain) * torch.tanh(c)
    scaled_signal = signal_scale * applied_signal_batch
    drift = state_batch @ diffusion.T + nonlinear + beta * scaled_signal - lambda_decay * state_batch
    s = state_batch + dt * drift
    if noise_scale is not None and float(noise_scale) > 0:
        s = s + noise_scale * torch.randn_like(s)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
    eps = state_norm_eps
    if torch.is_tensor(eps):
        eps = eps.to(device=s.device, dtype=s.dtype)
    s = s / (nrm + eps)
    return torch.clamp(s, -10.0, 10.0)


def positional_coupling_delta(
    S: torch.Tensor,
    position_gamma: torch.Tensor,
    position_asym: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sum_j w_ij (S_j - S_i), zero diagonal.
    Base: exp(-gamma * |i-j|); optional asymmetry scales left vs right neighbors (sign j-i).
    """
    if S.dim() == 3:
        return torch.stack(
            [
                positional_coupling_delta(S[b], position_gamma, position_asym)
                for b in range(S.size(0))
            ],
            dim=0,
        )
    W, _D = S.shape
    device = S.device
    dtype = S.dtype
    idx = torch.arange(W, device=device, dtype=dtype)
    rel = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    weights = torch.exp(-position_gamma * rel) * (1.0 - torch.eye(W, device=device, dtype=dtype))
    if position_asym is not None:
        ji = idx.unsqueeze(0) - idx.unsqueeze(1)
        sign_ji = torch.sign(ji)
        sign_ji = torch.where(ji == 0, torch.zeros_like(sign_ji), sign_ji)
        asym_fac = 1.0 + POSITION_ASYM_STRENGTH * torch.tanh(position_asym) * sign_ji
        # Keep strictly positive (avoid inverted coupling if asym + tanh are large).
        weights = weights * asym_fac.clamp(min=0.2)
    wsum = weights.sum(dim=1, keepdim=True)
    return (weights @ S) - wsum * S


def _sequence_is_weak_or_repetitive(token_ids):
    """True if all tokens are identical or one token accounts for >50% of the span (anti-repetition training)."""
    if not token_ids:
        return True
    n = len(token_ids)
    counts = Counter(token_ids)
    max_freq = max(counts.values())
    if max_freq / n > 0.5:
        return True
    return False


def build_sequence_dataset(tokens, window_size=6):
    """
    tokens: List[int] (single sentence, order preserved)
    returns: List of (context, target)
    context: List[int] of length window_size
    target: int (next token)
    Skips windows where the (context + target) span is all-one-token or >50% one token.
    """
    data = []
    for i in range(len(tokens) - window_size):
        context = tokens[i : i + window_size]
        target = tokens[i + window_size]
        span = list(context) + [target]
        if _sequence_is_weak_or_repetitive(span):
            continue
        data.append((context, target))
    return data


def load_corpus(path: Path) -> list[str]:
    """Load non-empty lines from a UTF-8 text file; skip blank lines and #-comments."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Create it or pass --corpus /path/to/file.txt (one sentence per line)."
        )
    out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def corpus_coverage_report(
    sentences: list[str],
    vocab: set[str],
    window_size: int,
) -> None:
    """Print token OOV rate and how many lines yield at least one training window."""
    n_lines = len(sentences)
    raw_tokens = 0
    kept_tokens = 0
    oov_tokens = 0
    n_too_short = 0
    n_usable = 0
    for s in sentences:
        words_raw = s.lower().split()
        raw_tokens += len(words_raw)
        words_in = [w for w in words_raw if w in vocab]
        oov_tokens += len(words_raw) - len(words_in)
        kept_tokens += len(words_in)
        if len(words_in) < window_size + 1:
            n_too_short += 1
        else:
            n_usable += 1
    oov_rate = oov_tokens / raw_tokens if raw_tokens else 0.0
    print(
        f"Corpus coverage: {n_lines} lines  |  {n_usable} usable (≥{window_size + 1} in-vocab tokens)  "
        f"|  {n_too_short} too short after OOV drop  |  OOV tokens={oov_tokens}/{raw_tokens} "
        f"({100.0 * oov_rate:.1f}%)",
        flush=True,
    )


def train_val_split(
    sentences: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if val_fraction <= 0 or len(sentences) < 2:
        return list(sentences), []
    rng = random.Random(seed)
    s = list(sentences)
    rng.shuffle(s)
    n_val = max(1, int(len(s) * val_fraction))
    n_val = min(n_val, len(s) - 1)
    return s[:-n_val], s[-n_val:]


def sentences_with_training_windows(
    sentences: list[str],
    vocab: set[str],
    window_size: int,
) -> list[str]:
    """Lines that yield at least one (context, target) pair after OOV removal."""
    out: list[str] = []
    for s in sentences:
        words = [w for w in s.lower().split() if w in vocab]
        if len(words) >= window_size + 1:
            out.append(s)
    return out


def build_dataset_from_sentences(
    sentences: list[str],
    model: TorchAttractorLanguageModel,
    window_size: int,
) -> list:
    w2i = model._word_to_idx
    dataset = []
    for sentence in sentences:
        words = [w for w in sentence.split() if w in w2i]
        if len(words) < window_size + 1:
            continue
        ids = [w2i[w] for w in words]
        dataset.extend(build_sequence_dataset(ids, window_size=window_size))
    return dataset


@torch.no_grad()
def mean_cross_entropy_eval(
    model: TorchAttractorLanguageModel,
    dataset: list,
) -> float:
    """Validation CE: same logit shaping as training, without noise or entropy-floor branch."""
    if not dataset:
        return float("nan")
    was_training = model.training
    model.eval()
    total = 0.0
    for context, target_id in dataset:
        logits = model.forward_training_window(context)
        prev_id = context[-1]
        logits = logits + BIGRAM_TRAIN_WEIGHT * torch.matmul(
            model.embedder.weight, model.embedder.weight[prev_id]
        )
        logits[prev_id] -= 2.0
        for t in context[-3:]:
            logits[t] -= 1.0
        target = torch.tensor([target_id], device=logits.device, dtype=torch.long)
        loss_ce = F.cross_entropy(
            logits.unsqueeze(0), target, label_smoothing=LABEL_SMOOTHING
        )
        total += float(loss_ce)
    if was_training:
        model.train()
    return total / len(dataset)


@torch.no_grad()
def mean_trajectory_contrastive_eval(
    model: TorchAttractorLanguageModel,
    dataset: list,
    batch_size: int = TRAJECTORY_BATCH_SIZE_DEFAULT,
) -> float:
    """Mean trajectory contrastive loss over the dataset (batched)."""
    if not dataset:
        return float("nan")
    was_training = model.training
    model.eval()
    total = 0.0
    n_seen = 0
    i = 0
    while i < len(dataset):
        chunk = dataset[i : i + batch_size]
        if len(chunk) < 2:
            chunk = chunk + chunk
        contexts = [c for c, _t in chunk]
        targets = [t for _c, t in chunk]
        S_pred = torch.stack([model.embed_window(c) for c in contexts], dim=0)
        S_pred, _ = model.run_window_dynamics(
            S_pred, collect_metrics=False, record_tension_log=False
        )
        S_tgt = torch.stack(
            [
                model.embed_window(model.shifted_next_window(c, t))
                for c, t in zip(contexts, targets, strict=True)
            ],
            dim=0,
        )
        S_tgt, _ = model.run_window_dynamics(
            S_tgt, collect_metrics=False, record_tension_log=False
        )
        total += float(model.trajectory_contrastive_loss(S_pred, S_tgt).item()) * len(
            contexts
        )
        n_seen += len(contexts)
        i += batch_size
    if was_training:
        model.train()
    return total / max(n_seen, 1)


def _aux_ce_loss_batch(
    model: TorchAttractorLanguageModel,
    logits: torch.Tensor,
    contexts: list[list[int]],
    targets: list[int],
) -> torch.Tensor:
    """Mean per-example CE − entropy bonus (matches single-window training shaping)."""
    device = logits.device
    acc = torch.zeros((), device=device, dtype=logits.dtype)
    B = logits.size(0)
    for bi in range(B):
        lo = logits[bi]
        prev_id = contexts[bi][-1]
        lo = lo + BIGRAM_TRAIN_WEIGHT * torch.matmul(
            model.embedder.weight, model.embedder.weight[prev_id]
        )
        lo = lo.clone()
        lo[prev_id] -= 2.0
        for t in contexts[bi][-3:]:
            lo[t] -= 1.0
        lo = lo + TRAIN_LOGIT_NOISE * torch.randn_like(lo)
        probs_floor = F.softmax(lo, dim=-1)
        ent_s = -(probs_floor * torch.log(probs_floor + 1e-9)).sum()
        if float(ent_s.detach()) < ENTROPY_FLOOR:
            lo = lo + torch.randn_like(lo) * ENTROPY_FLOOR_NOISE
            probs_for_entropy = F.softmax(lo, dim=-1)
        else:
            probs_for_entropy = probs_floor
        tgt = torch.tensor([targets[bi]], device=device, dtype=torch.long)
        loss_ce = F.cross_entropy(
            lo.unsqueeze(0), tgt, label_smoothing=LABEL_SMOOTHING
        )
        entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-8)).sum()
        acc = acc + (loss_ce - ENTROPY_WEIGHT * entropy)
    return acc / B


WINDOW_SIZE = 6
NUM_EPOCHS = 25
ENTROPY_WEIGHT = 0.03  # subtracted from CE; keep small vs CE scale or the objective chases flat distributions
CORPUS_EPOCH_COPIES = 2  # duplicate sentence list per epoch for more windows

# Phase 0: fixed prompts for comparable generations across runs (see docs/BASELINE.md).
BASELINE_PROMPT_1 = (
    "the quick brown fox jumps over the lazy dog and then what happens in the system of mind and reason"
)
BASELINE_PROMPT_2 = "mind reason cause effect system"
BASELINE_PROMPT_3 = "effect cause reason mind system"


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _format_phase0_baseline_block(
    *,
    corpus_path: Path,
    seed: int,
    val_fraction: float,
    epoch_copies: int,
    window_size: int,
    num_dynamics_steps: int,
    loss_mode: str,
    token_aux_ce: float,
    last_epoch: int,
    last_mean_loss: float,
    last_train_ce: float,
    last_val_ce: float | None,
    last_train_traj_contrast: float | None,
    last_val_traj_contrast: float | None,
    last_n_windows: int,
    last_epoch_sec: float,
    train_sec_total: float,
    gen1: str,
    gen2: str,
    gen3: str,
) -> str:
    val_s = f"{last_val_ce:.4f}" if last_val_ce is not None else "n/a (no val)"
    traj_train = (
        f"{last_train_traj_contrast:.6f}"
        if last_train_traj_contrast is not None
        else "n/a"
    )
    traj_val = (
        f"{last_val_traj_contrast:.6f}"
        if last_val_traj_contrast is not None
        else "n/a"
    )
    return (
        f"--- Phase 0 baseline (copy into docs/BASELINE.md) ---\n"
        f"time_utc: {datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}\n"
        f"git: {_git_short_hash()}\n"
        f"corpus: {corpus_path}\n"
        f"seed: {seed}  val_fraction: {val_fraction}  epoch_copies: {epoch_copies}\n"
        f"loss_mode: {loss_mode}  token_aux_ce: {token_aux_ce}\n"
        f"window_size: {window_size}  num_dynamics_steps: {num_dynamics_steps}  num_epochs: {NUM_EPOCHS}\n"
        f"last_epoch: {last_epoch}/{NUM_EPOCHS}  windows: {last_n_windows}  epoch_sec: {last_epoch_sec:.1f}\n"
        f"train_sec_total: {train_sec_total:.1f}\n"
        f"mean_loss (objective): {last_mean_loss:.4f}\n"
        f"train_CE: {last_train_ce:.4f}  val_CE: {val_s}\n"
        f"train_traj_contrast: {traj_train}  val_traj_contrast: {traj_val}\n"
        f"\n--- generation baseline prompt 1 ---\n{gen1}\n"
        f"\n--- generation baseline prompt 2 ---\n{gen2}\n"
        f"\n--- generation baseline prompt 3 ---\n{gen3}\n"
        f"--- end baseline ---\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attractor dynamics language model (see README)."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"Training text: one sentence per line; lines starting with # ignored. "
        f"Default: {DEFAULT_CORPUS_PATH}",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Hold out this fraction of lines for validation CE each epoch (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling and train/val split.",
    )
    parser.add_argument(
        "--epoch-copies",
        type=int,
        default=CORPUS_EPOCH_COPIES,
        help="Repeat the training sentence list this many times per epoch before shuffling.",
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=None,
        help="Write Phase 0 baseline snapshot (metrics + fixed generations) to this file (UTF-8).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Sliding context length W; dataset yields (W tokens, next token).",
    )
    parser.add_argument(
        "--num-dynamics-steps",
        type=int,
        default=MAX_WINDOW_STEPS,
        help="Max outer steps per window (tension-adaptive run_window_dynamics; may exit early).",
    )
    parser.add_argument(
        "--trajectory-batch-size",
        type=int,
        default=TRAJECTORY_BATCH_SIZE_DEFAULT,
        help="Batch size for trajectory contrastive training (need >=2 for negatives).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run window/context sanity checks and exit (no training).",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("trajectory", "ce"),
        default="trajectory",
        help="trajectory: contrastive(evolved pred vs teacher state) + optional token CE aux; "
        "ce: classic next-token cross-entropy only.",
    )
    parser.add_argument(
        "--token-aux-ce",
        type=float,
        default=TOKEN_AUX_CE_WEIGHT_DEFAULT,
        help="When loss-mode=trajectory, weight on auxiliary readout CE (0 disables).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=0,
        help="Multiply LR by --lr-gamma every N epochs (0 = no decay).",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,
        help="LR multiplier when --lr-decay-every is set.",
    )
    parser.add_argument(
        "--epoch-metrics-csv",
        type=Path,
        default=None,
        help="Append one CSV row per epoch (loss, CE, traj contrast, mean final-step tension, lr, …) for plotting.",
    )
    parser.add_argument(
        "--log-hard-batch-loss-above",
        type=float,
        default=0.0,
        help="Trajectory mode: print a hint for batches with loss above this (0 = off).",
    )
    args = parser.parse_args()
    corpus_path = args.corpus if args.corpus is not None else DEFAULT_CORPUS_PATH
    random.seed(args.seed)
    window_size = args.window_size
    if window_size < 2:
        raise SystemExit("--window-size must be >= 2")
    if args.loss_mode == "trajectory" and args.trajectory_batch_size < 2:
        raise SystemExit("--trajectory-batch-size must be >= 2 for contrastive training")

    print(f"Vocab size: {len(FULL_VOCAB)}", flush=True)
    model = TorchAttractorLanguageModel(
        FULL_VOCAB,
        train_window_size=window_size,
        max_window_steps=args.num_dynamics_steps,
    )
    if args.quick_test:
        run_quick_window_tests(model)
        return
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    lr_scheduler = None
    if args.lr_decay_every > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_every,
            gamma=args.lr_gamma,
        )
    vocab_set = set(model.vocab)

    sentences = load_corpus(corpus_path)
    print(f"Loaded corpus: {corpus_path}  ({len(sentences)} lines)", flush=True)
    corpus_coverage_report(sentences, vocab_set, window_size)

    usable = sentences_with_training_windows(sentences, vocab_set, window_size)
    if not usable:
        raise RuntimeError(
            "No corpus lines have enough in-vocabulary tokens to form a training window. "
            "Add text using words from the model vocab, or lower --window-size."
        )
    n_skip = len(sentences) - len(usable)
    if n_skip:
        print(
            f"Training/validation use only lines with ≥{window_size + 1} in-vocab tokens "
            f"({len(usable)} lines; {n_skip} lines skipped).",
            flush=True,
        )

    train_sents, val_sents = train_val_split(usable, args.val_fraction, args.seed)
    if val_sents:
        print(
            f"Train/val split: {len(train_sents)} train lines, {len(val_sents)} val lines "
            f"(fraction={args.val_fraction:g}, seed={args.seed})",
            flush=True,
        )
    val_dataset = build_dataset_from_sentences(val_sents, model, window_size)
    if val_sents and len(val_dataset) < 32:
        print(
            f"Warning: validation set is tiny ({len(val_dataset)} windows from {len(val_sents)} lines); "
            "val CE is mostly noise — use train CE and samples until you hold out more lines.",
            flush=True,
        )

    print(
        f"Pre-training ({NUM_EPOCHS} epochs, sliding window size={window_size}, "
        f"num_dynamics_steps={args.num_dynamics_steps}, "
        f"epoch_copies={args.epoch_copies}, "
        f"loss_mode={args.loss_mode}, token_aux_ce={args.token_aux_ce}, "
        f"trajectory_batch_size={args.trajectory_batch_size}, lr={args.lr}, "
        f"lr_decay_every={args.lr_decay_every})...",
        flush=True,
    )
    if args.loss_mode == "trajectory" and args.token_aux_ce <= 0:
        print(
            "Warning: trajectory contrastive loss does not depend on readout_window; with "
            "token_aux_ce=0 the readout gets no gradients — generation quality may collapse. "
            f"Use --token-aux-ce > 0 (default {TOKEN_AUX_CE_WEIGHT_DEFAULT}) unless you decode only by distance.",
            flush=True,
        )
    t_train0 = time.perf_counter()
    last_mean_loss = 0.0
    last_train_ce = 0.0
    last_val_ce: float | None = None
    last_train_traj_contrast: float | None = None
    last_val_traj_contrast: float | None = None
    last_n_windows = 0
    last_epoch_sec = 0.0
    last_epoch_num = 0
    w2i = model._word_to_idx
    for epoch in range(NUM_EPOCHS):
        training_sentences = list(train_sents * args.epoch_copies)
        random.shuffle(training_sentences)
        dataset = []
        for sentence in training_sentences:
            words = [w for w in sentence.split() if w in w2i]
            if len(words) < window_size + 1:
                continue
            ids = [w2i[w] for w in words]
            dataset.extend(build_sequence_dataset(ids, window_size=window_size))
        random.shuffle(dataset)

        n = len(dataset)
        t_ep0 = time.perf_counter()
        print(f"  epoch {epoch + 1}/{NUM_EPOCHS}  |  {n} windows", flush=True)
        loss_sum = 0.0
        mean_final_step_tension = float("nan")
        max_batch_loss_epoch = float("nan")
        if args.loss_mode == "trajectory":
            bs = max(2, args.trajectory_batch_size)
            nb_total = (n + bs - 1) // bs
            report_every = max(1, nb_total // 10)
            final_tension_values: list[float] = []
            max_batch_loss = -1.0
            for bi, batch_start in enumerate(range(0, n, bs)):
                chunk = dataset[batch_start : batch_start + bs]
                if len(chunk) < 2:
                    chunk = chunk + chunk
                contexts = [c for c, t in chunk]
                targets = [t for c, t in chunk]
                loss_traj, logits = model.trajectory_contrastive_loss_and_logits(
                    contexts, targets
                )
                loss = loss_traj
                if args.token_aux_ce > 0.0:
                    loss = loss + args.token_aux_ce * _aux_ce_loss_batch(
                        model, logits, contexts, targets
                    )
                curve = model._last_window_tension_curve
                if curve:
                    final_tension_values.append(curve[-1])
                li = float(loss.detach())
                loss_sum += li
                if li > max_batch_loss:
                    max_batch_loss = li
                if (
                    args.log_hard_batch_loss_above > 0
                    and li >= args.log_hard_batch_loss_above
                ):
                    words0 = " ".join(model.vocab[tid] for tid in contexts[0])
                    print(
                        f"    [hard batch] batch={bi + 1}/{nb_total} loss={li:.4f}  "
                        f"first_ctx={words0!r}",
                        flush=True,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ts = model._last_adaptive_window_steps
                if bi % report_every == 0 or batch_start + bs >= n:
                    if curve:
                        curve_s = "[" + ", ".join(f"{x:.4f}" for x in curve) + "]"
                        print(
                            f"    [batch {bi + 1}/{nb_total}] loss={loss.item():.4f}  "
                            f"Tension curve: {curve_s}  |  Steps: {ts}",
                            flush=True,
                        )
                    else:
                        print(
                            f"    [batch {bi + 1}/{nb_total}] loss={loss.item():.4f}",
                            flush=True,
                        )
            mean_loss = loss_sum / max(nb_total, 1)
            if final_tension_values:
                mean_final_step_tension = float(statistics.mean(final_tension_values))
            if nb_total > 0:
                max_batch_loss_epoch = max_batch_loss
        else:
            report_every = max(1, n // 10)
            for step, (context, target_id) in enumerate(dataset):
                logits = model.forward_training_window(context)
                prev_id = context[-1]
                logits = logits + BIGRAM_TRAIN_WEIGHT * torch.matmul(
                    model.embedder.weight, model.embedder.weight[prev_id]
                )
                logits[prev_id] -= 2.0
                for t in context[-3:]:
                    logits[t] -= 1.0
                logits = logits + TRAIN_LOGIT_NOISE * torch.randn_like(logits)
                probs_floor = F.softmax(logits, dim=-1)
                ent_s = -(probs_floor * torch.log(probs_floor + 1e-9)).sum()
                if float(ent_s.detach()) < ENTROPY_FLOOR:
                    logits = logits + torch.randn_like(logits) * ENTROPY_FLOOR_NOISE
                    probs_for_entropy = F.softmax(logits, dim=-1)
                else:
                    probs_for_entropy = probs_floor
                target = torch.tensor([target_id], device=logits.device, dtype=torch.long)
                loss_ce = F.cross_entropy(
                    logits.unsqueeze(0), target, label_smoothing=LABEL_SMOOTHING
                )
                entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-8)).sum()
                loss = loss_ce - ENTROPY_WEIGHT * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                if step % report_every == 0 or step == n - 1:
                    pct = 100.0 * (step + 1) / max(n, 1)
                    print(
                        f"    [{step + 1}/{n}] {pct:5.1f}%  loss={loss.item():.4f}",
                        flush=True,
                    )
            mean_loss = loss_sum / max(n, 1)

        ep_sec = time.perf_counter() - t_ep0
        train_ce = mean_cross_entropy_eval(model, dataset)
        val_msg = ""
        vce: float | None = None
        if val_dataset:
            vce = mean_cross_entropy_eval(model, val_dataset)
            val_msg = f"  |  val CE={vce:.4f}"
        train_traj_contrast: float | None = None
        val_traj_contrast: float | None = None
        lr_now = optimizer.param_groups[0]["lr"]
        if args.loss_mode == "trajectory":
            train_traj_contrast = mean_trajectory_contrastive_eval(
                model, dataset, batch_size=args.trajectory_batch_size
            )
            if val_dataset:
                val_traj_contrast = mean_trajectory_contrastive_eval(
                    model, val_dataset, batch_size=args.trajectory_batch_size
                )
            tm_s = f"{train_traj_contrast:.6f}" if train_traj_contrast is not None else "n/a"
            vm_s = f"{val_traj_contrast:.6f}" if val_traj_contrast is not None else "n/a"
            mft_s = (
                f"  mean_final_T={mean_final_step_tension:.4f}"
                if math.isfinite(mean_final_step_tension)
                else ""
            )
            mb_s = (
                f"  max_batch_loss={max_batch_loss_epoch:.4f}"
                if math.isfinite(max_batch_loss_epoch)
                else ""
            )
            print(
                f"  epoch {epoch + 1} done  |  {ep_sec:.1f}s  |  lr={lr_now:g}  |  "
                f"mean loss={mean_loss:.4f}  |  train traj contrast={tm_s}  train CE={train_ce:.4f}  "
                f"val traj contrast={vm_s}{mft_s}{mb_s}{val_msg}",
                flush=True,
            )
        else:
            print(
                f"  epoch {epoch + 1} done  |  {ep_sec:.1f}s  |  lr={lr_now:g}  |  "
                f"mean loss={mean_loss:.4f}  |  train CE={train_ce:.4f}{val_msg}",
                flush=True,
            )
        if args.epoch_metrics_csv is not None:
            mpath = Path(args.epoch_metrics_csv)
            mpath.parent.mkdir(parents=True, exist_ok=True)
            new_file = not mpath.exists() or mpath.stat().st_size == 0
            row = [
                epoch + 1,
                args.loss_mode,
                f"{mean_loss:.6f}",
                f"{train_ce:.6f}",
                f"{vce:.6f}" if vce is not None else "",
                f"{train_traj_contrast:.6f}" if train_traj_contrast is not None else "",
                f"{val_traj_contrast:.6f}" if val_traj_contrast is not None else "",
                f"{mean_final_step_tension:.6f}"
                if math.isfinite(mean_final_step_tension)
                else "",
                f"{max_batch_loss_epoch:.6f}"
                if math.isfinite(max_batch_loss_epoch)
                else "",
                f"{lr_now:.8f}",
            ]
            with mpath.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(
                        [
                            "epoch",
                            "loss_mode",
                            "mean_loss",
                            "train_ce",
                            "val_ce",
                            "train_traj_contrast",
                            "val_traj_contrast",
                            "mean_final_step_tension",
                            "max_batch_loss",
                            "lr",
                        ]
                    )
                w.writerow(row)
        if lr_scheduler is not None:
            lr_scheduler.step()
        last_mean_loss = mean_loss
        last_train_ce = train_ce
        last_val_ce = vce
        last_train_traj_contrast = (
            train_traj_contrast if args.loss_mode == "trajectory" else None
        )
        last_val_traj_contrast = (
            val_traj_contrast if args.loss_mode == "trajectory" else None
        )
        last_n_windows = n
        last_epoch_sec = ep_sec
        last_epoch_num = epoch + 1

    train_sec_total = time.perf_counter() - t_train0
    print(f"Pre-training done in {train_sec_total:.1f}s total.")

    print("\nPrompt 1:")
    gen_baseline_1 = model.generate(BASELINE_PROMPT_1)
    print(gen_baseline_1)
    print("\nPrompt 2:")
    gen_baseline_2 = model.generate(BASELINE_PROMPT_2)
    print(gen_baseline_2)
    print("\n(Order sensitivity check — same words, different order:)")
    gen_baseline_3 = model.generate(BASELINE_PROMPT_3)
    print(gen_baseline_3)

    baseline_block = _format_phase0_baseline_block(
        corpus_path=corpus_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        epoch_copies=args.epoch_copies,
        window_size=window_size,
        num_dynamics_steps=args.num_dynamics_steps,
        loss_mode=args.loss_mode,
        token_aux_ce=args.token_aux_ce,
        last_epoch=last_epoch_num,
        last_mean_loss=last_mean_loss,
        last_train_ce=last_train_ce,
        last_val_ce=last_val_ce,
        last_train_traj_contrast=last_train_traj_contrast,
        last_val_traj_contrast=last_val_traj_contrast,
        last_n_windows=last_n_windows,
        last_epoch_sec=last_epoch_sec,
        train_sec_total=train_sec_total,
        gen1=gen_baseline_1,
        gen2=gen_baseline_2,
        gen3=gen_baseline_3,
    )
    print("\n" + baseline_block, flush=True)
    if args.baseline_out is not None:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_out.write_text(baseline_block, encoding="utf-8")
        print(f"Wrote baseline snapshot to {args.baseline_out}", flush=True)
    print("\nDebug attractor tracking (one prompt):")
    model.generate(
        "the system stays stable because the reason is clear",
        max_tokens=12,
        debug_track=True,
    )
    print("\nTrajectory sensitivity (compare_prompts):")
    compare_prompts(
        model,
        "mind reason cause effect system",
        "effect cause reason mind system",
    )
    compare_prompts(
        model,
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog jumps over the quick brown fox",
    )


if __name__ == "__main__":
    main()
