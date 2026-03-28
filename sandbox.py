import argparse
import datetime
import math
import random
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

# --- GOAT-TS-style tension (adaptive dynamics + symplectic readout) ---
# T ≈ |ΔE_state| + λ(1 - cos(fast,slow)) + μ·H(logits); used to adapt inner steps and modulate noise.
TENSION_LAMBDA = 0.35
TENSION_MU = 0.08
TENSION_TOL = 0.85
MAX_CONVERGENCE_STEPS = 12
TENSION_BREAK_THRESH = 2.5
TENSION_NOISE_GAIN = 0.15
GEN_TENSION_TEMP_SCALE = 0.035


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
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # O(1) word → index (list.index is O(V) and dominated training/data prep).
        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(vocab)}
        self.state_dim = state_dim
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

    def encode_prompt(self, prompt: str):
        """Run dynamics on prompt tokens only; return (fast_state, slow_state)."""
        self.reset_readout_trajectory()
        w2i = self._word_to_idx
        tokens = [w for w in prompt.lower().split() if w in w2i] or ["the"]
        input_ids = [w2i[w] for w in tokens]
        fast_state, slow_state = None, None
        for tid in input_ids:
            sig = self.get_signal(tid, fast_state, slow_state)
            fast_state, slow_state = self.evolve_token(fast_state, slow_state, sig)
        return fast_state, slow_state

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

    def generate(self, prompt: str, max_tokens=40, debug_track=False):
        w2i = self._word_to_idx
        tokens = [w for w in prompt.lower().split() if w in w2i] or ["the"]
        input_ids = [w2i[w] for w in tokens]

        self.track_attractors = debug_track
        if debug_track:
            self._attractor_counts = Counter()

        was_training = self.training
        self.eval()
        self.reset_readout_trajectory()
        fast_state, slow_state = None, None
        with torch.inference_mode():
            for tid in input_ids:
                sig = self.get_signal(tid, fast_state, slow_state)
                fast_state, slow_state = self.evolve_token(fast_state, slow_state, sig)

            generated = tokens[:]
            generated_ids = list(input_ids)
            base_gen_temp = self.generation_temperature
            if torch.is_tensor(base_gen_temp):
                base_gen_temp = float(base_gen_temp.detach())
            tol_f = float(self.tension_tol)
            for _ in range(max_tokens):
                logits = self.next_token_logits(fast_state, slow_state)
                gen_temp = base_gen_temp * (
                    1.0 + GEN_TENSION_TEMP_SCALE * max(0.0, self._last_tension_val - tol_f)
                )
                next_id = sample_next_token_id(
                    logits,
                    gen_temp,
                    GEN_TOP_K,
                    generated_ids,
                    GEN_REPEAT_LOGIT_PENALTY,
                    GEN_NO_REPEAT_LAST_EXTRA,
                )
                next_word = self.vocab[next_id]
                generated.append(next_word)
                generated_ids.append(next_id)
                sig = self.get_signal(next_id, fast_state, slow_state)
                fast_state, slow_state = self.evolve_token(fast_state, slow_state, sig)

        if debug_track:
            print(
                f"[norms] last ||fast||={self._last_state_norm:.4f}  ||slow||={self._last_slow_norm:.4f}  "
                f"||combined||={self._last_combined_norm:.4f}"
            )
            self._print_attractor_diversity(top_k=5)
            self.track_attractors = False

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
    """Encode two prompts and report distance between final weighted combined states (path dependence)."""
    model.eval()
    with torch.inference_mode():
        f1, s1 = model.encode_prompt(prompt1)
        f2, s2 = model.encode_prompt(prompt2)
    c1 = model.combined_state(f1, s1)
    c2 = model.combined_state(f2, s2)
    dist = torch.linalg.vector_norm(c1 - c2).item()
    cos = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0), dim=1).item()
    print(
        f"[compare_prompts] L2(combined)={dist:.6f}  cosine={cos:.6f}  "
        f"||c1||={torch.linalg.vector_norm(c1).item():.4f}  ||c2||={torch.linalg.vector_norm(c2).item():.4f}"
    )


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
        model.reset_readout_trajectory()
        fast_state, slow_state = None, None
        for t_id in context:
            sig = model.get_signal(t_id, fast_state, slow_state)
            fast_state, slow_state = model.evolve_token(fast_state, slow_state, sig)
        logits = model.next_token_logits(fast_state, slow_state)
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
    last_epoch: int,
    last_mean_loss: float,
    last_train_ce: float,
    last_val_ce: float | None,
    last_n_windows: int,
    last_epoch_sec: float,
    train_sec_total: float,
    gen1: str,
    gen2: str,
    gen3: str,
) -> str:
    val_s = f"{last_val_ce:.4f}" if last_val_ce is not None else "n/a (no val)"
    return (
        f"--- Phase 0 baseline (copy into docs/BASELINE.md) ---\n"
        f"time_utc: {datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}\n"
        f"git: {_git_short_hash()}\n"
        f"corpus: {corpus_path}\n"
        f"seed: {seed}  val_fraction: {val_fraction}  epoch_copies: {epoch_copies}\n"
        f"window_size: {WINDOW_SIZE}  num_epochs: {NUM_EPOCHS}\n"
        f"last_epoch: {last_epoch}/{NUM_EPOCHS}  windows: {last_n_windows}  epoch_sec: {last_epoch_sec:.1f}\n"
        f"train_sec_total: {train_sec_total:.1f}\n"
        f"mean_loss (objective): {last_mean_loss:.4f}\n"
        f"train_CE: {last_train_ce:.4f}  val_CE: {val_s}\n"
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
    args = parser.parse_args()
    corpus_path = args.corpus if args.corpus is not None else DEFAULT_CORPUS_PATH
    random.seed(args.seed)

    print(f"Vocab size: {len(FULL_VOCAB)}", flush=True)
    model = TorchAttractorLanguageModel(FULL_VOCAB)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    vocab_set = set(model.vocab)

    sentences = load_corpus(corpus_path)
    print(f"Loaded corpus: {corpus_path}  ({len(sentences)} lines)", flush=True)
    corpus_coverage_report(sentences, vocab_set, WINDOW_SIZE)

    usable = sentences_with_training_windows(sentences, vocab_set, WINDOW_SIZE)
    if not usable:
        raise RuntimeError(
            "No corpus lines have enough in-vocabulary tokens to form a training window. "
            "Add text using words from the model vocab, or lower WINDOW_SIZE."
        )
    n_skip = len(sentences) - len(usable)
    if n_skip:
        print(
            f"Training/validation use only lines with ≥{WINDOW_SIZE + 1} in-vocab tokens "
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
    val_dataset = build_dataset_from_sentences(val_sents, model, WINDOW_SIZE)

    print(
        f"Pre-training ({NUM_EPOCHS} epochs, sliding window size={WINDOW_SIZE}, "
        f"epoch_copies={args.epoch_copies})...",
        flush=True,
    )
    t_train0 = time.perf_counter()
    last_mean_loss = 0.0
    last_train_ce = 0.0
    last_val_ce: float | None = None
    last_n_windows = 0
    last_epoch_sec = 0.0
    last_epoch_num = 0
    for epoch in range(NUM_EPOCHS):
        training_sentences = list(train_sents * args.epoch_copies)
        random.shuffle(training_sentences)
        dataset = []
        w2i = model._word_to_idx
        for sentence in training_sentences:
            words = [w for w in sentence.split() if w in w2i]
            if len(words) < WINDOW_SIZE + 1:
                continue
            ids = [w2i[w] for w in words]
            dataset.extend(build_sequence_dataset(ids, window_size=WINDOW_SIZE))
        random.shuffle(dataset)

        n = len(dataset)
        t_ep0 = time.perf_counter()
        print(f"  epoch {epoch + 1}/{NUM_EPOCHS}  |  {n} windows", flush=True)
        loss_sum = 0.0
        report_every = max(1, n // 10)

        for step, (context, target_id) in enumerate(dataset):
            # One reset per (context, target): evolve through full window without resetting mid-context.
            model.reset_readout_trajectory()
            fast_state, slow_state = None, None
            for t_id in context:
                sig = model.get_signal(t_id, fast_state, slow_state)
                fast_state, slow_state = model.evolve_token(fast_state, slow_state, sig)
            logits = model.next_token_logits(fast_state, slow_state)
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

        ep_sec = time.perf_counter() - t_ep0
        mean_loss = loss_sum / max(n, 1)
        # Same CE as val (no logit noise): comparable to val CE. mean_loss above is CE - ENTROPY_WEIGHT*H.
        train_ce = mean_cross_entropy_eval(model, dataset)
        val_msg = ""
        vce: float | None = None
        if val_dataset:
            vce = mean_cross_entropy_eval(model, val_dataset)
            val_msg = f"  |  val CE={vce:.4f}"
        print(
            f"  epoch {epoch + 1} done  |  {ep_sec:.1f}s  |  "
            f"mean loss={mean_loss:.4f}  |  train CE={train_ce:.4f}{val_msg}",
            flush=True,
        )
        last_mean_loss = mean_loss
        last_train_ce = train_ce
        last_val_ce = vce
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
        last_epoch=last_epoch_num,
        last_mean_loss=last_mean_loss,
        last_train_ce=last_train_ce,
        last_val_ce=last_val_ce,
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
