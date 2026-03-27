import math

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

_corpus_words = set()
for line in _CORPUS_LINES:
    _corpus_words.update(line.lower().split())

_seen: set[str] = set()
BASE_VOCAB: list[str] = []
for w in sorted(_corpus_words):
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
print(f"Vocab size: {len(FULL_VOCAB)}")

# ==================== FIXED TORCH MODEL (shape bugs corrected) ====================
class TorchAttractorLanguageModel(nn.Module):
    def __init__(self, vocab, state_dim=512):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.state_dim = state_dim
        self.dynamics = SimpleAttractorDynamics(state_dim)
        self.embedder = nn.Embedding(self.vocab_size, state_dim)
        self.norm = nn.LayerNorm(state_dim, elementwise_affine=False)
        # Unconstrained raw; effective temperature = softplus(raw) > 0 (learnable temp can hit 0 otherwise -> inf logits).
        t0 = 0.12
        self.temperature_raw = nn.Parameter(torch.tensor(math.log(math.exp(t0) - 1.0)))

    def effective_temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw).clamp(min=1e-6)

    def get_signal(self, token_id: int) -> torch.Tensor:
        emb = self.embedder(torch.tensor([token_id]))
        emb = self.norm(emb)
        n = torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        return (emb / n).squeeze(0)

    def converge(self, state, signal, num_steps=30):
        s = state.clone() if state is not None else torch.zeros(self.state_dim)
        for _ in range(num_steps):
            s = self.dynamics(s, signal)
        return s

    def next_token_logits(self, state):
        all_signals = torch.stack([self.get_signal(i) for i in range(self.vocab_size)])
        dists = torch.cdist(state.unsqueeze(0), all_signals).squeeze(0)
        logits = -dists / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        return logits

    def generate(self, prompt: str, max_tokens=40):
        tokens = [w for w in prompt.lower().split() if w in self.vocab] or ["the"]
        input_ids = [self.vocab.index(w) for w in tokens]

        state = None
        with torch.no_grad():
            for tid in input_ids:
                sig = self.get_signal(tid)
                state = self.converge(state, sig)

            generated = tokens[:]
            for _ in range(max_tokens):
                logits = self.next_token_logits(state)
                logits = logits - logits.max()
                probs = F.softmax(logits, dim=-1)
                if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                    probs = torch.ones_like(probs) / self.vocab_size
                next_id = torch.multinomial(probs, 1).item()
                next_word = self.vocab[next_id]
                generated.append(next_word)
                sig = self.get_signal(next_id)
                state = self.converge(state, sig)
        return " ".join(generated)


class SimpleAttractorDynamics(nn.Module):
    def __init__(self, dim=512, dt=0.04, cubic_scale=0.008):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cubic_scale = cubic_scale
        self.diffusion = nn.Parameter(make_diffusion_matrix(dim))

    def forward(self, state, signal):
        return step_state(state, self.diffusion, signal, self.dt, self.cubic_scale)


def make_diffusion_matrix(dim):
    torch.manual_seed(42)
    q = torch.linalg.qr(torch.randn(dim, dim))[0]
    u = torch.rand(dim)
    eigenvalues = -0.2 - (0.05 + 0.3 * u)
    return (q * eigenvalues) @ q.T


def step_state(state, diffusion, applied_signal, dt, cubic_scale):
    c = state - state.mean()
    nonlinear = cubic_scale * (c ** 3)
    drift = state @ diffusion.T + nonlinear + applied_signal
    s = state + dt * drift
    return torch.clamp(torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), -80.0, 80.0)


# ==================== RUN (pre-training + generation) ====================
model = TorchAttractorLanguageModel(FULL_VOCAB)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

corpus = [
    "the problem appears but the solution exists because the reason is clear therefore the system stays stable",
    "mind understands the cause and the effect creates a stable system",
    "the quick brown fox jumps over the lazy dog and then the pattern flows into the future",
] * 20

print("Pre-training (3 epochs)...")
for epoch in range(3):
    for sentence in corpus:
        words = [w for w in sentence.split() if w in model.vocab]
        if len(words) < 3:
            continue
        ids = [model.vocab.index(w) for w in words]
        state = None
        loss = 0.0
        for i in range(len(ids) - 1):
            sig = model.get_signal(ids[i])
            state = model.converge(state, sig)
            logits = model.next_token_logits(state)
            target = torch.tensor([ids[i + 1]])
            loss += F.cross_entropy(logits.unsqueeze(0), target)
        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

print("Pre-training done.")

print("\nPrompt 1:")
print(model.generate("the quick brown fox jumps over the lazy dog and then what happens in the system of mind and reason"))
print("\nPrompt 2:")
print(model.generate("mind reason cause effect system"))
