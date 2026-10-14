"""Pretrained LLM SMETimes-style battery forecasting, not a tree proxy.

Reuse the existing audited history/scaler contract. Numeric history is patched,
statistics become text cards encoded by the SAME frozen language backbone,
adaptive fusion precedes a second backbone pass, and a deterministic head
regresses SOH at independent future-cycle queries. Dense low-rank experts
replace the historical ineffective L1 gate regularizer with a real balance loss.
Base LLM weights stay frozen; optional numeric-only LoRA adapts the final
attention layer while text cards deliberately retain base-checkpoint encoding.

This is a battery regression adaptation, not an exact reproduction of the
original reconstruction decoder or a claim of sparse expert computation.
"""

import numpy as np
from contextlib import contextmanager
import torch
from torch import nn
from battery_smetimes.physical_types import require_same_registered_family
from battery_smetimes.research_sequences import pack_contexts
from transformers import AutoConfig, AutoModel, AutoTokenizer


class NumericLoRALinear(nn.Module):
    """Low-rank update for the numeric backbone pass; pretrained base stays frozen."""
    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base,nn.Linear) or rank < 1:
            raise ValueError('numeric LoRA requires a linear projection and positive rank')
        self.base=base
        self.in_features,self.out_features=base.in_features,base.out_features
        self.enabled=True
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=np.sqrt(5))
        for p in self.base.parameters():p.requires_grad=False

    @property
    def weight(self):return self.base.weight

    def forward(self,x):
        output=self.base(x)
        if self.enabled:
            residual=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)
            output=output+residual.to(output.dtype)
        return output


def statistics_cards(cell, origin, patch_size=5, history_slots=100):
    take = cell["cycle"] <= origin
    cycles = cell["cycle"][take]
    # Prepared observations and numeric model inputs use float32. Canonicalize
    # before reductions so appending float64 future rows cannot change visible
    # text through dtype promotion and rounded statistic strings.
    raw = np.asarray(cell["raw"][take], dtype=np.float32)
    if len(raw) > history_slots or not len(raw) or np.isinf(raw).any():
        raise ValueError("invalid observed statistics-card input")
    # Only known feature-layout information is used here, not dataset/cell names
    # in the text, capacity, SOH, future signals or final life.
    domain = cell["metadata"]["dataset"]
    slices = ((slice(23, 28), slice(28, 33), slice(33, 38)) if domain == "XJTU"
              else (slice(0, 11), slice(11, 22), slice(22, 33)))
    if 'signal_schema' in cell['metadata']:
        slices=tuple(cell['metadata']['signal_schema']['text_channel_columns'][key] for key in ('Voltage','Current','Temperature'))
    cards = []
    for start in range(0, len(raw), patch_size):
        end = min(start + patch_size, len(raw))
        items = [f"Observed cycles {int(cycles[start])} to {int(cycles[end-1])}; {end-start} observations."]
        for channel, columns, unit in zip(("Voltage", "Current", "Temperature"), slices, ("V", "A", "C")):
            values = raw[start:end, columns]
            finite = values[np.isfinite(values)]
            if not len(finite):
                items.append(f"{channel} unavailable.")
            else:
                row_mean = np.array([r[np.isfinite(r)].mean() if np.isfinite(r).any() else np.nan for r in values])
                valid = np.flatnonzero(np.isfinite(row_mean))
                change = float(row_mean[valid[-1]] - row_mean[valid[0]]) if len(valid) > 1 else 0.
                items.append(f"{channel} mean {finite.mean():.4g} {unit}, standard deviation {finite.std():.4g}, observed change {change:.4g}.")
        cards.append(" ".join(items))
    return cards


class FrozenSMETimesForecaster(nn.Module):
    def __init__(self, checkpoint, history_width, seed=20260930, pretrained=True,
                 use_statistics=True, experts=4, rank=64, patch_size=5,
                 dropout=.1, max_text_tokens=128, device="cuda", dtype=torch.bfloat16,
                 normalize_router=False, output_head="point", pooling="mean", numeric_dynamics=False,
                 adaptive_fusion=False, summary_width=0, residual_experts=False, history_slots=100,
                 initial_knee_bias=1., numeric_lora_rank=0):
        super().__init__()
        self.pretrained, self.use_statistics = pretrained, use_statistics
        self.num_experts, self.patch_size = experts, patch_size
        self.max_text_tokens = max_text_tokens
        self.normalize_router = normalize_router
        if pooling not in ("mean", "attention", "dual_scale", "tokens"):
            raise ValueError("unknown LLM history pooling")
        self.pooling = pooling
        self.numeric_dynamics = numeric_dynamics
        self.adaptive_fusion = adaptive_fusion
        self.summary_width = summary_width
        self.residual_experts = residual_experts
        if history_slots < 1 or history_slots % patch_size:
            raise ValueError('history capacity must be a positive multiple of patch size')
        self.history_slots = history_slots
        if adaptive_fusion and not use_statistics:
            raise ValueError("adaptive fusion requires the statistics branch")
        if output_head not in ("point", "integrated_rate", "basis_rate"):
            raise ValueError("unknown LLM trajectory head")
        self.output_head = output_head
        if not np.isfinite(initial_knee_bias) or (output_head != 'integrated_rate' and initial_knee_bias != 1.):
            raise ValueError('initial knee bias requires a finite integrated-rate initialization')
        self.backbone_dtype = dtype
        if not isinstance(numeric_lora_rank,int) or numeric_lora_rank < 0 or (numeric_lora_rank and not pretrained):
            raise ValueError('numeric LoRA requires nonnegative integer rank and pretrained backbone')
        self.numeric_lora_rank=numeric_lora_rank
        torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        config.use_cache = False
        if pretrained:
            self.backbone = AutoModel.from_pretrained(checkpoint, config=config,
                local_files_only=True, dtype=dtype).to(device)
        else:
            # E2: no checkpoint weights, INCLUDING the text-card encoding path.
            self.backbone = AutoModel.from_config(config, dtype=dtype).to(device=device, dtype=dtype)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dimension = config.hidden_size
        # Adapter/head initialization is identical in pretrained and random
        # versions, despite differing RNG consumption while loading backbones.
        torch.manual_seed(seed + 100)
        self.numeric_projection = nn.Sequential(nn.Linear(patch_size * history_width, dimension), nn.LayerNorm(dimension))
        self.fusion_logit = nn.Parameter(torch.zeros(())) if use_statistics else None
        self.gate = nn.Linear(dimension, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dimension, rank, bias=False), nn.GELU(),
                                      nn.Linear(rank, dimension, bias=False)) for _ in range(experts)])
        self.head = nn.Sequential(nn.Linear(dimension + 12, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        if output_head in ("integrated_rate", "basis_rate"):
            # Only LLM-derived encoding + known conditions generate trajectory
            # parameters. Time has a fixed physical scale, not a cell endpoint.
            output_width = 7 if output_head == 'integrated_rate' else 15
            self.head = nn.Sequential(nn.Linear(dimension + 9, 128), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(128, output_width))
            with torch.no_grad():
                bias = ([0.,-2.,-2.,initial_knee_bias,-1.,0.,0.] if output_head == 'integrated_rate'
                        else [0.,-2.] + [-4.] * 11 + [0.,0.])
                self.head[-1].bias.copy_(torch.tensor(bias))
            self.register_buffer("query_mean",torch.zeros(3))
            self.register_buffer("query_scale",torch.ones(3))
        self.balance_loss = None
        if pooling == "attention":
            # Start as the matched mean-pooling control. Scores use actual LLM
            # hidden states; masked patches never contribute to the summary.
            self.pool_score = nn.Linear(dimension, 1, bias=False)
            nn.init.zeros_(self.pool_score.weight)
        if numeric_dynamics:
            if history_width % 2 != 1:
                raise ValueError("dynamics require numeric + availability + source-cycle schema")
            self.dynamics_projection = nn.Linear(patch_size * history_width, dimension, bias=False)
            nn.init.zeros_(self.dynamics_projection.weight)
        if adaptive_fusion:
            # Preserve the control's initial fusion and shared parameters.
            # The patch-specific gate precedes the actual pretrained LLM.
            self.fusion_gate = nn.Linear(2 * dimension, 1, bias=False)
            nn.init.zeros_(self.fusion_gate.weight)
        if pooling == "dual_scale":
            # Keep chronological early/late contrast of actual LLM expert
            # outputs. Zero residual at initialization matches mean control.
            self.temporal_contrast = nn.Sequential(nn.Linear(dimension,rank,bias=False),
                nn.GELU(),nn.Linear(rank,dimension,bias=False))
            nn.init.zeros_(self.temporal_contrast[-1].weight)
        if summary_width:
            self.summary_projection = nn.Linear(summary_width,dimension,bias=False)
            nn.init.zeros_(self.summary_projection.weight)
        if numeric_lora_rank:
            # Install after peripheral initialization; a separate CPU RNG stream
            # preserves the matched control's adapter parameters and dropout RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+202)
                attention=self.backbone.layers[-1].self_attn
                for name in ('q_proj','v_proj'):
                    setattr(attention,name,NumericLoRALinear(getattr(attention,name),numeric_lora_rank))
        self.to(device)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen inference behavior in both branches
        return self

    @contextmanager
    def base_text_encoding(self):
        # Text cards deliberately use the unchanged base checkpoint. Their
        # detached cache is valid even as numeric-only LoRA weights are updated.
        modules=[m for m in self.backbone.modules() if isinstance(m,NumericLoRALinear)]
        before=[m.enabled for m in modules]
        try:
            for m in modules:m.enabled=False
            yield
        finally:
            for m,enabled in zip(modules,before):m.enabled=enabled

    def encode_cards(self, card_groups):
        device = self.numeric_projection[0].weight.device
        max_patches = self.history_slots // self.patch_size
        output = torch.zeros(len(card_groups), max_patches, self.backbone.config.hidden_size, device=device)
        mapping, flat = [], []
        for i, cards in enumerate(card_groups):
            if len(cards) > max_patches:
                raise ValueError("too many observed statistics cards")
            for j, card in enumerate(cards):
                mapping.append((i, j))
                flat.append(card)
        if not self.use_statistics:
            return output
        with torch.no_grad(), self.base_text_encoding():
            for start in range(0, len(flat), 4):
                tokens = self.tokenizer(flat[start:start + 4], padding=True, truncation=True,
                    max_length=self.max_text_tokens, return_tensors="pt").to(device)
                hidden = self.backbone(**tokens, use_cache=False).last_hidden_state.float()
                mask = tokens["attention_mask"].to(hidden.dtype)
                pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                for k, item in enumerate(pooled):
                    i, j = mapping[start + k]
                    output[i, j] = item
        return output.detach()

    def encode(self, history, length, card_embeddings, statistics=None):
        if history.shape[1] != self.history_slots:
            raise ValueError("source-budget history/patch schema mismatch")
        if torch.any(length <= 0) or torch.any(length > self.history_slots):
            raise ValueError('invalid observed history length')
        observed = torch.arange(self.history_slots, device=history.device)[None] < length[:, None]
        patches = (history * observed[..., None]).reshape(len(history), self.history_slots // self.patch_size, -1)
        # Additional storage padding must not change the bfloat16 backbone's
        # kernel shapes for an otherwise identical <=100-observation input.
        # Keep the original 20-patch computation for backward replay; larger
        # inputs retain every observed patch, including a partial last patch.
        active_patches = min(patches.shape[1], max(100 // self.patch_size,
            int(((length.max() + self.patch_size - 1) // self.patch_size).item())))
        patches = patches[:, :active_patches]
        card_embeddings = card_embeddings[:, :active_patches]
        numeric = self.numeric_projection(patches)
        if self.summary_width:
            if statistics is None or statistics.shape != (len(history),self.summary_width):
                raise ValueError("label-free historical summary missing or schema mismatch")
            numeric = numeric + self.summary_projection(statistics)[:,None,:]
        if self.numeric_dynamics:
            dynamics = historical_rates(history,length).reshape(len(history),self.history_slots//self.patch_size,-1)
            dynamics = dynamics[:, :active_patches]
            # An explicit trainable input adapter, not a prediction bypass.
            # This residual goes through fusion and the pretrained LLM.
            numeric = numeric + self.dynamics_projection(dynamics)
        if self.use_statistics:
            logit = self.fusion_logit
            if self.adaptive_fusion:
                pair = torch.cat((torch.nn.functional.layer_norm(numeric, (numeric.shape[-1],)),
                    torch.nn.functional.layer_norm(card_embeddings, (card_embeddings.shape[-1],))), -1)
                logit = logit + self.fusion_gate(pair)
            alpha = torch.sigmoid(logit)
            embedded = alpha * numeric + (1 - alpha) * card_embeddings
        else:
            embedded = numeric
        patch_count = (length + self.patch_size - 1) // self.patch_size
        valid = torch.arange(embedded.shape[1], device=embedded.device)[None] < patch_count[:, None]
        hidden = self.backbone(inputs_embeds=embedded.to(self.backbone_dtype),
            attention_mask=valid.long(), use_cache=False).last_hidden_state.float()
        # Optional single-factor control: stabilize router logit scale without
        # changing expert inputs, pooling, output head or pretrained LLM path.
        router_input = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],)) if self.normalize_router else hidden
        gates = torch.softmax(self.gate(router_input), -1)
        expert_output = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        output = (gates[..., None] * expert_output).sum(-2)
        if self.residual_experts:
            # Preserve the actual pretrained LLM representation while experts
            # learn a low-rank correction, rather than replacing it entirely.
            # Parameter-free normalization controls decoder input scale. This
            # is an LLM-output residual, never a raw-feature prediction bypass.
            output = torch.nn.functional.layer_norm(hidden + output, (hidden.shape[-1],))
        usage = (gates * valid[..., None]).sum((0, 1)) / valid.sum()
        # Non-constant differentiable load balance. The trainer must add it
        # to its actual objective (no multiply-by-zero pseudo-regularization).
        self.balance_loss = self.num_experts * torch.sum((usage - 1 / self.num_experts) ** 2)
        if self.pooling == "tokens":
            # The repair decoder reads chronological LLM/expert outputs. Keep
            # the validity mask instead of erasing their temporal differences.
            return output, valid
        if self.pooling == "attention":
            scores = self.pool_score(torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))).squeeze(-1)
            weights = torch.softmax(scores.masked_fill(~valid, float("-inf")), dim=1)
            return (output * weights[..., None]).sum(1)
        mean = (output * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        if self.pooling == "dual_scale":
            position=torch.arange(output.shape[1],device=output.device)[None]
            early=valid & (position < ((patch_count+1)//2)[:,None])
            late=valid & ~early
            first=(output*early[...,None]).sum(1)/early.sum(1,keepdim=True).clamp_min(1)
            last=(output*late[...,None]).sum(1)/late.sum(1,keepdim=True).clamp_min(1)
            contrast=(last-first)*(late.any(1)[:,None])
            return mean+self.temporal_contrast(contrast)
        return mean

    def decode(self, encoding, conditions, queries, return_parameters=False):
        if self.output_head in ("integrated_rate", "basis_rate"):
            parameters = self.head(torch.cat((encoding,conditions),-1))
            # Undo the TRAIN-fitted standardizer: feature 1 is horizon/100.
            physical = queries*self.query_scale+self.query_mean
            time = physical[...,1].clamp_min(0)
            prediction = (integrated_soh(parameters,time) if self.output_head == 'integrated_rate'
                          else basis_soh(parameters,time))
            return (prediction, parameters) if return_parameters else prediction
        if return_parameters:
            raise ValueError('curve parameters require integrated-rate head')
        value = torch.cat((encoding[:, None].expand(-1, queries.shape[1], -1),
                           conditions[:, None].expand(-1, queries.shape[1], -1), queries), -1)
        return .9 + .1 * self.head(value).squeeze(-1)

    def bind_query_scaler(self, scaler):
        if self.output_head in ("integrated_rate", "basis_rate"):
            with torch.no_grad():
                self.query_mean.copy_(torch.as_tensor(scaler.mean_,device=self.query_mean.device))
                self.query_scale.copy_(torch.as_tensor(scaler.scale_,device=self.query_scale.device))

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen": sum(p.numel() for p in self.parameters() if not p.requires_grad)}


def historical_rates(history,length):
    """Label-free rates per actual source cycle, with pair-availability masks.

    Contract: standardized measurements, finite-availability indicators, and
    source-cycle/100. Missing measurements, first observations and padding
    have no fabricated rate; gaps retain their actual duration.
    """
    width=(history.shape[-1]-1)//2
    if 2*width+1 != history.shape[-1]:
        raise ValueError('invalid historical rate schema')
    observed=torch.arange(history.shape[1],device=history.device)[None]<length[:,None]
    pair_observed=observed[:,1:] & observed[:,:-1]
    gap=100*(history[:,1:,-1]-history[:,:-1,-1])
    if torch.any(pair_observed & (gap<=0)):
        raise ValueError('observed source cycles must increase')
    availability=history[...,width:2*width]>.5
    pair=pair_observed[...,None] & availability[:,1:] & availability[:,:-1]
    delta=(history[:,1:,:width]-history[:,:-1,:width])/gap.clamp_min(1e-6)[...,None]
    out=torch.zeros_like(history)
    out[:,1:,:width]=torch.where(pair,delta,torch.zeros_like(delta))
    out[:,1:,width:2*width]=pair.to(history.dtype)
    out[:,1:,-1]=torch.where(pair_observed,gap/100,torch.zeros_like(gap))
    return out


def integrated_soh(parameters,time):
    """Analytic integral of smooth degradation rate plus limited recovery.

    Time is horizon / 100 source cycles. No interpolation, requested endpoint,
    target label, floor or padding extension enters this deterministic head.
    """
    p=parameters[:,None,:]
    level=1.+.1*p[...,0]
    base=.01*torch.nn.functional.softplus(p[...,1])
    acceleration=.02*torch.nn.functional.softplus(p[...,2])
    knee=5.*torch.nn.functional.softplus(p[...,3])
    width=.2+torch.nn.functional.softplus(p[...,4])
    integral=width*(torch.nn.functional.softplus((time-knee)/width)-
                    torch.nn.functional.softplus(-knee/width))
    recovery=.005*torch.tanh(p[...,5])*(-torch.expm1(-time/(.2+torch.nn.functional.softplus(p[...,6]))))
    return level-base*time-acceleration*integral+recovery


def basis_soh(parameters, time):
    """Smooth integrated rates on fixed physical horizons, learned from LLM.

    Eleven overlapping transitions supply gradients at early and late horizons
    without a single learned knee having to move across a long low-gradient
    region. Coefficients are cell-conditioned, knots never use record endpoints.
    No floors or extrapolation by repeated last values.
    """
    centres = time.new_tensor([.25,.5,1.,2.,3.,4.,6.,8.,12.,16.,24.])
    width = .2
    integral = width * (torch.nn.functional.softplus((time[...,None]-centres)/width)
                        - torch.nn.functional.softplus(-centres/width))
    p = parameters[:,None,:]
    coefficients = .02 * torch.nn.functional.softplus(p[...,2:13])
    recovery = .005 * torch.tanh(p[...,13]) * (-torch.expm1(-time/(.2+torch.nn.functional.softplus(p[...,14]))))
    return 1.+.1*p[...,0] - .01*torch.nn.functional.softplus(p[...,1])*time - (coefficients*integral).sum(-1) + recovery


def adapter_state(network):
    return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()
            if not k.startswith("backbone.") or k.endswith(('.lora_A','.lora_B'))}


def load_adapter(network, state):
    missing, unexpected = network.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("backbone.") or key.endswith(('.lora_A','.lora_B')) for key in missing):
        raise ValueError("adapter checkpoint schema mismatch")


def adapter_optimizer(network,learning_rate,weight_decay,numeric_lora_lr_scale=1.):
    """Optionally limit pretrained representation updates, retaining peripheral LR."""
    if not np.isfinite(numeric_lora_lr_scale) or numeric_lora_lr_scale<=0:
        raise ValueError('LoRA learning-rate multiplier must be finite and positive')
    if not network.numeric_lora_rank and numeric_lora_lr_scale!=1.:
        raise ValueError('LoRA LR multiplier requires numeric LoRA')
    trainable=[(k,p) for k,p in network.named_parameters() if p.requires_grad]
    if numeric_lora_lr_scale==1.:
        return torch.optim.AdamW([p for _,p in trainable],lr=learning_rate,weight_decay=weight_decay)
    lora=[p for k,p in trainable if k.startswith('backbone.')]
    peripheral=[p for k,p in trainable if not k.startswith('backbone.')]
    return torch.optim.AdamW([{'params':peripheral,'lr':learning_rate},
        {'params':lora,'lr':learning_rate*numeric_lora_lr_scale}],weight_decay=weight_decay)


def cards_embedding(bundle, cell, origin):
    require_same_registered_family(cell["physical_family"], [bundle["physical_family"]], bundle["registry"])
    cards = statistics_cards(cell, origin, bundle["network"].patch_size, bundle['network'].history_slots)
    key = (cell["physical_family"], int(origin), tuple(cards))
    cache = bundle.setdefault("card_cache", {})
    if key not in cache:
        # This cache is bound to ONE fixed backbone/seed, never shared between
        # E1/E2 or candidates. Only visible V/I/T is encoded, without fitting.
        cache[key] = bundle["network"].encode_cards([cards])[0].cpu()
    return cache[key]


def attach_cards(bundle, contexts, cells):
    for row in contexts:
        row["card_embeddings"] = cards_embedding(bundle, cells[row["cell_id"]], row["origin"])


def fit_llm_epoch(network, contexts, optimizer, seed, epoch, batch_size, device, balance_coefficient=.01):
    network.train()
    order = np.random.default_rng(seed + epoch).permutation(len(contexts))
    sum_loss, sum_weight, balance_total, steps = 0., 0., 0., 0
    for start in range(0, len(order), batch_size):
        rows = [contexts[i] for i in order[start:start + batch_size]]
        batch = pack_contexts(rows, device)
        embeddings = torch.stack([r["card_embeddings"] for r in rows]).to(device)
        encoded = network.encode(batch["history"], batch["length"], embeddings,
            batch['statistics'] if network.summary_width else None)
        prediction = network.decode(encoded, batch["conditions"], batch["query"])
        individual = (((prediction - batch["truth"]) ** 2) * batch["valid"]).sum(1) / batch["valid"].sum(1)
        loss = (individual * batch["weight"]).mean() + balance_coefficient * network.balance_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite frozen-LLM training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_((p for p in network.parameters() if p.requires_grad), 1.)
        optimizer.step()
        sum_loss += float((individual.detach() * batch["weight"]).sum())
        sum_weight += float(batch["weight"].sum())
        balance_total += float(network.balance_loss.detach())
        steps += 1
    network.last_epoch_diagnostics = {"mean_balance_loss": balance_total / steps,
                                     "balance_coefficient": balance_coefficient}
    return sum_loss / sum_weight


def predict_llm(bundle, cell, origin, future_cycles):
    require_same_registered_family(cell["physical_family"], [bundle["physical_family"]], bundle["registry"])
    network, processor, device = bundle["network"], bundle["processor"], bundle["device"]
    values = processor.model_input(cell, origin)
    query = processor.future_query(origin, future_cycles)
    embeddings = cards_embedding(bundle, cell, origin)[None].to(device)
    network.eval()
    with torch.inference_mode():
        encoded = network.encode(torch.as_tensor(values["history"][None], device=device),
            torch.as_tensor([values["length"]], device=device), embeddings,
            torch.as_tensor(values['statistics'][None],device=device) if network.summary_width else None)
        conditions = torch.as_tensor(values["conditions"][None], device=device)
        output = [network.decode(encoded, conditions, torch.as_tensor(part[None], device=device)).cpu().numpy()[0]
                  for part in np.array_split(query, max(1, int(np.ceil(len(query) / 1024))))]
    prediction = np.concatenate(output)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite frozen-LLM prediction")
    return prediction
