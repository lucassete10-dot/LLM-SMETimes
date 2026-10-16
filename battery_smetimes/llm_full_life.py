"""Full-life TRAIN windows; causal prefix-only inference at any source origin."""
import json
from pathlib import Path

import numpy as np

from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.research_llm import cards_embedding, predict_llm
from battery_smetimes.research_sequences import HistoryPreprocessor, TrainingScale
from battery_smetimes.research_sequences import pack_contexts
from battery_smetimes.forecast import query_matrix
from scripts.audit_llm_curve_repair import load_frozen, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_full_life(index_path):
    index_path = Path(index_path)
    index = read(index_path)
    if not index['complete']:
        raise ValueError('Full-life extraction incomplete')
    _, cells, plan, registry, _ = load_frozen(index['base_settings'])
    if index['train_ids']!=plan['train'] or index['physical_plan']!=plan:
        raise ValueError('Physical split drift')
    if {r['cell_id'] for r in index['cells']}!=set(plan['train']):
        raise ValueError('Full TRAIN coverage missing')
    for record in index['cells']:
        if fingerprint(record['path'])!=record['sha256']:
            raise ValueError('Full-life signals changed')
        cell = cells[record['cell_id']]
        with np.load(record['path'], allow_pickle=False) as data:
            cell['cycle'], cell['raw'] = data['cycle'].copy(), data['raw'].copy()
    return index, cells, plan, registry


def visible_window(cell, origin, slots=100):
    # No label, remaining life, future signal, or end-of-life access.
    indices = np.flatnonzero(np.asarray(cell['cycle'])<=origin)[-slots:]
    return {**cell, 'cycle':np.asarray(cell['cycle'])[indices],
            'raw':np.asarray(cell['raw'])[indices]}


def training_origins(cell, step=100):
    # Grid is in actual source cycles, never normalized by the battery lifetime.
    last = int(np.floor(float(cell['cycle'][-1])/step)*step)
    candidates = list(range(100, max(100,last)+1, step))
    result, previous_count = [], -1
    for origin in candidates:
        count = int(np.sum(cell['cycle']<=origin))
        future = np.isfinite(cell['soh']) & (cell['soh']>0) & (cell['label_cycle']>origin)
        if count==previous_count or future.sum()<2:
            continue
        try:
            signal_prefix(visible_window(cell,origin),origin)
        except ValueError as error:
            if 'insufficient' in str(error) or 'not established' in str(error):
                continue
            raise
        result.append(origin)
        previous_count = count
    if 100 not in result:
        raise ValueError('Every TRAIN cell must retain its origin100 task')
    return result


class FullLifePreprocessor(HistoryPreprocessor):
    def fit(self, cells, ids, step=100):
        self.history_slots = 100
        self.train_ids = list(ids)
        self.origins = [100]  # public evaluation contract, not the training grid
        self.training_origins_by_cell = {cid:training_origins(cells[cid],step) for cid in ids}
        self.raw_counts = {cid:len(cells[cid]['cycle']) for cid in ids}
        self.raw_width = cells[ids[0]]['raw'].shape[1]
        raw = np.concatenate([cells[cid]['raw'] for cid in ids])
        weight = np.concatenate([np.full(self.raw_counts[cid],1/self.raw_counts[cid]) for cid in ids])
        self.raw_scale = TrainingScale().fit(raw,weight)
        statistics, weights, queries = [], [], []
        for cid in ids:
            cell = cells[cid]
            origins = self.training_origins_by_cell[cid]
            for origin in origins:
                statistics.append(signal_prefix(visible_window(cell,origin),origin))
                weights.append(1/len(origins))
                x = cell['label_cycle'][np.isfinite(cell['soh']) & (cell['soh']>0) & (cell['label_cycle']>origin)]
                queries.append(query_matrix(np.zeros(1),origin,x)[:,-3:])
        self.stats_width = len(statistics[0])
        self.stats_scale = TrainingScale().fit(np.asarray(statistics), np.asarray(weights))
        self.query_scale = TrainingScale().fit(np.concatenate(queries))
        return self

    def model_input(self, cell, origin):
        return super().model_input(visible_window(cell,origin,self.history_slots),origin)


def full_life_contexts(cells, processor):
    contexts = []
    for cid in processor.train_ids:
        cell = cells[cid]
        origins = processor.training_origins_by_cell[cid]
        for origin in origins:
            take = np.isfinite(cell['soh']) & (cell['soh']>0) & (cell['label_cycle']>origin)
            cycle, truth = cell['label_cycle'][take], cell['soh'][take]
            context = dict(**processor.model_input(cell,origin), cell_id=cid,origin=origin,
                           query=processor.future_query(origin,cycle), truth=truth.astype(np.float32),
                           valid=np.ones(len(truth),np.float32), target_cycles=cycle,
                           weight=(1. if len(origins)==1 else .5 if origin==100 else .5/(len(origins)-1)))
            contexts.append(context)
    normalization = len(contexts)/len(processor.train_ids)
    for context in contexts:
        context['weight'] *= normalization
    return contexts


def attach_full_life_cards(bundle, contexts, cells):
    for context in contexts:
        context['card_embeddings'] = cards_embedding(bundle,visible_window(cells[context['cell_id']],context['origin']),context['origin'])


def pack_full_life(contexts, device):
    maximum = max(len(row['truth']) for row in contexts)
    padded = []
    for row in contexts:
        count = len(row['truth'])
        copy = dict(row)
        copy['query'] = np.pad(row['query'],((0,maximum-count),(0,0)))
        for name in ('truth','valid'):
            copy[name] = np.pad(row[name],(0,maximum-count))
        padded.append(copy)
    return pack_contexts(padded,device)


def predict_full_life(bundle, cell, origin, future_cycles):
    return predict_llm(bundle,visible_window(cell,origin),origin,future_cycles)

