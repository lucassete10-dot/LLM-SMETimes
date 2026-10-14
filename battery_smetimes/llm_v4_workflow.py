"""Finite, single-factor follow-up to the preserved v3 experiment."""
from battery_smetimes.llm_v3_workflow import screen_pass, confirmation_pass, summarize_validation

SEEDS = (20260930, 20260931, 20260932)
RATE_KNOTS = [0., .05, .1, .2, .25, .35, .5, .75, 1., 1.5, 2., 2.5, 3., 3.5,
              4., 5., 6., 7., 8., 10., 12., 14., 16., 20., 24., 28., 32.]
OVERRIDES = {'K':{'rate_knots_horizon_100cycles':RATE_KNOTS},
             'L':{'drop_loss_weight':1., 'tail_loss_weight':1.5}}


def screen_winner(candidates, reference):
    base = summarize_validation(reference)
    eligible = [name for name in OVERRIDES if screen_pass(summarize_validation(candidates[name]), base)]
    return min(eligible, key=lambda name:(summarize_validation(candidates[name])['score'],
                                        list(OVERRIDES).index(name))) if eligible else None


def accepted_confirmation(candidates, references):
    if len(candidates)!=3 or len(references)!=3:
        raise ValueError('three matched seeds required')
    # Summaries alone cannot enforce exact physical membership.
    if any(c['policy']!=r['policy'] for c,r in zip(candidates, references)):
        raise ValueError('confirmation policies/membership differ')
    return confirmation_pass([summarize_validation(e) for e in candidates],
                             [summarize_validation(e) for e in references])
