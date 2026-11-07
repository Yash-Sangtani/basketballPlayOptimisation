"""Stage 6F-C: do the defence agents choose schemes real defences rarely choose in those states? (pre-registered)
    python rl/audit_support.py --out ../reports/rl-stage6f/support.json"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, pickle
import league as L
import league_eval as LE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=100); ap.add_argument('--blocks', type=int, default=450); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl'); ap.add_argument('--split', default='holdout')
    ap.add_argument('--policies', default='def_seed0=../reports/rl-stage6c/run3/ckpt_0060.pkl,def_seed3=../reports/rl-stage6c/run4/ckpt_0060.pkl,def_seed5=../reports/rl-stage6c/run5/ckpt_0060.pkl,loose_league=../reports/rl-stage6d/run3/ckpt_0100.pkl,tight_league=../reports/rl-stage6d/run_tight/ckpt_0050.pkl,prior=../reports/rl-stage6c/prior_def_ckpt.pkl')
    a = ap.parse_args()
    st, G, table, starts, gi = LE.setup(a)
    res = {}
    for item in a.policies.split(','):
        nm, path = item.split('=')
        ck = pickle.load(open(path, 'rb'))
        o, d = L.make_agents(st, ck, learn=True, seed=3)           # learn=True only so that the decisions are buffered; nothing is updated
        A = (o if ck.get('off') is not None else None, d)
        eng = L.run_match(st, G, table, starts, 9, a.blocks, A=A, B=A)
        pr, lp = [], []
        for b in d.buf:
            pr.append(np.exp(b['logp0'][np.arange(len(b['a'])), b['a']]))
        pr = np.concatenate(pr)
        usage = np.bincount(np.concatenate([b['a'] for b in d.buf]), minlength=d.K) / len(pr)
        res[nm] = dict(n_decisions=int(len(pr)), mean_prior_prob_of_chosen=float(pr.mean()), share_below_5pct=float((pr < 0.05).mean()), share_below_10pct=float((pr < 0.10).mean()), usage=usage.round(3).tolist())
        print(f"{nm:14s} decisions {len(pr)}  mean prior prob of the chosen scheme {pr.mean():.3f}  share < 5% {(pr < 0.05).mean():.3f}  share < 10% {(pr < 0.10).mean():.3f}", flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
