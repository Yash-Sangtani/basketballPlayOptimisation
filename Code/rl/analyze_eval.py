"""Print the metric table of league_eval 'gain' json files and the pre-registered guard ratios."""
import json, sys
keys = ['n_poss', 'ppp', 'fga_per_poss', 'fg3a_per_poss', 'tov_per_poss', 'stl_per_poss', 'fta_per_poss', 'orb_per_poss', 'rim_share', 'p_make_mean', 'def1_mean']
for f in sys.argv[1:]:
    d = json.load(open(f))
    print(f, f"(n={d['n']} chains x {d['blocks'] * 0.4:.0f} s)")
    print('%-8s' % '', *['%9s' % k[:9] for k in keys], '%9s' % 'pts/100s')
    secs = d['n'] * d['blocks'] * 0.4 / 100.0
    for n in ('base', 'ref', 'agent'):
        if n in d:
            m = d[n]
            print('%-8s' % n, *['%9.3f' % m[k] for k in keys], '%9.1f' % (m['ppp'] * m['n_poss'] / secs / 2))
    if 'agent' in d:
        b, a = d['base'], d['agent']
        print('agent / imitation:', {k[:6]: round(a[k] / b[k], 2) for k in ('tov_per_poss', 'stl_per_poss', 'fta_per_poss', 'fg3a_per_poss', 'fga_per_poss', 'n_poss')})
        print(f"ppp diff vs imitation {d['ppp_diff']:+.3f} {[round(x, 3) for x in d['ppp_diff_ci95']]}", end='')
        if 'agent_minus_ref' in d:
            print(f"; minus zero-residual prior {d['agent_minus_ref']:+.3f} {[round(x, 3) for x in d['agent_minus_ref_ci95']]}  (prior vs imitation {d['ref_diff_vs_imitation']:+.3f})")
        else:
            print()
        # decomposition of the ppp change (rough): turnover, free throw and field goal quality terms
        dppp = a['ppp'] - b['ppp']
        d_tov = -(a['tov_per_poss'] - b['tov_per_poss']) * 0 + (-(a['tov_per_poss'] - b['tov_per_poss']) * b['ppp'])
        d_fta = (a['fta_per_poss'] - b['fta_per_poss']) * 0.76
        fg_pts = lambda m: m['fga_per_poss'] * m['p_make_mean'] * (2 + m['fg3a_per_poss'] / max(m['fga_per_poss'], 1e-6) * 0.9)
        print(f"   rough split of the {dppp:+.3f} change: turnovers {d_tov * -1:+.3f}, free throws {d_fta:+.3f}, field goals {fg_pts(a) - fg_pts(b):+.3f}")
