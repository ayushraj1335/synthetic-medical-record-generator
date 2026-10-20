"""Opacus accounting helpers. Reports are scoped to the DP-SGD mechanism."""
def privacy_report(privacy_engine, delta, status='completed', preprocessing_private=False):
    if privacy_engine is None:
        return {'status': status, 'dp_enabled': False, 'epsilon': None, 'delta': delta,
                'guarantee': 'No differential privacy guarantee: training was non-private.'}
    try:
        eps = float(privacy_engine.get_epsilon(delta=delta))
        guarantee = ('End-to-end claim unavailable: preprocessing/category vocabulary was fitted on private data.'
                     if not preprocessing_private else 'DP-SGD guarantee under Poisson sampling and add/remove record adjacency.')
        history = getattr(getattr(privacy_engine, 'accountant', None), 'history', None)
        return {'status': status, 'dp_enabled': True, 'epsilon': eps, 'delta': delta,
                'guarantee': guarantee, 'accountant': 'Opacus RDP accountant',
                'accounting_history_noise_sample_rate_steps': [list(item) for item in history] if history is not None else None,
                'adjacency_assumption': 'add/remove one record',
                'secure_rng': False,
                'rng_note': 'Opacus secure RNG is disabled; this is suitable for experiments, not a deployment-grade DP mechanism.'}
    except Exception as exc:
        return {'status': status, 'dp_enabled': True, 'epsilon': None, 'delta': delta,
                'accounting_error': str(exc), 'guarantee': 'Accounting failed; no verified privacy result.'}
