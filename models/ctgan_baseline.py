"""Optional CTGAN adapter. This baseline is explicitly non-private."""
class CTGANBaseline:
    def __init__(self, epochs=300): self.epochs=epochs; self.model=None
    def fit(self, frame, categorical_columns):
        try: from ctgan import CTGAN
        except ImportError as exc: raise RuntimeError('Install the optional ctgan package to use this baseline.') from exc
        self.model=CTGAN(epochs=self.epochs, verbose=False); self.model.fit(frame, categorical_columns); return self
    def sample(self, n):
        if self.model is None: raise RuntimeError('CTGAN baseline is not fitted')
        return self.model.sample(n)
    @property
    def privacy_label(self): return 'Non-private baseline; no differential privacy guarantee.'
