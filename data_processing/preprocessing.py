"""Schema-aware preprocessing. '?' in the source file is treated as missing."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

IDENTIFIERS = {'encounter_id', 'patient_nbr'}

@dataclass
class TabularPreprocessor:
    numeric_cols: list[str] | None = None
    categorical_cols: list[str] | None = None
    scaler: MinMaxScaler | None = None
    categories: dict[str, list[str]] | None = None
    columns: list[str] | None = None
    encoded_slices: dict[str, slice] | None = None
    dimension: int = 0

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.drop(columns=[c for c in IDENTIFIERS if c in frame], errors='ignore').copy()
        df = df.replace({'?': np.nan, '': np.nan})
        # Preserve numeric columns and represent their missingness as a categorical value.
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors='coerce')
            else:
                df[c] = df[c].astype('object').where(df[c].notna(), '__MISSING__').astype(str)
        return df

    def fit(self, frame: pd.DataFrame):
        df = self.clean(frame)
        self.columns = list(df.columns)
        self.numeric_cols = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols = [c for c in df if c not in self.numeric_cols]
        self.scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        if self.numeric_cols:
            nums = df[self.numeric_cols].copy()
            self.numeric_medians = nums.median().fillna(0)
            self.scaler.fit(nums.fillna(self.numeric_medians))
        else:
            self.numeric_medians = pd.Series(dtype=float)
        self.categories = {}
        self.encoded_slices = {}
        start = 0
        for c in self.categorical_cols:
            values = sorted(set(df[c].astype(str)))
            if '__MISSING__' not in values: values.append('__MISSING__')
            if '__UNK__' not in values: values.append('__UNK__')
            self.categories[c] = values
            self.encoded_slices[c] = slice(start, start + len(values)); start += len(values)
        for c in self.numeric_cols:
            self.encoded_slices[c] = slice(start, start + 1); start += 1
        self.dimension = start
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.columns is None: raise RuntimeError('preprocessor is not fitted')
        df = self.clean(frame).reindex(columns=self.columns)
        blocks = []
        for c in self.categorical_cols:
            vals = df[c].astype(str).where(df[c].astype(str).isin(self.categories[c]), '__UNK__')
            idx = pd.Categorical(vals, categories=self.categories[c]).codes
            onehot = np.zeros((len(df), len(self.categories[c])), dtype=np.float32)
            onehot[np.arange(len(df)), idx] = 1
            blocks.append(onehot)
        if self.numeric_cols:
            nums = df[self.numeric_cols].fillna(self.numeric_medians)
            blocks.append(self.scaler.transform(nums).astype(np.float32))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        n = encoded.shape[0]
        out = {}
        offset = 0
        for c in self.categorical_cols:
            cats = self.categories[c]; part = encoded[:, offset:offset+len(cats)]; offset += len(cats)
            out[c] = np.asarray(cats, dtype=object)[np.argmax(part, axis=1)]
        if self.numeric_cols:
            vals = encoded[:, offset:offset+len(self.numeric_cols)]
            out.update({c: self.scaler.inverse_transform(vals)[:, i] for i, c in enumerate(self.numeric_cols)})
        result = pd.DataFrame(out, columns=self.columns)
        for c in self.categorical_cols:
            result[c] = result[c].replace({'__MISSING__': np.nan, '__UNK__': np.nan})
        return result

    def summary(self, raw: pd.DataFrame) -> dict:
        df = self.clean(raw)
        numeric = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
        return {'rows': len(raw), 'columns': len(raw.columns), 'training_features': self.columns,
                'excluded_identifiers': [c for c in raw if c in IDENTIFIERS], 'numeric_features': numeric,
                'categorical_features': [c for c in df if c not in numeric],
                'missing_by_column': raw.replace({'?': np.nan, '': np.nan}).isna().sum().to_dict()}
