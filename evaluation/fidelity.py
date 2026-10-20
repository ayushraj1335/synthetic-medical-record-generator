import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

def compare(real: pd.DataFrame, synthetic: pd.DataFrame, numeric_cols=None, categorical_cols=None):
    numeric_cols=numeric_cols or [c for c in real if pd.api.types.is_numeric_dtype(real[c])]
    categorical_cols=categorical_cols or [c for c in real if c not in numeric_cols]
    numeric={}
    for c in numeric_cols:
        a=pd.to_numeric(real[c],errors='coerce').dropna(); b=pd.to_numeric(synthetic[c],errors='coerce').dropna()
        numeric[c]={'real_mean':float(a.mean()) if len(a) else None,'synthetic_mean':float(b.mean()) if len(b) else None,
                    'wasserstein':float(wasserstein_distance(a,b)) if len(a) and len(b) else None}
    categorical={}
    for c in categorical_cols:
        a=real[c].fillna('__NA__').astype(str).value_counts(normalize=True); b=synthetic[c].fillna('__NA__').astype(str).value_counts(normalize=True)
        keys=set(a.index)|set(b.index); categorical[c]={'total_variation':float(.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in keys))}
    common=[c for c in numeric_cols if c in real and c in synthetic]
    correlations={'real':real[common].corr().fillna(0).to_dict() if common else {},'synthetic':synthetic[common].corr().fillna(0).to_dict() if common else {}}
    return {'numeric':numeric,'categorical':categorical,'correlations':correlations}

def duplicate_metrics(real, synthetic):
    r=real.astype(str).drop_duplicates(); s=synthetic.astype(str)
    real_rows=set(map(tuple,r.to_numpy())); synth_rows=list(map(tuple,s.to_numpy()))
    return {'synthetic_duplicate_rate':float(s.duplicated().mean()) if len(s) else 0.,
            'exact_match_rate_to_training':float(np.mean([row in real_rows for row in synth_rows])) if synth_rows else 0.}

def nearest_neighbor_distances(real, synthetic, max_rows=2000):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    n=min(len(real),max_rows); m=min(len(synthetic),max_rows)
    if n==0 or m==0: return {'mean_distance':None,'note':'No rows available'}
    a=real.iloc[:n].select_dtypes(include='number').fillna(0); b=synthetic.iloc[:m].select_dtypes(include='number').fillna(0)
    cols=list(set(a.columns)&set(b.columns))
    if not cols: return {'mean_distance':None,'note':'No numeric columns'}
    scaler=StandardScaler().fit(a[cols]); x=scaler.transform(a[cols]); y=scaler.transform(b[cols])
    nn=NearestNeighbors(n_neighbors=1).fit(x)
    d=nn.kneighbors(y,return_distance=True)[0][:,0]
    return {'mean_distance':float(d.mean()),'median_distance':float(np.median(d)),'rows_compared':m}
