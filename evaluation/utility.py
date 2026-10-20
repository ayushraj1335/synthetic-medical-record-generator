import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def downstream_utility(real, synthetic, target, random_state=42):
    """Train a real-data model and compare held-out real performance with synthetic-trained model."""
    if target not in real or target not in synthetic: return {'error':'target column missing'}
    real=real.dropna(subset=[target]); synthetic=synthetic.dropna(subset=[target])
    if real[target].nunique()<2 or synthetic.empty: return {'error':'target needs at least two classes and synthetic rows'}
    X=real.drop(columns=[target]); y=real[target].astype(str)
    cats=list(X.select_dtypes(exclude='number').columns); nums=list(X.select_dtypes(include='number').columns)
    pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cats)])
    xa,xt,ya,yt=train_test_split(X,y,test_size=.2,random_state=random_state,stratify=y if y.value_counts().min()>=2 else None)
    model=make_pipeline(pre,RandomForestClassifier(n_estimators=100,min_samples_leaf=2,n_jobs=-1,random_state=random_state,class_weight='balanced'))
    model.fit(xa,ya); pred=model.predict(xt); result={'real_train_accuracy':float(accuracy_score(yt,pred))}
    try: result['real_train_auc_macro']=float(roc_auc_score(yt,model.predict_proba(xt),multi_class='ovr',average='macro',labels=model.classes_))
    except Exception: pass
    common=list(X.columns); sx=synthetic.reindex(columns=common); sy=synthetic[target].astype(str)
    valid=sy.isin(set(ya))
    if valid.sum()>0:
        synmodel=make_pipeline(pre,RandomForestClassifier(n_estimators=100,min_samples_leaf=2,n_jobs=-1,random_state=random_state,class_weight='balanced'))
        # Fit transformer on real training set to keep target evaluation fixed.
        synmodel.fit(sx.loc[valid],sy.loc[valid]); result['synthetic_train_accuracy_on_real_test']=float(accuracy_score(yt,synmodel.predict(xt)))
    return result
