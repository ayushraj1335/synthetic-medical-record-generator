import numpy as np
import pandas as pd
import pytest
from data_processing.preprocessing import TabularPreprocessor
from models.diffusion import TabDDPM
from privacy.accountant import privacy_report
from evaluation.fidelity import compare, duplicate_metrics

@pytest.fixture
def sample():
    return pd.DataFrame({'encounter_id':[1,2,3,4],'patient_nbr':[10,11,12,13], 'age':[20,30,40,50], 'gender':['F','M','F','M'], 'lab':[1.,np.nan,3.,4.]})

def test_identifiers_excluded_and_roundtrip(sample):
    p=TabularPreprocessor().fit(sample); x=p.transform(sample); y=p.inverse_transform(x)
    assert 'encounter_id' not in p.columns and 'patient_nbr' not in p.columns
    assert x.shape==(4,p.dimension) and y.columns.tolist()==['age','gender','lab']
    assert y['gender'].tolist()==sample.gender.tolist()

def test_diffusion_train_and_sample():
    data=np.random.default_rng(4).normal(size=(12,3)).astype('float32')
    model=TabDDPM(3,steps=4,hidden_dim=16,device='cpu'); model.fit(data,epochs=1,batch_size=4)
    generated=model.sample(5,seed=7)
    assert generated.shape==(5,3) and np.isfinite(generated).all()

def test_evaluation_and_nonprivate_report(sample):
    p=TabularPreprocessor().fit(sample); real=p.clean(sample); synth=p.inverse_transform(p.transform(sample))
    assert 'age' in compare(real,synth)['numeric']
    metrics=duplicate_metrics(real,synth)
    assert 0. <= metrics['exact_match_rate_to_training'] <= 1.
    report=privacy_report(None,1e-5)
    assert report['epsilon'] is None and report['dp_enabled'] is False

def test_generation_requires_training():
    with pytest.raises(RuntimeError): TabDDPM(2,steps=4).sample(1)
