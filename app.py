import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from config import DiffusionConfig
from data_processing.preprocessing import TabularPreprocessor, IDENTIFIERS
from models.diffusion import TabDDPM
from models.dp_trainer import train_private
from models.ctgan_baseline import CTGANBaseline
from privacy.accountant import privacy_report
from evaluation.fidelity import compare, duplicate_metrics, nearest_neighbor_distances
from evaluation.utility import downstream_utility
from evaluation.privacy_tests import run_empirical_privacy_checks

st.set_page_config(page_title='Private Synthetic Medical Data', layout='wide')
st.title('Synthetic Medical Record Generator')
st.caption('TabDDPM-style PyTorch diffusion with optional Opacus DP-SGD. Synthetic output is not for clinical use.')
csvs=sorted(Path('.').glob('*.csv'))
if not csvs:
    st.error('No CSV file found in the project folder. Place the dataset here and restart.'); st.stop()
chosen=st.sidebar.selectbox('Dataset',csvs,format_func=lambda p:p.name)
@st.cache_data(show_spinner='Loading CSV…')
def read_csv(path): return pd.read_csv(path, low_memory=False)
raw=read_csv(str(chosen))
if 'preprocessor' not in st.session_state or st.session_state.get('source') != str(chosen):
    prep=TabularPreprocessor().fit(raw)
    st.session_state.update(preprocessor=prep, source=str(chosen), model=None, privacy=None, synthetic=None, real=raw)
prep=st.session_state.preprocessor
tabs=st.tabs(['Dataset','Training','Generation','Evaluation','Privacy Report'])
with tabs[0]:
    st.subheader(chosen.name); st.write(f'{len(raw):,} rows × {len(raw.columns)} columns')
    st.write('Original identifiers excluded from training:', [c for c in raw if c in IDENTIFIERS])
    st.write('Features used', prep.columns); st.write('Numeric', prep.numeric_cols); st.write('Categorical', prep.categorical_cols)
    st.dataframe(pd.DataFrame({'dtype':raw.dtypes.astype(str),'missing':raw.replace({'?':np.nan,'':np.nan}).isna().sum(),'unique':raw.nunique(dropna=False)}), use_container_width=True)
    st.dataframe(raw.head(10),use_container_width=True)
with tabs[1]:
    epochs=st.number_input('Epochs',1,1000,20); batch=st.number_input('Batch size',16,4096,256,step=16)
    steps=st.number_input('Diffusion steps',10,1000,100); hidden=st.number_input('Hidden width',32,2048,256,step=32)
    dp=st.checkbox('Enable Opacus differential privacy',False)
    epsilon=st.number_input('Target epsilon (used to choose noise)',0.1,100.0,8.0,step=.1,disabled=not dp)
    delta=st.number_input('Delta',min_value=1e-10,max_value=1e-2,value=1e-5,format='%.1e',disabled=not dp)
    clip=st.number_input('Per-sample clipping norm',.01,100.,1.,disabled=not dp)
    if st.button('Train diffusion model',type='primary'):
        progress=st.progress(0); info=st.empty(); engine=None
        try:
            encoded=prep.transform(raw)
            model=TabDDPM(prep.dimension,steps=steps,hidden_dim=hidden)
            if dp:
                engine=train_private(model,encoded,epochs=int(epochs),batch_size=int(batch),noise_multiplier=None,target_epsilon=float(epsilon),max_grad_norm=float(clip),delta=float(delta),callback=lambda e,l:(progress.progress(e/int(epochs)),info.write(f'Epoch {e}/{epochs} · loss {l:.4f}')))
            else:
                model.fit(encoded,epochs=int(epochs),batch_size=int(batch),callback=lambda e,l:(progress.progress(e/int(epochs)),info.write(f'Epoch {e}/{epochs} · loss {l:.4f}')))
            report=privacy_report(engine,delta,status=model.training_status)
            st.session_state.update(model=model,privacy=report,trained=True,dp=dp)
            st.success('Training completed.'); st.json(report)
        except Exception as exc:
            st.session_state.update(model=None,privacy={'status':'failed','dp_enabled':dp,'epsilon':None,'delta':delta,'error':str(exc),'guarantee':'Training/accounting failed; no verified privacy result.'},trained=False)
            st.error(f'Training failed: {exc}')
    with st.expander('Optional CTGAN baseline (non-private)'):
        ctgan_epochs=st.number_input('CTGAN epochs',10,2000,300,step=10,key='ctgan_epochs')
        if st.button('Train CTGAN baseline'):
            try:
                baseline=CTGANBaseline(epochs=int(ctgan_epochs))
                baseline.fit(prep.clean(raw),prep.categorical_cols)
                st.session_state.ctgan=baseline
                st.success('CTGAN baseline trained. This model is non-private.')
            except Exception as exc: st.error(f'CTGAN training failed: {exc}')
with tabs[2]:
    model=st.session_state.get('model'); n=st.number_input('Records to generate',1,100000,1000)
    available=['Diffusion'] + (['CTGAN (non-private)'] if st.session_state.get('ctgan') is not None else [])
    source=st.selectbox('Generator',available)
    if st.button('Generate records',disabled=(model is None and source=='Diffusion')):
        try:
            if source=='Diffusion':
                generated=model.sample(int(n),seed=42); st.session_state.synthetic=prep.inverse_transform(generated)
            else:
                st.session_state.synthetic=st.session_state.ctgan.sample(int(n))
            st.session_state.synthetic_source=source
            st.success(f'Generated {int(n):,} synthetic records.')
        except Exception as exc: st.error(f'Generation failed: {exc}')
    if st.session_state.get('synthetic') is not None:
        synth=st.session_state.synthetic; st.dataframe(synth.head(20),use_container_width=True)
        if st.session_state.get('synthetic_source')=='CTGAN (non-private)': st.warning('These records came from the non-private CTGAN baseline. No differential privacy guarantee applies.')
        st.download_button('Download synthetic CSV',synth.to_csv(index=False).encode('utf-8'),file_name='synthetic_medical_records.csv',mime='text/csv')
with tabs[3]:
    synth=st.session_state.get('synthetic')
    if synth is None: st.info('Train a model and generate records to view comparisons.')
    else:
        real=prep.clean(raw); summary=compare(real,synth,numeric_cols=prep.numeric_cols,categorical_cols=prep.categorical_cols)
        st.subheader('Feature distribution comparison')
        for c,m in summary['numeric'].items():
            st.write(c, m)
            fig,ax=plt.subplots(); ax.hist(pd.to_numeric(real[c],errors='coerce').dropna(),bins=30,density=True,alpha=.55,label='Training'); ax.hist(pd.to_numeric(synth[c],errors='coerce').dropna(),bins=30,density=True,alpha=.55,label='Synthetic'); ax.legend(); ax.set_title(c); st.pyplot(fig); plt.close(fig)
        for c,m in summary['categorical'].items():
            st.write(c, m)
            freq=pd.concat([real[c].fillna('__NA__').astype(str).value_counts(normalize=True).rename('Training'),synth[c].fillna('__NA__').astype(str).value_counts(normalize=True).rename('Synthetic')],axis=1).fillna(0).sort_values('Training',ascending=False).head(12)
            st.bar_chart(freq)
        st.subheader('Correlation matrices'); st.json(summary['correlations'])
        st.subheader('Duplicates and nearest neighbors'); st.json({**duplicate_metrics(real,synth),**nearest_neighbor_distances(real,synth)})
        target=st.selectbox('Optional downstream target', [None]+prep.columns)
        if target and st.button('Evaluate downstream utility'):
            st.json(downstream_utility(real,synth,target))
        with st.expander('Empirical privacy checks (not a DP proof)'): st.json(run_empirical_privacy_checks(real,synth))
with tabs[4]:
    if st.session_state.get('synthetic_source')=='CTGAN (non-private)': st.warning('The currently displayed synthetic dataset was generated by CTGAN, which is non-private. The training report below applies only to the diffusion training run.')
    report=st.session_state.get('privacy')
    if report is None: st.info('No training run has produced a privacy report yet.')
    else:
        st.json(report)
        if report.get('dp_enabled'):
            st.caption('Formal definition: an algorithm M is (ε,δ)-DP if for neighboring datasets D,D′ and every measurable output set S, P[M(D)∈S] ≤ exp(ε)P[M(D′)∈S]+δ. This implementation accounts DP-SGD under add/remove-one-record adjacency, Poisson sampling, Gaussian noise, clipping and the reported number of steps. The data-fitted scaler and category vocabulary are not privatized; therefore reported ε is the denoiser DP-SGD accounting result and does not establish an end-to-end guarantee for the released decoded records.')
        st.caption('Empirical privacy checks can reveal some leakage but cannot prove differential privacy. CTGAN, when installed, is an optional non-private baseline.')
