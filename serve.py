"""ATMOS Location-Aware API — Render deployment."""
import os, sqlite3, threading, warnings
warnings.filterwarnings('ignore')

# ── Download bundle from Google Drive if not present ──────────────────────────
BUNDLE_PATH = os.environ.get('ATMOS_BUNDLE_PATH', 'atmos_bundle/bundle.joblib')
if not os.path.exists(BUNDLE_PATH):
    import gdown
    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)
    GDRIVE_ID = os.environ.get('ATMOS_BUNDLE_GDRIVE_ID', '1fZ67lbHcm0bQnydEwdBz8XO6vPwygkTB')
    print(f'Downloading bundle from Google Drive ({GDRIVE_ID})...')
    gdown.download(id=GDRIVE_ID, output=BUNDLE_PATH, quiet=False)
    print('Bundle downloaded.')

import numpy as np, pandas as pd, joblib, uvicorn
import lightgbm as lgb
from sklearn.linear_model import Ridge
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Required class definitions for unpickling the bundle ──────────────────────
def fourier(idx, K=4):
    t = np.asarray(idx.dayofyear, float)/365.25*2*np.pi
    return np.column_stack([fn(k*t) for k in range(1,K+1) for fn in (np.sin,np.cos)])

def anomaly_feats(an):
    return pd.DataFrame({'a1':an.ffill(limit=2),'a3':an.rolling(3,min_periods=1).mean(),
                         'a7':an.rolling(7,min_periods=3).mean(),'a30':an.rolling(30,min_periods=7).mean(),
                         'a_trend':an.ffill(limit=2) - an.rolling(7,min_periods=3).mean()})

ACOLS = ['a1','a3','a7','a30','a_trend']

class DailyForecasterCity:
    def __init__(self, target, H=30, K=4, lo=5.0, hi=500.0):
        self.target, self.H, self.K = target, H, K
        self.lo, self.hi = lo, hi
        self.resid_q = {}; self.models = {}; self.clim = None
    def fit(self, daily_city):
        s = daily_city[self.target].interpolate(limit=3, limit_direction='both')
        v = s.dropna()
        if len(v) < 60: return self
        self.clim = Ridge(alpha=1.0).fit(fourier(v.index, self.K), v.values)
        an = pd.Series(s.values - self.clim.predict(fourier(s.index, self.K)), index=s.index)
        Fd = anomaly_feats(an).dropna()
        for h in range(1, self.H+1):
            D = Fd.join(an.shift(-h).rename('y'), how='inner').dropna()
            if len(D) < 80: self.models[h] = []; continue
            Xb, y = D[ACOLS].values, D['y'].values
            cands = [Ridge(alpha=5.0).fit(Xb, y)]
            cands.append(lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                verbose=-1, random_state=42).fit(Xb, y))
            self.models[h] = cands
        return self
    def predict(self, recent, origin):
        origin = pd.Timestamp(origin).normalize()
        s = recent if isinstance(recent, pd.Series) else recent[self.target]
        s = s.reindex(pd.date_range(origin - pd.Timedelta(days=29), origin))
        if int(s.notna().sum()) >= 3:
            an = pd.Series(s.values - self.clim.predict(fourier(s.index, self.K)), index=s.index)
            f = anomaly_feats(an).iloc[-1].fillna(0.0)
        else:
            f = pd.Series(0.0, index=ACOLS)
        fut = pd.date_range(origin + pd.Timedelta(days=1), periods=self.H)
        base = self.clim.predict(fourier(fut, self.K))
        xa = f[ACOLS].values.astype(float)
        yhat = np.empty(self.H)
        for h in range(1, self.H+1):
            cands = self.models.get(h, [])
            preds = [float(m.predict(xa[None])[0]) for m in cands]
            yhat[h-1] = base[h-1] + (np.mean(preds) if preds else 0.0)
        yhat = np.clip(yhat, self.lo, self.hi)
        band = 15 + 0.5 * np.arange(1, self.H+1)
        return pd.DataFrame({'date':fut,'h':np.arange(1,self.H+1),'climatology':base,
                             'forecast':yhat,'low':np.clip(yhat-band,self.lo,self.hi),
                             'high':np.clip(yhat+band,self.lo,self.hi)}).set_index('date')

DB_PATH     = os.environ.get('ATMOS_DB_PATH', 'atmos_live.db')
PORT        = int(os.environ.get('PORT', 8000))
bundle = joblib.load(BUNDLE_PATH)
hourly_per_city = bundle['hourly_per_city']
final_daily = bundle['daily_per_city']
CITIES = bundle['cities']
DEFAULT_CITY = bundle.get('default_city', 'Dhaka')
AQI_LEVELS = bundle['levels']

# Any device_id is accepted — auth is by shared sensor key, not per-device key
SENSOR_KEY = os.environ.get('ATMOS_DEVICE_KEY', 'sensor-key-change-me')
READ_KEY   = os.environ.get('ATMOS_READ_KEY',   'frontend-read-key-change-me')

POLLUTANTS = ['pm2_5','pm10','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide','ozone']
POLLUTANT_LIMITS = {'pm2_5':1000,'pm10':2000,'carbon_monoxide':60000,'nitrogen_dioxide':3000,'sulphur_dioxide':3000,'ozone':1000}
MIN_SAMPLES_PER_HOUR = 1; MAX_READING_AGE_S = 21600
FT3_TO_M3=0.0283168466; CFM_TO_M3H=1.699011
TZ = ZoneInfo('Asia/Dhaka')
_EDGES = np.array([50,100,150,200,300])
_BP = {'pm2_5':[(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)],'pm10':[(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)],'co':[(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,40.4,301,400),(40.5,50.4,401,500)],'so2':[(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300),(605,804,301,400),(805,1004,401,500)],'no2':[(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)],'o3':[(0.000,0.054,0,50),(0.055,0.070,51,100),(0.071,0.085,101,150),(0.086,0.105,151,200),(0.106,0.200,201,300)]}
_TRUNC = {'pm2_5':1,'pm10':0,'co':1,'so2':0,'no2':0,'o3':3}; _DIV = {'co':1145.0,'no2':1.882,'so2':2.620,'o3':1963.0}
def _trunc(x,d): f=10.0**d; return np.round(np.floor(np.asarray(x,float)*f+1e-9)/f,d)
def _sub(c,n):
    c=np.asarray(c,float); b=_BP[n]
    lo=np.array([x[0] for x in b]); hi=np.array([x[1] for x in b])
    ilo=np.array([x[2] for x in b],float); ihi=np.array([x[3] for x in b],float)
    i=np.clip(np.searchsorted(lo,c,side='right')-1,0,len(b)-1)
    return np.where(np.isnan(c),np.nan,(ihi[i]-ilo[i])/(hi[i]-lo[i])*(c-lo[i])+ilo[i])
def compute_aqi(**kw):
    m={'pm2_5':kw.get('pm2_5'),'pm10':kw.get('pm10'),'co':kw.get('carbon_monoxide'),'no2':kw.get('nitrogen_dioxide'),'so2':kw.get('sulphur_dioxide'),'o3':kw.get('ozone')}
    ss=[]
    for n,v in m.items():
        if v is None: continue
        x=np.asarray(v,float); x=np.where(np.isfinite(x)&(x>=0),x,np.nan)
        if n in _DIV: x=x/_DIV[n]
        x=_trunc(x,_TRUNC[n]); x=np.where(x<=_BP[n][-1][1],x,np.nan); ss.append(_sub(x,n))
    if not ss: return float('nan')
    st=np.stack(np.broadcast_arrays(*ss),axis=0); an=np.all(np.isnan(st),axis=0)
    a=np.floor(np.where(np.isnan(st),-1,st).max(axis=0)+0.5); return float(np.where(an,np.nan,a))
def aqi_to_level(aqi):
    aqi=float(aqi)
    if not np.isfinite(aqi) or aqi<0: return {'aqi':None,'level':None,'level_index':-1,'color':None,'advice':None}
    idx=int(np.searchsorted(_EDGES, np.floor(aqi+0.5), side='left')); idx=max(0,min(idx,5))
    return {'aqi':round(aqi,1),'level':AQI_LEVELS[idx]['level'],'level_index':idx,'color':AQI_LEVELS[idx]['color'],'advice':AQI_LEVELS[idx]['advice']}
def normalize_city(loc):
    if loc in CITIES: return loc
    for c in CITIES:
        if c.lower()==str(loc).lower(): return c
    return DEFAULT_CITY
conn=sqlite3.connect(DB_PATH,check_same_thread=False); lock=threading.RLock()
conn.execute("CREATE TABLE IF NOT EXISTS samples(device_id TEXT, location TEXT, ts REAL, pm2_5 REAL, pm10 REAL, carbon_monoxide REAL, nitrogen_dioxide REAL, sulphur_dioxide REAL, ozone REAL, UNIQUE(device_id, ts))"); conn.commit()
PM25_AT_AQI={50:12.0,100:35.4,150:55.4,200:150.4}
SCENARIOS={'indoor_sealed':dict(lam=0.3,pen=0.7,k=0.15,desc='Sealed'),'indoor_normal':dict(lam=0.7,pen=0.8,k=0.10,desc='Normal'),'indoor_windows_open':dict(lam=3.0,pen=1.0,k=0.10,desc='Windows open'),'outdoor_semi_open':dict(lam=8.0,pen=1.0,k=0.0,desc='Semi-open'),'outdoor_open':dict(lam=20.,pen=1.0,k=0.0,desc='Open air')}
def room_volume_m3(a,h): return float(a)*float(h)*FT3_TO_M3
def devices_needed(pm25_out,sc,vol,cadr,target_pm25=35.4,hours=24.0,max_d=60):
    p=SCENARIOS[sc]; lam,pen,k=p['lam'],p['pen'],p['k']
    c0=pen*lam/(lam+k)*pm25_out
    if c0<=target_pm25: return {'n':0,'note':'Already within target'}
    q=cadr*CFM_TO_M3H/vol
    for n in range(1,max_d+1):
        rate=lam+k+n*q; css=pen*lam*pm25_out/rate; ct=css+(c0-css)*np.exp(-rate*hours)
        if ct<=target_pm25: return {'n':n,'note':''}
    return {'n':None,'note':f'Not reachable in {hours}h'}
def hourly_features(df):
    f=pd.DataFrame(index=df.index)
    for p in POLLUTANTS+['aqi']:
        s=df[p]; f[p]=s
        for lag in (1,2,3,6,12,24): f[f'{p}_lag{lag}']=s.shift(lag)
        for w in (3,6,24): f[f'{p}_mean{w}']=s.rolling(w,min_periods=w).mean()
        f[f'{p}_std6']=s.rolling(6,min_periods=6).std()
    f['aqi_d1']=df['aqi']-df['aqi'].shift(1); f['aqi_d3']=df['aqi']-df['aqi'].shift(3)
    h=df.index.hour; doy=df.index.dayofyear
    f['h_sin']=np.sin(2*np.pi*h/24); f['h_cos']=np.cos(2*np.pi*h/24)
    f['d_sin']=np.sin(2*np.pi*doy/365.25); f['d_cos']=np.cos(2*np.pi*doy/365.25); f['dow']=df.index.dayofweek
    return f
def _hourly_frame(device_id, city=None):
    with lock:
        if city: rows=conn.execute('SELECT ts,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone FROM samples WHERE device_id=? AND location=? ORDER BY ts',(device_id,city)).fetchall()
        else: rows=conn.execute('SELECT ts,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone FROM samples WHERE device_id=? ORDER BY ts',(device_id,)).fetchall()
    if not rows: return pd.DataFrame(columns=POLLUTANTS, dtype=float)
    d=pd.DataFrame(rows,columns=['ts']+POLLUTANTS)
    idx=pd.DatetimeIndex(pd.to_datetime(d.pop('ts'),unit='s',utc=True)).tz_convert(TZ).tz_localize(None)
    d.index=idx; g=d.groupby(idx.floor('h'))
    mean=g[POLLUTANTS].mean(); cnt=g[POLLUTANTS].count()
    mean=mean.where(cnt>=MIN_SAMPLES_PER_HOUR)
    cur=pd.Timestamp(datetime.now(TZ).replace(tzinfo=None)).floor('h')
    return mean[mean.index<cur]
class Reading(BaseModel):
    device_id: str; location: str; observed_at: datetime
    pm2_5: float | None = None; pm10: float | None = None; carbon_monoxide: float | None = None
    nitrogen_dioxide: float | None = None; sulphur_dioxide: float | None = None; ozone: float | None = None
api=FastAPI(title='ATMOS API',version=bundle['version'])
api.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['*'])
@api.get('/healthz')
def health(): return {'ok':True,'model_version':bundle['version'],'cities':CITIES,'time':datetime.now(TZ).isoformat()}
@api.post('/api/v1/readings')
def ingest(r:Reading,x_api_key:str=Header(default='')):
    if x_api_key != SENSOR_KEY: raise HTTPException(403,'bad key')
    if r.observed_at.tzinfo is None: raise HTTPException(422,'need tz')
    age=datetime.now(TZ).timestamp()-r.observed_at.timestamp()
    if age<-120: raise HTTPException(422,'future')
    if age>MAX_READING_AGE_S: raise HTTPException(422,'old')
    vals=[]
    for p in POLLUTANTS:
        v=getattr(r,p)
        if v is not None and (not np.isfinite(v) or v<0 or v>POLLUTANT_LIMITS[p]): v=None
        vals.append(v)
    if all(v is None for v in vals): raise HTTPException(422,'no valid')
    city=normalize_city(r.location)
    with lock:
        cur=conn.execute('INSERT OR IGNORE INTO samples VALUES(?,?,?,?,?,?,?,?,?)',(r.device_id,city,r.observed_at.timestamp(),*vals)); conn.commit()
    return {'accepted':True,'duplicate':cur.rowcount==0,'location':city}
@api.get('/api/v1/forecast/hourly')
def hourly_ep(device_id:str,location:str=DEFAULT_CITY,x_api_key:str=Header(default='')):
    if x_api_key!=READ_KEY: raise HTTPException(401,'bad read key')
    city=normalize_city(location); hf=_hourly_frame(device_id,city).dropna(subset=POLLUTANTS)
    resp={'device_id':device_id,'location':city,'status':'warming_up','hours_available':len(hf),'hours_required':25,'current':None,'forecasts':[]}
    if len(hf)==0: resp['message']='No data'; return resp
    run=hf.tail(25); last=run.iloc[-1]
    aqi_now=float(compute_aqi(**{p:float(last[p]) for p in POLLUTANTS}))
    pm25_now=float(last['pm2_5']) if pd.notna(last['pm2_5']) else None
    pm10_now=float(last['pm10']) if pd.notna(last['pm10']) else None
    cur=aqi_to_level(aqi_now); cur['hour_start']=run.index[-1].isoformat()
    cur['pm2_5']=round(pm25_now,1) if pm25_now is not None else None
    cur['pm10']=round(pm10_now,1) if pm10_now is not None else None
    resp['current']=cur
    if len(run)<25: resp['message']=f'Collecting {len(run)}/25'; return resp
    F_row=hourly_features(run); cm=hourly_per_city[city]
    for col in cm['features']:
        if col not in F_row.columns: F_row[col]=0
    X=F_row[cm['features']].iloc[[-1]]; fc=[]
    for hz in (1,3,6):
        d_aqi=[float(b.predict(X)[0]) for b in cm['aqi'][hz]]
        pred_aqi=float(np.clip(aqi_now+np.mean(d_aqi),0,500))
        pred_pm25=None; pred_pm10=None
        if pm25_now is not None and hz in cm.get('pm2_5',{}):
            d_pm=[float(b.predict(X)[0]) for b in cm['pm2_5'][hz]]
            pred_pm25=round(float(np.clip(pm25_now+np.mean(d_pm),0,500)),1)
        if pm10_now is not None and hz in cm.get('pm10',{}):
            d_p10=[float(b.predict(X)[0]) for b in cm['pm10'][hz]]
            pred_pm10=round(float(np.clip(pm10_now+np.mean(d_p10),0,1000)),1)
        e=aqi_to_level(pred_aqi); e['horizon_hours']=hz
        e['hour_start']=(run.index[-1]+pd.Timedelta(hours=hz)).isoformat()
        e['change_from_current']=round(pred_aqi-aqi_now,1)
        e['trend']='worsening' if pred_aqi-aqi_now>5 else 'improving' if pred_aqi-aqi_now<-5 else 'stable'
        e['pm2_5']=pred_pm25; e['pm10']=pred_pm10; fc.append(e)
    resp['status']='ok'; resp['forecasts']=fc; return resp
@api.get('/api/v1/forecast/daily')
def daily_ep(device_id:str,location:str=DEFAULT_CITY,days:int=30,x_api_key:str=Header(default='')):
    if x_api_key!=READ_KEY: raise HTTPException(401,'bad read key')
    city=normalize_city(location); hf=_hourly_frame(device_id,city)
    today=pd.Timestamp(datetime.now(TZ).replace(tzinfo=None)).normalize()
    if len(hf)>0:
        full=hf.reindex(pd.date_range(hf.index.min(),hf.index.max(),freq='h'))
        full['aqi']=compute_aqi(**{p:full[p].values for p in POLLUTANTS})
        g=full.resample('D'); d=pd.DataFrame({'aqi':g['aqi'].mean(),'pm2_5':g['pm2_5'].mean(),'pm10':g['pm10'].mean()})
        d.loc[g['aqi'].count()<MIN_SAMPLES_PER_HOUR,['aqi','pm2_5','pm10']]=np.nan; d=d[d.index<today]
    else: d=pd.DataFrame(columns=['aqi','pm2_5','pm10'],dtype=float)
    n_hist=int(d['aqi'].notna().sum()) if len(d) else 0; origin=today-pd.Timedelta(days=1)
    ra=d['aqi'] if n_hist>=3 else d['aqi'].iloc[0:0]; rp=d['pm2_5'] if n_hist>=3 else d['pm2_5'].iloc[0:0]; r10=d['pm10'] if n_hist>=3 else d['pm10'].iloc[0:0]
    p_aqi=final_daily[city]['aqi'].predict(ra,origin).iloc[:days]
    p_pm=final_daily[city]['pm2_5'].predict(rp,origin).iloc[:days]
    p_p10=final_daily[city]['pm10'].predict(r10,origin).iloc[:days]
    items=[]
    for i in range(len(p_aqi)):
        lv=aqi_to_level(p_aqi['forecast'].iloc[i])
        items.append({'date':p_aqi.index[i].date().isoformat(),'day':i+1,'aqi':lv['aqi'],'level':lv['level'],'level_index':lv['level_index'],'color':lv['color'],'advice':lv['advice'],'aqi_low':round(float(p_aqi['low'].iloc[i]),1),'aqi_high':round(float(p_aqi['high'].iloc[i]),1),'pm2_5':round(float(p_pm['forecast'].iloc[i]),1),'pm2_5_low':round(float(p_pm['low'].iloc[i]),1),'pm2_5_high':round(float(p_pm['high'].iloc[i]),1),'pm10':round(float(p_p10['forecast'].iloc[i]),1),'pm10_low':round(float(p_p10['low'].iloc[i]),1),'pm10_high':round(float(p_p10['high'].iloc[i]),1)})
    return {'device_id':device_id,'location':city,'basis':'live_sensor_history' if n_hist>=3 else 'seasonal_only','history_days_used':n_hist,'forecast':items}
@api.get('/api/v1/device-requirement')
def device_requirement(device_id:str,location:str=DEFAULT_CITY,scenario:str='indoor_normal',target_aqi:int=100,hours:float=24,area_ft2:float=700,height_ft:float=10,cadr_cfm:float=120,x_api_key:str=Header(default='')):
    if x_api_key!=READ_KEY: raise HTTPException(401,'bad read key')
    if scenario not in SCENARIOS: raise HTTPException(422,'bad scenario')
    if target_aqi not in PM25_AT_AQI: raise HTTPException(422,'bad target')
    city=normalize_city(location); target_pm25=PM25_AT_AQI[target_aqi]; vol=room_volume_m3(area_ft2,height_ft)
    hf=_hourly_frame(device_id,city); today=pd.Timestamp(datetime.now(TZ).replace(tzinfo=None)).normalize()
    if len(hf)>0:
        full=hf.reindex(pd.date_range(hf.index.min(),hf.index.max(),freq='h'))
        full['aqi']=compute_aqi(**{p:full[p].values for p in POLLUTANTS})
        g=full.resample('D'); d=pd.DataFrame({'aqi':g['aqi'].mean(),'pm2_5':g['pm2_5'].mean(),'pm10':g['pm10'].mean()})
        d.loc[g['aqi'].count()<MIN_SAMPLES_PER_HOUR,['aqi','pm2_5','pm10']]=np.nan; d=d[d.index<today]
    else: d=pd.DataFrame(columns=['aqi','pm2_5','pm10'],dtype=float)
    n_hist=int(d['aqi'].notna().sum()) if len(d) else 0; origin=today-pd.Timedelta(days=1)
    recent=d['pm2_5'] if n_hist>=3 else d['pm2_5'].iloc[0:0]
    pm_fc=final_daily[city]['pm2_5'].predict(recent,origin).iloc[:30]
    days_out=[]; typical=[]
    for i in range(len(pm_fc)):
        pm_val=float(pm_fc['forecast'].iloc[i]); pm_hi=float(pm_fc['high'].iloc[i])
        r=devices_needed(pm_val,scenario,vol,cadr_cfm,target_pm25,hours)
        r_hi=devices_needed(pm_hi,scenario,vol,cadr_cfm,target_pm25,hours)
        days_out.append({'date':pm_fc.index[i].date().isoformat(),'day':i+1,'forecast_outdoor_pm2_5':round(pm_val,1),'forecast_high_case_pm2_5':round(pm_hi,1),'devices_needed':r['n'],'devices_needed_high_case':r_hi['n'],'note':r['note']})
        if r['n'] is not None: typical.append(r['n'])
    return {'device_id':device_id,'location':city,'room':{'area_ft2':area_ft2,'height_ft':height_ft,'volume_m3':round(vol,1)},'device_cadr_cfm':cadr_cfm,'scenario':scenario,'scenario_description':SCENARIOS[scenario]['desc'],'target':{'aqi':target_aqi,'pm2_5':target_pm25,'within_hours':hours},'basis':'live_sensor_history' if n_hist>=3 else 'seasonal_only','history_days_used':n_hist,'summary':{'typical_devices':int(np.median(typical)) if typical else None,'max_devices_forecast':max(typical) if typical else None,'max_devices_high_case':max([d['devices_needed_high_case'] or 0 for d in days_out]) if days_out else None},'days':days_out,'disclaimer':'Physics model with assumed air-exchange values.'}
if __name__=='__main__': uvicorn.run(api,host='0.0.0.0',port=PORT,log_level='info')
