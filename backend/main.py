"""AURA backend: personal baseline + drift engine. Risk estimate, not a diagnosis."""
import numpy as np, pandas as pd, os, sqlite3, hashlib, secrets, io, datetime, json, re, time, urllib.request
from typing import Optional, List
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

_e=os.path.join(os.path.dirname(os.path.abspath(__file__)),".env")
if os.path.exists(_e):
    for _l in open(_e):
        if "=" in _l and not _l.startswith("#"): _k,_v=_l.strip().split("=",1); os.environ.setdefault(_k,_v)
FEATS = ["rhr","sleep","steps","bmi","sbp","glucose","stress","water","smoker","alcohol","age","hba1c","chol","family","d_rhr","d_sleep","d_steps","d_bmi"]
lin = lambda x,a,b: float(np.clip((x-a)/(b-a),0,1)*100)
sig = lambda z: 1/(1+np.exp(-z))

def latent(r):  # clinically-inspired synthetic risk function (replace with real dataset labels in production)
    return (-5.0+.03*(r["age"]-40)+.03*(r["rhr"]-70)+.018*(r["sbp"]-120)+.02*(r["glucose"]-95)+.1*(r["bmi"]-24)
        -.3*(r["sleep"]-7)-.00016*(r["steps"]-7000)+.18*(r["stress"]-5)-.4*(r["water"]-2.2)+.8*r["smoker"]+.25*r["alcohol"]
        +.8*(r["hba1c"]-5.4)+.006*(r["chol"]-190)+.5*r["family"]+.07*r["d_rhr"]-.5*r["d_sleep"]-.0003*r["d_steps"]+.6*r["d_bmi"])

def train(n=6000, seed=7):
    g=np.random.default_rng(seed)
    d=pd.DataFrame({"rhr":g.normal(70,9,n),"sleep":g.normal(7,1,n).clip(3,10),"steps":g.normal(7000,2800,n).clip(500,16000),
      "bmi":g.normal(25,4,n).clip(16,42),"sbp":g.normal(120,14,n),"glucose":g.normal(95,14,n),"stress":g.uniform(1,10,n),
      "water":g.normal(2.2,.7,n).clip(.5,4.5),"smoker":g.binomial(1,.2,n),"alcohol":g.integers(0,4,n),"age":g.uniform(20,70,n),
      "hba1c":g.normal(5.4,.5,n),"chol":g.normal(190,35,n),"family":g.binomial(1,.3,n),"d_rhr":g.normal(0,6,n),
      "d_sleep":g.normal(0,.8,n),"d_steps":g.normal(0,1500,n),"d_bmi":g.normal(0,.6,n)})
    y=(g.random(n)<sig(latent(d))).astype(int)
    sc=StandardScaler().fit(d[FEATS]); m=LogisticRegression(C=1.0,max_iter=2000).fit(sc.transform(d[FEATS]),y)
    return sc,m
SC,MODEL=train()

# ---------- real public datasets: validated sub-models (cross-validated AUC) ----------
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data"); RM={}; CARD=[]
def real_models():
    for key,f,cols,tgt,name,use,kw in [("met","pima.csv",["glu","bmi","age"],"y","Pima Indians Diabetes (UCI / NIDDK)","Metabolic risk",dict(header=None,names=["preg","glu","bp","skin","ins","bmi","dpf","age","y"])),
                                   ("card","heart.csv",["age","trestbps","chol"],"target","Heart Disease (UCI Cleveland)","Cardiovascular risk",dict(encoding="utf-8-sig"))]:
        try:
            d=pd.read_csv(os.path.join(D,f),**kw)
            if key=="met": d=d[(d.glu>0)&(d.bmi>0)]
            m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)); a=cross_val_score(m,d[cols],d[tgt],cv=5,scoring="roc_auc")
            lg=lambda p:np.log(p/(1-p)); RM[key]=(m.fit(d[cols],d[tgt]),cols,lg({'met':.10,'card':.08}[key])-lg(float(d[tgt].mean()))); CARD.append(dict(dataset=name,use=use,rows=len(d),auc=round(float(a.mean()),2),features=", ".join(cols)))
        except Exception as e: print("dataset skipped:",f,e)
real_models()
def real_p(key,**v):
    if key not in RM: return None
    m,cols,sh=RM[key]; p=float(m.predict_proba(pd.DataFrame([{c:v[c] for c in cols}]))[0,1]); return float(sig(np.log(p/(1-p))+sh))*100

# ---------- synthetic demo profiles (NOT real patient data) ----------
PROFILES={
 "stable":dict(label="Stable",st=dict(age=29,h=1.75,hba1c=5.1,chol=165,family=0,smoker=0,alcohol=0),
   a=dict(rhr=64,sleep=7.7,steps=9000,weight=68,sbp=112,glucose=86,stress=3,water=2.6),b=dict(rhr=64,sleep=7.6,steps=8800,weight=68.1,sbp=112,glucose=86,stress=3,water=2.6)),
 "drift":dict(label="Hidden Drift",st=dict(age=34,h=1.68,hba1c=5.5,chol=185,family=1,smoker=0,alcohol=1),
   a=dict(rhr=72,sleep=7.5,steps=6600,weight=69.0,sbp=122,glucose=94,stress=4,water=2.4),b=dict(rhr=86,sleep=5.3,steps=3900,weight=71.2,sbp=130,glucose=102,stress=8,water=1.7)),
 "high":dict(label="Elevated Risk",st=dict(age=52,h=1.75,hba1c=6.1,chol=235,family=1,smoker=1,alcohol=2),
   a=dict(rhr=80,sleep=6.0,steps=4000,weight=88,sbp=138,glucose=108,stress=7,water=1.6),b=dict(rhr=92,sleep=5.0,steps=2800,weight=91,sbp=148,glucose=118,stress=9,water=1.2)),
}
STATE={}; USERS={}
B=lambda df:min(15,len(df)//2)
ST=lambda k:PROFILES[k]['st'] if k in PROFILES else USERS[k]['st']
NAME=lambda k:PROFILES[k]['label'] if k in PROFILES else USERS[k]['name']
NEED=10
def gen(k,n=45):
    P=PROFILES[k]; g=np.random.default_rng(abs(hash(k))%999 if False else {"stable":1,"drift":2,"high":3}[k]); rows=[]
    for d in range(n):
        f=np.clip((d-14)/(n-15),0,1)**1.15; r={}
        for c in P["a"]:
            v=P["a"][c]+(P["b"][c]-P["a"][c])*f; r[c]=v+g.normal(0,{"rhr":1.2,"sleep":.25,"steps":450,"weight":.25,"sbp":2,"glucose":2,"stress":.4,"water":.15}[c])
        rows.append(r)
    return pd.DataFrame(rows)
def get(k):
    if k not in STATE: STATE[k]=gen(k)
    return STATE[k]

IF_CACHE={}
def iforest(k,df):
    key=(k,len(df),round(float(df.iloc[:B(df)].values.sum()),3))
    if key not in IF_CACHE:
        b=df.iloc[:B(df)][["rhr","sleep","steps","weight","stress"]]; mu,sd=b.mean(),b.std()+1e-6
        IF_CACHE[key]=(IsolationForest(n_estimators=200,random_state=0).fit((b-mu)/sd),mu,sd)
    return IF_CACHE[key]

def evaluate(k,df,idx,ov=None):
    ov=ov or {}; st=dict(ST(k)); st.update({x:ov[x] for x in("smoker","alcohol") if x in ov})
    base=df.iloc[:B(df)].mean(); cur=df.iloc[max(0,idx-6):idx+1].mean()
    for x in("sleep","steps","stress","weight","water"):
        if x in ov: cur[x]=ov[x]
    bmi=cur.weight/st["h"]**2; b0=base.weight/st["h"]**2
    r=dict(rhr=cur.rhr,sleep=cur.sleep,steps=cur.steps,bmi=bmi,sbp=cur.sbp,glucose=cur.glucose,stress=cur.stress,water=cur.water,smoker=st["smoker"],
      alcohol=st["alcohol"],age=st["age"],hba1c=st["hba1c"],chol=st["chol"],family=st["family"],d_rhr=cur.rhr-base.rhr,d_sleep=cur.sleep-base.sleep,
      d_steps=cur.steps-base.steps,d_bmi=bmi-b0)
    x=SC.transform(pd.DataFrame([r])[FEATS]); prob=float(MODEL.predict_proba(x)[0,1])*100
    m,mu,sd=iforest(k,df); s=-m.score_samples(((cur[["rhr","sleep","steps","weight","stress"]]-mu)/sd).to_frame().T)[0]
    anom=lin(s,.45,.75)
    pc=lambda a,b:(a-b)/b*100
    trend=.3*lin(pc(cur.rhr,base.rhr),0,14)+.3*lin(-pc(cur.sleep,base.sleep),0,25)+.25*lin(-pc(cur.steps,base.steps),0,40)+.15*lin(pc(cur.weight,base.weight),0,4)
    cats=dict(Cardiovascular=.35*lin(cur.sbp,110,150)+.3*lin(cur.rhr,60,100)+.2*lin(st["chol"],160,260)+.15*100*st["smoker"],
      Metabolic=.3*lin(cur.glucose,85,126)+.3*lin(st["hba1c"],5,6.5)+.25*lin(bmi,22,32)+.15*100*st["family"],
      Lifestyle=.4*lin(8000-cur.steps,0,6000)+.25*lin(cur.stress,3,9)+.2*lin(st["alcohol"],0,3)+.15*100*st["smoker"],
      Recovery=.5*lin(8-cur.sleep,0,3.5)+.3*lin(cur.rhr-base.rhr,0,18)+.2*lin(cur.stress,3,9),
      Stress=.55*lin(cur.stress,2,9)+.25*lin(8-cur.sleep,0,3)+.2*lin(2.8-cur.water,0,1.4))
    pm=real_p("met",glu=cur.glucose,bmi=bmi,age=st["age"]); pc_=real_p("card",age=st["age"],trestbps=cur.sbp,chol=st["chol"])
    if pm is not None: cats["Metabolic"]=.5*cats["Metabolic"]+.5*pm
    if pc_ is not None: cats["Cardiovascular"]=.5*cats["Cardiovascular"]+.5*pc_
    sym=USERS.get(k,{}).get("sym",set()) if idx==len(df)-1 else set()
    for t in sym: c_=SYM[t][1]; cats[c_]=min(100,cats[c_]+8)
    index=int(min(100,round(16+.25*min(100,prob)+.12*anom+.18*trend+.3*np.mean(list(cats.values()))+min(10,3*len(sym)))))
    # feature contributions (logit units) vs. personal baseline state
    r0=dict(r,d_rhr=0,d_sleep=0,d_steps=0,d_bmi=0,rhr=base.rhr,sleep=base.sleep,steps=base.steps,bmi=b0,stress=df.iloc[:B(df)].stress.mean(),water=df.iloc[:B(df)].water.mean(),sbp=base.sbp,glucose=base.glucose)
    dz=MODEL.coef_[0]*(x[0]-SC.transform(pd.DataFrame([r0])[FEATS])[0])
    ct=dict(zip(FEATS,dz)); grp=dict(sleep=ct["sleep"]+ct["d_sleep"],hr=ct["rhr"]+ct["d_rhr"],activity=ct["steps"]+ct["d_steps"],weight=ct["bmi"]+ct["d_bmi"],
      stress=ct["stress"],water=ct["water"],bp=ct["sbp"],glucose=ct["glucose"])
    return dict(index=int(index),prob=round(prob,1),anomaly=round(anom),trend=round(trend),cats={a:round(b) for a,b in cats.items()},contrib=grp,cur=cur,base=base)

SYM={"fatigue":("tired|fatigue|exhaust|drained|low energy|sluggish","Stress"),"poor_sleep":("sleeping bad|poor sleep|can't sleep|cannot sleep|insomnia|sleepless|sleeping badly|restless","Recovery"),
 "stress":("stress|anxious|anxiety|overwhelm|tense|worried","Stress"),"palpitations":("palpitation|racing heart|heart racing|pounding heart","Cardiovascular"),"dizziness":("dizz|lightheaded|light-headed","Cardiovascular"),
 "thirst":("very thirsty|excess(ive)? thirst|frequent urination|urinating a lot|peeing a lot","Metabolic"),"headache":("headache|migraine","Stress")}
URGENT="chest pain|chest tightness|can't breathe|cannot breathe|short(ness)? of breath|faint|passed out|numb(ness)? (in|on)|slurred"
def band(i): return "Low" if i<30 else "Moderate" if i<60 else "High" if i<80 else "Critical"
META=dict(sleep=("Sleep","h","sleep",1),hr=("Resting HR","BPM","rhr",1),activity=("Activity","steps","steps",0),weight=("Weight","kg","weight",1),
  stress=("Stress","/10","stress",1),water=("Water","L","water",1),bp=("Systolic BP","mmHg","sbp",0),glucose=("Glucose","mg/dL","glucose",0))

def full(k):
    df=get(k); n=len(df); b=B(df); cks=[b-1+(n-b)*i//3 for i in range(4)]; ev=[evaluate(k,df,i) for i in cks]; now=ev[-1]
    tl=[dict(label=l,index=e["index"],state=sx) for l,e,sx in zip([f"{n-1-i} days ago" if i<n-1 else "Today" for i in cks],ev,["Baseline" if j==0 else "Stable" if e["index"]-ev[0]["index"]<8 else "Early change" if e["index"]-ev[0]["index"]<18 else "Pattern detected" if e["index"]-ev[0]["index"]<35 else "Significant drift" for j,e in enumerate(ev)])]
    drv=[]
    for g,(lab,u,col,dp) in META.items():
        b,c=now["base"][col],now["cur"][col]; drv.append(dict(key=g,label=lab,unit=u,base=(round(b,dp) if dp else int(round(b))),now=(round(c,dp) if dp else int(round(c))),pct=round((c-b)/b*100),score=float(now["contrib"][g])))
    pos=sum(d["score"] for d in drv if d["score"]>0) or 1
    for d in drv: d["share"]=round(max(d["score"],0)/pos*100)
    drv=sorted(drv,key=lambda d:-d["score"])
    drift=now["index"]-ev[0]["index"]>=15 and now["trend"]>=25
    top=[d for d in drv if d["score"]>0][:4]
    ph=lambda d:f'{d["label"].lower()} {"increased" if d["pct"]>0 else "decreased"} {abs(d["pct"])}%'
    story=(f'Your profile was relatively stable 30 days ago (index {ev[0]["index"]}). Over the period, '+", ".join(ph(d) for d in top[:3])+
      f'. Together these changes moved your estimated risk index from {ev[0]["index"]} to {now["index"]}.') if drift else (
      f'Your estimated risk index is {now["index"]} ({band(now["index"])}), compared with {ev[0]["index"]} 30 days ago. No significant combined drift from your personal baseline was detected.')
    c=now["cur"]; cr=now["cur"]; alert=("A recent reading is in a range that may need prompt medical attention. Please contact a healthcare professional." if (cr.sbp>=180 or cr.glucose>=250 or cr.rhr>=130) else
      "You reported symptoms that can need prompt medical attention. Please contact a healthcare professional or emergency services." if USERS.get(k,{}).get("urgent") else None)
    conf=("High" if n>=30 else "Medium" if n>=14 else "Low")
    return dict(confidence=conf,days=n,alert=alert,symptoms=sorted(USERS.get(k,{}).get("sym",[])),key=k,name=NAME(k),index=now["index"],band=band(now["index"]),delta=now["index"]-ev[0]["index"],drift=drift,cats=now["cats"],
      timeline=tl,drivers=drv,story=story,anomaly=now["anomaly"],trend=now["trend"],model_prob=now["prob"],
      series={c_:[round(v,1) for v in df[c_]] for c_ in("rhr","sleep","steps","weight")},baseline_days=B(df),
      current=dict(sleep=round(c.sleep,1),steps=int(c.steps),stress=round(c.stress,1),weight=round(c.weight,1),water=round(c.water,1),
        smoker=int(ST(k)["smoker"]),alcohol=int(ST(k)["alcohol"])),synthetic=k in PROFILES)


DB=os.environ.get("AURA_DB",os.path.join(os.path.dirname(os.path.abspath(__file__)),"aura.db"))
def q(sql,a=(),one=False):
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    try:
        cur=c.execute(sql,a); c.commit(); return cur.fetchone() if one else cur.fetchall()
    finally: c.close()
for t in ["create table if not exists users(id integer primary key,email text unique,pw blob,salt blob,name text,age real,h real,hba1c real,chol real,family int,smoker int,alcohol int)",
  "create table if not exists sessions(token text primary key,uid int)",
  "create table if not exists checkins(uid int,date text,text text,tags text,urgent int)",
  "create table if not exists translations(lang text,src text,out text,primary key(lang,src))",
  "create table if not exists logs(uid int,date text,rhr real,sleep real,steps real,weight real,stress real,water real,sbp real,glucose real,primary key(uid,date))"]: q(t)
hp=lambda pw,salt:hashlib.scrypt(pw.encode(),salt=salt,n=2**14,r=8,p=1)
def session(uid):
    t=secrets.token_hex(24); q("insert into sessions values(?,?)",(t,uid)); return t
def uid_of(auth):
    r=q("select uid from sessions where token=?",((auth or "").replace("Bearer ",""),),True)
    if not r: raise HTTPException(401,"Please sign in")
    return r[0]
def load(uid):
    u=q("select * from users where id=?",(uid,),True); rows=q("select * from logs where uid=? order by date",(uid,)); k=f"u{uid}"
    df=pd.DataFrame([dict(r) for r in rows],columns=["uid","date","rhr","sleep","steps","weight","stress","water","sbp","glucose"]).drop(columns=["uid","date"])
    if len(df): df["sbp"]=df.sbp.ffill().fillna(120); df["glucose"]=df.glucose.ffill().fillna(95); df["water"]=df.water.ffill().fillna(2.2)
    STATE[k]=df.astype(float).reset_index(drop=True)
    cutoff=(datetime.date.today()-datetime.timedelta(days=14)).isoformat(); ck=q("select tags,urgent from checkins where uid=? and date>=?",(uid,cutoff))
    USERS[k]=dict(sym={t for r in ck for t in r['tags'].split(',') if t},urgent=any(r['urgent'] for r in ck),name=u["name"],st=dict(age=u["age"],h=u["h"],hba1c=u["hba1c"],chol=u["chol"],family=u["family"],smoker=u["smoker"],alcohol=u["alcohol"])); return k
def resolve(k,auth):
    if k=="me":
        k=load(uid_of(auth))
        if len(STATE[k])<NEED: raise HTTPException(400,f"Log at least {NEED} days first")
        return k
    if k not in PROFILES: raise HTTPException(404,"Unknown profile")
    return k

app=FastAPI(title="AURA API",description="Prototype for health-risk awareness; not a diagnosis.")
class Signup(BaseModel):
    email:str; password:str; name:str; age:float; height_cm:float; family:int=0; smoker:int=0; alcohol:int=0; hba1c:Optional[float]=None; chol:Optional[float]=None
class Login(BaseModel):
    email:str; password:str
class Entry(BaseModel):
    date:Optional[str]=None; rhr:float; sleep:float; steps:float; weight:float; stress:float; water:Optional[float]=None; sbp:Optional[float]=None; glucose:Optional[float]=None
class Check(BaseModel):
    text:str
class Csv(BaseModel):
    csv:str
class Sim(BaseModel):
    profile:str="drift"; sleep:float; steps:float; stress:float; weight:float; water:float=2.4; smoker:int=0; alcohol:int=0
class Ask(BaseModel):
    profile:str="drift"; q:str

@app.post("/api/auth/signup")
def signup(s:Signup):
    em=s.email.strip().lower()
    if "@" not in em or len(s.password)<8 or not 10<=s.age<=110 or not 120<=s.height_cm<=230 or not s.name.strip(): raise HTTPException(400,"Check your details (password 8+ characters)")
    salt=secrets.token_bytes(16)
    try: q("insert into users(email,pw,salt,name,age,h,hba1c,chol,family,smoker,alcohol) values(?,?,?,?,?,?,?,?,?,?,?)",(em,hp(s.password,salt),salt,s.name.strip(),s.age,s.height_cm/100,s.hba1c or 5.3,s.chol or 180,s.family,s.smoker,s.alcohol))
    except sqlite3.IntegrityError: raise HTTPException(409,"That email is already registered")
    return dict(token=session(q("select id from users where email=?",(em,),True)[0]))
@app.post("/api/auth/login")
def login(l:Login):
    u=q("select * from users where email=?",(l.email.strip().lower(),),True)
    if not u or not secrets.compare_digest(hp(l.password,u["salt"]),u["pw"]): raise HTTPException(401,"Wrong email or password")
    return dict(token=session(u["id"]))
@app.post("/api/auth/logout")
def logout(authorization:str=Header(None)): q("delete from sessions where token=?",((authorization or "").replace("Bearer ",""),)); return {}
def save(uid,e:Entry):
    try: d=datetime.date.fromisoformat(e.date) if e.date else datetime.date.today()
    except ValueError: raise HTTPException(400,"Bad date")
    if not(30<e.rhr<220 and 0<=e.sleep<=24 and 0<=e.steps<=100000 and 20<e.weight<400 and 1<=e.stress<=10): raise HTTPException(400,"A value is outside a plausible range")
    q("insert or replace into logs values(?,?,?,?,?,?,?,?,?,?)",(uid,d.isoformat(),e.rhr,e.sleep,e.steps,e.weight,e.stress,e.water,e.sbp,e.glucose))
@app.get("/api/me")
def me(authorization:str=Header(None)):
    uid=uid_of(authorization); k=load(uid); n=len(STATE[k])
    return dict(status="building",logged=n,need=NEED,user=USERS[k]["name"]) if n<NEED else full(k)
@app.post("/api/me/log")
def me_log(e:Entry,authorization:str=Header(None)): save(uid_of(authorization),e); return {}
@app.post("/api/me/import")
def me_import(c:Csv,authorization:str=Header(None)):
    uid=uid_of(authorization)
    try: d=pd.read_csv(io.StringIO(c.csv)).head(2000)
    except Exception: raise HTTPException(400,"Could not read that CSV")
    d.columns=[x.strip().lower() for x in d.columns]
    if not {"rhr","sleep","steps","weight","stress"}<=set(d.columns): raise HTTPException(400,"CSV needs columns: date,rhr,sleep,steps,weight,stress (optional: water,sbp,glucose)")
    base=datetime.date.today()-datetime.timedelta(days=len(d)-1)
    for i,r in enumerate(d.to_dict("records")):
        r={k:(None if pd.isna(v) else v) for k,v in r.items()}
        try: dt=pd.to_datetime(r.get("date")).date().isoformat() if r.get("date") else (base+datetime.timedelta(days=i)).isoformat()
        except Exception: raise HTTPException(400,f"Bad date in row {i+2}")
        save(uid,Entry(date=dt,**{k:r.get(k) for k in("rhr","sleep","steps","weight","stress","water","sbp","glucose")}))
    return dict(imported=len(d))
@app.delete("/api/me")
def me_del(authorization:str=Header(None)):
    uid=uid_of(authorization)
    for t,c in(("logs","uid"),("sessions","uid"),("users","id")): q(f"delete from {t} where {c}=?",(uid,))
    return {}
@app.get("/api/model-card")
def model_card(): return dict(sources=CARD,note="Sub-models use only the features shared with each public dataset (cross-validated AUC, 5-fold). Probabilities are prior-shifted to an assumed population prevalence (diabetes 10%, heart disease 8%) because the datasets over-represent patients. Public datasets are small, historical and not representative of every population; the dataset glucose is an OGTT value and approximates home readings. The risk index also blends a synthetic-trained baseline model and personal drift signals.")
@app.post("/api/checkin")
def checkin(c:Check,authorization:str=Header(None)):
    t=c.text.lower()[:600]; tags=[k for k,(p,_) in SYM.items() if re.search(p,t)]; urgent=bool(re.search(URGENT,t))
    saved=False
    if authorization:
        try: uid=uid_of(authorization); q("insert into checkins values(?,?,?,?,?)",(uid,datetime.date.today().isoformat(),c.text[:600],",".join(tags),int(urgent))); saved=True
        except HTTPException: pass
    return dict(tags=tags,labels={k:SYM[k][1] for k in tags},urgent=urgent,saved=saved)
LANGN=dict(en="English",hi="Hindi",kn="Kannada",ta="Tamil",te="Telugu",bn="Bengali",mr="Marathi",gu="Gujarati",ml="Malayalam",pa="Punjabi (Gurmukhi script)")
HITS={}
def ai_translate(lang,texts):
    key=os.environ.get("ANTHROPIC_API_KEY")
    if not key: raise HTTPException(503,"Live translation is not configured (set ANTHROPIC_API_KEY)")
    body=json.dumps(dict(model=os.environ.get("AURA_MODEL","claude-haiku-5-5"),max_tokens=8000,system=f"Translate every string in the JSON array into {LANGN[lang]}. Keep numbers, units, emoji, arrows and the product name AURA unchanged. Keep medical wording cautious and add no new claims. Return ONLY a JSON array of the same length.",
      messages=[dict(role="user",content=json.dumps(texts,ensure_ascii=False))])).encode()
    rq=urllib.request.Request("https://api.anthropic.com/v1/messages",body,{"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        out=json.loads(urllib.request.urlopen(rq,timeout=60).read())["content"][0]["text"].strip()
        arr=json.loads(re.sub(r"^```(?:json)?|```$","",out).strip())
    except Exception: raise HTTPException(502,"Translation service error")
    if not isinstance(arr,list) or len(arr)!=len(texts): raise HTTPException(502,"Translation service returned an unexpected result")
    return [str(x) for x in arr]
class Tr(BaseModel):
    lang:str; texts:List[str]
@app.post("/api/translate")
def translate(t:Tr,request:Request):
    if t.lang not in LANGN or len(t.texts)>60 or any(len(x)>3000 for x in t.texts): raise HTTPException(400,"Unsupported request")
    got={}
    if t.texts: got={r["src"]:r["out"] for r in q(f"select src,out from translations where lang=? and src in ({','.join('?'*len(t.texts))})",(t.lang,*t.texts))}
    miss=[x for x in dict.fromkeys(t.texts) if x not in got]
    if miss:
        ip=request.client.host if request.client else "?"; now=time.time(); h=[x for x in HITS.get(ip,[]) if now-x<3600]
        if len(h)>=30: raise HTTPException(429,"Translation limit reached, please try again later")
        HITS[ip]=h+[now]
        for src,o in zip(miss,ai_translate(t.lang,miss)): got[src]=o; q("insert or replace into translations values(?,?,?)",(t.lang,src,o))
    return dict(out=[got[x] for x in t.texts])
@app.get("/api/plan/{k}")
def plan(k:str,authorization:str=Header(None)):
    k=resolve(k,authorization); df=get(k); n=len(df); cur=evaluate(k,df,n-1); c=cur["cur"]; st=ST(k); bmi=c.weight/st["h"]**2
    opts=[("sleep","More sleep","+1 hour",dict(sleep=min(8.5,c.sleep+1)),c.sleep<8),("steps","Walk more","+3,000 steps/day",dict(steps=min(10000,c.steps+3000)),c.steps<9000),
      ("stress","Lower stress","−2 points",dict(stress=max(2,c.stress-2)),c.stress>3),("weight","Healthy weight","−2 kg",dict(weight=c.weight-2),bmi>=23),
      ("water","Hydration","+0.5 L/day",dict(water=min(3,c.water+.5)),c.water<2.8),("alcohol","Less alcohol","one level lower",dict(alcohol=max(0,int(st["alcohol"])-1)),st["alcohol"]>0),
      ("smoker","Quit smoking","stop",dict(smoker=0),st["smoker"]>0)]
    out=[]
    for key,label,txt,ov,ok in opts:
        if not ok: continue
        d=cur["index"]-evaluate(k,df,n-1,ov)["index"]
        if d>0: out.append(dict(key=key,label=label,change=txt,delta=int(d),sim=int(cur["index"]-d),ov={a:round(float(b),1) for a,b in ov.items()}))
    out.sort(key=lambda x:-x["delta"]); top=out[:3]; mv={}
    for i in top: mv.update(i["ov"])
    comb=dict(sim=int(evaluate(k,df,n-1,mv)["index"]),ov=mv) if top else None
    return dict(today=cur["index"],items=out[:4],combined=comb)
RNG=dict(glucose=(40,600),hba1c=(3,15),sbp=(70,250),chol=(80,500),bmi=(10,70))
PAT=dict(glucose=r"(?:glucose|sugar)[^\d]{0,30}(\d{2,3}(?:\.\d)?)",hba1c=r"hba1c[^\d]{0,20}(\d{1,2}(?:\.\d)?)",chol=r"(?:total\s+)?cholesterol[^\d]{0,25}(\d{2,3})",sbp=r"(?:blood\s+pressure|\bbp\b)[^\d]{0,15}(\d{2,3})\s*/\s*\d{2,3}",bmi=r"\bbmi\b[^\d]{0,15}(\d{2}(?:\.\d)?)")
def clean(d): return {k:round(float(v),1) for k,v in d.items() if k in RNG and v is not None and RNG[k][0]<=float(v)<=RNG[k][1]}
def regex_labs(t):
    t=t.lower(); out={}
    for k,p in PAT.items():
        m=re.search(p,t)
        if m: out[k]=float(m.group(1))
    return clean(out)
def ai_labs(b64,mt):
    key=os.environ.get("ANTHROPIC_API_KEY")
    if not key: raise HTTPException(503,"Reading scans and photos needs ANTHROPIC_API_KEY. You can use the sample report instead.")
    blk=dict(type="document" if mt=="application/pdf" else "image",source=dict(type="base64",media_type=mt,data=b64))
    pr="This is a medical lab report. Extract ONLY values explicitly printed in it: glucose (mg/dL; convert from mmol/L x18 if needed), hba1c (%), sbp (systolic blood pressure, mmHg), chol (total cholesterol mg/dL; convert from mmol/L x38.67 if needed), bmi. Return ONLY a JSON object with keys glucose, hba1c, sbp, chol, bmi; use null for anything not present. Never guess or infer."
    body=json.dumps(dict(model=os.environ.get("AURA_MODEL","claude-haiku-5-5"),max_tokens=400,messages=[dict(role="user",content=[blk,dict(type="text",text=pr)])])).encode()
    rq=urllib.request.Request("https://api.anthropic.com/v1/messages",body,{"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        txt=json.loads(urllib.request.urlopen(rq,timeout=60).read())["content"][0]["text"]; return clean(json.loads(re.search(r"\{.*\}",txt,re.S).group(0)))
    except Exception: raise HTTPException(502,"Could not read that report")
class Rep(BaseModel):
    data:str; media_type:str
class Labs(BaseModel):
    glucose:Optional[float]=None; hba1c:Optional[float]=None; sbp:Optional[float]=None; chol:Optional[float]=None
@app.get("/api/report/sample")
def report_sample(): return dict(values=dict(glucose=112.0,hba1c=5.9,sbp=134.0,chol=214.0,bmi=26.4),source="sample")
@app.post("/api/report/extract")
def report_extract(r:Rep,request:Request):
    if r.media_type not in("application/pdf","image/png","image/jpeg","image/webp") or len(r.data)>8_000_000: raise HTTPException(400,"Upload a PDF or a PNG/JPG/WebP image under 5 MB")
    ip="r"+(request.client.host if request.client else "?"); now=time.time(); h=[x for x in HITS.get(ip,[]) if now-x<3600]
    if len(h)>=10: raise HTTPException(429,"Report limit reached, please try again later")
    HITS[ip]=h+[now]; vals={}; src="ai"
    if r.media_type=="application/pdf":
        try:
            import pypdf,base64; vals=regex_labs(" ".join((p.extract_text() or "") for p in pypdf.PdfReader(io.BytesIO(base64.b64decode(r.data))).pages[:6])); src="text"
        except Exception: vals={}
    if len(vals)<2:
        try: vals=ai_labs(r.data,r.media_type); src="ai"
        except HTTPException:
            if not vals: raise
    if not vals: raise HTTPException(422,"No lab values found in that file")
    return dict(values=vals,source=src)
@app.post("/api/me/labs")
def me_labs(l:Labs,authorization:str=Header(None)):
    uid=uid_of(authorization); v=clean(l.model_dump())
    if not v: raise HTTPException(400,"No valid values to apply")
    if "hba1c" in v: q("update users set hba1c=? where id=?",(v["hba1c"],uid))
    if "chol" in v: q("update users set chol=? where id=?",(v["chol"],uid))
    for k in("sbp","glucose"):
        if k in v: q(f"update logs set {k}=? where uid=? and {k} is null",(v[k],uid))
    return dict(applied=v)
@app.get("/api/profiles")
def profiles(): return {k:v["label"] for k,v in PROFILES.items()}
@app.get("/api/profile/{k}")
def profile(k:str): return full(resolve(k,None))
@app.post("/api/simulate")
def simulate(s:Sim,authorization:str=Header(None)):
    k=resolve(s.profile,authorization); df=get(k); n=len(df); e=evaluate(k,df,n-1,s.model_dump()); cur=evaluate(k,df,n-1); prev=evaluate(k,df,max(B(df),n-11))
    slope=max(0,(cur["index"]-prev["index"])/max(1,min(10,n-1-B(df)))*.25); xs=list(range(0,31,3))
    return dict(today=cur["index"],simulated=e["index"],cats=e["cats"],x=xs,current=[min(100,round(cur["index"]+slope*x)) for x in xs],
      sim=[round(cur["index"]+(e["index"]-cur["index"])*(1-np.exp(-x/9))) for x in xs],label="AI scenario estimate — not a guaranteed medical outcome")
@app.post("/api/ask")
def ask(a:Ask,authorization:str=Header(None)):
    f=full(resolve(a.profile,authorization)); t0=a.q.lower(); top=[d for d in f["drivers"] if d["score"]>0][:3]
    ch=lambda d:f'{d["label"]} {d["base"]}→{d["now"]} {d["unit"]} ({d["pct"]:+}%)'
    if "month" in t0 or "change" in t0: t=f["story"]
    elif any(w in t0 for w in("improve","first","factor","biggest")):
        t=f'The largest contributor to your estimated risk right now is {top[0]["label"].lower()} ({ch(top[0])}), about {top[0]["share"]}% of the model-attributed increase. Use Future You to test it.' if top else "No factor is currently pushing your estimate above your baseline."
    elif any(w in t0 for w in("why","increas","risk")):
        t=("Your estimate moved because: "+"; ".join(ch(d) for d in top)+f". The combined pattern (trend signal {f['trend']}/100, anomaly {f['anomaly']}/100) matters more than any single reading.") if top else "Your readings are close to your personal baseline, so risk is steady."
    else: t=f'Your AURA index is {f["index"]}/100 ({f["band"]}). Ask why it changed, what changed this month, or what to improve first.'
    return dict(answer=t+" (Model-generated estimate, not a diagnosis.)")
@app.get("/api/summary/{k}")
def summary(k:str,authorization:str=Header(None)):
    f=full(resolve(k,authorization)); L=[f'AURA HEALTH SUMMARY — {f["name"]}'+(" (SYNTHETIC SAMPLE DATA)" if f["synthetic"] else ""),f'Risk index: {f["index"]}/100 ({f["band"]}); {f["delta"]:+} over the logged period','Categories: '+", ".join(f"{a} {b}" for a,b in f["cats"].items()),
      "Key trends:"]+[f'- {d["label"]}: {d["base"]} → {d["now"]} {d["unit"]} ({d["pct"]:+}%)' for d in f["drivers"][:4]]+["Story: "+f["story"],"Questions to consider discussing with a clinician:",
      "- Could my sleep/activity/heart-rate trend have an underlying cause?","- Which checks or lifestyle changes are appropriate for me?","","This report is for informational and decision-support purposes and is not a medical diagnosis."]
    return dict(text="\n".join(L))
@app.get("/")
def home(): return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","frontend","index.html"))
