from main import train, FEATS
sc,m=train(); print("Standardised coefficients (log-odds per SD):")
for f,c in sorted(zip(FEATS,m.coef_[0]),key=lambda t:-abs(t[1])): print(f"{f:10s}{c:+.2f}")
