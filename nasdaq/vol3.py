"""vol2 결과 검증: (a) 3등분 문턱을 '그날 이전 예측값'으로만 정함(미래참조 제거) (b) 뒤섞기 시험 (c) 왜 그런가: ② 방향/평균"""
import io, contextlib, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    import vol2 as v
Nn, O = v.Nn, v.O
# (a) 확장창 문턱: 각 날짜에서 그 이전 예측값들의 33/67% 분위
vp = Nn.vpred
lo = vp.expanding(min_periods=60).quantile(1/3).shift(1); hi = vp.expanding(min_periods=60).quantile(2/3).shift(1)
grade = pd.Series(np.select([vp <= lo, vp >= hi], [0, 2], 1), index=vp.index).where(lo.notna())
J = O.join(pd.DataFrame({"g": grade, "r2": Nn.r2, "vp": vp}), how="inner").dropna(subset=["g"])
p = J["우리_08시_판당"]
print(f"[a] 미래참조 없는 등급 (우리 밤 {len(J)}밤)")
for g_, nm in ((0, "예측 작음"), (1, "예측 중간"), (2, "예측 큼")):
    s = J[J.g == g_]
    print(f"    {nm}: {len(s)}밤 판당 {s['우리_08시_판당'].mean():+.3f}% 이김 {(s['우리_08시_판당']>0).mean()*100:4.1f}% | 나스닥 ② 평균 {s.r2.mean()*100:+.3f}% 상승 {(s.r2>0).mean()*100:4.1f}%")
for nm, mm in (("앞절반", J.index < J.index[len(J)//2]), ("뒤절반", J.index >= J.index[len(J)//2]), ("애프터", J.index >= "2026-09-15")):
    s = J[mm]; print(f"    {nm}: " + " / ".join(f"{s[s.g==g_]['우리_08시_판당'].mean():+.2f}%({(s.g==g_).sum()})" for g_ in (0, 1, 2)))
# (b) 뒤섞기: (큼 - 작음) 차이가 우연으로 나올 확률
diff = p[J.g == 2].mean() - p[J.g == 0].mean()
rng = np.random.default_rng(1); g = J.g.values; pv = p.values
sh = [pv[rng.permutation(len(pv))][g == 2].mean() - pv[rng.permutation(len(pv))][g == 0].mean() for _ in range(5000)]
print(f"\n[b] 판당 차이(예측 큼 − 작음) {diff:+.3f}%p, 뒤섞기 p = {np.mean(np.array(sh) >= diff):.4f}")
# 판 크기 조절 (미래참조 없는 등급)
for nm, w in {"전부 1배": 1.0, "작음 0.5배": np.where(g == 0, .5, 1), "작음 0·쉼": np.where(g == 0, 0, 1),
              "작음 0.5·큼 1.5": np.select([g == 0, g == 2], [.5, 1.5], 1)}.items():
    w = np.broadcast_to(w, pv.shape); r = pv * w
    print(f"    {nm:14s} 합계 {r.sum():+6.1f}%  투입 1당 {r.sum()/w.sum():+.3f}%  밤당 표준편차 {r.std():.2f}")
# (c) 어떤 지표가 '예측 큼'을 만드나 (등급별 평균)
N2 = Nn.join(grade.rename("g")).dropna(subset=["g"])
print("\n[c] 등급별 지표 평균 (20:00 기준)")
for c in ("vix", "vix_term", "ma20gap", "p_nq", "h5", "absgap", "mon"):
    print(f"    {c:9s} " + " / ".join(f"{N2[N2.g==g_][c].mean():+.4f}" for g_ in (0, 1, 2)))
out = pd.DataFrame({"d": Nn.index.strftime("%Y%m%d"), "예측출렁임_%": (vp * 100).round(3),
                    "등급": grade.map({0: "예측작음", 1: "예측중간", 2: "예측큼"}), "실제_②_%": (Nn.r2 * 100).round(3)})
out.to_csv("night_volgrade.csv", index=False, encoding="utf-8-sig")
print("\n저장: night_volgrade.csv (날짜별 예측 출렁임·등급, 미래참조 없음)")
