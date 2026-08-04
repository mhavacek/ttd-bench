# Citace k dohledání

Fronta zdrojů, které text potřebuje a které zatím **nemáme fyzicky staženy a
přečteny**. Dokud položka nemá PDF v `paper/refs/`, nesmí být v
`paper/references.bib` a odpovídající věta se v článku nepíše — v textu zůstane
`[NEEDS-REF: …]`.

Formát položky: **tvrzení** → **jaká práce** → **kde v článku na tom stojí**.

---

## 0. Přečteno, ale chybí soubor a plná metadata (blokuje první .bib položky)

Pět prací, které smím citovat, ale ke kterým **v repozitáři není PDF**. Bez PDF
v `paper/refs/` nemůže vzniknout ani jedna položka bibliografie. Potřebuji je
dodat, nebo potvrdit, kde je mám stáhnout.

| # | práce | k čemu v článku |
|---|---|---|
| R1 | Havacek et al., *Deep learning for security-relevant event detection in visual data*, Artificial Intelligence Review, 2026 | §2.2 stav detekce zbraní; §1 rámec |
| R2 | *Rethinking Metrics and Benchmarks of Video Anomaly Detection* (5/2025), zavádí LaAP | §1 pozice vůči literatuře; §2.1; §5 vztah ke kritikám |
| R3 | *Benchmark AUC Is Not Deployable Reliability* (6/2026) | §2.1; §5 |
| R4 | *We Need to Rethink Benchmarking in Anomaly Detection* (7/2025) | §2.1; §5 |
| R5 | *From Frames to Events* (4/2026), event-level tIoU/F1 | §2.1; §3.1 zdůvodnění event-level formulace |

U R2–R5 potřebuji navíc **přesné bibliografické údaje** (autoři, plný název,
venue, DOI/arXiv ID) — ze zkratek v zadání je sestavit nelze a hádat je zakázané.

---

## 1. Introduction

**I1 — provozní požadavek na reakční čas.**
Tvrzení: detekce zbraně v CCTV je systém reálného času a užitečnost se měří
časem do poplachu, ne AUC.
Potřebuji: práci nebo normu/technickou zprávu, která uvádí konkrétní provozní
požadavek na reakční dobu bezpečnostního systému (autor/název/DOI neznám).
Stojí na tom: úvodní odstavec §1.
Bez zdroje: větu přeformulovat na neopřenou motivaci bez čísla, nebo škrtnout.

---

## 2. Related work

**RW1 — benchmarky detekce zbraní ve videu.**
Tvrzení: existující práce vykazují per-frame/per-clip metriky a TTD nevykazují.
Potřebuji: 3–5 reprezentativních prací detekce zbraní ve videu (pokud je pokrývá
přehledovka R1, může stačit odkaz na ni + 2 primární zdroje).
Stojí na tom: §2.2, druhá věta.
Bez zdroje: tvrzení oslabit na „v pracích shrnutých v [R1] se TTD nevykazuje",
a to jen pokud to R1 skutečně říká — nutno ověřit v textu R1.

**RW2 — streaming perception / latency-aware evaluace.**
Tvrzení: existuje linie práce, která bere v úvahu, že se scéna během inference
mění, ale nepokrývá cestu nasazení.
Potřebuji: zakládající práci streaming perception (tuším Li et al., *Towards
Streaming Perception*, ECCV 2020 — **neověřeno, necitovat, dokud PDF nemám**).
Stojí na tom: celý §2.3.
Bez zdroje: §2.3 se **nepíše** (odstavec vypadne, related work bude tříodstavcové).

**RW3 — mTTA v anticipaci nehod.**
Tvrzení: sousední doména používá mean time-to-accident, které průměruje přes
prahy, takže má tutéž díru — chybí provozní bod.
Potřebuji: původní práci zavádějící mTTA + jednu recentní, která ho používá.
Stojí na tom: §2.4, jeden odstavec; a §5 bod 4.
Bez zdroje: §2.4 vypadne. Tvrzení o mTTA **nesmí** zůstat bez citace, protože
je to tvrzení o cizí metrice.

---

## 3. Methods

**M1 — CONSORT jako vzor pro vykazování dispozice.**
Tvrzení: klinické studie musí vykázat, co se stalo s každým subjektem; replay
benchmarky to nedělají téměř nikdy.
Potřebuji: CONSORT statement (Schulz et al. 2010 nebo aktuální revizi).
Stojí na tom: §4.1 a zdůvodnění obr. 2; formulace je dnes v docstringu
`consort_figure.py`.
Bez zdroje: analogii uvést bez odkazu jako vlastní volbu formátu.

**M2 — survival analýza / Cox s robustními SE clusterovanými.**
Tvrzení: shluknutá pozorování vyžadují robustní SE.
Potřebuji: standardní referenci (Lin & Wei 1989 nebo učebnice).
Stojí na tom: §3.7 bod 3.
Bez zdroje: uvést jen implementaci (`lifelines`) bez metodologické citace —
přijatelné, ale slabší.

**M3 — Kaplan–Meier.**
Potřebuji: původní práci nebo standardní učebnici.
Stojí na tom: §3.7 bod 1.

---

## 4. Discussion

**D1 — provozní bod / kalibrace na rozpočet falešných poplachů.**
Tvrzení: srovnání modelů při pevném prahu srovnává kalibraci, ne architekturu.
Potřebuji: práci o kalibraci detektorů nebo o FP-matched srovnání.
Stojí na tom: §4.7 závěr a §5 bod 4.
Bez zdroje: tvrzení zůstane podepřené vlastními daty `[SW]` — to stačí, citace
je nice-to-have.

---

## Uzavřené položky

*(zatím žádné)*
