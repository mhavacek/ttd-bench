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

**I2 — „FAR za kamerohodinu vykazují 2 z 35 studií".**
Tvrzení: provozní jednotka falešných poplachů se v oboru téměř nevykazuje;
podklad pro to, proč je D9 příspěvek, ne patch.
Potřebuji: R1 (vlastní přehledovka) jako PDF **a ověření, že to R1 skutečně
takto uvádí** — číslo dnes žije jen v `ANALYSIS-PLAN-PHASE2.md` §D9.
Stojí na tom: §1 (odůvodnění jednotky), §4.7 nová sekce.
Bez zdroje: číslo z textu vypustit a nechat jen kvalitativní „vykazuje se
zřídka" — bez čísla, protože číslo bez ověřitelného zdroje je totéž jako
vymyšlená citace.

**I3 — čtyři evaluační kritiky (R2–R5) v úvodu.**
Tvrzení: „čtyři evaluační kritiky za 14 měsíců konvergují ke stejné diagnóze."
Potřebuji: PDF R2–R5 a jejich přesná data vydání (tvrzení o časovém okně
„14 měsíců" je ověřitelné jen s nimi).
Stojí na tom: §1 druhý odstavec.
Bez zdroje: odstavec se nepíše; úvod přeskočí rovnou na páteřní tezi.

---

## 2. Related work

**RW1 — benchmarky detekce zbraní ve videu.**
Tvrzení: existující práce vykazují per-frame/per-clip metriky a TTD nevykazují.
Potřebuji: 3–5 reprezentativních prací detekce zbraní ve videu (pokud je pokrývá
přehledovka R1, může stačit odkaz na ni + 2 primární zdroje).
Stojí na tom: §2.2, druhá věta.
Bez zdroje: tvrzení oslabit na „v pracích shrnutých v [R1] se TTD nevykazuje",
a to jen pokud to R1 skutečně říká — nutno ověřit v textu R1.

**RW2 — UZAVŘENO.** `li2020streaming` ověřen a §2.3 napsána.

**RW3 — mTTA v anticipaci nehod. ČÁSTEČNĚ UZAVŘENO, ale tvrzení bylo ŠPATNĚ.**
Původní práci máme (`chan2016anticipating`), jenže **ona provozní bod uvádí** —
reportuje křivku average TTA vs. recall a konkrétní bod 1,8559 s při 80 %
recall a 56,14 % precision. Kritika „mTTA průměruje přes prahy" tedy na Chan
et al. nesedí a nesmí jim být připsána; §2.4 je teď napsaná tak, že je cituje
jako **pozitivní precedens**.
Zbývá: **jedna recentní práce anticipace nehod, která reportuje mTTA
zprůměrované přes prahy** — bez ní se ta kritika nesmí vyslovit vůbec.
Stojí na tom: §2.4 druhý odstavec; §5 bod 4.

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

**2026-08-16 — sedm PDF doplněno, klíče a `.bib` hotové.**
R1 `havacek2026air` (dodal uživatel), R2 `liu2025rethinking` (arXiv 2505.19022),
R3 `rashidi2026auc` (2606.29506), R4 `rochner2025benchmarking` (2507.15584),
R5 `rashvand2026frames` (2604.09327), RW2 `li2020streaming` (2005.10420),
RW3 `chan2016anticipating` (ACCV 2016, kapitola vyříznuta ze sborníku LNCS 10114,
str. 136–153; opraveno 2026-09-21 podle tištěných čísel stran v PDF, dřív tu
chybně stálo 145–162, `.bib` byl správně). U R2–R5 se název shoduje přesně se zadáním, identita je jistá.

**I2 UZAVŘENA — a tvrzení bylo ŠPATNĚ.** Ověřeno proti R1: přehledovka NEuvádí
„FAR/h vykazují 2 z 35 studií". Uvádí, že z **36** reprezentativních studií
vykazuje TTD **0 %**, FAR-typ hlášení falešných poplachů **3 z 36 (8 %)** a
falešné poplachy za provozní hodinu **žádná studie**. Opraveno v D9
i v úvodu; číslo 2/35 se v článku nesmí objevit.

**2026-09-13 — I1, RW1 a RW3 uzavřeny bez nového zdroje** (pre-submission review,
nálezy B1–B3, `paper/review/REPORT.md`).
- **I1:** úvod netvrdí žádný číselný provozní požadavek, takže zdroj není potřeba.
  Věta zmírněna na „rarely what the literature reports", značka odstraněna.
- **RW1:** §2.2 zúžena na vzorek R1 („the review's sample offers no TTD value").
  Primární práce o detekci zbraní zůstávají žádoucí, hlavně **Olmos, Tabik, Herrera,
  *Neurocomputing* 275 (2018), doi:10.1016/j.neucom.2017.05.012**. Je to blízký
  precedent (čas do aktivace alarmu po pěti po sobě jdoucích TP) a bez něj zůstává
  nález B6 otevřený. PDF je za paywallem, přístup přes knihovnu VŠB-TUO.
- **RW3:** věta o mTTA průměrovaném přes prahy z §2.4 vypuštěna.
- **TRECVID 2019 overview staženo 2026-09-13 se schválením uživatele** →
  `refs/awad2020trecvid.pdf` (Awad et al., arXiv:2009.09984v1, 39 stran, 15,2 MB).
  Klíč `awad2020trecvid`, autoři opsáni z první strany PDF. Použito pro vymezení
  jednotky falešných poplachů za čas videa (nález B4). Doklad, s. 20:
  „RFA(τ) = NFA(τ) / VideoDurInMinutes"; ActEV18 hodnocen „on the operating
  points; Pmiss at RFA = 0.15 and Pmiss at RFA = 1". TRECVID SED („per
  camera-hour") tento PDF nedokládá, proto se necituje.

---

## Kandidáti ke stažení (dohledáno 2026-09-21, review M13, M14, B6)

**Stav k 2026-09-21: 12 položek staženo se souhlasem uživatele** (všechny kromě
`fawcett1999activity`), metadata ověřena v PDF a doplněna do `.bib`, věty v
textu napsány z přečtených pasáží (review M13, M14, B6). `fawcett1999activity`
zůstává: ACM blokuje automatické stažení, a proto se necituje. Kontrola
`check_section_numbers.py` teď strojově hlídá pravidlo „citovaný klíč ⇔ .bib ⇔
PDF v refs/“.
Metadata jsou ověřená na stránce arXivu, CVF, JOSS nebo ACM. Velikost je
z HTTP hlavičky. Tvrzení do textu se píšou až po přečtení staženého PDF.

**Volně dostupné:**

| navržený klíč | práce | k čemu | zdroj | velikost |
|---|---|---|---|---|
| aremu2022ssivd | Aremu et al., *SSIVD-Net …* (zavádí SCVD), arXiv:2207.12850v8 | §3.5, §3.7.5 dataset | arxiv.org/pdf/2207.12850 | 6,6 MB |
| sultani2018realworld | Sultani, Chen, Shah, *Real-World Anomaly Detection in Surveillance Videos*, CVPR 2018 | §3.5, §3.7.5 UCF-Crime | openaccess.thecvf.com | 3,4 MB |
| zhao2024detrs | Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR), CVPR 2024 | §3.4 | openaccess.thecvf.com | 1,5 MB |
| tian2025yolov12 | Tian, Ye, Doermann, *YOLOv12: Attention-Centric Real-Time Object Detectors*, arXiv:2502.12524 | §3.4 | arxiv.org | 2,9 MB |
| jocher2026yolo26 | Jocher et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748. Oficiální citace Ultralytics podle CITATION.cff jejich repozitáře; YOLOv8 vlastní článek nemá | §3.4 (yolov8, yolov26m, backend) | arxiv.org | 9,6 MB |
| lin2014coco | Lin et al., *Microsoft COCO: Common Objects in Context*, arXiv:1405.0312 | §3.4 (COCO-pretrained) | arxiv.org | 8,1 MB |
| davidsonpilon2019lifelines | Davidson-Pilon, *lifelines: survival analysis in Python*, JOSS 4(40) 1317, 2019 | §3.7.2–3 (implementace KM, Cox a robustních SE) | theoj.org | 0,14 MB |
| olmos2017handgun | Olmos, Tabik, Herrera, *Automatic Handgun Detection Alarm in Videos Using Deep Learning*, arXiv:1702.05147 (preprint článku v Neurocomputing 2018) | §2.2, B6 | arxiv.org | 3,3 MB |
| doshi2020online | Doshi, Yilmaz, *Online Anomaly Detection in Surveillance Videos with Asymptotic Bounds on False Alarm Rate*, arXiv:2010.07110 (preprint článku v Pattern Recognition) | §2.1, §5.5 | arxiv.org | 3,2 MB |
| lavin2015nab | Lavin, Ahmad, *Evaluating Real-time Anomaly Detection Algorithms – the Numenta Anomaly Benchmark*, arXiv:1510.03336 (ICMLA 2015) | §2.1 | arxiv.org | 2,4 MB |
| fawcett1999activity | Fawcett, Provost, *Activity monitoring: noticing interesting changes in behavior*, KDD '99, s. 53–62 | §2 (AMOC) | dl.acm.org (ACM je od 2026 open access; automatické stažení blokuje) | ? |
| hoai2012mmed | Hoai, De la Torre, *Max-Margin Early Event Detectors*, CVPR 2012 | §2.4 | robots.ox.ac.uk/~minhhoai (stránka autora) | 0,69 MB |
| yang2022streaming | Yang et al., *Real-Time Object Detection for Streaming Perception*, CVPR 2022 | §2.3 | openaccess.thecvf.com | 5,8 MB |

**Jen přes knihovnu VŠB (za paywallem):**
- Kaplan & Meier 1958 (JASA);
- Cox 1972 (JRSS B);
- Lin & Wei 1989 (JASA);
- Garwood 1936 (Biometrika);
- Schuirmann 1987 (J. Pharmacokinet. Biopharm.). Na Zenodu (rec. 1232484) je kopie označená CC0, ale taková licence u článku ze Springeru z roku 1987 je nevěrohodná, proto ji nedoporučuju.

Volné kopie KM a Cox leží na stránkách cizích kurzů. Jako zdroj je nepoužíváme.

**Doplněno 2026-09-21 odpoledne:**

- **`fawcett1999activity` staženo.** Uživatel PDF stáhl ručně z ACM DL a já ho zkopíroval do `refs/`. Ověřeno: KDD '99, s. 53–62, 10 stran.
  - Důsledek pro text: AMOC už v roce 1999 páruje včasnost alarmu s mírou falešných poplachů normalizovanou za hodinu.
  - §5.5 proto už netvrdí, že tohle spárování je náš přínos. Přínos je zúžený na samotné měření: čas v sekundách od onsetu, cenzorovaný, přes nasazenou streamovou cestu, při kalibrované míře poplachů.
  - §2.1 a §5.4 citují AMOC a jejich argument, že vítěz v ROC nemusí být lepší monitor.
- **Primární statistické zdroje nebudou.** Jde o Kaplan–Meier 1958, Cox 1972, Lin & Wei 1989, Garwood 1936 a Schuirmann 1987.
  - Uživatel je v knihovně nesežene, rozhodnuto 2026-09-21. M2 a M3 se tím uzavírají bez primární citace.
  - Text metody jen jmenuje. Survival modely cituje přes implementaci (`davidsonpilon2019lifelines`, §3.7.2). Garwoodův interval, TOST a klastrované robustní SE zůstávají bez citace.
  - Riziko: IEEE Access chce text „carefully referenced" a recenzent si o ty citace může říct.
