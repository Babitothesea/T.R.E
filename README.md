# SentimentCam

Analisi del sentiment in tempo reale dalla webcam, con **rilevamento simultaneo di 3 o piu' volti**.

Stack deliberatamente minimale: **OpenCV** (YuNet per i volti) + **ONNX Runtime** (`emotion-ferplus-8` per le espressioni). Tre pacchetti Python, due pesi `.onnx` da 33 MB in tutto, zero GPU richiesta.

![stack](https://img.shields.io/badge/stack-OpenCV%20%2B%20ONNX%20Runtime-6c7ae0) ![license](https://img.shields.io/badge/code-MIT-6c7ae0) ![deps](https://img.shields.io/badge/dipendenze-3-6c7ae0)

---

## Perche' questo stack

La scelta non e' stilistica: e' imposta dall'hardware.

- **Nessun framework di deep learning.** Tutto gira su due grafi ONNX eseguiti dal
  runtime integrato in OpenCV e da ONNX Runtime. Si installa in pochi secondi e
  non richiede ambienti di training.
- **`emotion-ferplus-8` e' addestrato su FER+**, le annotazioni crowdsourced di
  FER2013 (paper arXiv:1608.01041). E' cio' che distingue un'espressione triste da
  una neutra: un classificatore generico zero-shot non ci arriva, mentre un modello
  dedicato alle espressioni costa 33 MB e 20 ms per volto.
- **Nessun bisogno di GPU.** La GTX 850M di questa macchina (Kepler, 2014) non e'
  supportata dai runtime moderni; tutto e' pensato per CPU, che qui e' l'unica
  opzione reale.

### Attenzione alla versione di onnxruntime

`requirements.txt` blocca `onnxruntime<1.21` per un motivo verificato: le versioni
1.21, 1.22 e 1.30 **non si caricano** su un Windows con VC++ runtime 14.24 (Haswell
del 2014) e terminano con un access violation (`0xC0000005`). La 1.20.1 funziona.

Se un giorno cambiasse macchina e volessi la versione piu' recente,
`tools/find_ort.ps1` prova le versioni in ordine decrescente e si ferma alla prima
compatibile.

## Come funziona

```
frame webcam (BGR)
   │
   ├─► YuNet  ·  cv2.FaceDetectorYN  ·  runtime ONNX integrato in OpenCV
   │      └─► box + 5 landmark per volto, nessun limite al numero di volti
   │
   ├─► MultiFaceTracker  ·  IoU greedy
   │      └─► ID stabile #1, #2, #3… + media mobile delle emozioni
   │
   ├─► ritaglio allineato  ·  similarity transform Umeyama sui 5 landmark
   │      └─► 64×64 frontale e stabile anche a testa ruotata
   │
   ├─► emotion-ferplus-8  ·  ONNX Runtime
   │      └─► 8 classi: neutro felice sorpreso triste arrabbiato disgustato spaventato disdegno
   │
   └─► emozioni → valenza [-1,+1] → overlay, pannello, grafico
```

Il flusso resta interamente in BGR: sia il detector sia il ritaglio lavorano su array OpenCV,
senza conversioni di canale. Il ritaglio e' in scala di grigi 64×64, che e' il formato nativo
atteso dal modello.

**Componenti e scelte:**

| Componente | Scelta | Motivo |
|---|---|---|
| Rilevamento volti | YuNet (OpenCV Zoo) via `cv2.FaceDetectorYN` | 230 KB, realtime, 5 landmark per l'allineamento, nessuna dipendenza pesante |
| Espressioni | `emotion-ferplus-8` (ONNX Model Zoo) | 33 MB, addestrato su FER+, 20 ms per volto su CPU debole |
| Stabilita' temporale | EMA sulle probabilita' (`--smooth`) | Un singolo frame e' rumoroso: l'EMA evita che l'etichetta salti ogni 100 ms |
| Identita' | IoU greedy, nessun embedding | 3+ volti sono gia' gestiti; gli embedding ArcFace costerebbero ~200 ms/frame senza guadagno apprezzabile |

## Installazione

```bash
pip install -r requirements.txt
```

Al primo utilizzo vengono scaricati due pesi in `weights/`, verificati con SHA-256:

- `face_detection_yunet_2023mar.onnx` — 230 KB, MIT, OpenCV Zoo
- `emotion-ferplus-8.onnx` — 33 MB, MIT, ONNX Model Zoo

## Utilizzo

```bash
python run.py                          # webcam 0, fino a 6 volti
python run.py --source 1               # seconda webcam
python run.py --max-faces 10           # piu' volti contemporanei
python run.py --source video.mp4       # analizza un video
python run.py --source foto.jpg        # analizza una singola immagine
python run.py --det-interval 2         # piu' fluidita' su CPU lente
```

### Comandi durante l'uso

| Tasto | Azione |
|---|---|
| `q` / `Esc` | esci |
| `s` | salva uno screenshot in `screenshots/` |
| `r` | azzera tracking e storico |
| `m` | attiva/disattiva il mirror |
| `h` | mostra/nasconde il pannello |
| `+` / `-` | numero massimo di volti (fino a 32) |

### Opzioni principali

```
--max-faces N            volti analizzati insieme (default 6, minimo 3)
--det-conf 0.6           soglia del detector: piu' bassa = piu' volti, anche piccoli
--det-interval N         rileva i volti ogni N frame, riusando i box nel mezzo
--max-infer-per-frame N  quanti volti classificare per frame (default 2)
--smooth 0.55            smorzamento temporale 0..1
--width / --height       risoluzione webcam (il principale acceleratore)
--emotion-model          emotion-ferplus-8 (fp32) o emotion-ferplus-12-int8
--no-window              non apre la finestra (test / CI)
--max-frames N           ferma dopo N frame
--save-video out.mp4     salva il video annotato
```

## Prestazioni misurate

Su **i7-4510U** (Haswell 2014, 2 core + HT, 2,0 GHz), Windows 10, 3 volti inquadrati:

| Configurazione | FPS |
|---|---|
| 960×540, `--det-interval 1` | 9,7 |
| 960×540, `--det-interval 2` | **15,6** |
| 960×540, `--det-interval 3` | 19,2 |
| webcam reale 640×480, `--det-interval 1` | 6,2 |
| ciclo completo, tutti e 3 i volti classificati ogni frame | 6,7 |

La webcam di questa macchina accetta al massimo **640×480**: chiedere 960×540 non
aumenta la qualita', resta il default sensato. Con `--det-interval 2` si sale a
circa 10 fps, e da li' il limite diventa la webcam, non il modello.

Costo dei singoli componenti:

| Operazione | Tempo |
|---|---|
| YuNet 1280×720 | 281 ms |
| YuNet 960×540 | 132 ms |
| YuNet 640×360 | 67 ms |
| YuNet 480×270 | 40 ms |
| emotion-ferplus-8, per volto | 20 ms |

La detection scala linearmente con i pixel, quindi `--width/--height` e' la leva
principale. Su hardware recente si puo' tornare senza problemi a 1280×720.

Il modello int8 (`emotion-ferplus-12-int8`) **non** e' piu' veloce su CPU Haswell
(20 ms contro 20 ms): manca il supporto VNNI, che rende la quantizzazione INT8
un guadagno solo su CPU piu' recenti. Resta disponibile per GPU o CPU moderne.

## Le otto emozioni e il sentiment

| Emozione | Valenza | | Emozione | Valenza |
|---|---|---|---|---|
| felice | `+1.00` | | spaventato | `-0.70` |
| sorpreso | `+0.30` | | triste | `-0.80` |
| neutro | `0.00` | | disgustato | `-0.80` |
| disdegno | `-0.40` | | arrabbiato | `-0.90` |

La valenza di un volto e' la media pesata delle otto probabilita', riscalata per
l'intensita': `valenza = Σ pᵢ·vᵢ · (1 − p_neutro)`. Un volto perfettamente neutro
restituisce `0`, non "positivo per assenza di negativity".

Il sentiment di gruppo e' la media delle valenze dei volti, pesata per l'area del box:
un volto in primo piano conta di piu'.

## Limiti da conoscere

- **FER+ over-predice "neutral" su espressioni sottili.** Misurato: una foto
  apertamente nominata "sad" su Wikimedia viene letta `neutro 96%`. L'espressione
  e' infatti tenue (bocca chiusa, ciglia poco corrugate) e questa e' una risposta
  ragionevole, non un bug. Verificato con `tools/ablation.py`, che confronta
  allineamento vs ritaglio centrato e diverse normalizzazioni: il modello e'
  coerente e non e' un errore di pre-processing.
- **Il pre-processing conta piu' di quanto sembri.** Normalizzare l'input
  (`(x-127.5)/127.5` o `x/255`) **distrugge** il modello: tutte le facce tornano
  `neutro 74% / triste 21%` identiche, qualunque espressione. I pesi vanno
  alimentati con valori grezzi 0-255 in scala di grigi, come documenta la model
  card. Il codice lo fa e c'e' un commento che spiega perche'.
- **Volti troppo piccoli** (< 48 px di lato, `--min-face-px`) vengono scartati di
  proposito: sotto quella soglia FER+ non ha segnale affidabile.
- Serve **illuminazione frontale**: le ombre sugli occhi confondono sia il detector
  sia il modello di espressioni.

## Struttura del progetto

```
run.py                      entry point
requirements.txt            opencv, numpy, onnxruntime
sentimentcam/
  config.py                 parametri, CLI, valenze e colori delle emozioni
  weights.py                download e verifica SHA-256 dei pesi ONNX
  geometry.py               similarity transform Umeyama, ritaglio allineato
  detector.py               YuNet via cv2.FaceDetectorYN
  emotion.py                emotion-ferplus via ONNX Runtime
  tracker.py                tracking IoU multi-volto, stato emozionale, valenza
  ui.py                     overlay sui volti, pannello, grafico di andamento
  app.py                    loop principale, tasti, screenshot, export video
tests/
  test_core.py              50 verifiche della logica, senza pesi
tools/
  validate_emotions.py      benchmark end-to-end e controlli di coerenza
  ablation.py               confronto allineamento/centering e normalizzazioni
  probe_fer.py              sanity check su foto fisse, fp32 vs int8
  dump_crops.py             salva i ritagli per ispezione visiva
  fetch_candidates.py       scarica candidati Wikimedia e li compone in contact sheet
  find_orphans.py           elenca i pacchetti installati non piu' necessari
  find_ort.ps1              trova la versione di onnxruntime compatibile col PC
```

## Test

```bash
python tests/test_core.py
```

50 verifiche: IoU, valenza, stabilita' degli ID con 3+ volti, scadenza delle track,
media mobile, rendering della UI, similarity transform, validazione dei parametri.

L'ultimo test e' una guardia: **blocca a livello di import** i moduli dei framework
pesanti (`torch`, `libreyolo`, `ultralytics`, `tensorflow`, `mediapipe`) e verifica
che `detector`, `emotion` e `app` si carichino comunque. Se un domani un'import che
richiede 2 GB di dipendenze per una riga di codice, il test fallisce li invece di
lasciare il colpo scoprire in produzione.

`tools/find_orphans.py` fa il contorno: se un domani aggiungere un pacchetto,
elenca quelli che diventano orfani e li rimuove solo con `--apply`.

## Licenze

Codice: MIT. Pesi: YuNet MIT (OpenCV Zoo), emotion-ferplus MIT (ONNX Model Zoo).
Verifica comunque la licenza specifica su Hugging Face prima di un uso commerciale.
