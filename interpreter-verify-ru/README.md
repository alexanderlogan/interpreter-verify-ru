# Interpreter-Verify-RU

**A local, HIPAA-compliant medical translation and terminology verification tool for professional Russian-English interpreters.**

Interpreter-Verify-RU listens to medical telehealth sessions, transcribes and translates bidirectionally between Russian and English, and flags pharmaceutical terms, false cognates ("false friends"), and clinical risks in real time. All processing happens locally on your machine. No patient data ever leaves the device.

---

## What This Application Does

1. **Captures audio** from your telehealth platform (Zoom, Teams, etc.) via Windows system audio loopback
2. **Transcribes speech** using Whisper AI, automatically detecting whether each segment is English or Russian
3. **Translates** Russian to English (for the provider) and English to Russian (for the patient) using a local LLM
4. **Flags pharmaceutical terms** by matching against a curated database of 44+ Russian/US medications
5. **Alerts on false friends** (words that sound similar in both languages but have dangerously different meanings, e.g., "ангина" means tonsillitis in Russian, not angina pectoris)
6. **Displays everything** in a floating overlay window on your screen

**Who is this for?** Professional medical interpreters working remote English-Russian telehealth sessions. The application is a work tool that provides draft translations and pharmaceutical intelligence. It does not replace the interpreter; it augments them.

---

## Architecture Overview

```
[Telehealth Audio]
        |
        v
[WASAPI Loopback Capture] ---- captures all system audio
        |
        v
[Silero VAD] ---- detects speech segments, filters silence
        |
        v
[Faster-Whisper] ---- transcribes + auto-detects language (EN/RU)
        |
        +---> [Pharma Map Lookup] ---- flags drug names and false friends
        |
        v
[Ollama / Qwen 2.5 7B] ---- translates + audits terminology (local LLM)
        |
        v
[Application Window] ---- shows translation, flags, and alerts
```

**Key design principle:** Everything runs locally. Whisper transcribes on your CPU. Ollama runs the translation LLM on your CPU. The pharma database is a local JSON file. No internet connection is required during operation.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 (64-bit) | Windows 11 |
| CPU | Intel Core i5 (8th gen+) | Intel Core i7/i9 |
| RAM | 16 GB | 32 GB |
| GPU | Not required (Intel UHD OK) | NVIDIA RTX 3060+ (optional, improves speed) |
| Storage | 15 GB free | 20 GB free |
| Audio | Any telehealth platform with speaker output | Same |

**Performance expectations (CPU-only, 16 GB RAM):**
- Transcription: 1-2 seconds per utterance
- Translation + drug check: 3-8 seconds per utterance
- Total time-to-screen: 4-10 seconds (you will have already spoken before the translation appears)

---

## Installation

### Prerequisites

Before installing the application, you need three things on your machine:

#### 1. Python 3.10 or newer

Check if you have it:
```powershell
python --version
```

If not installed, download from [python.org](https://www.python.org/downloads/). During installation, **check "Add Python to PATH"**.

#### 2. Ollama (local LLM runtime)

Install via PowerShell:
```powershell
winget install Ollama.Ollama
```

Then pull the translation model (~4.5 GB download):
```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
```

Verify it works:
```powershell
ollama run qwen2.5:7b-instruct-q4_K_M "Translate to English: У пациента высокое давление"
```

You should see a translation appear after a few seconds. Press Ctrl+D to exit.

#### 3. Git (for version control)

Install via PowerShell:
```powershell
winget install Git.Git
```

Close and reopen PowerShell after installing.

### Application Installation

```powershell
# Clone the repository
git clone https://github.com/alexanderlogan/interpreter-verify-ru.git
cd interpreter-verify-ru

# Create a virtual environment (keeps dependencies isolated)
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### First Run

```powershell
# Make sure you are in the project folder with venv activated
cd interpreter-verify-ru
.\venv\Scripts\Activate.ps1

# Start the application
python main.py
```

The application window will appear. Start your telehealth session, and the app will begin capturing and translating audio.

---

## How to Use

1. **Start Ollama** (it usually runs automatically in the background after installation)
2. **Start a telehealth session** (Zoom, Teams, etc.)
3. **Launch Interpreter-Verify-RU** (`python main.py`)
4. **Position the application window** where you can see it while working
5. The app will:
   - Show transcribed text as speakers talk
   - Display translations (RU to EN and EN to RU)
   - Highlight pharmaceutical terms in the text
   - Show alerts for false friends and drug warnings
6. **You interpret as normal.** Glance at the app for:
   - Unfamiliar drug names and their US equivalents
   - False friend warnings
   - Draft translations for reference

### Keyboard Shortcuts (planned)

| Key | Action |
|-----|--------|
| F1 | Start/pause listening |
| F2 | Toggle always-on-top |
| Esc | Minimize to tray |

---

## Project Structure

```
interpreter-verify-ru/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Application entry point
├── config.py                 # Configuration settings
├── src/
│   ├── audio/
│   │   ├── capture.py        # WASAPI loopback audio capture
│   │   └── vad.py            # Voice Activity Detection
│   ├── transcription/
│   │   └── whisper_engine.py # Whisper transcription engine
│   ├── translation/
│   │   └── ollama_engine.py  # Ollama/Qwen translation + audit
│   ├── pharma/
│   │   ├── pharma_lookup.py  # Drug name matching engine
│   │   └── pharma_map.json   # Medication database
│   └── ui/
│       └── overlay.py        # Application window (Tkinter)
├── tests/
│   ├── test_audio.py         # Audio capture tests
│   ├── test_whisper.py       # Transcription tests
│   ├── test_translation.py   # Translation tests
│   └── test_pharma.py        # Pharma lookup tests
└── docs/
    ├── ARCHITECTURE.md        # Technical architecture details
    ├── PHARMA_GUIDE.md        # Guide to the medication database
    └── CHANGELOG.md           # Version history
```

---

## Development Roadmap

This application is being built incrementally, one working feature at a time. Each phase produces a testable, usable program that is committed to Git before moving to the next phase. If something breaks, we roll back to the last working version.

### Phase 1: Audio Capture (v0.1.0)
**Goal:** Prove we can capture system audio from the telehealth platform.
- [ ] WASAPI loopback capture working
- [ ] Record 10 seconds of audio, play it back
- [ ] Audio device selection (if multiple outputs exist)
- **Test:** Run the app, play a YouTube video, verify captured audio matches

### Phase 2: Transcription (v0.2.0)
**Goal:** Transcribe captured audio to text with language detection.
- [ ] Whisper distil-large-v3 integration
- [ ] Auto-detect English vs. Russian per segment
- [ ] Print transcripts to console with language labels
- [ ] Handle silence gracefully (no empty transcripts)
- **Test:** Speak English and Russian alternately, verify correct transcription and language tags

### Phase 3: Translation (v0.3.0)
**Goal:** Translate transcribed text using local LLM.
- [ ] Ollama API integration (localhost:11434)
- [ ] RU to EN translation with medical system prompt
- [ ] EN to RU translation with medical system prompt
- [ ] Async/non-blocking (audio pipeline keeps running during translation)
- **Test:** Speak a Russian medical sentence, verify English translation appears. Then reverse.

### Phase 4: Pharmaceutical Intelligence (v0.4.0)
**Goal:** Detect and flag pharmaceutical terms and false friends.
- [ ] Load pharma_map.json at startup
- [ ] Exact match detection in transcripts
- [ ] Fuzzy match detection (for misspellings, Whisper errors)
- [ ] False friend alerting
- [ ] Drug warning display (FDA status, clinical warnings)
- **Test:** Say "Корвалол" in a sentence, verify it flags with warnings. Say "ангина", verify false friend alert.

### Phase 5: Application Window (v0.5.0)
**Goal:** Display everything in a usable floating window.
- [ ] Tkinter overlay window
- [ ] Scrolling transcript with translations
- [ ] Color-coded pharmaceutical highlights
- [ ] Alert panel for warnings and false friends
- [ ] Always-on-top toggle
- [ ] Start/pause button
- **Test:** Run a mock telehealth session, verify all information displays correctly.

### Phase 6: Polish and Packaging (v0.6.0 to v1.0.0)
**Goal:** Make the application installable and professional.
- [ ] Windows executable (.exe) via PyInstaller
- [ ] Installer package
- [ ] Settings panel (model selection, audio device, window position)
- [ ] Session logging (save transcript to file, no PHI in logs)
- [ ] Performance optimization (caching, conditional LLM calls)
- [ ] Error handling and recovery

---

## Git Workflow (for beginners)

This project uses Git to track changes and allow rollback to working versions. Here is the workflow:

### Daily Work Pattern

```powershell
# Before starting work, make sure you are on the main branch
git status

# After making changes that WORK, save them:
git add .
git commit -m "Brief description of what you changed"

# Push to GitHub (backup + version history)
git push
```

### Key Git Commands

| Command | What it does |
|---------|-------------|
| `git status` | Shows what files have changed |
| `git add .` | Stages all changes for commit |
| `git commit -m "message"` | Saves a snapshot with a description |
| `git push` | Uploads to GitHub |
| `git log --oneline` | Shows version history |
| `git checkout .` | Undo all uncommitted changes (go back to last commit) |
| `git tag v0.1.0` | Mark a version milestone |

### Rolling Back

If something breaks and you want to go back to the last working version:

```powershell
# See recent commits
git log --oneline

# Go back to a specific commit (replace abc1234 with the commit ID)
git checkout abc1234

# Or discard all changes since last commit
git checkout .
```

### Golden Rule

**Commit after every working change.** Do not wait until the end of the day. If you get Phase 1 audio capture working, commit immediately. Then if Phase 2 breaks something, you can roll back to the Phase 1 commit.

---

## Pharmaceutical Database

The `pharma_map.json` file contains:
- **44 medications** mapped between Russian and US trade names
- **12 false friends** with risk levels and interpreter guidance
- **55 unique active ingredients**

### Categories
- **Russian Popular (12):** Medications with no US equivalent or unique risk (Корвалол, Анальгин, Но-шпа, etc.)
- **US Common (2):** Frequently discussed US drugs with Russian name variants
- **Antibiotics (12):** Амоксиклав, Сумамед, Цефтриаксон, Левомицетин, etc.
- **Antihypertensives (10):** Энап, Конкор, Капотен, Лозап, Нифедипин, etc.
- **NSAIDs/Analgesics (8):** Найз, Кеторол, Диклофенак, Мелоксикам, etc.

### Highest Risk Items
| Drug | Risk | Why |
|------|------|-----|
| Левомицетин (Chloramphenicol) | Fatal aplastic anemia | OTC in Russia, rarely used systemically in US |
| Найз (Nimesulide) | Hepatotoxicity | Never approved by FDA, withdrawn in parts of EU |
| Аркоксиа (Etoricoxib) | Cardiovascular events | FDA specifically declined approval in 2007 |
| Валидол | Masks cardiac symptoms | No cardiac efficacy, patients may use instead of nitroglycerin |
| Корвалол | Undisclosed barbiturate | Contains Phenobarbital (Schedule IV), affects anesthesia |

### Highest Risk False Friends
| Russian Term | Sounds Like | Actually Means | Risk |
|-------------|-------------|----------------|------|
| Ангина | Angina (chest pain) | Tonsillitis / sore throat | CRITICAL |
| Наркоз | Narcotics | General anesthesia | HIGH |
| Инсульт | Insult | Stroke (CVA) | HIGH |
| Стаж | Stage (of disease) | Duration / how long they have had it | HIGH |

---

## Privacy and Compliance

- **All processing is local.** Audio is processed by Whisper on your CPU. Translation is done by Ollama on your CPU. No data is sent to any server.
- **No internet required** during operation (only for initial setup and model downloads).
- **No audio is stored** by default. Transcripts can optionally be saved to local files.
- **No patient identifiers** are included in any logs or saved data.
- **HIPAA consideration:** This tool is designed for use by a professional interpreter as a work aid. It does not replace clinical judgment. All translations should be verified by the interpreter before being communicated.

---

## Troubleshooting

### "Ollama is not running"
Ollama runs as a background service. If it is not running:
```powershell
ollama serve
```
Leave this PowerShell window open while using the application.

### "No audio devices found"
Make sure your telehealth application is producing audio through speakers (not headphones with exclusive mode). The app captures the default Windows audio output.

### "Translations are slow"
On CPU-only hardware, expect 3-8 seconds per translation. To speed things up:
- Close unnecessary applications to free RAM
- Use the `medium` Whisper model instead of `distil-large-v3` (faster, slightly less accurate)
- Ensure Ollama is pre-warmed (the first translation is slowest while the model loads)

### "Whisper misheard a drug name"
The pharma lookup uses fuzzy matching to catch misspellings, but Whisper may occasionally misrecognize rare terms. The medication database will improve over time as you encounter and add new terms.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Audio Capture | PyAudioWPatch (WASAPI) | Captures system audio on Windows |
| Voice Detection | Silero VAD | Filters silence, segments speech |
| Transcription | Faster-Whisper (distil-large-v3) | Speech-to-text with language detection |
| Translation + Audit | Ollama + Qwen 2.5 7B (Q4_K_M) | Local LLM for translation and term verification |
| Drug Matching | rapidfuzz + pharma_map.json | Fuzzy pharmaceutical term detection |
| User Interface | Tkinter | Floating application window |
| Packaging | PyInstaller (Phase 6) | Windows executable |

---

## License

This project is for personal professional use. The pharmaceutical database is curated from public sources (Vidal.ru, WHO ATC, FDA.gov, PubMed).

---

## Disclaimer

This application is a professional work tool for qualified medical interpreters. It does not provide medical advice, diagnosis, or treatment recommendations. All translations and pharmaceutical information should be verified by the interpreter before being communicated. The developer assumes no liability for clinical decisions made based on this tool's output.
