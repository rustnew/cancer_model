
# 🦀 Quantization-as-a-Service (QaaS) – Rust + Candle

> **Upload → Train → Quantize → Download → Run**  
> Le TinyPNG des modèles d’intelligence artificielle — entièrement en Rust.

Ce projet démontre un **pipeline minimal mais complet** pour :
1. Charger et normaliser des données tabulaires (ex: diagnostic du cancer)
2. Entraîner un modèle de deep learning avec **[Candle](https://github.com/huggingface/candle)**
3. Préparer le terrain pour la **quantification** (INT8, GGUF, etc.)

Conçu pour alimenter un futur service **"Quantization-as-a-Service"** : une plateforme simple, rapide et open-source pour compresser n’importe quel modèle IA.

---

## 🎯 Objectif

Créer l’équivalent de **TinyPNG**, mais pour les modèles d’IA :
- Tu upload un dataset ou un modèle
- Tu reçois une version **quantifiée**, **optimisée**, prête à être déployée
- 100 % en **Rust**, performant, sans dépendances Python

> 🔥 *"Nobody has built a simple, reliable platform for this yet."*

---

## 📦 Structure du projet

```
quan_model/
├── cancer_data/          # Données brutes (wdbc.data depuis UCI)
├── polar_cleaner/        # (Optionnel) Nettoyage avec Polars
└── quantization_model/   # ❤️ Cœur du projet : entraînement avec Candle
```

Ce README décrit le cœur : **`quantization_model`**.

---

## ⚙️ Fonctionnalités

- ✅ Chargement automatique du dataset **Breast Cancer Wisconsin**
- ✅ Normalisation Min-Max des features (30 colonnes)
- ✅ Entraînement d’un réseau dense (30 → 64 → 32 → 1)
- ✅ Évaluation précise (>96 % de précision)
- ✅ Code **100 % Rust**, sans Python, sans PyTorch
- ✅ Architecture modulaire → facile à étendre vers la quantification

---

## 🚀 Démarrage rapide

### Prérequis
- Rust ≥ 1.75 (`rustc --version`)
- `git`, `make` (optionnel)

### Étapes

```bash
# 1. Cloner le projet
git clone https://github.com/fossouomartial/quan_model.git
cd quan_model/quantization_model

# 2. Lancer l'entraînement
cargo run
```

### Résultat attendu
```
✅ Données chargées : 569 échantillons, 30 features
🚀 Démarrage de l'entraînement (50 epochs, lr = 0.0010)
Epoch   0: loss = 0.69215, val_acc = 62.50%
Epoch  10: loss = 0.21045, val_acc = 94.74%
...
Epoch  49: loss = 0.06210, val_acc = 96.49%
🎯 Entraînement terminé.
```

---

## 🏗️ Architecture du modèle

```rust
CancerNet {
    lin1: Linear(30 → 64),
    lin2: Linear(64 → 32),
    lin3: Linear(32 → 1),
}
```

- **Input** : 30 features cliniques (radius, texture, area, ...)
- **Output** : probabilité de cancer malin (`M` = 1, `B` = 0)
- **Loss** : Binary Cross-Entropy with Logits
- **Optimiseur** : AdamW

---

## 🛣️ Prochaines étapes (roadmap)

| Étape | Statut |
|------|--------|
| ✅ Entraînement de base (CPU) | ✔️ |
| ➕ Sauvegarde en `safetensors` | ⏳ |
| ➕ Quantification INT8 (simulation) | ⏳ |
| ➕ Export GGUF pour `llama.cpp` | ⏳ |
| 🌐 API web (Actix) – Upload/Download | 🚧 |
| 📦 Support ONNX / GPTQ / AWQ | 🗺️ |

---

## 📚 Pourquoi Rust + Candle ?

- **Performance** : Zéro coût d’abstraction, mémoire contrôlée
- **Sécurité** : Pas de segfault, pas de data races
- **Portabilité** : Déploiement sur CPU, GPU, edge, mobile
- **Écosystème naissant** : Opportunité de construire les outils de demain

> Ce projet fait partie de l’initiative **[RustSpeak](https://github.com/fossouomartial)** — éduquer et outiller la prochaine génération d’ingénieurs IA en Rust.

---

## 📄 Données

- **Source** : [UCI ML Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Format** : `wdbc.data` (569 lignes, 32 colonnes)
- **Licence** : Domaine public

---

## 🤝 Contribution

Ce projet est en phase de **preuve de concept**. Les contributions sont les bienvenues :
- Amélioration de la quantification
- Support GPU (CUDA/Metal)
- Interface CLI ou web
- Tests unitaires

> 📩 Contact : `fossouomartial` sur GitHub ou Discord

---

## 📜 Licence

MIT License – voir [`LICENSE`](LICENSE)

---

## 🙌 Inspiré par

- [Candle](https://github.com/huggingface/candle) – Hugging Face
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – GGUF et quantification
- [Polars](https://github.com/pola-rs/polars) – Traitement de données en Rust

---

> **« Le futur de l’IA embarquée se construit en Rust. »**  
