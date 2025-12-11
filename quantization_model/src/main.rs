use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{loss, ops, Linear, Module, VarMap, VarBuilder, AdamW, Optimizer};
use std::fs::File;
use std::io::{BufRead, BufReader};

const N_FEATURES: usize = 30;
const N_CLASSES: usize = 1;

// === Modèle configurable
#[derive(Debug)]
struct CancerNet {
    lin1: Linear,
    lin2: Linear,
    lin3: Linear,
}

impl CancerNet {
    fn new(vs: VarBuilder, hidden1: usize, hidden2: usize) -> candle_core::Result<Self> {
        let lin1 = candle_nn::linear(N_FEATURES, hidden1, vs.pp("lin1"))?;
        let lin2 = candle_nn::linear(hidden1, hidden2, vs.pp("lin2"))?;
        let lin3 = candle_nn::linear(hidden2, N_CLASSES, vs.pp("lin3"))?;
        Ok(Self { lin1, lin2, lin3 })
    }
}

impl Module for CancerNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.lin1)?.relu()?;
        let xs = xs.apply(&self.lin2)?.relu()?;
        xs.apply(&self.lin3)
    }
}

// === Chargement brut
fn load_raw_data(path: &str) -> candle_core::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| candle_core::Error::Msg(format!("Ligne {}: {}", idx, e)))?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 32 {
            eprintln!("⚠️  Ligne {} ignorée (trop courte)", idx);
            continue;
        }

        labels.push(if parts[1] == "M" { 1.0 } else { 0.0 });

        let mut sample = Vec::with_capacity(N_FEATURES);
        for i in 2..=31 {
            let val: f32 = parts[i].parse().unwrap_or(0.0);
            sample.push(val);
        }
        assert_eq!(sample.len(), N_FEATURES, "Incohérence dans le nombre de features");
        features.push(sample);
    }
    Ok((features, labels))
}

// === Normalisation Min-Max
fn min_max_normalize(data: Vec<Vec<f32>>) -> Vec<f32> {
    let n = data.len();
    let mut flat = vec![0.0f32; n * N_FEATURES];

    for j in 0..N_FEATURES {
        let col: Vec<f32> = (0..n).map(|i| data[i][j]).collect();
        let min_val = col.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = col.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        for i in 0..n {
            let norm = if range > 0.0 {
                (data[i][j] - min_val) / range
            } else {
                0.0
            };
            flat[i * N_FEATURES + j] = norm;
        }
    }
    flat
}

// === Évaluation
fn evaluate(model: &CancerNet, x_test: &Tensor, y_test: &Tensor, device: &Device) -> candle_core::Result<f32> {
    let logits = model.forward(x_test)?;
    let probs = ops::sigmoid(&logits)?;
    let threshold = Tensor::from_slice(&[0.5f32], (1,), device)?;
    let threshold = threshold.broadcast_as(probs.shape())?;
    let preds = probs.gt(&threshold)?;

    let y_bool = y_test.gt(&threshold)?;
    let correct = preds.eq(&y_bool)?.to_dtype(DType::F32)?.sum_all()?;
    let accuracy = correct.to_vec0::<f32>()? / y_test.dims()[0] as f32;
    Ok(accuracy)
}

// === Configuration
struct Config {
    epochs: usize,
    learning_rate: f64,
    hidden1: usize,
    hidden2: usize,
    train_ratio: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 1e-3,
            hidden1: 64,
            hidden2: 32,
            train_ratio: 0.8,
        }
    }
}

// === Main
fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let cfg = Config::default();

    // Chargement et préparation
    let (raw_features, labels) = load_raw_data("../cancer_data/wdbc.data")?;
    let normalized_features = min_max_normalize(raw_features);
    let n = labels.len();
    println!("✅ Données chargées : {} échantillons, {} features", n, N_FEATURES);

    let x = Tensor::from_slice(&normalized_features, (n, N_FEATURES), &device)?;
    let y = Tensor::from_slice(&labels, (n, N_CLASSES), &device)?;

    let n_train = (n as f32 * cfg.train_ratio) as usize;
    let x_train = x.i((..n_train, ..))?;
    let x_test = x.i((n_train.., ..))?;
    let y_train = y.i((..n_train, ..))?;
    let y_test = y.i((n_train.., ..))?;

    // Modèle et optimiseur
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = CancerNet::new(vs, cfg.hidden1, cfg.hidden2)?;
    let mut opt = AdamW::new_lr(varmap.all_vars(), cfg.learning_rate)?;

    // Entraînement
    println!("🚀 Démarrage de l'entraînement ({} epochs, lr = {:.4})", cfg.epochs, cfg.learning_rate);
    for epoch in 0..cfg.epochs {
        let logits = model.forward(&x_train)?;
        let loss = loss::binary_cross_entropy_with_logit(&logits.flatten_all()?, &y_train.flatten_all()?)?;
        opt.backward_step(&loss)?;

        if epoch % 10 == 0 || epoch == cfg.epochs - 1 {
            let acc = evaluate(&model, &x_test, &y_test, &device)?;
            println!(
                "Epoch {:3}: loss = {:.5}, val_acc = {:.2}%",
                epoch,
                loss.to_vec0::<f32>()?,
                acc * 100.0
            );
        }
    }

    println!("🎯 Entraînement terminé.");
    Ok(())
}