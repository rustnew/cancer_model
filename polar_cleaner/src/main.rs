use polars::prelude::*;
use polars::lazy::dsl::col;
use std::fs::File;

fn load_and_name_data() -> PolarsResult<DataFrame> {
    let file = File::open("../cancer_data/wdbc.data")?;
    let df = CsvReader::new(file)
        .with_has_header(false)          // ✅ Nouvelle API
        .with_separator(b',')
        .finish()?;

    let feature_groups = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry", "fractal_dimension"
    ];
    let mut col_names = vec!["id".to_string(), "diagnosis".to_string()];
    for stat in ["_mean", "_se", "_worst"] {
        for &fg in &feature_groups {
            col_names.push(format!("{}{}", fg.replace(' ', "_"), stat));
        }
    }
    df.rename(&col_names)
}

fn encode_labels(mut df: DataFrame) -> PolarsResult<(DataFrame, Series)> {
    // Récupère la colonne "diagnosis" comme Utf8
    let diagnosis_series = df.column("diagnosis")?.utf8()?; // ✅ Important : .utf8()!
    
    let labels: BooleanChunked = diagnosis_series
        .into_iter()
        .map(|opt_s| opt_s.map(|s| s == "M"))
        .collect();

    let labels_u8: Series = labels.cast(&DataType::UInt8)?;
    
    df = df.drop(["id", "diagnosis"])?; // ✅ drop accepte un slice
    Ok((df, labels_u8))
}

fn validate_data(df: &DataFrame) -> PolarsResult<()> {
    for name in df.get_column_names() {
        let s = df.column(name)?;
        if s.null_count() > 0 {
            eprintln!("⚠️  {} valeurs manquantes dans {}", s.null_count(), name);
        }
    }
    Ok(())
}

fn normalize_features(df: DataFrame) -> PolarsResult<DataFrame> {
    let mut exprs = Vec::with_capacity(df.width());
    for name in df.get_column_names() {
        let min_expr = col(name).min();
        let max_expr = col(name).max();
        let normalized = (col(name) - min_expr) / (max_expr - min_expr);
        exprs.push(normalized.alias(name));
    }
    df.lazy().select(exprs).collect()
}

fn save_prepared_data(df: &DataFrame, labels: &Series) -> PolarsResult<()> {
    let x_file = File::create("X.parquet")?;
    ParquetWriter::new(x_file).finish(df)?;

    let y_df = DataFrame::new(vec![labels.clone().rename("label")])?;
    let y_file = File::create("y.parquet")?;
    ParquetWriter::new(y_file).finish(&y_df)?;

    println!("✅ Données sauvegardées : X.parquet, y.parquet");
    Ok(())
}

fn main() -> PolarsResult<()> {
    let df = load_and_name_data()?;
    validate_data(&df)?;
    let (features, labels) = encode_labels(df)?;
    let normalized = normalize_features(features)?;
    save_prepared_data(&normalized, &labels)?;
    println!("🎉 Pipeline de nettoyage terminé avec succès !");
    Ok(())
}