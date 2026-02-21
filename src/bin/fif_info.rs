fn main() -> anyhow::Result<()> {
    use exg::fiff::open_raw;
    use ndarray::Array2;

    let path = std::path::Path::new("data/sample1_raw.fif");
    let raw = open_raw(path)?;
    let data_f64 = raw.read_all_data()?;
    let data_f32: Array2<f32> = data_f64.mapv(|v| v as f32);

    println!("sfreq={} nchan={} n_times={}", raw.info.sfreq, raw.info.n_chan, raw.n_times());
    println!("data shape: {:?}", data_f32.dim());

    // Print first 5 values per channel
    for (i, ch) in raw.info.chs.iter().enumerate() {
        let row = data_f32.row(i);
        let vals: Vec<String> = row.iter().take(5).map(|v| format!("{:.4e}", v)).collect();
        println!("  ch{i:02} {:8} loc=[{:.5},{:.5},{:.5}]  data[0..5]=[{}]",
            ch.name, ch.loc[0], ch.loc[1], ch.loc[2], vals.join(","));
    }
    Ok(())
}
