#!/usr/bin/env python3
import os, json, glob

def load_results(pattern):
    files = glob.glob(pattern, recursive=True)
    data = {}
    for f in files:
        try:
            with open(f) as fp:
                results = json.load(fp)
            model = os.path.basename(os.path.dirname(f))
            data[model] = results
        except:
            continue
    return data

def write_markdown(data, out_path="results_summary.md"):
    with open(out_path, "w") as md:
        md.write("# Model Performance Summary\n\n")
        md.write("| Model | Accuracy (%) | F1-Score | MAE | RMSE | Time (s) |\n")
        md.write("|-------|--------------|----------|-----|------|----------|\n")
        for model, res in data.items():
            md.write(f"| {model} | "
                     f"{res.get('accuracy',0)*100:.2f} | "
                     f"{res.get('f1_score',0):.2f} | "
                     f"{res.get('mae',0):.2f} | "
                     f"{res.get('rmse',0):.2f} | "
                     f"{res.get('training_time',0):.1f} |\n")
    print(f"Report written to {out_path}")

if __name__=="__main__":
    # Load results from all models
    patterns = [
        "models/*/results.json",
        "models/main_model/results.json"
    ]
    all_results = {}
    for p in patterns:
        all_results.update(load_results(p))
    # Write report
    os.makedirs("results", exist_ok=True)
    write_markdown(all_results, out_path="results/results_summary.md")
