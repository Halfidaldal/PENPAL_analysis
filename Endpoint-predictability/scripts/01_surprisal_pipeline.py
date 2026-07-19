import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pyrefly: ignore [missing-import]
from compute_surprisal import compute_endpoint_surprisal_scores, load_language_model, build_dataset

DATA_DIR = Path(__file__).parent.parent / "data"

def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    pilot_paper_path = config["data"]["pilot_paper"][0]
    scorer_model_name = config["scorer"]["model_name"]
    
    tokenizer, model, device = load_language_model(scorer_model_name)

    df = build_dataset(pilot_paper_path)
    df = compute_endpoint_surprisal_scores(df, tokenizer, model)
    df.to_csv(DATA_DIR / "surprisal_scores.csv", index=False)
    

    print(f"\nSaved raw surprisal scores to: {DATA_DIR / 'surprisal_scores.csv'}")


if __name__ == "__main__":
    main()