# AMoC v4 Persona Age Experiments

A pipeline to generate AMoC knowledge-graph triplets for persona texts, plot graphs, remove outliers, and run stats—optimized for SLURM + vLLM.

## End-to-end flow
Persona CSV (possibly chunked)
        |
        v
  amoc.cli.main
    - parse args (models, chunk file, TP, plotting flags)
    - load spaCy
    - call process_persona_csv
        |
        v
  AgeAwareAMoCEngine.run
    - build persona description (age + text)
    - instantiate AMoCv4
    - AMoCv4.analyze (story sentences)
         - init graph from first sentence
         - iterate sentences: add/infer edges, update active flags, optional per-sentence plots
         - return triplets (with active flag)
        |
        v
  Runner writes triplets CSV
    - filename: model_<model>_triplets_<chunk>.csv
    - columns: original_index, age_refined, persona_text, model_name,
               subject, relation, object, regime, active
    - optional final plot: amoc_graph_<model>_<persona>_<age>_final.png
        |
        v
Parallel over chunks via SLURM array
        |
        v
After extraction:
Merge chunks:
  merge_csv_triplet_chunks.py
    - group by model/regime
    - merge → model_<model>_triplets_<regime>.csv
        |
        v
Outlier removal & stats:
  remove_outliers_and_run_plots.sh
    - remove_outliers.py (reads merged files)
    - stats/plots per regime/age bins

## Directories
- Persona inputs: `input/` 
- Extraction outputs: `results/extracted_triplets_final_plot/`
- Graphs after each sentence: `results/Qwen3-30b/graphs_per_sentence/`

## File naming:
  - Triplets per chunk: `model_<model>_triplets_<chunkfilename>.csv`
  - Merged per regime: `model_<model>_triplets_<regime>.csv`
  - Plots: `amoc_graph_<model>_<persona>_<age>[_sentN|_final].png`

## Important Flags
- `--plot-final-graph`: one graph per persona (final state).
- `--plot-largest-component-only`: plot only the largest component (display only).
- `--replace-pronouns`: enable pronoun resolution (off by default).
- `--tp`: tensor parallel size for vLLM.



