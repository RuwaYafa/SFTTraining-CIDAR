import statistics
import logging
import pandas as pd
import argparse
import os
from evaluate import load

logger = logging.getLogger(__name__)

def bert_score(refs, outputs):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=outputs, references=refs, lang="ar")
    return results

def bleu_score(refs, preds):
    bleu = load("bleu")
    results = bleu.compute(predictions=preds, references=refs)
    return results

def rouge_score(refs, outputs):
    rouge = load("rouge")
    results = rouge.compute(predictions=outputs, references=refs)
    return results

def compute_metrics(ref_list, output_list):
    rouge_results = rouge_score(output_list, ref_list)
    bleu_results = bleu_score([[ref] for ref in ref_list], output_list) # BLEU expects list of lists for references

    avg_rouge1 = rouge_results['rouge1']
    avg_rouge2 = rouge_results['rouge2']
    avg_rougeL = rouge_results['rougeL']

    bertscore = bert_score(list(ref_list), list(output_list))
    bert_f1s = bertscore['f1']
    avg_bertscore = statistics.mean(bert_f1s)

    metrics = {
        'ROUGE1': avg_rouge1,
        'ROUGE2': avg_rouge2,
        'ROUGEL': avg_rougeL,
        'BertScore': avg_bertscore,
        'BLEU': bleu_results['bleu'],
        'BLEU_JSON': bleu_results
    }

    score_df = pd.DataFrame({
        'ROUGE1': [rouge_results['rouge1']] * len(ref_list), # Repeat overall score for each row
        'ROUGE2': [rouge_results['rouge2']] * len(ref_list),
        'ROUGEL': [rouge_results['rougeL']] * len(ref_list),
        'BertScore': bert_f1s
    })

    return metrics, score_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_file', type=str, required=True, help='Path to reference CSV file.')
    parser.add_argument('--generated_file', type=str, required=True, help='Path to generated JSON file.')
    parser.add_argument('--per_row_output_file', type=str, required=True, help='Path to save detailed metrics per row.')
    parser.add_argument('--overall_output_file', type=str, required=True, help='Path to save overall metrics.')

    args = parser.parse_args()

    print(f"Loading reference file: {args.reference_file}")
    ref_df = pd.read_json(args.reference_file)[:5]
    print(f"Reference file columns: {ref_df.columns.tolist()}")
    print(f"Reference file content:\n{ref_df.head()}")

    print(f"Loading generated file: {args.generated_file}")
    gen_df = pd.read_json(args.generated_file)[:5]
    print(f"Generated file columns: {gen_df.columns.tolist()}")
    print(f"Generated file content:\n{gen_df.head()}")

    if len(ref_df) != len(gen_df):
        raise ValueError("Reference and generated files have a different number of rows.")

    ref_outputs = ref_df['model_output'].fillna('').tolist()
    gen_outputs = gen_df['generated_output'].fillna('').tolist()
    instructions = ref_df['instruction'].tolist()

    print("Computing metrics using Hugging Face Evaluate...")
    metrics, detailed_scores = compute_metrics(ref_outputs, gen_outputs)

    # Printing the result
    print("Result of evaluation:", metrics)

    # Prepare saving
    os.makedirs(os.path.dirname(args.per_row_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.overall_output_file), exist_ok=True)

    # Save detailed per-example scores
    detailed_results_path = args.per_row_output_file
    detailed_df = pd.DataFrame({
        'instruction': instructions,
        'model_output': ref_outputs,
        'generated_output': gen_outputs,
        'ROUGE1': detailed_scores['ROUGE1'],
        'ROUGE2': detailed_scores['ROUGE2'],
        'ROUGEL': detailed_scores['ROUGEL'],
        'BertScore': detailed_scores['BertScore']
    })
    detailed_df.to_csv(detailed_results_path, index=False)
    print(f"Saved detailed results to {detailed_results_path}")

    # Save overall metrics
    overall_metrics_path = args.overall_output_file
    overall_df = pd.DataFrame([metrics])
    overall_df.to_json(overall_metrics_path, orient='index')
    print(f"Saved overall metrics to {overall_metrics_path}")

if __name__ == "__main__":
    main()

# Result of evaluation: {'ROUGE1': 0.0, 'ROUGE2': 0.0, 'ROUGEL': 0, 'BertScore': 0.9999999880790711, 'BLEU': 1.0, 'BLEU_JSON': {'bleu': 1.0, 'precisions': [1.0, 1.0, 1.0, 1.0], 'brevity_penalty': 1.0, 'length_ratio': 1.0, 'translation_length': 635, 'reference_length': 635}}
# import statistics
# import logging
# import pandas as pd
# import argparse
# import os
# from evaluate import load
# from rouge_score import rouge_scorer

# logger = logging.getLogger(__name__)

# def bert_score(refs, outputs):
#     bertscore = load("bertscore")
#     results = bertscore.compute(predictions=outputs, references=refs, lang="ar")
#     return results

# def bleu_score(refs, preds):
#     bleu = load("bleu")
#     results = bleu.compute(predictions=preds, references=refs)
#     return results

# def rouge(ref, output):
#     f1_score_map = {}
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, )
#     scores = scorer.score(ref, output)
#     for k, v in scores.items():
#         f1_score_map[k] = v.fmeasure
#     return f1_score_map

# def compute_metrics(ref_list, output_list):
#     rouge1_scores = []
#     rouge2_scores = []
#     rougeL_scores = []
#     bleu_inputs = []
#     bleu_refs = []

#     for ref, output in zip(ref_list, output_list):
#         r_score = rouge(ref, output)
#         rouge1_scores.append(r_score['rouge1'])
#         rouge2_scores.append(r_score['rouge2'])
#         rougeL_scores.append(r_score['rougeL'])

#         bleu_inputs.append(output)
#         bleu_refs.append([ref])

#     avg_rouge1 = statistics.mean(rouge1_scores)
#     avg_rouge2 = statistics.mean(rouge2_scores)
#     avg_rougeL = statistics.mean(rougeL_scores)

#     bertscore = bert_score(list(ref_list), list(output_list)) # Convert generators to lists
#     bert_f1s = bertscore['f1']
#     avg_bertscore = statistics.mean(bert_f1s)

#     avg_bleu = bleu_score(bleu_refs, bleu_inputs)

#     metrics = {
#         'ROUGE1': avg_rouge1,
#         'ROUGE2': avg_rouge2,
#         'ROUGEL': avg_rougeL,
#         'BertScore': avg_bertscore,
#         'BLEU': avg_bleu['bleu'],
#         'BLEU_JSON': avg_bleu
#     }

#     score_df = pd.DataFrame({
#         'ROUGE1': rouge1_scores,
#         'ROUGE2': rouge2_scores,
#         'ROUGEL': rougeL_scores,
#         'BertScore': bert_f1s
#     })

#     return metrics, score_df

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--reference_file', type=str, required=True, help='Path to reference CSV file.')
#     parser.add_argument('--generated_file', type=str, required=True, help='Path to generated JSON file.')
#     parser.add_argument('--per_row_output_file', type=str, required=True, help='Path to save detailed metrics per row.')
#     parser.add_argument('--overall_output_file', type=str, required=True, help='Path to save overall metrics.')

#     args = parser.parse_args()

#     print(f"Loading reference file: {args.reference_file}")
#     ref_df = pd.read_json(args.reference_file)[:5]
#     print(f"Reference file columns: {ref_df.columns.tolist()}")
#     print(f"Reference file content:\n{ref_df}")

#     print(f"Loading generated file: {args.generated_file}")
#     gen_df = pd.read_json(args.generated_file)[:5]
#     print(f"Generated file columns: {gen_df.columns.tolist()}")
#     print(f"Generated file content:\n{gen_df}")

#     # Assuming both DataFrames have the same number of rows and the order matters
#     if len(ref_df) != len(gen_df):
#         raise ValueError("Reference and generated files have a different number of rows.")

#     ref_outputs = ref_df['model_output'].fillna('').tolist()
#     gen_outputs = gen_df['model_output'].fillna('').tolist()
#     instructions = ref_df['instruction'].tolist() # Assuming 'instruction' is relevant for output

#     print("Computing metrics...")
#     metrics, detailed_scores = compute_metrics(ref_outputs, gen_outputs)

#     # Printing the result
#     print("Result of evaluation:", metrics)

#     # Prepare saving
#     os.makedirs(os.path.dirname(args.per_row_output_file), exist_ok=True)
#     os.makedirs(os.path.dirname(args.overall_output_file), exist_ok=True)

#     # Save detailed per-example scores
#     detailed_results_path = args.per_row_output_file
#     detailed_df = pd.DataFrame({
#         'instruction': instructions,
#         'model_output': ref_outputs,
#         'generated_output': gen_outputs,
#         'ROUGE1': detailed_scores['ROUGE1'],
#         'ROUGE2': detailed_scores['ROUGE2'],
#         'ROUGEL': detailed_scores['ROUGEL'],
#         'BertScore': detailed_scores['BertScore']
#     })
#     detailed_df.to_csv(detailed_results_path, index=False)
#     print(f"Saved detailed results to {detailed_results_path}")

#     # Save overall metrics
#     overall_metrics_path = args.overall_output_file
#     overall_df = pd.DataFrame([metrics])
#     overall_df.to_json(overall_metrics_path, index=False, orient='records') # Changed orient for better readability
#     print(f"Saved overall metrics to {overall_metrics_path}")

# if __name__ == "__main__":
#     main()









# import statistics
# import logging
# import pandas as pd
# import argparse
# import os
# from evaluate import load
# from rouge_score import rouge_scorer

# logger = logging.getLogger(__name__)

# def bert_score(refs, outputs):
#     bertscore = load("bertscore")
#     results = bertscore.compute(predictions=outputs, references=refs, lang="ar")
#     return results

# def bleu_score(refs, preds):
#     bleu = load("bleu")
#     results = bleu.compute(predictions=preds, references=refs)
#     return results

# def rouge(ref, output):
#     f1_score_map = {}
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(ref, output)
#     for k, v in scores.items():
#         f1_score_map[k] = v.fmeasure
#     return f1_score_map

# def compute_metrics(df, ref_col='output_ref', output_col='output_gen'):
#     # print(f"margeddf---------------{df}")
#     rouge1_scores = []
#     rouge2_scores = []
#     rougeL_scores = []
#     bleu_inputs = []
#     bleu_refs = []
#     df.fillna('', inplace=True)

#     for i, row in df.iterrows():
#         ref = row[ref_col]
#         output = row[output_col]

#         r_score = rouge(ref, output)
#         rouge1_scores.append(r_score['rouge1'])
#         rouge2_scores.append(r_score['rouge2'])
#         rougeL_scores.append(r_score['rougeL'])

#         bleu_inputs.append(output)
#         bleu_refs.append([ref])

#     avg_rouge1 = statistics.mean(rouge1_scores)
#     avg_rouge2 = statistics.mean(rouge2_scores)
#     avg_rougeL = statistics.mean(rougeL_scores)

#     outputs = df[output_col].values
#     refs = df[ref_col].values
#     bertscore = bert_score(refs, outputs)
#     bert_f1s = bertscore['f1']
#     avg_bertscore = statistics.mean(bert_f1s)

#     avg_bleu = bleu_score(bleu_refs, bleu_inputs)

#     metrics = {
#         'ROUGE1': avg_rouge1,
#         'ROUGE2': avg_rouge2,
#         'ROUGEL': avg_rougeL,
#         'BertScore': avg_bertscore,
#         'BLEU': avg_bleu['bleu'],
#         'BLEU_JSON': avg_bleu
#     }

#     score_df = pd.DataFrame({
#         'ROUGE1': rouge1_scores,
#         'ROUGE2': rouge2_scores,
#         'ROUGEL': rougeL_scores,
#         'BertScore': bert_f1s
#     })
    




    

#     return metrics, score_df

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--reference_file', type=str, required=True, help='Path to reference CSV file.')
#     parser.add_argument('--generated_file', type=str, required=True, help='Path to generated JSON file.')
#     parser.add_argument('--per_row_output_file', type=str, required=True, help='Path to save detailed metrics per row.')
#     parser.add_argument('--overall_output_file', type=str, required=True, help='Path to save overall metrics.')
    
#     args = parser.parse_args()

#     print(f"Loading reference file: {args.reference_file}")
#     # ref_df = pd.read_csv(args.reference_file)
#     ref_df = pd.read_json(args.reference_file)[:5]


#     print(f"Reference file columns: {ref_df.columns.tolist()}")
#     print(f"Reference file columns:----- {ref_df}")


#     print(f"Loading generated file: {args.generated_file}")
#     gen_df = pd.read_json(args.generated_file)
#     print(f"Generated file columns: {gen_df.columns.tolist()}")
#     print(f"Generated file columns:----- {gen_df}")



#     # ref_df['merge_index'] = range(len(ref_df))
#     # gen_df['merge_index'] = range(len(gen_df))
#     # merged_df = pd.merge(ref_df, gen_df, on='merge_index', suffixes=('model_output', 'generated_output'))
#     # merged_df = merged_df.drop(columns=['merge_index']) #remove the extra column


#     print("Merging reference and generated outputs...")
#     merged_df = pd.merge(ref_df, gen_df, on="instruction", suffixes=('model_output', 'generated_output'))
    
    
    
#     # merged_df = [];
#     # merged_df['output'] = ref_df['output']
#     # merged_df['generated_output'] = gen_df['generated_output']
#     # print("Unique instructions in ref_df:", ref_df['instruction'].unique())
#     # print("Unique instructions in gen_df:", gen_df['instruction'].unique())
#     # print("Columns after merging:", merged_df.columns.tolist())
#     # print(f"merged_df---------------{merged_df}")



#     print("Computing metrics...")
#     metrics, detailed_scores = compute_metrics(merged_df, ref_col = "model_output", output_col="generated_output")

#     # Printing the result
#     print("Result of >>>>", metrics)

#     # Prepare saving
#     os.makedirs(os.path.dirname(args.per_row_output_file), exist_ok=True)
#     os.makedirs(os.path.dirname(args.overall_output_file), exist_ok=True)

#     # Save detailed per-example scores
#     detailed_results_path = args.per_row_output_file
#     merged_with_scores = pd.concat([merged_df[["instruction", "model_output", "generated_output"]], detailed_scores], axis=1)
#     merged_with_scores.to_csv(detailed_results_path, index=False)
#     print(f"Saved detailed results to {detailed_results_path}")

#     # Save overall metrics
#     overall_metrics_path = args.overall_output_file
#     overall_df = pd.DataFrame([metrics])
#     overall_df.to_json(overall_metrics_path, index=False)
#     print(f"Saved overall metrics to {overall_metrics_path}")

# if __name__ == "__main__":
#     main()
