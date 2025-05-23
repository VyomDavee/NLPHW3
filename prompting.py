import os, argparse, random
from tqdm import tqdm
import re
import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs, load_alignment_data
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, ptype=0, train_x=None, train_y=None, schema_info=None, alignment_dict=None):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    '''
    schema_context = ""
    if schema_info and 'tables' in schema_info:
        schema_context = "Database Schema:\n"
        for table_name, table_info in schema_info['tables'].items():
            schema_context += f"Table: {table_name}\n"
            if isinstance(table_info, dict) and 'columns' in table_info:
                columns = table_info['columns']
                schema_context += f"Columns: {', '.join(columns)}\n"
            else:
                schema_context += f"Columns: {', '.join(table_info)}\n"
        schema_context += "\n"
    
    # Add alignment information
    alignment_context = ""
    if alignment_dict:
        alignment_context = "Term Mappings (natural language -> database):\n"
 
        for nl_term, db_term in alignment_dict.items():
            alignment_context += f"- \"{nl_term}\" refers to \"{db_term}\"\n"
        alignment_context += "\n"
    
    # Select prompt template based on ptype
    if ptype == 0:
        system_prompt = "You are a helpful assistant that translates natural language questions about a flight database into SQL queries."
        
        if k == 0:
            prompt = f"{system_prompt}\n\n{schema_context}{alignment_context}Translate the following natural language instruction into a SQL query:\n\n{sentence}\n\nOutput only the SQL Query. SQL:"
        else:
            prompt = f"{system_prompt}\n\n{schema_context}{alignment_context}Here are some examples of natural language instructions and their corresponding SQL queries:\n\n"
            
            # Generate k random examples (could be smarter with example selection)
            indices = random.sample(range(len(train_x)), k)
            for idx in indices:
                prompt += f"Instruction: {train_x[idx]}\nSQL: {train_y[idx]}\n\n"
            
            prompt += f"Now translate this new instruction into a SQL query:\n\nInstruction: {sentence}\n\nOutput only the SQL Query. SQL:"
    #different prompt type
    elif ptype == 1:
        system_prompt = "You are a database expert tasked with converting natural language questions into correct SQL queries for a flight database."
        
        if k == 0:
            prompt = f"{system_prompt}\n\n{schema_context}{alignment_context}TASK: Convert the natural language question below into a valid SQL query.\n\n- Make sure your query is valid SQL syntax\n- Output just the SQL query without any explanations\n- Use only tables and columns that exist in the database schema\n- Use the provided term mappings to convert natural language terms to database terms\n\nQuestion: {sentence}\n\nSQL Query:"
        else:
            prompt = f"{system_prompt}\n\n{schema_context}{alignment_context}TASK: Convert natural language questions into valid SQL queries.\n\nHere are {k} examples:\n\n"
            
            # Generate k random examples
            indices = random.sample(range(len(train_x)), k)
            for idx in indices:
                prompt += f"Question: {train_x[idx]}\nSQL Query: {train_y[idx]}\n\n"
            
            prompt += f"Now convert this question into a SQL query:\n\nQuestion: {sentence}\n\nOutput only the SQL Query:"
    
    return prompt


def exp_kshot(tokenizer, model, inputs, k, ptype=0, train_x=None, train_y=None, schema_info=None, alignment_dict=None):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    raw_outputs = []
    extracted_queries = []

    for i, sentence in tqdm(enumerate(inputs), total=len(inputs), desc=f"{k}-shot prompting"):
        prompt = create_prompt(sentence, k, ptype, train_x, train_y, schema_info, alignment_dict)

        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **input_ids, 
                max_new_tokens=128,
                do_sample=False,
                temperature=0.7,
                num_beams=3,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
        raw_outputs.append(response)

        extracted_query = extract_sql_query(response)

        # Apply term substitutions based on alignment dictionary
        processed_query = extracted_query
        if alignment_dict:
            for nl_term, db_term in alignment_dict.items():
                processed_query = re.sub(r'\b' + re.escape(nl_term) + r'\b', db_term, processed_query, flags=re.IGNORECASE)
        
        extracted_queries.append(processed_query)
        
        # Print the first few examples for debugging
        if i < 2:
            print(f"\nExample {i}:")
            print(f"Instruction: {sentence}")
            print(f"Generated SQL: {processed_query}")
    
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    # Compute metrics using the utility functions
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, 
        model_sql_path, 
        gt_record_path, 
        model_record_path
    )
    
    # Calculate error rate
    error_count = sum(1 for msg in model_error_msgs if msg)
    error_rate = error_count / len(model_error_msgs) if model_error_msgs else 0
    
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        config=nf4_config).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16).to(DEVICE)
    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    # Load database schema
    schema_path = os.path.join(data_folder, 'flight_database.schema')
    schema_info = read_schema(schema_path)
    
    # Load alignment data
    alignment_path = os.path.join(data_folder, 'alignment.txt')
    alignment_dict = load_alignment_data(alignment_path)
    print(f"Loaded {len(alignment_dict)} term mappings from alignment file")

    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)
    os.makedirs('logs', exist_ok=True)

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        raw_outputs, extracted_queries = exp_kshot(
            tokenizer, model, eval_x, shot, ptype, train_x, train_y, schema_info, alignment_dict
        )

        # You can add any post-processing if needed
        # You can compute the records with `compute_records``

        # Save the generated queries
        model_sql_path = os.path.join(f'results/{model_name}_{ptype}_{shot}shot_{experiment_name}_{eval_split}.sql')
        model_record_path = os.path.join(f'records/{model_name}_{ptype}_{shot}shot_{experiment_name}_{eval_split}.pkl')
        
        # Save the generated queries and compute their records
        save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
        
        # Evaluate the results, only for dev since answers are available
        if eval_split == "dev":
            gt_sql_path = os.path.join(f'data/{eval_split}.sql')
            gt_record_path = os.path.join(f'records/{eval_split}_gt_records.pkl')
            
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x, eval_y,
                gt_sql_path,
                model_sql_path,
                gt_record_path,
                model_record_path
            )
            
            print(f"{eval_split} set results: ")
            print(f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
            print(f"{error_rate*100:.2f}% of the generated outputs led to SQL errors")
            
            # Save logs
            log_path = f'logs/{model_name}_{ptype}_{shot}shot_{experiment_name}_{eval_split}.log'
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)


if __name__ == "__main__":
    main()