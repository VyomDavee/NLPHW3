import os
import numpy as np

def load_file(file_path):
    """Load a file and return its contents as a list of lines."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def tokenize_sql(query):
    """Tokenize SQL query, handling special characters appropriately."""
    # Replace special characters with space-padded versions
    for char in "(),;=<>":
        query = query.replace(char, f" {char} ")
    
    # Handle comparison operators and aliases
    query = query.replace("!=", " != ")
    query = query.replace(">=", " >= ")
    query = query.replace("<=", " <= ")
    query = query.replace(".", " . ")
    
    # Split by whitespace and filter out empty tokens
    tokens = [token for token in query.split() if token]
    return tokens

def calc_mean_length(sentences, is_sql=False):
    """Calculate mean length of sentences in tokens."""
    if is_sql:
        lengths = [len(tokenize_sql(sentence)) for sentence in sentences]
    else:
        lengths = [len(sentence.split()) for sentence in sentences]
    return np.mean(lengths), np.std(lengths)

def calc_vocab_size(sentences, is_sql=False):
    """Calculate vocabulary size of a set of sentences."""
    all_words = set()
    for sentence in sentences:
        if is_sql:
            tokens = tokenize_sql(sentence)
        else:
            tokens = sentence.split()
        all_words.update([word.lower() for word in tokens])
    return len(all_words)

def analyze_dataset(data_folder, split):
    """Analyze statistics for a dataset split."""
    nl_path = os.path.join(data_folder, f"{split}.nl")
    sql_path = os.path.join(data_folder, f"{split}.sql")
    
    # Load data
    nl_sentences = load_file(nl_path)
    sql_queries = load_file(sql_path)
    
    # Calculate statistics
    nl_mean_len, nl_std_len = calc_mean_length(nl_sentences, is_sql=False)
    sql_mean_len, sql_std_len = calc_mean_length(sql_queries, is_sql=True)
    
    nl_vocab_size = calc_vocab_size(nl_sentences, is_sql=False)
    sql_vocab_size = calc_vocab_size(sql_queries, is_sql=True)
    
    # Print results
    print(f"\n=== Statistics for {split.upper()} set ===")
    print(f"Number of examples: {len(nl_sentences)}")
    print(f"Mean NL sentence length: {nl_mean_len:.2f} tokens (± {nl_std_len:.2f})")
    print(f"Mean SQL query length: {sql_mean_len:.2f} tokens (± {sql_std_len:.2f})")
    print(f"NL vocabulary size: {nl_vocab_size} unique tokens")
    print(f"SQL vocabulary size: {sql_vocab_size} unique tokens")
    
    return {
        'count': len(nl_sentences),
        'nl_mean_len': nl_mean_len,
        'nl_std_len': nl_std_len,
        'sql_mean_len': sql_mean_len,
        'sql_std_len': sql_std_len,
        'nl_vocab_size': nl_vocab_size,
        'sql_vocab_size': sql_vocab_size
    }

def main():
    data_folder = 'data'  # Adjust if needed
    
    # Analyze both train and dev sets
    train_stats = analyze_dataset(data_folder, 'train')
    dev_stats = analyze_dataset(data_folder, 'dev')
    
    # Calculate statistics for the combined dataset
    print("\n=== Combined Dataset Statistics ===")
    train_nl = load_file(os.path.join(data_folder, "train.nl"))
    dev_nl = load_file(os.path.join(data_folder, "dev.nl"))
    train_sql = load_file(os.path.join(data_folder, "train.sql"))
    dev_sql = load_file(os.path.join(data_folder, "dev.sql"))
    
    all_nl = train_nl + dev_nl
    all_sql = train_sql + dev_sql
    
    # Calculate combined vocabulary sizes
    all_nl_vocab_size = calc_vocab_size(all_nl, is_sql=False)
    all_sql_vocab_size = calc_vocab_size(all_sql, is_sql=True)
    
    print(f"Total examples: {len(all_nl)}")
    print(f"Combined NL vocabulary size: {all_nl_vocab_size} unique tokens")
    print(f"Combined SQL vocabulary size: {all_sql_vocab_size} unique tokens")
    
    # Print a comparison table
    print("\n=== Comparison Table ===")
    print(f"{'Statistic':<25} {'TRAIN':<15} {'DEV':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15}")
    print(f"{'Number of examples':<25} {train_stats['count']:<15} {dev_stats['count']:<15}")
    print(f"{'NL mean length':<25} {train_stats['nl_mean_len']:.2f} {dev_stats['nl_mean_len']:.2f}")
    print(f"{'SQL mean length':<25} {train_stats['sql_mean_len']:.2f} {dev_stats['sql_mean_len']:.2f}")
    print(f"{'NL vocabulary size':<25} {train_stats['nl_vocab_size']:<15} {dev_stats['nl_vocab_size']:<15}")
    print(f"{'SQL vocabulary size':<25} {train_stats['sql_vocab_size']:<15} {dev_stats['sql_vocab_size']:<15}")

if __name__ == "__main__":
    main()