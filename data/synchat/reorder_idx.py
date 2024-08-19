import json

# Function to modify idx values in a JSON Lines file
def modify_idx_in_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for new_idx, line in enumerate(infile):
            # Parse the JSON object from the line
            data = json.loads(line)
            
            # Modify the idx field
            data['idx'] = new_idx
            
            # Write the modified JSON object to the output file
            outfile.write(json.dumps(data) + '\n')

# Example usage
input_file = 'input.jsonl'
output_file = 'output.jsonl'
modify_idx_in_jsonl(input_file, output_file)

print(f'Modified idx values written to {output_file}')

