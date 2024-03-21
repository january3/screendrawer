#!/usr/bin/env python3
import re

def process_file(input_filename, output_filename):
    placeholder_pattern = re.compile(r'<placeholder (.+?)>')
    
    with open(input_filename, 'r') as file:
        lines = file.readlines()
    
    output_lines = []
    
    for line in lines:
        placeholder_match = placeholder_pattern.search(line)
        if placeholder_match:
            # Found a placeholder, now process the specified file
            file_to_insert = placeholder_match.group(1)
            try:
                with open(file_to_insert, 'r') as insert_file:
                    for insert_line in insert_file:
                        if '<remove>' not in insert_line:
                            output_lines.append(insert_line)
            except FileNotFoundError:
                print(f"Warning: File {file_to_insert} not found. Placeholder was ignored.")
        else:
            # Normal line, just add it to output
            output_lines.append(line)
    
    # Write the processed lines to a new file or overwrite the old one
    with open(output_filename, 'w') as output_file:
        output_file.writelines(output_lines)

# Example usage
process_file('sd_devel.py', 'sd.py')

