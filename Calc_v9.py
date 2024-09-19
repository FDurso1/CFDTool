# Primarily Developed by Francis Durso
# This is the mathematics file meant to provide a JSON formatted output for use with:
# CFD_Research_Prototype_v1.py

#import sys
from collections import Counter, defaultdict
import json
from itertools import combinations, product
from datetime import datetime
import pandas as pd
from math import comb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

xtimescutoff = 0          # EXP: How many more times a value combination must be present in one file than the other to be considered notable
variable_names = []       # EXP: Stores variable names from the header if present
max_name_length = 0       # EXP: Used to format the verbose output variable
ncc = 0                   # EXP: Number of columns in class data
nrc = 0                   # EXP: Number of rows in class data
nrn = 0                   # EXP: Number of rows in NONclass file
ncn = 0                   # EXP: Number of columsn in NONclass file (should equal ncc, only used to check this)

classFilePath = None
nonClassFilePath = None


classFileData = None        # EXP: Data inside the class file
nonClassFileData = None     # EXP: Data inside the nonclass file
allFileData = None          # EXP: All the data from both files

hasAHeader = False           # EXP: Does either file contain a header


def detect_header(df):
    """
    Detects if the first row is a header based on whether any of its values appear elsewhere in their columns.
    """
    first_row = df.iloc[0]
    for column in df.columns:
        if first_row[column] in df[column].iloc[1:].values:
            return False
    return True

def getsetFiles():
    global classFileData
    global variable_names
    global max_name_length
    global ncc, nrc
    global nonClassFileData
    global nrn, ncn
    global hasAHeader
    global xtimescutoff
    global allFileData
    global classFilePath, nonClassFilePath
    
    xtimescutoff = 10

    #
    # EXP: ClassFile
    #
    
    classFilePath = Path(classFilePath)
    nonClassFilePath = Path(nonClassFilePath)
    
    class_df = pd.read_csv(classFilePath,header=None)
    
    if detect_header(class_df):
        hasAHeader = True
        variable_names = class_df.iloc[0].tolist()
        lengths = [len(str(name)) for name in variable_names]
        # Calculate the maximum length
        max_name_length = max(lengths)
        classFileData = class_df.iloc[1:].values.tolist()
    else:
        hasAHeader = False
        classFileData = class_df.values.tolist()
    
    nrc = len(classFileData)
    ncc = len(classFileData[0])

    #
    # EXP: NonClassFile
    #
    
    non_class_df = pd.read_csv(nonClassFilePath, header=None)
    if detect_header(non_class_df):
        variable_names_2 = non_class_df.iloc[0].tolist()
        if hasAHeader and (variable_names != variable_names_2):
            error_message = {
                "error_type": "ValueError",
                "error_message": f"Both files must refer to the same variables\n{variable_names}\n\nis not\n\n{variable_names_2}"
            }
            error_msg_json = json.dumps(error_message)
        
        hasAHeader = True
        nonClassFileData = non_class_df.iloc[1:].values.tolist()
    else:
        nonClassFileData = non_class_df.values.tolist()
    
    nrn = len(nonClassFileData)
    ncn = len(nonClassFileData[0])

    #
    # EXP: User Error Check
    #
    if ncn != ncc:
        error_message = {
            "error_type": "ValueError",
            "error_message": "Both files must have the same number of variables"
        }
        error_msg_json = json.dumps(error_message)
        raise ValueError("number of variables")

    allFileData = classFileData + nonClassFileData    
    # EXP: End of Function



class CombinationCounter:           # Primary data structure for class and nonclass calculations
    def __init__(self, fileData):
        self.data = fileData
        self.one_way_counts = defaultdict(int)      # Dictionary where values map to how often they appear in the file
        self.two_way_counts = defaultdict(int)      # Same but keys are two-way value interactions
        self.three_way_counts = defaultdict(int)    # And three way value interactions
        self.distinct_values = defaultdict(set)     # Dictionary where feature index maps to sets with all values posessed by that feature

    def count_combinations(self):

        for row in self.data:
            self._count_specific_combinations(row, 1, self.one_way_counts)
            self._count_specific_combinations(row, 2, self.two_way_counts)
            self._count_specific_combinations(row, 3, self.three_way_counts)

    def _count_specific_combinations(self, values, r, count_dict):
        indexed_values = [(index, value) for index, value in enumerate(values)]
        for combo in combinations(indexed_values, r):
            count_dict[combo] += 1

    def get_combination_counts(self):               # Getter
        return {
            '1-way': dict(self.one_way_counts),
            '2-way': dict(self.two_way_counts),
            '3-way': dict(self.three_way_counts)
        }
    
    def get_value_frequencies_and_distinct_values(self):
        ncc = len(self.data[0])  # Number of columns
        val_freq = [Counter() for _ in range(ncc)]
        
        for row in self.data:            
            for j in range(ncc):
                value = row[j]
                if pd.notnull(value):
                    val_freq[j][value] += 1
                    self.distinct_values[j].add(value)
        return val_freq, self.distinct_values
    

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_combo = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, combo):
        node = self.root
        for item in sorted(combo, key=lambda x: x[0]):  # Sort by index
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.is_end_of_combo = True

    def is_any_subset(self, combo):
        def search(node, items, index):
            if node.is_end_of_combo:
                return True
            if index == len(items):
                return False
            if items[index] in node.children:
                if search(node.children[items[index]], items, index + 1):
                    return True
            return search(node, items, index + 1)
        
        return search(self.root, sorted(combo), 0)

def find_notable_combos(counts_class_a, counts_class_b, x, lower_order_notable, num_rows_class_a, num_rows_class_b):
    notable_combos = []
    notable_combos_2 = []
    count_unique_a = 0
    count_nonunique_a = 0
    
    trie = Trie()
    for combo, _, _ in lower_order_notable: # Faster lookups with a Trie to filter out lower-order notables
        trie.insert(combo)

    for combo, count_a in counts_class_a.items():
        count_b = counts_class_b.get(combo, 0)
        freq_a = count_a / num_rows_class_a
        freq_b = count_b / num_rows_class_b if num_rows_class_b > 0 else 0

        diff = freq_a / freq_b if freq_b != 0 else "INF"
        indices = [el[0] for el in combo]
        vals = [el[1] for el in combo]
        if freq_a >= x * freq_b:
            if not trie.is_any_subset(combo):
                if count_b == 0:
                    count_unique_a += 1
                else:
                    count_nonunique_a += 1
                notable_combos.append((combo, freq_a, freq_b))
                notable_combos_2.append([indices, vals, freq_a, diff])
    
    # Create an empty list with size of max_index + 1 to hold the sorted lists
    sorted_data = [[] for _ in range(ncc)]
    
    # Place each list in its correct position
    for sublist in notable_combos_2:
        index = sublist[0][0]
        sorted_data[index] = sublist
    
    return notable_combos, sorted_data, count_unique_a, count_nonunique_a


def count_entries_with_notable_combos(data, one_way_combos, two_way_combos, three_way_combos):
    one_count = 0
    two_count = 0
    three_count = 0
    all_count = 0

    # Convert combos into sets for easy subset checking
    one_way_combos = [frozenset(combo) for combo, _, _ in one_way_combos]
    two_way_combos = [frozenset(combo) for combo, _, _ in two_way_combos]
    three_way_combos = [frozenset(combo) for combo, _, _ in three_way_combos]

    one_way_set = set(one_way_combos)
    two_way_set = set(two_way_combos)
    three_way_set = set(three_way_combos)

    for row in data:
        # Create a set of (index, value) pairs for the row
        row_set = frozenset((index, row[index]) for index in range(len(row)))
        one_found = any(combo.issubset(row_set) for combo in one_way_set)
        two_found = any(combo.issubset(row_set) for combo in two_way_set)
        three_found = any(combo.issubset(row_set) for combo in three_way_set)

        if one_found or two_found or three_found:
            all_count += 1
        if one_found:
            one_count += 1
        if two_found:
            two_count += 1
        if three_found:
            three_count += 1            
    return all_count, one_count, two_count, three_count

def calculate_max_possible_interactions(distinct_values):
    """
    Calculate the maximum possible 2-way and 3-way interactions and store them in a dictionary.
    distinct_values is expected to be a dictionary where the keys are variable indices
    and the values are sets of distinct values for each variable.
    """
    max_2way_combos = 0
    max_3way_combos = 0
    all_possible_2way_combos = {}
    all_possible_3way_combos = {}

    # Calculate the number of possible 2-way combinations and store them in the dictionary
    for combo in combinations(distinct_values.keys(), 2):
        values_product = list(product(distinct_values[combo[0]], distinct_values[combo[1]]))
        max_2way_combos += len(values_product)
        all_possible_2way_combos[combo] = values_product

    # Calculate the number of possible 3-way combinations and store them in the dictionary
    for combo in combinations(distinct_values.keys(), 3):
        values_product = list(product(distinct_values[combo[0]], distinct_values[combo[1]], distinct_values[combo[2]]))
        max_3way_combos += len(values_product)
        all_possible_3way_combos[combo] = values_product

    return max_2way_combos, max_3way_combos, all_possible_2way_combos, all_possible_3way_combos


def count_common_combinations(counts_class, counts_nonclass):
    common_1way = len(set(counts_class['1-way']) & set(counts_nonclass['1-way']))
    common_2way = len(set(counts_class['2-way']) & set(counts_nonclass['2-way']))
    common_3way = len(set(counts_class['3-way']) & set(counts_nonclass['3-way']))
        
    return {
        '1-way': common_1way,
        '2-way': common_2way,
        '3-way': common_3way
    }


def find_missing_combos(all_possible_2way_class, all_possible_3way_class, 
                        all_possible_2way_nonclass, all_possible_3way_nonclass, 
                        combination_counts_class, combination_counts_nonclass):
    missing_2way_class = []
    missing_3way_class = []
    missing_2way_nonclass = []
    missing_3way_nonclass = []

    # Get existing 2-way and 3-way combinations for class and nonclass
    existing_2way_class = set(combination_counts_class['2-way'].keys())
    existing_3way_class = set(combination_counts_class['3-way'].keys())
    existing_2way_nonclass = set(combination_counts_nonclass['2-way'].keys())
    existing_3way_nonclass = set(combination_counts_nonclass['3-way'].keys())
    
    # Check for missing 2-way class combinations
    for indices, values_list in all_possible_2way_class.items():
        for values in values_list:
            combo = tuple(zip(indices, values))
            if combo not in existing_2way_class:
                missing_2way_class.append([values, list(indices)])

    # Check for missing 3-way class combinations
    for indices, values_list in all_possible_3way_class.items():
        for values in values_list:
            combo = tuple(zip(indices, values))
            if combo not in existing_3way_class:
                missing_3way_class.append([values, list(indices)])

    # Check for missing 2-way nonclass combinations
    for indices, values_list in all_possible_2way_nonclass.items():
        for values in values_list:
            combo = tuple(zip(indices, values))
            if combo not in existing_2way_nonclass:
                missing_2way_nonclass.append([values, list(indices)])

    # Check for missing 3-way nonclass combinations
    for indices, values_list in all_possible_3way_nonclass.items():
        for values in values_list:
            combo = tuple(zip(indices, values))
            if combo not in existing_3way_nonclass:
                missing_3way_nonclass.append([values, list(indices)])

    return missing_2way_class, missing_3way_class, missing_2way_nonclass, missing_3way_nonclass


def generate_output_statements(class_notable_combos, nonclass_notable_combos, variable_names=None, max_name_length=0):
    listOfPrint = []
    listOfPrintVerbose = []

    unique_combos = []
    notable_combos_list = []

    for combo, freq_class, freq_nonclass in class_notable_combos:
        if freq_nonclass == 0:
            unique_combos.append((combo, freq_class, 'Class'))
        else:
            notable_combos_list.append((combo, freq_class, freq_nonclass, freq_class / freq_nonclass))

    for combo, freq_nonclass, freq_class in nonclass_notable_combos:
        if freq_class == 0:
            unique_combos.append((combo, freq_nonclass, 'Nonclass'))
        else:
            notable_combos_list.append((combo, freq_nonclass, freq_class, freq_nonclass / freq_class))

    # Sort unique combos by frequency in the respective file
    unique_combos.sort(key=lambda x: x[1], reverse=True)

    # Sort notable combos by frequency difference in decreasing order
    notable_combos_list.sort(key=lambda x: x[3], reverse=True)

    for combo, freq, class_type in unique_combos:
        combo_str = ", ".join([f"({index}, {value})" for index, value in combo])
        statement = ""
        if class_type == "Class":
            statement = f"Unique  {class_type}    Combo: [{combo_str}], freq:{freq:.3f}"
        elif class_type == "Nonclass":
            statement = f"Unique  {class_type} Combo: [{combo_str}], freq:{freq:.3f}"
        listOfPrint.append(statement)

        if variable_names:
            combo_str_verbose = ", ".join([f"{variable_names[index]} = {value}" for index, value in combo])
            statement_verbose = ""
            if class_type == "Class":
                statement_verbose = f"Unique  {class_type}    Combo when {combo_str_verbose}, freq:{freq:.3f}"
            else:   
                statement_verbose = f"Unique  {class_type} Combo when {combo_str_verbose}, freq:{freq:.3f}"
            listOfPrintVerbose.append(statement_verbose)

    for combo, freq_a, freq_b, diff in notable_combos_list:
        class_type = 'Class' if freq_a > freq_b else 'Nonclass'
        combo_str = ", ".join([f"({index}, {value})" for index, value in combo])
        statement = ""
        if class_type == 'Class':
            statement = f"Notable {class_type}    Combo: [{combo_str}], freq:{freq_a:.3f}, freq_diff:{diff:.3f}"
        elif class_type == 'Nonclass':
            statement = f"Notable {class_type} Combo: [{combo_str}], freq:{freq_a:.3f}, freq_diff:{diff:.3f}"
        listOfPrint.append(statement)

        if variable_names:
            combo_str_verbose = ", ".join([f"{variable_names[index]} = {value}" for index, value in combo])
            statement_verbose = ""
            if class_type == "Class":
                statement_verbose = f"Notable {class_type}    Combo when {combo_str_verbose}, freq:{freq_a:.3f}, freq_diff:{diff:.3f}"
            elif class_type == "Nonclass":
                statement_verbose = f"Notable {class_type} Combo when {combo_str_verbose}, freq:{freq_a:.3f}, freq_diff:{diff:.3f}"
            listOfPrintVerbose.append(statement_verbose)
                
    return listOfPrint, listOfPrintVerbose



def find_common_entries(class_data, nonclass_data):
    """
    Find entries that appear in both class and nonclass files, and record their counts.

    Returns:
    - common_entries: Dictionary with entries as keys and their counts in both files as values.
    """
    class_data_tuples = [tuple(entry) for entry in class_data]
    nonclass_data_tuples = [tuple(entry) for entry in nonclass_data]

    # Count entries in class data
    class_counts = defaultdict(int)
    for entry in class_data_tuples:
        class_counts[entry] += 1

    # Count entries in nonclass data
    nonclass_counts = defaultdict(int)
    for entry in nonclass_data_tuples:
        nonclass_counts[entry] += 1

    # Find common entries and their counts
    common_entries = []
    for entry in class_counts:
        if entry in nonclass_counts:
            common_entries.append([
                list(entry),
                {'class_count': class_counts[entry], 'nonclass_count': nonclass_counts[entry]}
            ])

    return common_entries


def analyze_combos(counts_class_a, counts_class_b, nrc, nrn):
    """
    Analyzes 2-way and 3-way value combinations between two datasets and plots the results.

    Parameters:
    - counts_class_a: dict, combination counts for class dataset
    - counts_class_b: dict, combination counts for nonclass dataset
    - nrc: int, number of rows in the class dataset
    - nrn: int, number of rows in the nonclass dataset
    - threshold: float, threshold for spike detection (default is 0.05)
    - plot: bool, whether to plot the results (default is True)

    Returns:
    - ld2: list of differences in relative frequencies
    - lx2: list of relative frequencies for the class dataset
    - ln2: list of negative relative frequencies for the nonclass dataset
    """
    lx2 = []
    ln2 = []
    ld2 = []

    for r in [2, 3]:
        key = f'{r}-way'
        all_combos = set(counts_class_a[key].keys()).union(counts_class_b[key].keys())
        for combo in all_combos:
            count_a = counts_class_a[key].get(combo, 0)
            count_b = counts_class_b[key].get(combo, 0)
            freq_a = count_a / nrc if nrc > 0 else 0
            freq_b = count_b / nrn if nrn > 0 else 0

            df = freq_a - freq_b
            ld2.append(df)
            lx2.append(freq_a)
            ln2.append(-freq_b)

    # if plot:
    #     fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(18, 15))
    #     minor_ticks = np.arange(-1.00, 1.0, 0.1)
    #     fig.subplots_adjust(hspace=0.5)

    #     ax2.plot(lx2, label='Class Relative Frequencies')
    #     ax3.plot(ln2, label='Nonclass Negative Relative Frequencies')
    #     if lx2 != []:
    #         ax2.set_ylim(min(lx2), max(lx2))  # Set y-axis limits dynamically
    #     if ln2 != []:
    #         ax3.set_ylim(min(ln2), max(ln2))  # Set y-axis limits dynamically

    #     differences = [lx + ln for lx, ln in zip(lx2, ln2)]
    #     ax4.plot(differences, label='Differences (Class - Nonclass)')
        
    #     if differences != []:
    #         ax4.set_ylim(min(differences), max(differences))  # Set y-axis limits dynamically

        
    #     ax2.grid(True, which='both')
    #     ax3.grid(True, which='both')
    #     ax4.grid(True, which='both')
        
    #     ax2.set_yticks(minor_ticks, minor=True)
    #     ax3.set_yticks(minor_ticks, minor=True)
    #     ax4.set_yticks(minor_ticks, minor=True)
        
    #     ax2.legend()
    #     ax3.legend()
    #     ax4.legend()
        
    #    # ax4.set_ylim(-1, 1)

    #     plt.show()


    return ld2, lx2, ln2


def notable_coverage_graph(notable_class_combos, notable_nonclass_combos):
    # Extract the combos and their frequencies
    class_combos = {combo: (freq_a, freq_b) for combo, freq_a, freq_b in notable_class_combos}
    nonclass_combos = {combo: (freq_a, freq_b) for combo, freq_a, freq_b in notable_nonclass_combos}
        
    lx = []
    ln = []
    ls = []
    
    all_combos = set(class_combos.keys()).union(nonclass_combos.keys())
        
    for combo in all_combos:
        freq_a_class, _ = class_combos.get(combo, (0, 0))
        freq_b_nonclass, _ = nonclass_combos.get(combo, (0, 0))
        
        lx.append(freq_a_class)
        ln.append(-freq_b_nonclass)
        ls.append(freq_a_class - freq_b_nonclass)
    
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 9))
    # minor_ticks = np.arange(-1.00, 1.0, 0.1)
    # fig.subplots_adjust(hspace=0.5)

    # # First Graph
    # ax1.plot(lx, label='Class Frequencies')
    # ax1.set_title(f'Class Frequencies for Notable {t}-way Combos')
    # ax1.set_xlabel('Index')
    # ax1.set_ylabel('Frequency')
    # ax1.legend()
    # ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax1.set_yticks(minor_ticks, minor=True)
    # if lx != []:
    #    ax1.set_ylim(min(lx), max(lx))  # Set y-axis limits dynamically

    # # Second Graph
    # ax2.plot(ln, label='Nonclass Frequencies')
    # ax2.set_title(f'Nonclass Frequencies for Notable {t}-way Combos')
    # ax2.set_xlabel('Index')
    # ax2.set_ylabel('Frequency')
    # ax2.legend()
    # ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax2.set_yticks(minor_ticks, minor=True)
    # if ln != []:
    #     ax2.set_ylim(min(ln), max(ln))  # Set y-axis limits dynamically
    
    # # Third Graph
    # ax3.plot(ls, label='Difference of Class and Nonclass Frequencies')
    # ax3.set_title(f'Sum of Frequencies (Class - Nonclass) for Notable {t}-way Combos')
    # ax3.set_xlabel('Index')
    # ax3.set_ylabel('Diff of Frequency')
    # ax3.legend()
    # ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax3.set_yticks(minor_ticks, minor=True)
    # if ls != []:
    #    ax3.set_ylim(min(ls), max(ls))  # Set y-axis limits dynamically
    # plt.show()
    
    return lx, ln, ls



def filter_notable_combinations(combination_counts, notable_combos):
    notable_counts = {1: {}, 2: {}, 3: {}}
    
    for combo, freq_a, freq_b in notable_combos['1-way']:
        if combo in combination_counts['1-way']:
            notable_counts[1][combo] = combination_counts['1-way'][combo]
    
    for combo, freq_a, freq_b in notable_combos['2-way']:
        if combo in combination_counts['2-way']:
            notable_counts[2][combo] = combination_counts['2-way'][combo]
    
    for combo, freq_a, freq_b in notable_combos['3-way']:
        if combo in combination_counts['3-way']:
            notable_counts[3][combo] = combination_counts['3-way'][combo]
    
    return notable_counts


def aggregate_counts_by_variable_notable(notable_combination_counts):
    count1 = [0] * ncc
    count2 = [0] * ncc
    count3 = [0] * ncc

    for key, count in notable_combination_counts[1].items():
        index = key[0][0]
        count1[index] += 1

    for key, count in notable_combination_counts[2].items():
        index1, index2 = key[0][0], key[1][0]
        count2[index1] += 1
        count2[index2] += 1

    for key, count in notable_combination_counts[3].items():
        index1, index2, index3 = key[0][0], key[1][0], key[2][0]
        count3[index1] += 1
        count3[index2] += 1
        count3[index3] += 1

    return count1, count2, count3


def merge_combination_counts(counts_class, counts_nonclass):
    merged_counts = defaultdict(lambda: defaultdict(int))

    for key in counts_class:
        for combo, count in counts_class[key].items():
            merged_counts[key][combo] += count

    for key in counts_nonclass:
        for combo, count in counts_nonclass[key].items():
            merged_counts[key][combo] += count

    return merged_counts

data = {}
def runData():
    global classFileData, nonClassFileData
    global ncc, data
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Begin Run Data - {formatted_time}\n")
    
    counter_class = CombinationCounter(classFileData)
    counter_class.count_combinations()
    combination_counts_class = counter_class.get_combination_counts()

    counter_nonclass = CombinationCounter(nonClassFileData)
    counter_nonclass.count_combinations()
    combination_counts_nonclass = counter_nonclass.get_combination_counts()

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 0\n")
    
    common_entries = find_common_entries(classFileData, nonClassFileData)

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 1\n")

   # Find notable 1-way combinations
    notable_1way_class_combos, notable_1way_class_combos_ind, \
    num_uniq_1way_class, num_intersect_1way_c = find_notable_combos(
        combination_counts_class['1-way'],
        combination_counts_nonclass['1-way'],
        xtimescutoff,
        [],
        nrc,
        nrn
    )
    
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 2 - {formatted_time}\n")
    
    notable_1way_nonclass_combos, notable_1way_nonclass_combos_ind, \
    num_uniq_1way_nonclass, num_intersect_1way_n = find_notable_combos(
        combination_counts_nonclass['1-way'],
        combination_counts_class['1-way'],
        xtimescutoff,
        [],
        nrn,
        nrc
    )
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 3 - {formatted_time}\n")

    # Find notable 2-way combinations
    notable_2way_class_combos, notable_2way_class_combos_ind, \
    num_uniq_2way_class, num_intersect_2way_c = find_notable_combos(
        combination_counts_class['2-way'],
        combination_counts_nonclass['2-way'],
        xtimescutoff,
        notable_1way_class_combos,
        nrc,
        nrn
    )
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 4 - {formatted_time}\n")

    
    notable_2way_nonclass_combos, notable_2way_nonclass_combos_ind, \
    num_uniq_2way_nonclass, num_intersect_2way_n = find_notable_combos(
        combination_counts_nonclass['2-way'],
        combination_counts_class['2-way'],
        xtimescutoff,
        notable_1way_nonclass_combos,
        nrn,
        nrc
    )

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 5 - {formatted_time}\n")


    # Find notable 3-way combinations
    notable_3way_class_combos, notable_3way_class_combos_ind, \
    num_uniq_3way_class, num_intersect_3way_c = find_notable_combos(
        combination_counts_class['3-way'],
        combination_counts_nonclass['3-way'],
        xtimescutoff,
        notable_1way_class_combos + notable_2way_class_combos,
        nrc,
        nrn
    )
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 6 - {formatted_time}\n")

    
    notable_3way_nonclass_combos, notable_3way_nonclass_combos_ind, \
    num_uniq_3way_nonclass, num_intersect_3way_n = find_notable_combos(
        combination_counts_nonclass['3-way'],
        combination_counts_class['3-way'],
        xtimescutoff,
        notable_1way_nonclass_combos + notable_2way_nonclass_combos,
        nrn,
        nrc
    )    
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 7 - {formatted_time}\n")

    
    valFreqClass, distinct_values_class = counter_class.get_value_frequencies_and_distinct_values()
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 8 - {formatted_time}\n")

    
    valFreqNonClass, distinct_values_nonclass = counter_nonclass.get_value_frequencies_and_distinct_values()
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 9 - {formatted_time}\n")
    
    unique_to_class = []
    for key in distinct_values_class.keys():
        if key in distinct_values_nonclass:
            difference_set = distinct_values_class[key] - distinct_values_nonclass[key]
        else:
            difference_set = distinct_values_class[key]
        unique_to_class.append(list(difference_set))
        
    numUniqValsInClass = sum(len(lst) for lst in unique_to_class)

    # Get unique values in nonclass file but not in class file
    unique_to_nonclass = []
    for key in distinct_values_nonclass.keys():
        if key in distinct_values_class:
            difference_set = distinct_values_nonclass[key] - distinct_values_class[key]
        else:
            difference_set = distinct_values_nonclass[key]
        unique_to_nonclass.append(list(difference_set))
        
    numUniqValsInNonClass = sum(len(lst) for lst in unique_to_nonclass)
        
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 10 - {formatted_time}\n")

    
    numClass1Ways_notable = len(notable_1way_class_combos)
    numClass2Ways_notable = len(notable_2way_class_combos)
    numClass3Ways_notable = len(notable_3way_class_combos)
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 11 - {formatted_time}\n")

    
    numNonClass1Ways_notable = len(notable_1way_nonclass_combos)
    numNonClass2Ways_notable = len(notable_2way_nonclass_combos)
    numNonClass3Ways_notable = len(notable_3way_nonclass_combos)
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 12 - {formatted_time}\n")

    sum_notable_1way = numClass1Ways_notable + numNonClass1Ways_notable    
    sum_notable_2way = numClass2Ways_notable + numNonClass2Ways_notable
    sum_notable_3way = numClass3Ways_notable + numNonClass3Ways_notable

    totalNotable =  len(notable_1way_class_combos) + len(notable_1way_nonclass_combos) + \
                    len(notable_2way_class_combos) + len(notable_2way_nonclass_combos) + \
                    len(notable_3way_class_combos) + len(notable_3way_nonclass_combos)
                    
    
    occur_notable_class, occur_notable_class_1way, occur_notable_class_2way, occur_notable_class_3way = \
        count_entries_with_notable_combos(classFileData, notable_1way_class_combos, notable_2way_class_combos, notable_3way_class_combos)

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 13 - {formatted_time}\n")


    occur_notable_nonclass, occur_notable_nonclass_1way, occur_notable_nonclass_2way, occur_notable_nonclass_3way = \
        count_entries_with_notable_combos(nonClassFileData, notable_1way_nonclass_combos, notable_2way_nonclass_combos, notable_3way_nonclass_combos)

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 14 - {formatted_time}\n")


    # Count entries with notable combinations
    occur_1way = occur_notable_class_1way + occur_notable_nonclass_1way
    occur_2way = occur_notable_class_2way + occur_notable_nonclass_2way
    occur_3way = occur_notable_class_3way + occur_notable_nonclass_3way
    occurNotable = occur_notable_class + occur_notable_nonclass

    
    # Calculate max possible interactions for class data
    maxClass1Ways = sum(len(values) for values in distinct_values_class.values())
    maxClass2Ways, maxClass3Ways, all_possible_2way_class, all_possible_3way_class = calculate_max_possible_interactions(distinct_values_class)

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 15 - {formatted_time}\n")

    # Calculate max possible interactions for nonclass data
    maxNonClass1Ways = sum(len(values) for values in distinct_values_nonclass.values())
    maxNonClass2Ways, maxNonClass3Ways, all_possible_2way_nonclass, all_possible_3way_nonclass = calculate_max_possible_interactions(distinct_values_nonclass)

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 16 - {formatted_time}\n")

    # Calculate max possible interactions for combined data
    maxAll1Ways = sum(len(distinct_values_class.get(key, set()).union(distinct_values_nonclass.get(key, set()))) for key in set(distinct_values_class) | set(distinct_values_nonclass))
    maxAll2Ways, maxAll3Ways, all_possible_2way_combos, all_possible_3way_combos = calculate_max_possible_interactions({key: distinct_values_class.get(key, set()).union(distinct_values_nonclass.get(key, set())) for key in set(distinct_values_class) | set(distinct_values_nonclass)})

        
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 17 - {formatted_time}\n")

        
    common_combinations = count_common_combinations(combination_counts_class, combination_counts_nonclass)
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 18 - {formatted_time}\n")

                
    numClass2Ways = len(combination_counts_class['2-way'])
    numClass3Ways = len(combination_counts_class['3-way'])
    numNonClass2Ways = len(combination_counts_nonclass['2-way'])
    numNonClass3Ways = len(combination_counts_nonclass['3-way'])


    listOfPrintOneWays, listOfPrintOneWaysVerbose = generate_output_statements(
        notable_1way_class_combos,
        notable_1way_nonclass_combos,
        variable_names,
        max_name_length,
    )
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 19 - {formatted_time}\n")


    # Generate output statements for 2-way combinations
    listOfPrintTwoWays, listOfPrintTwoWaysVerbose = generate_output_statements(
        notable_2way_class_combos,
        notable_2way_nonclass_combos,
        variable_names,
        max_name_length,
    )
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 20 - {formatted_time}\n")


    # Generate output statements for 3-way combinations
    listOfPrintThreeWays, listOfPrintThreeWaysVerbose = generate_output_statements(
        notable_3way_class_combos,
        notable_3way_nonclass_combos,
        variable_names,
        max_name_length,
    )

    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 21 - {formatted_time}\n")
    
    missing_2way_class, missing_3way_class, missing_2way_nonclass, missing_3way_nonclass = find_missing_combos(
        all_possible_2way_class, all_possible_3way_class,
        all_possible_2way_nonclass, all_possible_3way_nonclass,
        combination_counts_class, combination_counts_nonclass
    )
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 22 - {formatted_time}\n")

    
    combined_combination_counts = merge_combination_counts(combination_counts_class, combination_counts_nonclass)


    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 23 - {formatted_time}\n")



    notable_combination_counts = filter_notable_combinations(combined_combination_counts, {
        '1-way': notable_1way_class_combos + notable_1way_nonclass_combos,
        '2-way': notable_2way_class_combos + notable_2way_nonclass_combos,
        '3-way': notable_3way_class_combos + notable_3way_nonclass_combos
    })
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 24 - {formatted_time}\n")


    count1_notable, count2_notable, count3_notable = aggregate_counts_by_variable_notable(notable_combination_counts)


    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Checkpoint 25 - {formatted_time}\n")

    
    class_file = str(classFilePath)
    non_class_file = str(nonClassFilePath)

    ld2, lx2, ln2 = analyze_combos(combination_counts_class, combination_counts_nonclass, nrc, nrn)    
     
     #2-way NVCs
    c2, n2, d2 = notable_coverage_graph(notable_2way_class_combos, notable_2way_nonclass_combos)
     
     #3-way NVCs
    c3, n3, d3 = notable_coverage_graph(notable_3way_class_combos, notable_3way_nonclass_combos)

  
    data = {
        "lines1":           [lx2, ln2, ld2],
        "lines2":           [c2, n2, d2],
        "lines3":           [c3, n3, d3], 
        #"lines-nonNotable-class":combination_counts_class,      # Lines part 1
        #"lines-nonNotable-nonclass":combination_counts_nonclass,# Lines part 1
        #"lines-3-class": notable_3way_class_combos,             # Lines part 3
        #"lines-3-nonclass": notable_3way_nonclass_combos,       # Lines part 3
        "ClassFile":        class_file,
        "NonClassFile":     non_class_file,
        "canVerbose":       hasAHeader,
        "numVariables":     ncc,
        "classRows":        nrc,
        "nonClassRows":     nrn,
        "numDiffValues":    maxAll1Ways,
        "numIntersectVals": common_combinations['1-way'],
        "numUniqClass":     numUniqValsInClass,
        "numUniqNonClass":  numUniqValsInNonClass,
        "numNotable":       totalNotable,
        "occurNotable":     occurNotable,
        "avgValPerVar":     maxAll1Ways / ncc,
        "grid": [[maxAll1Ways, maxClass1Ways, maxNonClass1Ways, maxClass1Ways, maxNonClass1Ways, common_combinations['1-way'], sum_notable_1way, (sum_notable_1way / totalNotable) if totalNotable != 0 else 0, occur_1way, (occur_1way / occurNotable) if occurNotable != 0 else 0],
                 [maxAll2Ways, maxClass2Ways, maxNonClass2Ways, numClass2Ways, numNonClass2Ways, common_combinations['2-way'], sum_notable_2way, (sum_notable_2way / totalNotable) if totalNotable != 0 else 0, occur_2way, (occur_2way / occurNotable) if occurNotable != 0 else 0],
                 [maxAll3Ways, maxClass3Ways, maxNonClass3Ways, numClass3Ways, numNonClass3Ways, common_combinations['3-way'], sum_notable_3way, (sum_notable_3way / totalNotable) if totalNotable != 0 else 0, occur_3way, (occur_3way / occurNotable) if occurNotable != 0 else 0]],   
        "notable_1way_combinations_class": notable_1way_class_combos_ind,   
        "notable_2way_combinations_class": notable_2way_class_combos_ind,       # Lines part 2
        #"notable_3way_combinations_class": [],
        "notable_1way_combinations_nonclass": notable_1way_nonclass_combos_ind,
        "notable_2way_combinations_nonclass": notable_2way_nonclass_combos_ind, # Lines part 2
        #"notable_3way_combinations_nonclass": [],
        "print1Ways": listOfPrintOneWays,
        "print1WaysVerbose": listOfPrintOneWaysVerbose,
        "print2Ways": listOfPrintTwoWays,
        "print2WaysVerbose": listOfPrintTwoWaysVerbose,
        "print3Ways": listOfPrintThreeWays,
        "print3WaysVerbose": listOfPrintThreeWaysVerbose,
        "variableNames": variable_names,
        "count1": count1_notable,   # Bar graph
        "count2": count2_notable,   # Bar graph
        "count3": count3_notable,   # Bar graph
        "valFreqClass": valFreqClass,
        "valFreqNonClass": valFreqNonClass,
        "commonEntries": common_entries,
        "missing2WayClassValCombos": missing_2way_class,
        "missing3WayClassValCombos": missing_3way_class,
        "missing2WayNonClassValCombos": missing_2way_nonclass,
        "missing3WayNonClassValCombos": missing_3way_nonclass,
        "occurNotableLst":  [[num_uniq_1way_class, num_uniq_1way_nonclass, num_intersect_1way_c + num_intersect_1way_n],
                             [num_uniq_2way_class, num_uniq_2way_nonclass, num_intersect_2way_c + num_intersect_2way_n],
                             [num_uniq_3way_class, num_uniq_3way_nonclass, num_intersect_3way_c + num_intersect_3way_n]]    # Venn
    }        
    
         
#    # All VCs
#     ld2, lx2, ln2 = analyze_combos(combination_counts_class, combination_counts_nonclass, nrc, nrn)    
    
#     #2-way NVCs
#     notable_coverage_graph(notable_2way_class_combos, notable_2way_nonclass_combos, 2)
    
#     #3-way NVCs
#     notable_coverage_graph(notable_3way_class_combos, notable_3way_nonclass_combos, 3)
    
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S")
    with open("progress.txt", "a") as file:
        file.write(f"Returning Data - {formatted_time}\n")
        
    formatted_output = json.dumps(data)
    print(formatted_output)
    
    
# from matplotlib_venn import venn2

# # EXP: Display the NVC and All Value Combination Venn Diagrams
# def vennDiagram():
#     global data
    
#     grid = data['grid']
#     notableGrid = data['occurNotableLst']
    
#     fig, axs = plt.subplots(3, 4, gridspec_kw={'width_ratios': [0.4, 1, 0.5, 1]})
#     fig.suptitle("T-Ways           All Value Combinations                 Notable Value Combinations")
    
    
#     axs[0, 0].set_axis_off()  # Turn off the axes for this subplot
#     axs[1, 0].set_axis_off()  # Turn off the axes for this subplot
#     axs[2, 0].set_axis_off()  # Turn off the axes for this subplot

#     axs[0, 2].set_axis_off()  # Turn off the axes for this subplot
#     axs[1, 2].set_axis_off()  # Turn off the axes for this subplot
#     axs[2, 2].set_axis_off()  # Turn off the axes for this subplot

#     axs[0, 3].set_axis_off()  # Turn off the axes for this subplot
#     axs[1, 3].set_axis_off()  # Turn off the axes for this subplot
#     axs[2, 3].set_axis_off()  # Turn off the axes for this subplot


#     axs[0, 0].text(0.5, 0.5, "1", ha='center', fontsize=14, fontweight='bold')
#     axs[1, 0].text(0.5, 0.5, "2", ha='center', fontsize=14, fontweight='bold')
#     axs[2, 0].text(0.5, 0.5, "3", ha='center', fontsize=14, fontweight='bold')
        
#     for i in range(4):
#         if i == 2:
#             # axs[2][1].text(0.5, 0.5, '')
#             # axs[2][2].text(0.5, 0.5, '')
#             # axs[2][3].text(0.5, 0.5, '')
#             continue
#         if i == 3:
#             i -= 1
            
#         classVIs = grid[i][3] - grid[i][5]
#         nonClassVIs = grid[i][4] - grid[i][5]
#         intersectVIs = grid[i][5]
        
#         notableClassVIs = notableGrid[i][0]
#         notableNonClassVIs = notableGrid[i][1]
#         notableIntersectVIs = notableGrid[i][2]
        
#         # vd = venn2(subsets=(classVIs, nonClassVIs, intersectVIs), set_labels=('',''), ax=axs[i][1])
#         # nvd = venn2(subsets=(notableClassVIs, notableNonClassVIs, notableIntersectVIs), set_labels=('',''), ax=axs[i][2])        
        
#         # Check if all values are zero for both Venn diagrams
#         if all(value == 0 for value in [classVIs, nonClassVIs, intersectVIs]):
#             vd_text = "No Applicable Data"
#         else:
#             vd_text = ''
        
#         if all(value == 0 for value in [notableClassVIs, notableNonClassVIs, notableIntersectVIs]):
#             nvd_text = "No Applicable Data"
#         else:
#             nvd_text = ''

#         # Creates Venn diagrams or displays text based on the condition
        
#         if vd_text:
#             axs[i][1].text(0.5, 0.5, vd_text, ha='center', va='center', fontsize=12)
#         else:
#             vd = venn2(subsets=(classVIs, nonClassVIs, intersectVIs), set_labels=('',''), ax=axs[i][1])
            
#         if nvd_text:
#             axs[i][3].text(0.5, 0.5, nvd_text, ha='center', va='center', fontsize=12)
#         else:
#             nvd = venn2(subsets=(notableClassVIs, notableNonClassVIs, notableIntersectVIs), set_labels=('',''), ax=axs[i][3])
            
#         labels = ['Class', 'NonClass']        
        
#         def label_by_id(label, ID, vd, nvd):
#             if vd:
#                 num = vd.get_label_by_id(ID).get_text() 
#                 vd.get_label_by_id(ID).set_text(label + "\n" + num)
#                 if ID == "10": vd.get_label_by_id(ID).set_x(-1.01)
#                 if ID == "01": vd.get_label_by_id(ID).set_x(1.01)
            
#             if nvd:
#                 num = nvd.get_label_by_id(ID).get_text() 
#                 nvd.get_label_by_id(ID).set_text(label + "\n" + num)
#                 if ID == "10": nvd.get_label_by_id(ID).set_x(-1.01)
#                 if ID == "01": nvd.get_label_by_id(ID).set_x(1.01)

#         for label, ID in zip(labels, ["10", "01"]):
#             label_by_id(label, ID, vd if not vd_text else None, nvd if not nvd_text else None)

#         plt.tight_layout()
#     plt.show()



if __name__ == "__main__":      # Testing purposes only
    import argparse
    parser = argparse.ArgumentParser(description='Process some file paths and a cutoff value.')
    parser.add_argument('classFilePath', type=str, help='Path to the class file')
    parser.add_argument('nonClassFilePath', type=str, help='Path to the non-class file')
    parser.add_argument('xtimescutoff', type=int, help='Cutoff value for processing')
    
    args = parser.parse_args()
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python Calc_v4.py <classFilePath> <nonClassFilePath> <xtimescutoff>")
        sys.exit(1)
    
    classFilePath = sys.argv[1]
    nonClassFilePath = sys.argv[2]
    xtimescutoff = int(sys.argv[3])    
    getsetFiles()
    runData()


def process_files(classFP, nonClassFP, xcutoff):
    global classFilePath, nonClassFilePath, xtimescutoff, data
    
    classFilePath = Path(classFP)
    nonClassFilePath = Path(nonClassFP)
    xtimescutoff = xcutoff        
    getsetFiles()
    runData()
    return data
    

'''

% Define your file paths and parameters
classFilePath = 'C:\path\to\your\class\file';
nonClassFilePath = 'C:\path\to\your\nonclass\file';
xtimescutoff = 10;  % Example value, adjust as needed

% Ensure file paths are formatted correctly for Python
classFilePath = string(classFilePath);
nonClassFilePath = string(nonClassFilePath);
xtimescutoff = py.int(xtimescutoff);

% Add the directory containing your Python script to the Python path
insert(py.sys.path, int32(0), 'path_to_directory_containing_Calc_v4');

% Import and call the Python function
Calc_v4 = py.importlib.import_module('Calc_v4');
result = Calc_v4.process_files(classFilePath, nonClassFilePath, xtimescutoff);

% Handle the result (if any)
disp(result);

'''