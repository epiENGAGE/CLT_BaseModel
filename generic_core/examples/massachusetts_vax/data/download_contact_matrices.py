"""
Written by Remy Pasco.
Downloads contact matrices for school, work, and total from the 
epydemix-data repo: https://github.com/epistorm/epydemix-data/tree/main
Function load_epydemix_population is available at
https://github.com/epistorm/epydemix/blob/main/epydemix/population/population.py

The data is saved in 2 formats: into separate csv files, and into a json files
so it can be added to other CLT inputs (as in 
https://github.com/LP-relaxation/CLT_BaseModel/blob/main/flu_instances/texas_input_files/common_subpop_params.json)

The epydemix package may need to be installed, using
pip install epydemix 

One user who was getting an error about "urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]", on a mac, needed to go to finder:  to folder Applications.  
Find the application for version of Python using, and double click on the file in that folder called "Install Certificates.command".  For others, 
copying the error and searching for that on the web is often a useful way to find solutions.  

List of possible locations available in
https://raw.githubusercontent.com/epistorm/epydemix-data/main/locations.csv
or 
https://github.com/epistorm/epydemix-data/tree/main/data
"""


from epydemix.population import load_epydemix_population # , Population
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Outputs land next to this script, whatever directory it is run from
# (previously cwd-relative, which only worked when run from MA_vax/).
CURRENT_DIRECTORY_PATH = Path(__file__).parent

# US_states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
#              "District-of-Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
#              "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
#              "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New-Hampshire", "New-Jersey",
#              "New-Mexico", "New-York", "North-Carolina", "North-Dakota", "Ohio", "Oklahoma", "Oregon",
#              "Pennsylvania", "Rhode-Island", "South-Carolina", "South-Dakota", "Tennessee", "Texas", "Utah",
#              "Vermont", "Virginia", "Washington", "West-Virginia", "Wisconsin", "Wyoming"]

SETTING_TO_JSON_KEY = {
    'work': 'work_contact_matrix', 
    'school': 'school_contact_matrix',
    'all': 'total_contact_matrix', 
    'community': 'community_contact_matrix', 
    'home': 'home_contact_matrix',
    }

###############################################################################
## Functions
def to_population_name(state_hyphen: str) -> str:
    """Convert 'New-York' -> 'United_States_New_York'."""
    core = state_hyphen.replace("-", "_")
    return f"United_States__{core}"


def check_state_exists(state_name):
    """ Check data exists for the chosen state, otherwise raise error"""
    try:
        _ = load_epydemix_population(
            population_name  = state_name,
            contacts_source  = "mistry_2021"
            )
        return True
    except Exception as e:
        print(f'An error occurred trying to fetch data for state: "{state_name}".')
        print('Error message:')
        print(e)
        
        return False
    

def check_age_groups_formatting(age_groups):
    if not(isinstance(age_groups, list)):
        msg = 'Age groups input is not a list.'
        raise ValueError(msg)
    
    # Check all entries are strings
    are_all_strings = all([
        isinstance(x, str) for x in age_groups
        ])
    if not(are_all_strings):
        msg = 'All age group entries need to be of type string.'
        raise ValueError(msg)
    
    # Check list starts at 0
    first_lower_bound = age_groups[0].split('-')[0]
    if first_lower_bound != '0':
        msg = 'The age groups must start at 0. First age group lower bound' +\
           f' is {first_lower_bound}.' 
        raise ValueError(msg)
        
    # Check last group is of the type 'x+' and starts at 84 or less
    last_group = age_groups[-1]
    if last_group[-1] != '+':
        msg = 'Last group should be of the form x+, eg: 65+, not "' + \
            str(last_group) + '".'
        raise ValueError(msg)
        
    last_age_start = int(last_group.replace('+', ''))
    if last_age_start > 84:
        msg = 'Last age group start should not be greater than 84.'
        raise ValueError(msg)
    
    
    # Check each group starts where previous one ends
    number_groups = len(age_groups)
    for group_i in age_groups:
        if not(isinstance(group_i, str)):
            msg = 'Invalid variable type for group:' + str(group_i) + \
                '. It should be a string.'
            raise ValueError(msg)
            
    
    for i in range(1, number_groups):
        if i == number_groups-1:
            new_group_start = age_groups[i].replace('+', '')
        else:
            new_group_start = int(age_groups[i].split('-')[0])
        
        previous_end = int(age_groups[i-1].split('-')[-1])
        
        
        if not(np.isclose(int(previous_end), int(new_group_start) - 1)):
            previous_group = age_groups[i-1]
            current_group = age_groups[i]
            msg = f'Issue with age groups, there is a gap between "{previous_group}"' +\
                f' and "{current_group}".'
            raise ValueError(msg)


def get_age_group_mappings(age_groups):
    """
    Go from age group interval in string formats (eg: "0-4") to list of
    strings in large contact matrix (eg: [ '0',  '1',  '2',  '3',  '4']).
    Returns a dictionary with age groups as keys, and lists of strings of 
    integers as values.
    
    """
    
    mapping = {}
    for group_i in age_groups:
        if group_i[-1] == '+':
            lower_bound = int(group_i.replace('+', ''))
            idx_list = [
                str(x) for x in range(lower_bound, 85)
                ]
            idx_list[-1] = '84+'
        
        else:
            lower_bound = int(group_i.split('-')[0])
            upper_bound = int(group_i.split('-')[1])
            idx_list = [
                str(x) for x in range(lower_bound, upper_bound + 1)
                ]
        
        mapping[group_i] = idx_list
        
    return mapping

    
def get_contact_data_single_state(state, age_group_mappings):
    """
    Check data for state exists.
    Download data. Save data in separate csv files, and into a combined
    json file.

    """
    state = str(state)
    state_name = to_population_name(state)
    
    # Check data exists before proceeding
    data_exists = check_state_exists(state_name)
    if not(data_exists):
        msg = f' !!!!!!! Cannot proceed for state "{state}". !!!!!!!'
        print(msg)
    
    
    # Download data
    contact_settings = ['work', 'school', 'all'] # ["school", "work", "home", "community", "all"]
    
    population = load_epydemix_population(
        population_name   = state_name,
        contacts_source   = "mistry_2021",
        layers            = contact_settings,
        age_group_mapping = age_group_mappings)
    
    
    matrix_dict = {}
    
    # Format and save data
    for mat_idx in range(len(contact_settings)):
        contact_matrix = population.contact_matrices[contact_settings[mat_idx]]
        df = pd.DataFrame(contact_matrix)

        filename = f"contact_matrix_{state}_Mistry2021_{contact_settings[mat_idx]}.csv"
        out_path = CURRENT_DIRECTORY_PATH / state / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, header=False) # no col/row labels, just matrix values
        print(f'Created {out_path}')

        matrix_dict[SETTING_TO_JSON_KEY[contact_settings[mat_idx]]] = contact_matrix.tolist()
    
    json_filename = f"contact_matrix_{state}_Mistry2021.json"
    json_path = CURRENT_DIRECTORY_PATH / state /json_filename
    with open(json_path, "w") as f:
        json.dump(matrix_dict, f, indent=4) 
    
    # Save age groups used
    age_groups_list = str(list(age_group_mappings.keys()))
    txt_filename = 'Age groups used.txt'
    txt_path = CURRENT_DIRECTORY_PATH / state /txt_filename
    with open(txt_path, 'w') as file:
        file.write(age_groups_list)
        

def get_contact_data_state_list(state_list, age_groups):
    check_age_groups_formatting(age_groups)
    age_group_mappings = get_age_group_mappings(age_groups)
    
    for state in state_list:
        print('\n\n######################## Starting for', str(state))
        get_contact_data_single_state(state, age_group_mappings)
        
## End of functions
###############################################################################


###############################################################################
##                                                                           ##
##  Required input: state names list and age groups                          ##
##                                                                           ##
###############################################################################
STATE_LIST = ['Massachusetts']

# Age groups should be of the form ["0-x"], ["x+1-y"], ... ["z+"] where z <= 84
# eg: ['0-4', '5-17', '18-49', '50-64', '65+']
AGE_GROUPS = ['0-0', '1-4', '5-12', '13-17', '18-49', '50-64', '65+']


###############################################################################
##                                                                           ##
##  Download data                                                            ##
##                                                                           ##
###############################################################################
get_contact_data_state_list(STATE_LIST, AGE_GROUPS)