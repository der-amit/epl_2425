import pandas as pd

def player_data_clean(filepath, new_cols, outputpath = None):
    df = pd.read_csv(filepath)
    df = df.drop(df.columns[0], axis = 1)
    df = df.iloc[1:].reset_index(drop = True)
    
    if len(df.columns) == len(new_cols):
        df.columns = new_cols
    else:
        print(f"Existing columns are: {len(df.columns)} but you passed: {len(new_cols)}")
        
    if outputpath:
        df.to_csv(outputpath, index = False)
        print(f"Cleaned data saved to: {outputpath}")
        
    return df

gca_cols = ['Player','Nation','Pos','Squad','Age','Born','90s',
            'SCA','SCA90', 
            'SCA_PassLive','SCA_PassDead','SCA_TO','SCA_Shot','SCA_Fld','SCA_DefAct',
            'GCA', 'GCA90', 
            'GCA_PassLive','GCA_PassDead','GCA_TO','GCA_Shot','GCA_Fld','GCA_DefAct', 'Matches']

defense_cols = ['Player','Nation',	'Pos', 'Squad', 'Age','Born', '90s', 
                'Tkl_attempted', 'TklW', 'Tkl_Def', 'Tkl_Mid', 'Tkl_Att',
           'Drib_Tkl', 'Drib_chal', 'Drib_Tkl%', 'Chal_Lost', 'Blocks', 'Shots_blocked', 
           'Pass_blocked', 'Interceptions', 'Tkl+Int', 'Clr', 'Err_shot','Matches']

pass_col = [
    'Player','Nation',	'Pos', 'Squad', 'Age','Born', '90s',	
    'Pass_Cmp',	'Pass_Att',	'Pass_Cmp%','Pass_TotDist'	,'Pass_PrgDist',
    'ShortPass_Cmp','ShortPass_Att','ShortPass_Cmp%','MedPass_Cmp',	'MedPass_Att',	'MedPass_Cmp%',	
    'LongPass_Cmp',	'LongPass_Att',	'LongPass_Cmp%',
    'Assists','xAG','xA','A-xAG', 'KeyPasses','Pass_FinalTrd','Pass_PenArea','Crs_PenArea','PrgP','Matches'
]

poss_col = [
    'Player','Nation',	'Pos', 'Squad', 'Age','Born', '90s',
    'Touches', 'Touches_DefPen', 'Touches_Def3rd', 'Touches_Mid3rd', 'Touches_Att3rd', 'Touches_AttPen', 'Touches_Live', 
    'TO_Att', 'TO_Succ', 'TO_Succ%', 'TO_Tkld', 'TO_Tkld%', 
    'Carries', 'TotDist', 'PrgDist', 'PrgC', 'Carry1/3', 'Carry_PA', 'Mis', 'Dis', 'Pass_Rec', 'Pass_PrgR', 'Matches'
]

#
gca_playerdata_2025 = player_data_clean('/Users/amitmishra/epl_2425/data/player_gca_2025_raw.csv', gca_cols, 
                                        'data/player_gca_2025_cleaned.csv')

defense_playerdata_2025 = player_data_clean('data/player_defense_2025_raw.csv', defense_cols, 
                                        'data/player_defense_2025_cleaned.csv')

poss_player_data_2025 = player_data_clean('data/player_possession_2025_raw.csv', poss_col, 
                                           'data/player_possession_2025_cleaned.csv')

pass_player_data_2025 = player_data_clean('data/player_passing_2025_raw.csv', pass_col, 'data/player_pass_2025_cleaned.csv')