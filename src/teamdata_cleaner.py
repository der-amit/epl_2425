import pandas as pd

def clean_table(file_path, new_columns, output_path = None):
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[0], axis = 1)
    df = df.iloc[1:].reset_index(drop = True)
    
    if len(df.columns) == len(new_columns):
        df.columns = new_columns
    else:
        print(f"Column mismatch! Table has {len(df.columns)} but {len(new_columns)} names were provided")
        
    if output_path:
        df.to_csv(output_path, index = False)
        print(f"Cleaned data saved to {output_path}")
        
    return df

#Define new columns
    
def_cols = ['Squad', 'Players_used', '90s', 'Tkl_attempted', 'TklW', 'Tkl_Def', 'Tkl_Mid', 'Tkl_Att',
           'Drib_Tkl', 'Drib_chal', 'Drib_Tkl%', 'Chal_Lost', 'Blocks', 'Shots_blocked', 
           'Pass_blocked', 'Interceptions', 'Tkl+Int', 'Clr', 'Err_shot']

gca_cols = ["Squad", "Players_used", "90s", "SCA", "SCA90", "SCA_Passlive", "SCA_PassDead", 
           "SCA_TO", "SCA_Sh", "SCA_Fld", "SCA_Def", "GCA", "GCA90", "GCA_PassLive", 
           "GCA_PassDead", "GCA_TO", "GCA_Sh", "GCA_Fld", "GCA_Def"]

shooting_col = ['Squad','Players_used','90s','Goals','Shots',
                'SoT','SoT%','Sh/90','SoT/90',	'G/Sh',	'G/SoT', 
                'Shot_Dist','Shot_FK','PK_made','PKatt',
                'xG','npxG','npxG/Sh','Goals-xG', 'np:Goals-xG']

standard_col = ['Squad', 'Players_used','Age',	'Poss',	'Matches',	'Starts',	
                'Min',	'90s',	'Goals','Assists', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 
                'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'Goals_90', 
                'Assists_90', 'G+A_90', 'G-PK_90', 'G+A-PK_90', 
                'xG_90', 'xAG_90', 'xG+xAG_90', 'npxG_90', 'npxG+xAG_90']

pass_col = [
    'Squad','Players_used', '90s',	'Pass_Cmp',	'Pass_Att',	'Pass_Cmp%','Pass_TotDist','Pass_PrgDist',
    'ShortPass_Cmp','ShortPass_Att','ShortPass_Cmp%','MedPass_Cmp','MedPass_Att','MedPass_Cmp%',	
    'LongPass_Cmp',	'LongPass_Att',	'LongPass_Cmp%',
    'Assists','xAG','xA','A-xAG', 'KeyPasses','Pass_FinalTrd','Pass_PenArea','Crs_PenArea','PrgP'
]

poss_col = [
    'Squad', 'Players_used', 'Poss', '90s', 
    'Touches', 'Touches_DefPen', 'Touches_Def3rd', 'Touches_Mid3rd', 'Touches_Att3rd', 'Touches_AttPen', 'Touches_Live', 
    'TO_Att', 'TO_Succ', 'TO_Succ%', 'TO_Tkld', 'TO_Tkld%', 
    'Carries', 'TotDist', 'PrgDist', 'PrgC', 'Carry1/3', 'Carry_PA', 'Mis', 'Dis', 'Pass_Rec', 'Pass_PrgR'
]


epl_defense_24 = clean_table('/Users/amitmishra/epl_2425/team_data/EPL_2425_stats_squads_defense_for.csv', def_cols, 'team_data/epl2425_defense_cleaned.csv')

epl_gca_24 = clean_table('/Users/amitmishra/epl_2425/team_data/EPL_2425_stats_squads_gca_for.csv', gca_cols,'team_data/epl2425_gca_cleaned.csv')

epl_shooting_24 = clean_table('/Users/amitmishra/epl_2425/team_data/EPL_2425_stats_squads_shooting_for.csv',shooting_col, 'team_data/epl2425_shooting_cleaned.csv')

epl_standard_24 = clean_table('/Users/amitmishra/epl_2425/team_data/EPL_2425_stats_squads_standard_for.csv', standard_col, 'team_data/epl2425_standard_cleaned.csv')

epl_passing_24 = clean_table('/Users/amitmishra/epl_2425/team_data/EPL_2425_stats_squads_passing_for.csv', pass_col, 
                             'team_data/epl2425_passing_cleaned.csv')

epl_poss_24 = clean_table('/Users/amitmishra/epl_2425/team_data/EPL_2425_stats_squads_possession_for.csv', poss_col,
                          'team_data/epl2425_possession_cleaned.csv')