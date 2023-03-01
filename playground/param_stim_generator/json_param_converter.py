import json
import pandas as pd
import allensdk.core.json_utilities as ju


json_param_path = '/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage/cell_487664663/487664663_fit.json'
bounds = '/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/biophys_optimize/fit_styles/f9_fit_style.json'

template = '/global/cscratch1/sd/zladd/axonstandardized/playground/param_stim_generator/params_reference/params_allen_full_BEFORE_NEW_MODEL.csv'



curr_params = ju.read(json_param_path)
bounds = ju.read(bounds)

param_df = pd.concat([pd.DataFrame.from_dict(bounds['channels']), pd.DataFrame.from_dict(bounds['addl_params'])])
df = pd.DataFrame.from_dict(curr_params['genome'])

template = pd.read_csv(template)
unnamed_cols = [col for col in template.columns if "Unnamed" in col]
# template = template.drop(unnamed_cols, axis=1)
pd.DataFrame(columns=template.columns)


final_df = pd.DataFrame(columns=template.columns)
final_df['Param name'] = param_df['parameter'] + '_' +  param_df['mechanism'] + '_' + param_df['section']
final_df['Base value'] = df['value']
final_df['Lower bound'] = param_df['min']
final_df['Upper bound'] = param_df['max']
unfilled_cols = [col for col in template.columns if not col in ['Param name','Lower bound','Upper bound', 'Base value']]

# columns we don't care about
for col in unfilled_cols:
    final_df[col] = template.loc[:len(final_df), col]

final_df.to_csv('params_reference/params_allen_full.csv', index=False)