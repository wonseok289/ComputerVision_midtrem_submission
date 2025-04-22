import argparse
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import rankdata
import numpy as np
import natsort
import os
import time
import psutil

def parse():
    parser = argparse.ArgumentParser(
        description='Generate tables from CSV file using make_tables function.')
    
    parser.add_argument('--input_csv', type=str,
                        help='Path to the input CSV file')
    parser.add_argument('--reference_model', type=str,
                        help='Reference column name for model (e.g., "yolov8n")')
    
    return parser.parse_args()


def reshape_df(input_csv_path, keep_columns, column_groups, groups_name, value_name):
    # Load CSV file
    df = pd.read_csv(input_csv_path, encoding='cp949')
    # Drop rows with missing values in specified columns
    df = df.dropna(subset=keep_columns + column_groups)
    # Preserve selected columns and reshape DataFrame
    df= df.melt(id_vars=keep_columns, value_vars=column_groups, var_name=groups_name, value_name=value_name)
    if "Iteration" in df.columns:
        df = df.copy()
        df["Iteration"] = pd.to_numeric(df["Iteration"], errors='coerce').astype('Int64')
    return df


def filter_dataframe_by_keys(df, row, column, rename_row, rename_col):
    # row와 column 값이 각각 rename_dict_row 및 rename_dict_col의 키에 해당하는 경우만 필터링
    filtered_df = df[
        df[row].isin(rename_row) & 
        df[column].isin(rename_col)
    ]
    return filtered_df


def filter_and_transform_formal_name(df, row, column, rename_row, rename_col, null_column):
    if isinstance(rename_col, dict) and isinstance(rename_row, dict):
        df = filter_dataframe_by_keys(df, row, column, list(rename_row.keys()), list(rename_col.keys()))
        df = transform_column_values(df, column, rename_col)
        df = transform_column_values(df, row, rename_row)
        null_column = rename_col[null_column]
    elif isinstance(rename_col, list) and isinstance(rename_row, list):
        df = filter_dataframe_by_keys(df, row, column, rename_row, rename_col)
    else:
        print('filter_and_transform_formal_name Error')
    return df, null_column


def transform_column_values(df, column, conversion_dict):
    """
    주어진 DataFrame의 특정 컬럼에 대해 값들을 사전에 정의된 딕셔너리를 사용하여 변환하는 함수.

    Parameters:
    df (pd.DataFrame): 변환할 DataFrame
    column (str): 변환할 대상 컬럼 이름
    conversion_dict (dict): 키-값 쌍으로 정의된 변환 딕셔너리

    Returns:
    pd.DataFrame: 변환된 DataFrame
    """
    if column in df.columns:
        df = df.copy()
        df[column] = df[column].map(conversion_dict).fillna(df[column])  # 일치하지 않는 값은 원래 값 유지
    return df


def reorder_dataframe(df, new_column_order, new_row_order) :
    if isinstance(new_column_order, dict) and isinstance(new_row_order, dict):
        new_column_order = list(new_column_order.values())
        new_row_order = list(new_row_order.values())
    elif isinstance(new_column_order, list) and isinstance(new_row_order, list):
        pass
    else:
        print('reorder_dataframe Error')
    if all(col in df.columns for col in new_column_order):
        df = df[new_column_order]
    else:
        raise ValueError("new_column_order에 포함된 모든 열이 DataFrame에 존재하지 않습니다.")

    # 행 순서 재정렬
    if len(df) == len(new_row_order):
        df = df.reindex(new_row_order)
    else:
        raise ValueError("new_row_order의 길이가 DataFrame의 행 수와 일치하지 않습니다.")

    return df


def get_unique_rows(df, row):
    """Get unique rows sorted using natsort."""
    return natsort.natsorted(df[row].unique())


def extract_values_by_column(row_subset, column, unique_columns):
    """Extract values for each unique column."""
    return {c: row_subset[row_subset[column] == c]['value'].tolist() for c in unique_columns}


def perform_paired_t_tests(null_values, values_by_column, unique_columns, null_column, significance_levels, common_length):
    """Perform paired t-tests and determine significance annotations."""
    significance_annotations = {}
    for c in unique_columns:
        if c != null_column and len(values_by_column[c]) >= common_length:
            t_stat, p_val = ttest_rel(null_values[:common_length], values_by_column[c][:common_length])
            # Determine significance annotation with △ or ▼
            annotation = ''
            for i, level in enumerate(significance_levels):
                if p_val < level:
                    if np.mean(values_by_column[c][:common_length]) > np.mean(null_values[:common_length]):
                        annotation = '△' * (i + 1)
                    else:
                        annotation = '▼' * (i + 1)
            significance_annotations[c] = annotation
    return significance_annotations


def calculate_ranks(value_matrix_dict, ranking_order='asc'):
    """
    Calculate ranks across different columns for each index (iteration).
    
    Args:
        value_matrix_dict (dict): Dictionary where keys are column names and values are lists of values.
        ranking_order (str): 'asc' for ascending, 'desc' for descending ranking.

    Returns:
        dict: Dictionary with average ranks for each column.
    """
    # Convert dictionary to matrix (list of rows where each row is one iteration's values across columns)
    columns = list(value_matrix_dict.keys())
    matrix = np.array([value_matrix_dict[col] for col in columns]).T  # Transpose to make rows represent iterations

    # Calculate ranks for each row
    ranks_per_row = [
        rankdata(row if ranking_order == 'asc' else [-val for val in row])
        for row in matrix
    ]
    
    # Average ranks across rows for each column
    avg_ranks = np.mean(ranks_per_row, axis=0)
    
    # Map average ranks back to column names
    return {columns[i]: avg_ranks[i] for i in range(len(columns))}


def apply_custom_formatting(mean_value, std_value, annotation, decimal_places, custom_fmt_template):
    plus_minus = "±"

    formatted_text = custom_fmt_template.format(
        mean_fmt=f"{mean_value:.{decimal_places}f}",
        std_fmt=f"{std_value:.{decimal_places}f}",
        significance=annotation,
    ).replace("±", plus_minus)

    return formatted_text


def analyze_row_column_combinations_with_ranking(df, row, column, unique_columns, ranking_order_list, null_column, significance_levels, decimal_places=2, custom_fmt_template='{underline_prefix}{bold_prefix}{mean_fmt}±{std_fmt} {significance}{bold_suffix}{underline_suffix}'):
    """Main function to analyze row-column combinations with ranking."""
    unique_rows = get_unique_rows(df, row)
    results = []

    for idx, r in enumerate(unique_rows):
        ranking_order = ranking_order_list[idx] if idx < len(ranking_order_list) else 'asc'
        row_subset = df[df[row] == r]
        
        values_by_column = extract_values_by_column(row_subset, column, unique_columns)
        common_length = min(len(values) for values in values_by_column.values() if len(values) > 0)
        if common_length == 0:
            continue  # Skip if no valid data is present

        value_matrix = [[values_by_column[c][idx] for c in unique_columns if len(values_by_column[c]) > idx] for idx in range(common_length)]
        
        ranked_results = calculate_ranks(values_by_column, ranking_order)
        null_values = values_by_column.get(null_column, [])
        significance_annotations = perform_paired_t_tests(null_values, values_by_column, unique_columns, null_column, significance_levels, common_length)

        # Calculate mean values for determining max mean per row
        mean_values = {c: np.mean(values_by_column[c]) for c in unique_columns if values_by_column[c]}
        sorted_mean_values = sorted(mean_values.items(), key=lambda item: item[1], reverse=True)
        max_mean_value = sorted_mean_values[0][1] if mean_values else None
        second_max_mean_value = sorted_mean_values[1][1] if len(sorted_mean_values) > 1 else None

        row_results = {}
        for c in unique_columns:
            ranks = ranked_results[c]
            if ranks:
                mean_rank = np.mean(ranks)
                mean_value = np.mean(values_by_column[c])
                std_value = np.std(values_by_column[c])
                annotation = significance_annotations.get(c, '')
                # Determine if bold or underline should be applied
                formatted_result = apply_custom_formatting(mean_value, std_value, annotation, decimal_places, custom_fmt_template)
                row_results[c] = formatted_result
            else:
                row_results[c] = "N/A"
        results.append(row_results)

    final_df = pd.DataFrame(results, index=unique_rows, columns=unique_columns)

    return final_df


def make_tables(input_csv_path, keep_columns, keep_measures, reduction, row, column, sheet, null_column, custom_fmt_template,
                significance_levels,decimal_places, transpose
               ):

    df = reshape_df(input_csv_path, keep_columns, keep_measures, groups_name='Measure Type', value_name='value')
    rename_row = dict(zip(df[row].unique(), df[row].unique()))
    rename_col = dict(zip(df[column].unique(), df[column].unique()))
    rename_sheet = dict(zip(df[sheet].unique(), df[sheet].unique()))
    ranking_order_list = ['desc'] * len(rename_row)
    
    second_df, null_column = filter_and_transform_formal_name(df, row, column, rename_row, rename_col, null_column)
    
    for a_sheet in rename_sheet.keys():
        third_df = second_df[(second_df[sheet] == a_sheet)]
        fourth_df = analyze_row_column_combinations_with_ranking(
            third_df, 
            row=row, 
            column=column, 
            unique_columns=third_df[column].unique(), 
            ranking_order_list=ranking_order_list,
            null_column=null_column, 
            significance_levels=significance_levels, 
            decimal_places=decimal_places, 
            custom_fmt_template=custom_fmt_template,
        )
        fourth_df = reorder_dataframe(fourth_df, rename_col, rename_row)
        fourth_df = fourth_df.T if transpose else fourth_df
        output_csv_path = input_csv_path.replace('.csv',f'_stat_{a_sheet}.csv')
        
        fourth_df.to_csv(output_csv_path, index=True, encoding='UTF-8')
        print(f"CSV 파일로 저장 완료: {output_csv_path}")


def main(args):
    input_csv_path = args.input_csv
    reference_column = args.reference_model

    keep_columns = ['Iteration', 'Dataset Name', 'Model Name']
    keep_measures = ['mAP@0.5', 'mAP@0.5:0.95', 'Mean Precision', 'Mean Recall']

    reduction = 'Iteration'
    row = 'Measure Type'
    column = 'Model Name'
    sheet = 'Dataset Name'

    custom_fmt_template = '{mean_fmt} ± {std_fmt} {significance}'
    significance_levels = [0.2, 0.1, 0.05, 0.01]
    decimal_places = 3
    transpose = True

    make_tables(input_csv_path, keep_columns, keep_measures, reduction, row, column, sheet,
                reference_column, custom_fmt_template, significance_levels, decimal_places, transpose)


if __name__ == '__main__':
    args = parse()
    main(args)