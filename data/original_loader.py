import os
import json

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# We have to do some import magic for this to work on a Mac
# https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

import seaborn as sns

plt.tight_layout()

RUN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = r'./'

CENSUS_CSV_FILE_NAME = "raw_census_data.csv"
CENSUS_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
CENSUS_DATA_COLUMNS = [
    'age', 'workclass', 'fnwgt',
    'education', 'education-num',
    'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week',
    'native-country', 'income'
]

WINE_CSV_FILE_NAME = "winequality-red.csv"


def download_census_data_and_save_as_csv():
    if CENSUS_CSV_FILE_NAME in os.listdir(DATA_PATH):
        raise Exception("File already exists")

    with open(os.path.join(DATA_PATH, CENSUS_CSV_FILE_NAME), "w") as raw_census_data:
        raw_census_data.write(','.join(CENSUS_DATA_COLUMNS) + '\n')
        raw_census_data.writelines(
            requests.get(CENSUS_DATA_URL).text
        )


def output_graphs(df, output_dir='', columns=[], gtype='hist'):
    for column in columns:
        if gtype == 'hist':
            values = df[column]
        elif gtype == 'bar':
            values = df[column].value_counts()
        else:
            raise Exception('Unsupported gtype')
        values.plot(kind=gtype, title=column.upper())
        plt.savefig(os.path.join(output_dir, '%s_histogram.png' % column), bbox_inches="tight")
        plt.close()


def plot_heatmap(df, columns, output_dir='', file_name='', title=''):
    sns.heatmap(
        df[columns].corr(),
        annot=True,
        fmt='.2f',
    )
    plt.title(title)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches="tight")
    plt.close()


class CensusDataLoader(object):
    """
    CensusDataLoader: a bespoke data pipeline for the Census data found at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    """

    def __init__(self, df, pipeline=[]):
        # TODO: consider making pipeline immutable
        # TODO: consider writing finished data frame to file as csv for ease of replication.
        """
        NOTE: self.pipeline shouldn't need to change
        since we are not building an API to run this pipeline.
        However, if the pipeline needs to change,
        a user can update the attribute with more functions
        that take a data frame as input and return a data frame
        as output.
        Parameters:
            df (pandas.DataFrame): data frame that will be operated on
            pipeline (list): list containing functions that take in a pandas.DataFrame
            and return an updated pandas.DataFrame. The updated df is used to update the
            self.df.
        Returns: void
        """
        self.df = df
        if pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = [
                self.trim_strings,
                self.drop_missing_values,
                self.update_marital_status,
                # self.create_category_num_columns,
                self.add_from_united_states_column,
                self.create_dummy_columns,
                self.bucket_age_column,
                self.add_income_num_column,
            ]

    def apply_pipeline(self):
        """
        Moves through the list of pipeline functions and applies.
        This  assumes idempotent changes. Calling this multiple times
        will result in wasteful ops, but does not change the df.
        Returns:
            self (pandas.DataFrame)
        """
        for fxn in self.pipeline:
            self.df = fxn(self.df)
        return self.df

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, value):
        """
        Consider adding checks for the columns here
        to ensure the df has not mutated outside of the
        initial or intended schema.
        """
        self.__df = value

    @staticmethod
    def trim_strings(df):
        """
        Trim each element if it is a string
        operates against this data frame
        Args:
            df (pandas.DataFrame): input data frame. Assumes the data frame
            is of the form of the data frame this CensusDataLoader was initialized.
        Returns:
            df (pandas.DataFrame)
        """
        return df.applymap(
            lambda item: item.strip() if isinstance(item, str) else item)

    @staticmethod
    def drop_missing_values(df):
        """
        Drop missing values which are denoted by '?' in the data set.
        Args:
            df (pandas.DataFrame): input data frame. Assumes the data frame
            is of the form of the data frame this CensusDataLoader was initialized.
        Returns:
            df (pandas.DataFrame)
        """
        df = df[df['workclass'] != '?']
        df = df[df['occupation'] != '?']
        df = df[df['native-country'] != '?']
        return df

    @staticmethod
    def update_marital_status(df):
        """
        Reduce the marital-status column to either single or married for simplicity
        Args:
            df (pandas.DataFrame)
        Returns:
            df (pandas.DataFrame)
        """
        df['marital-status'] = df['marital-status'].replace(
            ['Never-married', 'Divorced', 'Separated', 'Widowed'],
            'Single'
        )
        df['marital-status'] = df['marital-status'].replace(
            ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'],
            'Married'
        )
        return df

    @staticmethod
    def add_from_united_states_column(df):
        """
        Reduce native-country to from United States or not
        Args:
            df (pandas.DataFrame)
        Returns:
            df (pandas.DataFrame)
        """
        df['from_united_states'] = df['native-country'].apply(lambda country: 1 if country == 'United-States' else 0)
        return df

    @staticmethod
    def create_category_num_columns(df):
        """
        Transform categorical (class) data into a numerical representation.
        Args:
            df (pandas.DataFrame): input data frame. Assumes the data frame
            is of the form of the data frame this CensusDataLoader was initialized.
        Returns:
            df (pandas.DataFrame)
        """
        category_maps = {}
        try:
            with open(os.path.join(RUN_PATH, 'preprocessors/census_category_maps.json'), 'r') as input_categories:
                category_maps = json.loads(input_categories.read())
        except Exception:
            category_maps = {
                'workclass': {key: idx for idx, key in enumerate(set(df['workclass']))},
                'marital-status': {key: idx for idx, key in enumerate(set(df['marital-status']))},
                'occupation': {key: idx for idx, key in enumerate(set(df['occupation']))},
                'relationship': {key: idx for idx, key in enumerate(set(df['relationship']))},
                'race': {key: idx for idx, key in enumerate(set(df['race']))},
                'sex': {key: idx for idx, key in enumerate(set(df['sex']))},
                'native-country': {key: idx for idx, key in enumerate(set(df['native-country']))},
                'income': {'<=50K': 0, '>50K': 1}
            }

        with open(os.path.join(RUN_PATH, 'preprocessors/census_category_maps.json'), 'w') as output_category_maps:
            output_category_maps.write(json.dumps(category_maps, indent=4))

        for col, category_map in category_maps.items():
            df[col + '_num'] = df[col].map(category_map)
        return df

    @staticmethod
    def create_dummy_columns(df):
        """
        Create dummy columns for categorical variables with many values that do not have an order
        Args:
            df (pandas.DataFrame)
        Returns:
            df (pandas.DataFrame)
        """
        df = df.join(pd.get_dummies(df[['workclass']], prefix='workclass', drop_first=True))
        df = df.join(pd.get_dummies(df['marital-status'], prefix='marital-status', drop_first=True))
        df = df.join(pd.get_dummies(df['occupation'], prefix='occupation', drop_first=True))
        df = df.join(pd.get_dummies(df['relationship'], prefix='relationship', drop_first=True))
        df = df.join(pd.get_dummies(df['race'], prefix='race', drop_first=True))
        df = df.join(pd.get_dummies(df['sex'], prefix='sex', drop_first=True))
        # df = df.join(pd.get_dummies(df['native-country'], prefix='native-country', drop_first=True))
        return df

    @staticmethod
    def add_income_num_column(df):
        df['income_num'] = df['income'].map({'<=50K': -1, '>50K': 1})
        return df

    @staticmethod
    def bucket_age_column(df):
        """
        Buckets the age based on the CensusDataLoader._bucket_age_column_helper function.
        Args:
            df (pandas.DataFrame): input data frame
        Returns:
            df (pandas.DataFrame): updated data frame
        """
        df['age_num'] = df['age'].apply(
            lambda age: CensusDataLoader._bucket_age_column_helper(age)
        )
        return df

    @staticmethod
    def _bucket_age_column_helper(row_age):
        age_buckets = {
            0: lambda age: age < 20,
            1: lambda age: 20 <= age < 30,
            2: lambda age: 30 <= age < 40,
            3: lambda age: 40 <= age < 50,
            4: lambda age: 50 <= age < 60,
            5: lambda age: age >= 60
        }
        for age_num, evaluator in age_buckets.items():
            if evaluator(row_age):
                return age_num
        raise Exception("No age mapped")


class WineDataLoader(object):
    """
    A bespoke data pipeline operating on a pandas.DataFrame via a list of operations.
    Data found at https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009/version/2
    """

    def __init__(self, df, pipeline=[]):
        self.df = df
        if pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = [
                self.dropna,
                self.reclassify_quality_v2
            ]

    def apply_pipeline(self):
        """
        Moves through the list of pipeline functions and applies.
        This  assumes idempotent changes. Calling this multiple times
        will result in wasteful ops, but does not change the df.
        Returns:
            self (pandas.DataFrame)
        """
        for fxn in self.pipeline:
            self.df = fxn(self.df)
        try:
            with open(os.path.join(RUN_PATH, 'preprocessors/wine_full_column_list.txt'), 'w') as column_list:
                column_list.write('\n'.join(self.df.columns))
        except Exception as e:
            print('Exception writing census columns to file during preprocessing with error %s' % e)
        return self.df

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, value):
        """
        Consider adding checks for the columns here
        to ensure the df has not mutated outside of the
        initial or intended schema.
        """
        self.__df = value

    @staticmethod
    def dropna(df):
        return df.dropna()

    @staticmethod
    def reclassify_quality(df):
        bins = (2, 6.5, 8)
        group_names = ['Low Quality', 'High Quality']
        df['quality_num'] = pd.cut(df['quality'], bins=bins, labels=group_names)
        return df

    @staticmethod
    def reclassify_quality_v2(df):
        df['quality_num'] = df['quality'].map(lambda val: 0 if val < 6.5 else 1)
        return df


if __name__ == '__main__':
    # print(DATA_PATH)
    # download_census_data_and_save_as_csv()

    feature_cols = ['age_num', 'education-num', 'marital-status_Single',
                    'hours-per-week', 'capital-gain',
                    'capital-loss', 'sex_Male', 'from_united_states']

    target = 'income_num'

    census_df = pd.read_csv(os.path.join(DATA_PATH, CENSUS_CSV_FILE_NAME))
    data_loader = CensusDataLoader(census_df)

    loaded_data = data_loader.apply_pipeline()

    # TODO: Consider stratification
    x_train, x_test, y_train, y_test = train_test_split(
        loaded_data[feature_cols],
        loaded_data[target],
        random_state=0,
        test_size=0.2
    )

    x_train, x_validate, y_train, y_validate = train_test_split(
        x_train,
        y_train,
        random_state=0,
        test_size=0.5
    )

    test_set = pd.concat([x_test, y_test], axis=1)
    train_set = pd.concat([x_train, y_train], axis=1)
    validate_set = pd.concat([x_validate, y_validate], axis=1)

    test_set.to_csv('./{}_test.csv'.format('census'), index=False, header=False)
    train_set.to_csv('./{}_train.csv'.format('census'), index=False, header=False)
    validate_set.to_csv('./{}_validate.csv'.format('census'), index=False, header=False)
